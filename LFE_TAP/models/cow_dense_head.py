from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from LFE_TAP.models.cow_refine import VideoRefineTransformer, MODEL_CONFIGS


def coords_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)


def bilinear_sampler(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    # coords: [B, H, W, 2] in pixel-center coordinates.
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / max(W - 1, 1) - 1
    ygrid = 2 * ygrid / max(H - 1, 1) - 1
    return F.grid_sample(img, torch.cat([xgrid, ygrid], dim=-1), align_corners=True, padding_mode="border")


class DenseWarpTrackingHead(nn.Module):
    """CoWTracker-style dense warping head for fused TAPFormer features."""

    def __init__(
        self,
        feature_dim: int,
        down_ratio: int = 4,
        refine_model: str = "vits",
        refine_patch_size: int = 4,
        refine_blocks: Optional[int] = None,
        temporal_interleave_stride: int = 2,
        limit_flow: bool = False,
        max_flow_update_ratio: float = 0.15,
        max_flow_magnitude_ratio: float = 1.0,
        refine_checkpoint: bool = False,
        info_update_mode: str = "direct",
    ):
        super().__init__()
        if refine_model not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported cow refine model: {refine_model}")
        self.down_ratio = int(down_ratio)
        self.iter_dim = MODEL_CONFIGS[refine_model]["features"]
        self.limit_flow = bool(limit_flow)
        self.max_flow_update_ratio = float(max_flow_update_ratio)
        self.max_flow_magnitude_ratio = float(max_flow_magnitude_ratio)
        self.refine_checkpoint = bool(refine_checkpoint)
        self.info_update_mode = str(info_update_mode).lower().strip()
        if self.info_update_mode not in {"direct", "iterative"}:
            raise ValueError(
                f"Unsupported info_update_mode={info_update_mode}. Use one of: direct, iterative."
            )
        if self.limit_flow:
            if self.max_flow_update_ratio <= 0.0:
                raise ValueError("max_flow_update_ratio must be positive when limit_flow=True.")
            if self.max_flow_magnitude_ratio <= 0.0:
                raise ValueError("max_flow_magnitude_ratio must be positive when limit_flow=True.")

        self.fmap_conv = nn.Conv2d(feature_dim, self.iter_dim, 1)
        self.hidden_conv = nn.Conv2d(self.iter_dim * 2, self.iter_dim, 1)
        self.warp_linear = nn.Conv2d(3 * self.iter_dim + 2, self.iter_dim, 1)
        self.refine_net = VideoRefineTransformer(
            model_name=refine_model,
            input_dim=self.iter_dim,
            patch_size=refine_patch_size,
            temporal_interleave_stride=temporal_interleave_stride,
            num_blocks=refine_blocks,
        )
        self.refine_transform = nn.Conv2d(self.iter_dim * 2, self.iter_dim, 1)
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 4, 1),
        )
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, (self.down_ratio**2) * 9, 1),
        )

    def _upsample_single(self, flow: torch.Tensor, info: torch.Tensor, mask: torch.Tensor):
        B, _, H, W = flow.shape
        factor = self.down_ratio
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1).view(B, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1).view(B, info.shape[1], 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2).permute(0, 1, 4, 2, 5, 3)
        return (
            up_flow.reshape(B, 2, factor * H, factor * W),
            up_info.reshape(B, info.shape[1], factor * H, factor * W),
        )

    def _upsample_predictions(self, flow: torch.Tensor, info: torch.Tensor, weight: torch.Tensor):
        flow_ups, info_ups = [], []
        for t in range(flow.shape[1]):
            flow_up, info_up = self._upsample_single(flow[:, t], info[:, t], weight[:, t])
            flow_ups.append(flow_up)
            info_ups.append(info_up)
        return torch.stack(flow_ups, dim=1), torch.stack(info_ups, dim=1)

    def _flow_to_tracks(self, flow: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        B, T, _, H, W = flow.shape
        H_img, W_img = image_size
        if (H, W) != (H_img, W_img):
            flow = F.interpolate(
                flow.reshape(B * T, 2, H, W),
                size=(H_img, W_img),
                mode="bilinear",
                align_corners=True,
            ).reshape(B, T, 2, H_img, W_img)
        grid = coords_grid(B * T, H_img, W_img, flow.device, flow.dtype).view(B, T, 2, H_img, W_img)
        return (grid + flow).permute(0, 1, 3, 4, 2)

    def _init_flow(
        self,
        init_track: Optional[torch.Tensor],
        B: int,
        T: int,
        H: int,
        W: int,
        H_img: int,
        W_img: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if init_track is None:
            return torch.zeros(B, T, 2, H, W, device=device, dtype=dtype)

        coords = coords_grid(B * T, H_img, W_img, device, dtype).view(B, T, 2, H_img, W_img)
        init_track = init_track.to(device=device, dtype=dtype).permute(0, 1, 4, 2, 3)
        disp = init_track - coords
        disp = (disp / float(self.down_ratio)).view(B * T, 2, H_img, W_img)
        disp = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=True)
        return disp.view(B, T, 2, H, W)

    def _init_info(
        self,
        init_vis: Optional[torch.Tensor],
        init_conf: Optional[torch.Tensor],
        B: int,
        T: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        info = torch.zeros(B * T, 2, H, W, device=device, dtype=dtype)
        for idx, init_prob in enumerate((init_vis, init_conf)):
            if init_prob is None:
                continue
            prob = init_prob.to(device=device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
            prob = prob.view(B * T, 1, prob.shape[-2], prob.shape[-1])
            prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=True)
            info[:, idx : idx + 1] = torch.logit(prob)
        return info.view(B, T, 2, H, W)

    def _format_predictions(
        self,
        flow: torch.Tensor,
        info: torch.Tensor,
        net: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = flow.shape[:2]
        weight = 0.25 * self.upsample_weight(net.reshape(B * T, self.iter_dim, flow.shape[-2], flow.shape[-1])).view(
            B, T, -1, flow.shape[-2], flow.shape[-1]
        )
        flow_up, info_up = self._upsample_predictions(flow, info, weight)
        dense_tracks = self._flow_to_tracks(flow_up, image_size)
        return dense_tracks, info_up[:, :, 0], info_up[:, :, 1]

    def _flow_limits(self, flow: torch.Tensor) -> Tuple[float, float]:
        longer_side = float(max(flow.shape[-2], flow.shape[-1]))
        update_limit = longer_side * self.max_flow_update_ratio
        total_limit = longer_side * self.max_flow_magnitude_ratio
        return update_limit, total_limit

    def _apply_flow_update(self, flow: torch.Tensor, raw_delta: torch.Tensor) -> torch.Tensor:
        if not self.limit_flow:
            return flow + raw_delta

        update_limit, total_limit = self._flow_limits(flow)
        delta = update_limit * torch.tanh(raw_delta / update_limit)
        return (flow + delta).clamp(-total_limit, total_limit)

    def _apply_info_update(self, info: torch.Tensor, raw_info: torch.Tensor) -> torch.Tensor:
        if self.info_update_mode == "iterative":
            return info + raw_info
        return raw_info

    def _run_refine_net(self, refine_inp: torch.Tensor) -> torch.Tensor:
        def refine_forward(x: torch.Tensor) -> torch.Tensor:
            return self.refine_net(x)["out"]

        if self.training and refine_inp.requires_grad and self.refine_checkpoint:
            return checkpoint(refine_forward, refine_inp, use_reentrant=False)
        return refine_forward(refine_inp)

    def forward(
        self,
        features: torch.Tensor,
        image_size: Tuple[int, int],
        iters: Optional[int] = None,
        init_track: Optional[torch.Tensor] = None,
        init_vis: Optional[torch.Tensor] = None,
        init_conf: Optional[torch.Tensor] = None,
        first_frame_features: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        B, T, _, H, W = features.shape
        if iters is None:
            raise ValueError("DenseWarpTrackingHead.forward requires an explicit iters argument.")
        n_iters = int(iters)

        H_img, W_img = image_size
        fmap = self.fmap_conv(features.reshape(B * T, -1, H, W)).view(B, T, self.iter_dim, H, W)
        if first_frame_features is not None:
            frame0_base = self.fmap_conv(first_frame_features.reshape(B, -1, H, W)).view(B, 1, self.iter_dim, H, W)
        else:
            frame0_base = fmap[:, 0:1]
        frame0 = frame0_base.expand(B, T, -1, -1, -1)
        net = self.hidden_conv(torch.cat([frame0, fmap], dim=2).reshape(B * T, -1, H, W))
        net = net.view(B, T, self.iter_dim, H, W)
        flow = self._init_flow(init_track, B, T, H, W, H_img, W_img, features.device, features.dtype)
        info = self._init_info(init_vis, init_conf, B, T, H, W, features.device, features.dtype)

        track_preds, vis_preds, conf_preds = [], [], []
        dense_debug = None
        base_coords = coords_grid(B * T, H, W, features.device, features.dtype)
        for _ in range(n_iters):
            flow = flow.detach()
            coords = base_coords + flow.reshape(B * T, 2, H, W)
            warped = bilinear_sampler(
                fmap.reshape(B * T, self.iter_dim, H, W),
                coords.permute(0, 2, 3, 1),
            ).view(B, T, self.iter_dim, H, W)
            refine_inp = self.warp_linear(torch.cat([frame0, warped, net, flow], dim=2).reshape(B * T, -1, H, W))
            refine_inp = refine_inp.view(B, T, self.iter_dim, H, W)
            refine_out = self._run_refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_out, net], dim=2).reshape(B * T, -1, H, W))
            net = net.view(B, T, self.iter_dim, H, W)

            update = self.flow_head(net.reshape(B * T, self.iter_dim, H, W)).view(B, T, 4, H, W)
            flow = self._apply_flow_update(flow, update[:, :, :2])
            info = self._apply_info_update(info, update[:, :, 2:])
            dense_tracks, dense_vis, dense_conf = self._format_predictions(flow, info, net, image_size)
            track_preds.append(dense_tracks)
            vis_preds.append(dense_vis)
            conf_preds.append(dense_conf)
            if return_debug:
                dense_debug = {
                    "dense_tracks": dense_tracks,
                    "dense_vis": torch.sigmoid(dense_vis),
                    "dense_conf": torch.sigmoid(dense_conf),
                }

        if return_debug:
            return track_preds, vis_preds, conf_preds, dense_debug
        return track_preds, vis_preds, conf_preds
