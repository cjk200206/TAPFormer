from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ):
        super().__init__()
        if refine_model not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported cow refine model: {refine_model}")
        self.down_ratio = int(down_ratio)
        self.iter_dim = MODEL_CONFIGS[refine_model]["features"]

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

    def forward(self, features: torch.Tensor, image_size: Tuple[int, int], iters: Optional[int] = None):
        B, T, _, H, W = features.shape
        if iters is None:
            raise ValueError("DenseWarpTrackingHead.forward requires an explicit iters argument.")
        n_iters = int(iters)

        fmap = self.fmap_conv(features.reshape(B * T, -1, H, W)).view(B, T, self.iter_dim, H, W)
        frame0 = fmap[:, 0:1].expand(B, T, -1, -1, -1)
        net = self.hidden_conv(torch.cat([frame0, fmap], dim=2).reshape(B * T, -1, H, W))
        net = net.view(B, T, self.iter_dim, H, W)
        flow = torch.zeros(B, T, 2, H, W, device=features.device, dtype=features.dtype)

        track_preds, vis_preds, conf_preds = [], [], []
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
            refine_out = self.refine_net(refine_inp)["out"]
            net = self.refine_transform(torch.cat([refine_out, net], dim=2).reshape(B * T, -1, H, W))
            net = net.view(B, T, self.iter_dim, H, W)

            update = self.flow_head(net.reshape(B * T, self.iter_dim, H, W)).view(B, T, 4, H, W)
            flow = flow + update[:, :, :2]
            info = update[:, :, 2:]
            weight = 0.25 * self.upsample_weight(net.reshape(B * T, self.iter_dim, H, W)).view(B, T, -1, H, W)
            flow_up, info_up = self._upsample_predictions(flow, info, weight)
            track_preds.append(self._flow_to_tracks(flow_up, image_size))
            vis_preds.append(info_up[:, :, 0])
            conf_preds.append(info_up[:, :, 1])

        return track_preds, vis_preds, conf_preds
