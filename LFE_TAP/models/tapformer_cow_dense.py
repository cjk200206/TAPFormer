from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from LFE_TAP.models.cow_dense_head import DenseWarpTrackingHead
from LFE_TAP.models.fusionFormer import Fusionformer, FusionformerFrameAnchor


class TAPFormerCowDense(nn.Module):
    """TAPFormer fusion front-end with a CoWTracker-style dense warping head."""

    def __init__(
        self,
        window_size=16,
        stride=4,
        corr_radius=3,
        corr_levels=3,
        backbone="basic",
        num_heads=8,
        hidden_size=384,
        space_depth=3,
        time_depth=3,
        cow_refine_model="vits",
        cow_refine_patch_size=4,
        cow_refine_blocks=None,
        cow_temporal_interleave_stride=2,
        cow_tracking_down_ratio=2,
        cow_limit_flow=True,
        cow_max_flow_update_ratio=0.15,
        cow_max_flow_magnitude_ratio=1.0,
        cow_refine_checkpoint=False,
        cow_info_update_mode="direct",
        cow_tapir_init=False,
        cow_tapir_init_stride=16,
        cow_tapir_init_temperature=20.0,
        cow_tapir_init_radius=5,
        cow_tapir_init_chunk_size=64,
        cow_frontend_type="base",
        cow_anchor_state_mix=0.7,
        cow_anchor_skip_mix=0.7,
        **_,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = int(stride)
        self.cow_tracking_down_ratio = int(cow_tracking_down_ratio)
        self.latent_dim = 128
        self.model_resolution = (384, 512)
        self.cow_frontend_type = str(cow_frontend_type).lower().strip()
        if self.cow_frontend_type not in {"base", "frame_anchor"}:
            raise ValueError(
                f"Unsupported cow_frontend_type={cow_frontend_type}. Use one of: base, frame_anchor."
            )
        frontend_kwargs = dict(
            image_size=self.model_resolution,
            out_dim=self.latent_dim,
            mlp_dim=512,
            stride=self.stride,
            depth=2,
        )
        if self.cow_frontend_type == "frame_anchor":
            frontend_kwargs.update(
                anchor_state_mix=float(cow_anchor_state_mix),
                anchor_skip_mix=float(cow_anchor_skip_mix),
            )
            self.fusion_block = FusionformerFrameAnchor(**frontend_kwargs)
        else:
            self.fusion_block = Fusionformer(**frontend_kwargs)
        self.dense_head = DenseWarpTrackingHead(
            feature_dim=self.latent_dim,
            down_ratio=self.cow_tracking_down_ratio,
            refine_model=cow_refine_model,
            refine_patch_size=cow_refine_patch_size,
            refine_blocks=cow_refine_blocks,
            temporal_interleave_stride=cow_temporal_interleave_stride,
            limit_flow=bool(cow_limit_flow),
            max_flow_update_ratio=float(cow_max_flow_update_ratio),
            max_flow_magnitude_ratio=float(cow_max_flow_magnitude_ratio),
            refine_checkpoint=bool(cow_refine_checkpoint),
            info_update_mode=cow_info_update_mode,
            tapir_init=bool(cow_tapir_init),
            tapir_init_stride=int(cow_tapir_init_stride),
            tapir_init_temperature=float(cow_tapir_init_temperature),
            tapir_init_radius=int(cow_tapir_init_radius),
            tapir_init_chunk_size=int(cow_tapir_init_chunk_size),
        )

    def _validate_inputs(self, rgbs, events, queries, iters):
        B, T, C_event, H, W = events.shape
        _, N, _ = queries.shape
        _, _, C_img, _, _ = rgbs.shape
        if B != 1:
            raise AssertionError("TAPFormerCowDense currently follows TAPFormer training and expects batch_size == 1")
        if H % self.stride != 0 or W % self.stride != 0:
            raise AssertionError("Input height/width must be divisible by model.stride")
        if H % self.cow_tracking_down_ratio != 0 or W % self.cow_tracking_down_ratio != 0:
            raise AssertionError("Input height/width must be divisible by model.cow_tracking_down_ratio")
        if iters is None:
            raise ValueError("TAPFormerCowDense.forward requires an explicit iters argument.")
        return B, T, C_event, H, W, N, C_img

    def _prepare_query_xy(self, queries: torch.Tensor) -> torch.Tensor:
        queried_frames = queries[:, :, 0].long()
        if torch.any(queried_frames != 0):
            raise ValueError(
                "TAPFormerCowDense expects all query frames to be 0. "
                "Use a config with dataset.sample_vis_1st_frame: true."
            )
        return queries[..., 1:3]

    def _build_fmaps(self, fmaps_out, B: int, T: int, H: int, W: int, dtype: torch.dtype) -> torch.Tensor:
        H_stride, W_stride = H // self.stride, W // self.stride
        H_track, W_track = H // self.cow_tracking_down_ratio, W // self.cow_tracking_down_ratio
        fmap = fmaps_out[0] if isinstance(fmaps_out, list) else fmaps_out
        fmap = fmap.permute(0, 2, 3, 1)
        fmap = fmap / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmap), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=fmap.device, dtype=fmap.dtype),
            )
        )
        fmap = fmap.permute(0, 3, 1, 2).reshape(B * T, self.latent_dim, H_stride, W_stride)
        if (H_stride, W_stride) != (H_track, W_track):
            fmap = F.interpolate(
                fmap,
                size=(H_track, W_track),
                mode="bilinear",
                align_corners=True,
            )
        fmap = fmap.reshape(B, T, self.latent_dim, H_track, W_track)
        return fmap.to(dtype)

    def _encode_features(self, rgbs, events, img_ifnew=None):
        B, T, C_event, H, W = events.shape
        _, _, C_img, _, _ = rgbs.shape
        rgbs_norm = 2 * (rgbs / 255.0) - 1.0
        events_norm = 2 * events - 1.0
        dtype = rgbs_norm.dtype
        if img_ifnew is None:
            img_ifnew = torch.ones(T, device=rgbs.device, dtype=rgbs.dtype)

        fmaps_out = self.fusion_block(
            rgbs_norm.reshape(-1, C_img, H, W),
            events_norm.reshape(-1, C_event, H, W),
            img_ifnew,
        )
        return self._build_fmaps(fmaps_out, B, T, H, W, dtype)

    def _reset_fusion_state(self):
        transunet = getattr(self.fusion_block, "transunet", None)
        if transunet is None:
            return
        if hasattr(transunet, "xe_history"):
            transunet.xe_history = []
        if hasattr(transunet, "x_e_pre"):
            transunet.x_e_pre = None
        if hasattr(transunet, "x_out_"):
            transunet.x_out_ = None
        if hasattr(transunet, "x_anchor_"):
            transunet.x_anchor_ = None
        if hasattr(transunet, "x1_anchor_"):
            transunet.x1_anchor_ = None
        if hasattr(transunet, "x0_anchor_"):
            transunet.x0_anchor_ = None

    def _encode_window_features(self, rgbs, events, img_ifnew=None, reset_state=False):
        if reset_state:
            self._reset_fusion_state()
        return self._encode_features(rgbs, events, img_ifnew=img_ifnew)

    def _forward_window(
        self,
        features: torch.Tensor,
        query_xy: torch.Tensor,
        image_size: Tuple[int, int],
        iters: int,
        init_track: torch.Tensor | None = None,
        init_vis: torch.Tensor | None = None,
        init_conf: torch.Tensor | None = None,
        init_valid_mask: torch.Tensor | None = None,
        first_frame_features: torch.Tensor | None = None,
        return_debug: bool = False,
    ):
        dense_debug = None
        dense_outputs = self.dense_head(
            features,
            image_size=image_size,
            iters=iters,
            init_track=init_track,
            init_vis=init_vis,
            init_conf=init_conf,
            init_valid_mask=init_valid_mask,
            first_frame_features=first_frame_features,
            return_debug=return_debug,
        )
        if return_debug:
            dense_coords, dense_vis_logits, dense_conf_logits, dense_debug = dense_outputs
        else:
            dense_coords, dense_vis_logits, dense_conf_logits = dense_outputs

        coord_preds: List[torch.Tensor] = [self._sample_dense(pred, query_xy) for pred in dense_coords]
        vis_preds: List[torch.Tensor] = [self._sample_dense_scalar(pred, query_xy) for pred in dense_vis_logits]
        conf_preds: List[torch.Tensor] = [self._sample_dense_scalar(pred, query_xy) for pred in dense_conf_logits]

        coords_predicted = coord_preds[-1]
        vis_predicted = torch.sigmoid(vis_preds[-1])
        conf_predicted = torch.sigmoid(conf_preds[-1])
        return coords_predicted, vis_predicted, conf_predicted, coord_preds, vis_preds, conf_preds, dense_debug

    def _sample_dense(self, dense: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
        # dense: [B,T,H,W,C], query_xy: [B,N,2] in original image coordinates.
        B, T, H, W, C = dense.shape
        N = query_xy.shape[1]
        x = 2 * query_xy[..., 0] / max(W - 1, 1) - 1
        y = 2 * query_xy[..., 1] / max(H - 1, 1) - 1
        grid = torch.stack([x, y], dim=-1)[:, None].expand(B, T, N, 2)
        grid = grid.reshape(B * T, N, 1, 2)
        sampled = F.grid_sample(
            dense.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W),
            grid,
            align_corners=True,
            padding_mode="border",
        )
        return sampled.squeeze(-1).permute(0, 2, 1).reshape(B, T, N, C)

    def _sample_dense_scalar(self, dense: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
        return self._sample_dense(dense.unsqueeze(-1), query_xy)[..., 0]

    @torch.no_grad()
    def forward_dense_debug(self, rgbs, events, iters=None, img_ifnew=None):
        B, T, _, H, W = events.shape
        if B != 1:
            raise AssertionError("TAPFormerCowDense.forward_dense_debug expects batch_size == 1")
        if H % self.stride != 0 or W % self.stride != 0:
            raise AssertionError("Input height/width must be divisible by model.stride")
        if H % self.cow_tracking_down_ratio != 0 or W % self.cow_tracking_down_ratio != 0:
            raise AssertionError("Input height/width must be divisible by model.cow_tracking_down_ratio")
        if iters is None:
            raise ValueError("TAPFormerCowDense.forward_dense_debug requires an explicit iters argument.")
        features = self._encode_features(rgbs, events, img_ifnew=img_ifnew)
        dense_outputs = self.dense_head(
            features,
            image_size=(H, W),
            iters=int(iters),
            return_debug=True,
        )
        _, _, _, dense_debug = dense_outputs
        return dense_debug

    def forward(
        self,
        rgbs,
        events,
        queries,
        iters=None,
        img_ifnew=None,
        feat_init=None,
        is_train=False,
        reference_rgbs=None,
        reference_events=None,
    ):
        _, T, _, H, W, _, _ = self._validate_inputs(rgbs, events, queries, iters)
        queried_frames = queries[:, :, 0].long()
        query_xy = self._prepare_query_xy(queries)
        features = self._encode_features(rgbs, events, img_ifnew=img_ifnew)
        first_frame_features = None
        if reference_rgbs is not None and reference_events is not None:
            reference_ifnew = torch.ones(
                reference_rgbs.shape[1],
                device=reference_rgbs.device,
                dtype=reference_rgbs.dtype,
            )
            first_frame_features = self._encode_features(
                reference_rgbs,
                reference_events,
                img_ifnew=reference_ifnew,
            )
        coords_predicted, vis_predicted, conf_predicted, coord_preds, vis_preds, conf_preds, _ = self._forward_window(
            features,
            query_xy,
            image_size=(H, W),
            iters=int(iters),
            first_frame_features=first_frame_features,
        )

        if is_train:
            valid_mask = queried_frames[:, None] <= torch.arange(0, T, device=queries.device)[None, :, None]
            train_data = ([coord_preds], [vis_preds], [conf_preds], valid_mask)
        else:
            train_data = None

        return coords_predicted, vis_predicted, conf_predicted, train_data
