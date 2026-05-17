from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from LFE_TAP.models.cow_dense_head import DenseWarpTrackingHead
from LFE_TAP.models.fusionFormer import Fusionformer


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
        **_,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = int(stride)
        self.latent_dim = 128
        self.model_resolution = (384, 512)
        self.fusion_block = Fusionformer(
            image_size=self.model_resolution,
            out_dim=self.latent_dim,
            mlp_dim=512,
            stride=self.stride,
            depth=2,
        )
        self.dense_head = DenseWarpTrackingHead(
            feature_dim=self.latent_dim,
            down_ratio=self.stride,
            refine_model=cow_refine_model,
            refine_patch_size=cow_refine_patch_size,
            refine_blocks=cow_refine_blocks,
            temporal_interleave_stride=cow_temporal_interleave_stride,
        )

    def _build_fmaps(self, fmaps_out, B: int, T: int, H: int, W: int, dtype: torch.dtype) -> torch.Tensor:
        H_stride, W_stride = H // self.stride, W // self.stride
        fmap = fmaps_out[0] if isinstance(fmaps_out, list) else fmaps_out
        fmap = fmap.permute(0, 2, 3, 1)
        fmap = fmap / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmap), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=fmap.device, dtype=fmap.dtype),
            )
        )
        fmap = fmap.permute(0, 3, 1, 2).reshape(B, T, self.latent_dim, H_stride, W_stride)
        return fmap.to(dtype)

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

    def forward(self, rgbs, events, queries, iters=None, img_ifnew=None, feat_init=None, is_train=False):
        B, T, C_event, H, W = events.shape
        _, N, _ = queries.shape
        _, _, C_img, _, _ = rgbs.shape
        if B != 1:
            raise AssertionError("TAPFormerCowDense currently follows TAPFormer training and expects batch_size == 1")
        if H % self.stride != 0 or W % self.stride != 0:
            raise AssertionError("Input height/width must be divisible by model.stride")
        if iters is None:
            raise ValueError("TAPFormerCowDense.forward requires an explicit iters argument.")

        queried_frames = queries[:, :, 0].long()
        if torch.any(queried_frames != 0):
            raise ValueError(
                "TAPFormerCowDense expects all query frames to be 0. "
                "Use a config with dataset.sample_vis_1st_frame: true."
            )
        query_xy = queries[..., 1:3]

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
        features = self._build_fmaps(fmaps_out, B, T, H, W, dtype)
        dense_coords, dense_vis_logits, dense_conf_logits = self.dense_head(
            features,
            image_size=(H, W),
            iters=iters,
        )

        coord_preds: List[torch.Tensor] = [self._sample_dense(pred, query_xy) for pred in dense_coords]
        vis_preds: List[torch.Tensor] = [self._sample_dense_scalar(pred, query_xy) for pred in dense_vis_logits]
        conf_preds: List[torch.Tensor] = [self._sample_dense_scalar(pred, query_xy) for pred in dense_conf_logits]

        coords_predicted = coord_preds[-1]
        vis_predicted = torch.sigmoid(vis_preds[-1])
        conf_predicted = torch.sigmoid(conf_preds[-1])

        if is_train:
            valid_mask = queried_frames[:, None] <= torch.arange(0, T, device=queries.device)[None, :, None]
            train_data = ([coord_preds], [vis_preds], [conf_preds], valid_mask)
        else:
            train_data = None

        return coords_predicted, vis_predicted, conf_predicted, train_data
