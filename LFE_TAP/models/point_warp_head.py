from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from LFE_TAP.models.blocks import EfficientUpdateFormer
from LFE_TAP.models.embeddings import get_1d_sincos_pos_embed_from_grid
from LFE_TAP.models.tapformer import posenc


def sample_point_patches(
    features: torch.Tensor,
    coords: torch.Tensor,
    image_size: Tuple[int, int],
    radius: int,
) -> torch.Tensor:
    """Sample square feature patches around image-space point coordinates."""
    if features.ndim != 5 or coords.ndim != 4:
        raise ValueError("features/coords must have shape [B,T,C,H,W] and [B,T,N,2].")
    batch, frames, channels, feature_height, feature_width = features.shape
    if coords.shape[:2] != (batch, frames) or coords.shape[-1] != 2:
        raise ValueError("coords must match the feature batch/time dimensions.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    image_height, image_width = image_size
    if image_height <= 1 or image_width <= 1:
        raise ValueError("image_size dimensions must be greater than one.")
    image_scale = coords.new_tensor(
        [
            (feature_width - 1) / (image_width - 1),
            (feature_height - 1) / (image_height - 1),
        ]
    )
    feature_coords = coords * image_scale

    offset_y, offset_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=coords.device, dtype=coords.dtype),
        torch.arange(-radius, radius + 1, device=coords.device, dtype=coords.dtype),
        indexing="ij",
    )
    offsets = torch.stack([offset_x, offset_y], dim=-1).flatten(0, 1)
    sample_coords = feature_coords[..., None, :] + offsets
    norm_scale = coords.new_tensor(
        [max(feature_width - 1, 1), max(feature_height - 1, 1)]
    )
    grid = 2.0 * sample_coords / norm_scale - 1.0

    sampled = F.grid_sample(
        features.reshape(batch * frames, channels, feature_height, feature_width),
        grid.reshape(batch * frames, coords.shape[2], offsets.shape[0], 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.permute(0, 2, 3, 1).reshape(
        batch,
        frames,
        coords.shape[2],
        offsets.shape[0],
        channels,
    )


def sample_cost_patches(
    cost_map: torch.Tensor,
    coords: torch.Tensor,
    image_size: Tuple[int, int],
    radius: int,
) -> torch.Tensor:
    """Lookup local patches from a per-query global cost map."""
    if cost_map.ndim != 5 or coords.ndim != 4:
        raise ValueError("cost_map/coords must have shape [B,T,N,H,W] and [B,T,N,2].")
    batch, frames, queries, height, width = cost_map.shape
    if coords.shape[:3] != (batch, frames, queries):
        raise ValueError("coords must match cost_map batch/time/query dimensions.")
    image_height, image_width = image_size
    scale = coords.new_tensor(
        [(width - 1) / (image_width - 1), (height - 1) / (image_height - 1)]
    )
    centers = coords * scale
    offset_y, offset_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=coords.device, dtype=coords.dtype),
        torch.arange(-radius, radius + 1, device=coords.device, dtype=coords.dtype),
        indexing="ij",
    )
    offsets = torch.stack([offset_x, offset_y], dim=-1).flatten(0, 1)
    sample_coords = centers[..., None, :] + offsets
    norm = coords.new_tensor([max(width - 1, 1), max(height - 1, 1)])
    grid = 2.0 * sample_coords / norm - 1.0
    sampled = F.grid_sample(
        cost_map.reshape(batch * frames * queries, 1, height, width),
        grid.reshape(batch * frames * queries, 1, offsets.shape[0], 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(batch, frames, queries, offsets.shape[0])


class PointWarpTrackingHead(nn.Module):
    """Iteratively refine sparse tracks using point-local warped features."""

    def __init__(
        self,
        feature_dim: int = 128,
        window_size: int = 12,
        hidden_size: int = 384,
        space_depth: int = 3,
        time_depth: int = 3,
        state_dim: int = 128,
        warp_dim: int = 256,
        corr_levels: int = 3,
        patch_radius: int = 2,
        cost_levels: int = 3,
        cost_radius: int = 3,
        cost_dim: int = 256,
        detach_coords: bool = True,
        limit_update: bool = True,
        max_update_ratio: float = 0.15,
        max_magnitude_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        if int(hidden_size) != 384:
            raise ValueError("EfficientUpdateFormer currently requires hidden_size=384.")
        self.state_dim = int(state_dim)
        self.corr_levels = int(corr_levels)
        self.patch_radius = int(patch_radius)
        self.cost_levels = int(cost_levels)
        self.cost_radius = int(cost_radius)
        self.detach_coords = bool(detach_coords)
        self.limit_update = bool(limit_update)
        self.max_update_ratio = float(max_update_ratio)
        self.max_magnitude_ratio = float(max_magnitude_ratio)
        if min(self.corr_levels, self.cost_levels, self.state_dim, warp_dim, cost_dim) <= 0:
            raise ValueError("Feature levels and token dimensions must be positive.")
        if self.patch_radius < 0 or self.cost_radius < 0:
            raise ValueError("Patch radii must be non-negative.")
        if self.max_update_ratio <= 0 or self.max_magnitude_ratio <= 0:
            raise ValueError("point update limits must be positive.")

        self.warp_encoder = nn.Sequential(
            nn.Linear(4 * feature_dim, warp_dim),
            nn.GELU(),
            nn.Linear(warp_dim, warp_dim),
        )
        self.level_fusion = nn.Linear(self.corr_levels * warp_dim, warp_dim)
        cost_patch_dim = (2 * self.cost_radius + 1) ** 2
        self.cost_patch_encoder = nn.Sequential(
            nn.Linear(cost_patch_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.cost_level_fusion = nn.Linear(self.cost_levels * 128, cost_dim)
        # displacement PE: 2 * 21 = 42, bidirectional-motion PE: 4 * 21 = 84.
        self.input_dim = int(warp_dim) + int(cost_dim) + self.state_dim + 42 + 84 + 2
        if self.input_dim % 2 != 0:
            raise ValueError("point refinement token dimension must be even for time embedding.")
        self.updateformer2 = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.input_dim,
            hidden_size=hidden_size,
            output_dim=4 + self.state_dim,
            mlp_ratio=4.0,
            num_virtual_tracks=32,
            linear_layer_for_vis_conf=True,
        )
        time_grid = torch.linspace(0, self.window_size - 1, self.window_size)
        self.register_buffer(
            "time_emb",
            get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid),
            persistent=False,
        )

    def _time_embedding(self, frames: int, dtype: torch.dtype) -> torch.Tensor:
        embedding = self.time_emb
        if embedding.shape[1] != frames:
            embedding = F.interpolate(
                embedding.float().permute(0, 2, 1),
                size=frames,
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)
        return embedding.to(dtype)

    def _apply_coord_update(
        self,
        coords: torch.Tensor,
        query_xy: torch.Tensor,
        raw_delta: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        if not self.limit_update:
            return coords + raw_delta
        longer_side = float(max(image_size))
        update_limit = longer_side * self.max_update_ratio
        total_limit = longer_side * self.max_magnitude_ratio
        delta = update_limit * torch.tanh(raw_delta / update_limit)
        displacement = (coords - query_xy[:, None] + delta).clamp(-total_limit, total_limit)
        return query_xy[:, None] + displacement

    def _build_warp_evidence(
        self,
        features: List[torch.Tensor],
        anchor_patches: List[torch.Tensor],
        coords: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        level_evidence = []
        for feature, anchor_patch in zip(features, anchor_patches):
            target_patch = sample_point_patches(
                feature,
                coords,
                image_size=image_size,
                radius=self.patch_radius,
            )
            anchor = anchor_patch[:, None].expand(-1, feature.shape[1], -1, -1, -1)
            pair = torch.cat(
                [anchor, target_patch, anchor - target_patch, anchor * target_patch],
                dim=-1,
            )
            level_evidence.append(self.warp_encoder(pair).mean(dim=-2))
        return self.level_fusion(torch.cat(level_evidence, dim=-1))

    def _build_cost_evidence(
        self,
        cost_pyramid: List[torch.Tensor],
        coords: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        level_evidence = [
            self.cost_patch_encoder(
                sample_cost_patches(cost_map, coords, image_size, self.cost_radius)
            )
            for cost_map in cost_pyramid
        ]
        return self.cost_level_fusion(torch.cat(level_evidence, dim=-1))

    def forward(
        self,
        features: List[torch.Tensor],
        anchor_features: List[torch.Tensor],
        cost_pyramid: List[torch.Tensor],
        cost_strides: List[int],
        query_xy: torch.Tensor,
        coords: torch.Tensor,
        vis: torch.Tensor,
        conf: torch.Tensor,
        image_size: Tuple[int, int],
        iters: int,
        local_anchor: bool,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if len(features) != self.corr_levels or len(anchor_features) != self.corr_levels:
            raise ValueError(f"Expected {self.corr_levels} feature pyramid levels.")
        if len(cost_pyramid) != self.cost_levels or len(cost_strides) != self.cost_levels:
            raise ValueError(f"Expected {self.cost_levels} cost pyramid levels and strides.")
        batch, frames = features[0].shape[:2]
        for feature, anchor_feature in zip(features, anchor_features):
            if feature.shape[:2] != (batch, frames) or anchor_feature.shape[:2] != (batch, 1):
                raise ValueError("All feature levels must share batch/time dimensions and one anchor frame.")
        for cost_map, cost_stride in zip(cost_pyramid, cost_strides):
            if cost_map.shape[:3] != (batch, frames, query_xy.shape[1]):
                raise ValueError("All cost levels must share batch/time/query dimensions.")
            expected_size = (image_size[0] // cost_stride, image_size[1] // cost_stride)
            if cost_map.shape[-2:] != expected_size:
                raise ValueError("Cost map shape does not match its declared stride.")
        if coords.shape[:3] != (batch, frames, query_xy.shape[1]):
            raise ValueError("coords must have shape [B,T,N,2].")
        if iters < 0:
            raise ValueError("iters must be non-negative.")

        anchor_coords = query_xy[:, None]
        anchor_patches = [
            sample_point_patches(
                anchor_feature,
                anchor_coords,
                image_size=image_size,
                radius=self.patch_radius,
            )[:, 0]
            for anchor_feature in anchor_features
        ]
        hidden = features[0].new_zeros(batch, frames, query_xy.shape[1], self.state_dim)
        coord_preds, vis_preds, conf_preds = [], [], []
        spatial_scale = coords.new_tensor([image_size[1], image_size[0]])

        for _ in range(iters):
            if self.detach_coords:
                coords = coords.detach()
            sample_coords = coords
            warp_evidence = self._build_warp_evidence(
                features,
                anchor_patches,
                sample_coords,
                image_size,
            )
            cost_evidence = self._build_cost_evidence(
                cost_pyramid,
                sample_coords,
                image_size,
            )
            displacement = (coords - query_xy[:, None]) / spatial_scale
            displacement_emb = posenc(displacement, min_deg=0, max_deg=10)

            rel_forward = F.pad(coords[:, :-1] - coords[:, 1:], (0, 0, 0, 0, 0, 1))
            rel_backward = F.pad(coords[:, 1:] - coords[:, :-1], (0, 0, 0, 0, 1, 0))
            motion = torch.cat([rel_forward, rel_backward], dim=-1)
            motion = motion / spatial_scale.repeat(2)
            motion_emb = posenc(motion, min_deg=0, max_deg=10)

            tokens = torch.cat(
                [warp_evidence, cost_evidence, hidden, displacement_emb, motion_emb, vis, conf],
                dim=-1,
            ).permute(0, 2, 1, 3)
            tokens = tokens + self._time_embedding(frames, tokens.dtype)[:, None]
            raw = self.updateformer2(tokens)

            raw_delta = raw[..., :2].permute(0, 2, 1, 3)
            delta_hidden = raw[..., 2 : 2 + self.state_dim].permute(0, 2, 1, 3)
            delta_vis = raw[..., 2 + self.state_dim : 3 + self.state_dim].permute(0, 2, 1, 3)
            delta_conf = raw[..., 3 + self.state_dim :].permute(0, 2, 1, 3)
            coords = self._apply_coord_update(coords, query_xy, raw_delta, image_size)
            hidden = hidden + delta_hidden
            vis = vis + delta_vis
            conf = conf + delta_conf
            if local_anchor:
                coords = torch.cat([query_xy[:, None], coords[:, 1:]], dim=1)

            coord_preds.append(coords)
            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])

        return coord_preds, vis_preds, conf_preds
