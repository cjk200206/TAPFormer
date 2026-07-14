from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointGlobalInitializer(nn.Module):
    """Initialize sparse tracks with raw cosine correlation on a coarse grid."""

    def __init__(
        self,
        feature_dim: int = 128,
        base_stride: int = 8,
        stride: int = 16,
        cost_levels: int = 3,
        temperature: float = 20.0,
        softargmax_radius: int = 5,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.base_stride = int(base_stride)
        self.stride = int(stride)
        self.cost_levels = int(cost_levels)
        self.temperature = float(temperature)
        self.softargmax_radius = int(softargmax_radius)
        self.chunk_size = int(chunk_size)
        if self.base_stride <= 0 or self.stride < self.base_stride or self.cost_levels <= 0:
            raise ValueError("base_stride/stride/cost_levels must define a positive cost pyramid.")
        stride_ratio = self.stride / self.base_stride
        self.initializer_level = int(round(torch.log2(torch.tensor(stride_ratio)).item()))
        if 2**self.initializer_level != stride_ratio or self.initializer_level >= self.cost_levels:
            raise ValueError("stride must select an existing power-of-two cost pyramid level.")
        if self.softargmax_radius < 0 or self.chunk_size <= 0:
            raise ValueError("softargmax_radius must be non-negative and chunk_size positive.")

        self.descriptor_adapter = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        nn.init.dirac_(self.descriptor_adapter.weight)

    @staticmethod
    def _image_to_grid(query_xy: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        image_height, image_width = image_size
        if image_height <= 1 or image_width <= 1:
            raise ValueError("image_size dimensions must be greater than one.")
        scale = query_xy.new_tensor([image_width - 1, image_height - 1])
        return 2.0 * query_xy / scale - 1.0

    def _sample_anchor_descriptors(
        self,
        anchor: torch.Tensor,
        query_xy: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        grid = self._image_to_grid(query_xy, image_size).unsqueeze(2)
        descriptors = F.grid_sample(
            anchor,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return F.normalize(descriptors.squeeze(-1).transpose(1, 2), dim=-1, eps=1e-6)

    def _local_soft_argmax(
        self,
        logits: torch.Tensor,
        coarse_size: Tuple[int, int],
    ) -> torch.Tensor:
        coarse_height, coarse_width = coarse_size
        logits_float = logits.float()
        peaks = logits_float.argmax(dim=-1)
        peak_y = torch.div(peaks, coarse_width, rounding_mode="floor")
        peak_x = peaks.remainder(coarse_width)

        y, x = torch.meshgrid(
            torch.arange(coarse_height, device=logits.device),
            torch.arange(coarse_width, device=logits.device),
            indexing="ij",
        )
        x = x.flatten()
        y = y.flatten()
        if self.softargmax_radius == 0:
            return torch.stack([peak_x, peak_y], dim=-1).to(logits.dtype)

        distance_sq = (
            (x - peak_x[..., None]).square()
            + (y - peak_y[..., None]).square()
        )
        local_mask = distance_sq <= self.softargmax_radius**2
        masked_logits = logits_float.masked_fill(~local_mask, torch.finfo(logits_float.dtype).min)
        probabilities = torch.softmax(masked_logits, dim=-1)
        point_x = torch.sum(probabilities * x, dim=-1)
        point_y = torch.sum(probabilities * y, dim=-1)
        return torch.stack([point_x, point_y], dim=-1).to(logits.dtype)

    def _build_cost_pyramid(self, cost_base: torch.Tensor) -> List[torch.Tensor]:
        batch, frames, queries, height, width = cost_base.shape
        pyramid = [cost_base]
        for _ in range(1, self.cost_levels):
            previous = pyramid[-1]
            pooled = F.avg_pool2d(
                previous.reshape(batch * frames * queries, 1, height, width),
                kernel_size=2,
                stride=2,
            )
            height, width = pooled.shape[-2:]
            pyramid.append(pooled.reshape(batch, frames, queries, height, width))
        return pyramid

    def forward(
        self,
        features: torch.Tensor,
        anchor_features: torch.Tensor,
        query_xy: torch.Tensor,
        image_size: Tuple[int, int],
        local_anchor: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if features.ndim != 5 or anchor_features.ndim != 5:
            raise ValueError("features and anchor_features must have shape [B,T,C,H,W].")
        batch, frames, channels = features.shape[:3]
        if anchor_features.shape[:3] != (batch, 1, channels):
            raise ValueError("anchor_features must have shape [B,1,C,H,W] matching features.")
        if query_xy.ndim != 3 or query_xy.shape[0] != batch or query_xy.shape[-1] != 2:
            raise ValueError("query_xy must have shape [B,N,2].")

        image_height, image_width = image_size
        base_height = image_height // self.base_stride
        base_width = image_width // self.base_stride
        if base_height <= 1 or base_width <= 1:
            raise ValueError("base_stride produces a cost map smaller than 2x2.")

        target = F.adaptive_avg_pool2d(
            features.reshape(batch * frames, channels, *features.shape[-2:]),
            (base_height, base_width),
        )
        target = self.descriptor_adapter(target).view(
            batch, frames, channels, base_height, base_width
        )
        anchor = F.adaptive_avg_pool2d(anchor_features[:, 0], (base_height, base_width))
        anchor = self.descriptor_adapter(anchor)
        target = F.normalize(target, dim=2, eps=1e-6)
        anchor = F.normalize(anchor, dim=1, eps=1e-6)
        query_descriptors = self._sample_anchor_descriptors(anchor, query_xy, image_size)
        target_flat = target.flatten(3).permute(0, 1, 3, 2)

        cost_chunks = []
        num_queries = query_xy.shape[1]
        for start in range(0, num_queries, self.chunk_size):
            query_chunk = query_descriptors[:, start : start + self.chunk_size]
            cost_chunks.append(
                self.temperature * torch.einsum("bnc,btkc->btnk", query_chunk, target_flat)
            )
        cost_base = torch.cat(cost_chunks, dim=2).view(
            batch, frames, num_queries, base_height, base_width
        )
        cost_pyramid = self._build_cost_pyramid(cost_base)
        initializer_logits = cost_pyramid[self.initializer_level]
        coarse_height, coarse_width = initializer_logits.shape[-2:]
        coarse_coords = self._local_soft_argmax(
            initializer_logits.flatten(-2),
            (coarse_height, coarse_width),
        )
        scale = features.new_tensor(
            [
                (image_width - 1) / (coarse_width - 1),
                (image_height - 1) / (coarse_height - 1),
            ]
        )
        coords = coarse_coords * scale
        if local_anchor:
            coords = torch.cat([query_xy[:, None], coords[:, 1:]], dim=1)

        logits_shape = (batch, frames, num_queries, 1)
        vis = features.new_zeros(logits_shape)
        conf = features.new_zeros(logits_shape)
        auxiliary = {
            "cost_pyramid": cost_pyramid,
            "cost_strides": [self.base_stride * 2**level for level in range(self.cost_levels)],
        }
        return coords, vis, conf, auxiliary
