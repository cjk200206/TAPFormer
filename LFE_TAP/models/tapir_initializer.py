from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TapirCostVolumeInitializer(nn.Module):
    """TAPIR-style global cost-volume initialization on a coarse dense grid."""

    def __init__(
        self,
        stride: int = 16,
        temperature: float = 20.0,
        softargmax_radius: int = 5,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.stride = int(stride)
        self.temperature = float(temperature)
        self.softargmax_radius = int(softargmax_radius)
        self.chunk_size = int(chunk_size)
        if self.stride <= 0 or self.softargmax_radius <= 0 or self.chunk_size <= 0:
            raise ValueError("TAPIR initializer stride, radius, and chunk_size must be positive.")

        self.cost_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.cost_conv2 = nn.Conv2d(16, 1, 3, padding=1)
        self.uncertainty_conv = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.uncertainty_fc = nn.Linear(32, 16)
        self.uncertainty_out = nn.Linear(16, 2)

    def _cost_head(self, cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.cost_conv1(cost))
        position = self.cost_conv2(hidden)
        uncertainty = F.relu(self.uncertainty_conv(hidden)).mean(dim=(-2, -1))
        uncertainty = self.uncertainty_out(F.relu(self.uncertainty_fc(uncertainty)))
        return position, uncertainty

    def _run_cost_head(self, cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and cost.requires_grad:
            return checkpoint(self._cost_head, cost, use_reentrant=False)
        return self._cost_head(cost)

    def _local_soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = logits.shape
        probabilities = torch.softmax(
            logits.float().flatten(1) * self.temperature,
            dim=-1,
        ).view(batch, height, width)

        y, x = torch.meshgrid(
            torch.arange(height, device=logits.device, dtype=torch.float32) + 0.5,
            torch.arange(width, device=logits.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        coords = torch.stack([x, y], dim=-1)
        peak = probabilities.flatten(1).argmax(dim=-1)
        peak_coords = coords.view(-1, 2)[peak]
        local_mask = (
            (coords[None] - peak_coords[:, None, None]).square().sum(dim=-1)
            < float(self.softargmax_radius**2)
        )
        local_prob = probabilities * local_mask
        local_prob = local_prob / local_prob.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        points = (local_prob[..., None] * coords).sum(dim=(-3, -2))
        return points.to(logits.dtype)

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features, dim=2, eps=1e-6)

    def forward(
        self,
        features: torch.Tensor,
        anchor_features: torch.Tensor,
        image_size: Tuple[int, int],
        tracking_size: Tuple[int, int],
        local_anchor: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return flow and visible/confident logits at the tracking resolution."""
        batch, frames, channels, _, _ = features.shape
        image_height, image_width = image_size
        coarse_height = image_height // self.stride
        coarse_width = image_width // self.stride
        if coarse_height <= 0 or coarse_width <= 0:
            raise ValueError("TAPIR initializer stride is larger than the input image.")
        if image_height % self.stride != 0 or image_width % self.stride != 0:
            raise ValueError("Input height/width must be divisible by tapir_init_stride.")

        target = F.adaptive_avg_pool2d(
            features.reshape(batch * frames, channels, *features.shape[-2:]),
            (coarse_height, coarse_width),
        ).view(batch, frames, channels, coarse_height, coarse_width)
        anchor = anchor_features[:, 0] if anchor_features.ndim == 5 else anchor_features
        anchor = F.adaptive_avg_pool2d(anchor, (coarse_height, coarse_width)).unsqueeze(1)
        target = self._normalize(target)
        anchor = self._normalize(anchor)[:, 0]

        target_flat = target.flatten(3).permute(0, 1, 3, 2)
        anchor_flat = anchor.flatten(2).transpose(1, 2)
        num_queries = coarse_height * coarse_width
        flow_chunks, info_chunks = [], []

        y, x = torch.meshgrid(
            torch.arange(coarse_height, device=features.device, dtype=features.dtype) + 0.5,
            torch.arange(coarse_width, device=features.device, dtype=features.dtype) + 0.5,
            indexing="ij",
        )
        query_coords = torch.stack([x, y], dim=-1).view(num_queries, 2)
        scale = features.new_tensor(
            [features.shape[-1] / coarse_width, features.shape[-2] / coarse_height]
        )

        for start in range(0, num_queries, self.chunk_size):
            end = min(start + self.chunk_size, num_queries)
            query = anchor_flat[:, start:end]
            cost = torch.einsum("bqc,btkc->btqk", query, target_flat)
            cost = cost.reshape(batch * frames * (end - start), 1, coarse_height, coarse_width)
            position_logits, uncertainty = self._run_cost_head(cost)
            points = self._local_soft_argmax(position_logits)
            points = points.view(batch, frames, end - start, 2)
            flow_chunks.append((points - query_coords[start:end]) * scale)

            # TAPIR predicts occlusion and expected distance; CowDense uses the
            # opposite visible and confident logit conventions.
            info_chunks.append(-uncertainty.view(batch, frames, end - start, 2))

        flow = torch.cat(flow_chunks, dim=2).view(
            batch, frames, coarse_height, coarse_width, 2
        ).permute(0, 1, 4, 2, 3)
        info = torch.cat(info_chunks, dim=2).view(
            batch, frames, coarse_height, coarse_width, 2
        ).permute(0, 1, 4, 2, 3)

        track_height, track_width = tracking_size
        flow = F.interpolate(
            flow.reshape(batch * frames, 2, coarse_height, coarse_width),
            size=(track_height, track_width),
            mode="bilinear",
            align_corners=False,
        ).view(batch, frames, 2, track_height, track_width)
        info = F.interpolate(
            info.reshape(batch * frames, 2, coarse_height, coarse_width),
            size=(track_height, track_width),
            mode="bilinear",
            align_corners=False,
        ).view(batch, frames, 2, track_height, track_width)

        if local_anchor:
            flow = torch.cat([torch.zeros_like(flow[:, :1]), flow[:, 1:]], dim=1)
        return flow, info
