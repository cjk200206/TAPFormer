from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from LFE_TAP.models.fusionFormer import Fusionformer
from LFE_TAP.models.point_warp_head import PointLocalTrackingHead, PointWarpTrackingHead


class TAPFormerPointWarp(nn.Module):
    """Native TAPFormer fusion with propagated sparse local tracking."""

    def __init__(
        self,
        window_size: int = 12,
        stride: int = 4,
        hidden_size: int = 384,
        space_depth: int = 3,
        time_depth: int = 3,
        point_state_dim: int = 128,
        point_warp_dim: int = 256,
        point_corr_levels: int = 3,
        point_patch_radius: int = 2,
        point_local_corr_radius: int = 3,
        point_coarse_iters: int | None = None,
        point_refine_iters: int | None = None,
        point_refine_use_local_evidence: bool = True,
        point_refine_detach_local_evidence: bool = True,
        point_use_global_init: bool = False,
        point_cost_base_stride: int = 8,
        point_cost_levels: int = 3,
        point_cost_radius: int = 3,
        point_cost_dim: int = 256,
        point_initializer_stride: int = 16,
        point_initializer_temperature: float = 20.0,
        point_initializer_radius: int = 5,
        point_initializer_chunk_size: int = 64,
        point_detach_coords: bool = True,
        point_limit_update: bool = True,
        point_max_update_ratio: float = 0.15,
        point_max_magnitude_ratio: float = 1.0,
        point_support_mode: str = "none",
        **_,
    ) -> None:
        super().__init__()
        if str(point_support_mode).lower().strip() != "none":
            raise ValueError("The point-warp MVP only supports point_support_mode='none'.")
        if bool(point_use_global_init):
            raise ValueError("point_use_global_init is not supported in the local-update point-warp v1.")
        del (
            point_cost_base_stride,
            point_cost_levels,
            point_cost_radius,
            point_cost_dim,
            point_initializer_stride,
            point_initializer_temperature,
            point_initializer_radius,
            point_initializer_chunk_size,
        )

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.point_corr_levels = int(point_corr_levels)
        self.point_coarse_iters = None if point_coarse_iters is None else int(point_coarse_iters)
        self.point_refine_iters = None if point_refine_iters is None else int(point_refine_iters)
        self.point_refine_use_local_evidence = bool(point_refine_use_local_evidence)
        self.point_refine_detach_local_evidence = bool(point_refine_detach_local_evidence)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.point_corr_levels <= 0:
            raise ValueError("point_corr_levels must be positive.")
        if self.point_coarse_iters is not None and self.point_coarse_iters < 0:
            raise ValueError("point_coarse_iters must be non-negative.")
        if self.point_refine_iters is not None and self.point_refine_iters < 0:
            raise ValueError("point_refine_iters must be non-negative.")

        self.latent_dim = 128
        self.model_resolution = (384, 512)
        self.fusion_block = Fusionformer(
            image_size=self.model_resolution,
            out_dim=self.latent_dim,
            mlp_dim=512,
            stride=self.stride,
            depth=2,
        )
        self.local_head = PointLocalTrackingHead(
            feature_dim=self.latent_dim,
            window_size=self.window_size,
            hidden_size=hidden_size,
            space_depth=space_depth,
            time_depth=time_depth,
            state_dim=point_state_dim,
            corr_dim=point_warp_dim,
            corr_levels=self.point_corr_levels,
            corr_radius=point_local_corr_radius,
            detach_coords=point_detach_coords,
            limit_update=point_limit_update,
            max_update_ratio=point_max_update_ratio,
            max_magnitude_ratio=point_max_magnitude_ratio,
        )
        self.point_head = PointWarpTrackingHead(
            feature_dim=self.latent_dim,
            window_size=self.window_size,
            hidden_size=hidden_size,
            space_depth=space_depth,
            time_depth=time_depth,
            state_dim=point_state_dim,
            warp_dim=point_warp_dim,
            corr_levels=self.point_corr_levels,
            patch_radius=point_patch_radius,
            detach_coords=point_detach_coords,
            limit_update=point_limit_update,
            max_update_ratio=point_max_update_ratio,
            max_magnitude_ratio=point_max_magnitude_ratio,
            use_local_evidence=self.point_refine_use_local_evidence,
        )

    def _reset_fusion_state(self) -> None:
        transunet = getattr(self.fusion_block, "transunet", None)
        if transunet is None:
            return
        for name, value in (
            ("xe_history", []),
            ("x_e_pre", None),
            ("x_out_", None),
            ("x_anchor_", None),
            ("x1_anchor_", None),
            ("x0_anchor_", None),
        ):
            if hasattr(transunet, name):
                setattr(transunet, name, value)

    def _reshape_pyramid(
        self,
        outputs,
        batch: int,
        frames: int,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        outputs = outputs if isinstance(outputs, list) else [outputs]
        if not outputs:
            raise ValueError("Fusionformer returned an empty feature pyramid.")
        pyramid = []
        for feature in outputs:
            if feature.shape[0] != batch * frames:
                raise ValueError("Fusionformer output does not match the input batch/time dimensions.")
            feature = feature.reshape(batch, frames, *feature.shape[1:])
            pyramid.append(F.normalize(feature, dim=2, eps=1e-6).to(dtype))
        return pyramid

    def _prepare_pyramid(self, pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        pyramid = list(pyramid[: self.point_corr_levels])
        while len(pyramid) < self.point_corr_levels:
            previous = pyramid[-1]
            if min(previous.shape[-2:]) < 2:
                raise ValueError("point_corr_levels exceeds the available feature resolution.")
            batch, frames, channels, height, width = previous.shape
            pooled = F.avg_pool2d(
                previous.reshape(batch * frames, channels, height, width),
                kernel_size=2,
                stride=2,
            )
            pyramid.append(pooled.reshape(batch, frames, channels, *pooled.shape[-2:]))
        return pyramid

    def _encode(
        self,
        rgbs: torch.Tensor,
        events: torch.Tensor,
        img_ifnew: torch.Tensor | None,
    ) -> List[torch.Tensor]:
        batch, frames, event_channels, height, width = events.shape
        image_channels = rgbs.shape[2]
        rgbs = 2.0 * (rgbs / 255.0) - 1.0
        events = 2.0 * events - 1.0
        if img_ifnew is None:
            img_ifnew = torch.ones(frames, device=rgbs.device, dtype=rgbs.dtype)
        outputs = self.fusion_block(
            rgbs.reshape(batch * frames, image_channels, height, width),
            events.reshape(batch * frames, event_channels, height, width),
            img_ifnew,
        )
        return self._reshape_pyramid(outputs, batch, frames, rgbs.dtype)

    @staticmethod
    def _validate_queries(queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim != 3 or queries.shape[-1] != 3:
            raise ValueError("queries must have shape [B,N,3].")
        if torch.any(queries[..., 0].long() != 0):
            raise ValueError("TAPFormerPointWarp expects queries anchored at frame 0/reference.")
        return queries[..., 1:3]

    def _resolve_tracking_iters(self, iters: int) -> Tuple[int, int]:
        if iters < 0:
            raise ValueError("iters must be non-negative.")
        coarse_iters = 2 if self.point_coarse_iters is None else self.point_coarse_iters
        if self.point_refine_iters is None:
            refine_iters = max(int(iters) - coarse_iters, 1)
        else:
            refine_iters = self.point_refine_iters
        return int(coarse_iters), int(refine_iters)

    def _make_initial_state(
        self,
        feature_pyramid: List[torch.Tensor],
        query_xy: torch.Tensor,
        coords_init: torch.Tensor | None,
        vis_init: torch.Tensor | None,
        conf_init: torch.Tensor | None,
        init_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames = feature_pyramid[0].shape[:2]
        query_xy = query_xy.to(device=feature_pyramid[0].device, dtype=feature_pyramid[0].dtype)
        coords = query_xy[:, None].expand(batch, frames, -1, -1).clone()
        logits_shape = (batch, frames, query_xy.shape[1], 1)
        vis = feature_pyramid[0].new_zeros(logits_shape)
        conf = feature_pyramid[0].new_zeros(logits_shape)

        init_values = (coords_init, vis_init, conf_init)
        if any(value is not None for value in init_values):
            if not all(value is not None for value in init_values):
                raise ValueError("coords_init, vis_init and conf_init must be provided together.")
            expected_scalar_shape = (*coords.shape[:-1], 1)
            if coords_init.shape != coords.shape:
                raise ValueError(f"coords_init must have shape {tuple(coords.shape)}.")
            if vis_init.shape != expected_scalar_shape or conf_init.shape != expected_scalar_shape:
                raise ValueError(f"vis_init/conf_init must have shape {expected_scalar_shape}.")
            coords_init = coords_init.to(device=coords.device, dtype=coords.dtype)
            vis_init = vis_init.to(device=vis.device, dtype=vis.dtype)
            conf_init = conf_init.to(device=conf.device, dtype=conf.dtype)
            if init_mask is None:
                coords, vis, conf = coords_init, vis_init, conf_init
            else:
                if init_mask.shape != expected_scalar_shape:
                    raise ValueError(f"init_mask must have shape {expected_scalar_shape}.")
                init_mask = init_mask.to(device=coords.device, dtype=torch.bool)
                coords = torch.where(init_mask.expand_as(coords), coords_init, coords)
                vis = torch.where(init_mask, vis_init, vis)
                conf = torch.where(init_mask, conf_init, conf)
        elif init_mask is not None:
            raise ValueError("init_mask requires coords_init, vis_init and conf_init.")

        return coords, vis, conf

    def _track_from_pyramids(
        self,
        feature_pyramid: List[torch.Tensor],
        anchor_pyramid: List[torch.Tensor],
        query_xy: torch.Tensor,
        image_size: Tuple[int, int],
        iters: int,
        local_anchor: bool,
        coords_init: torch.Tensor | None = None,
        vis_init: torch.Tensor | None = None,
        conf_init: torch.Tensor | None = None,
        init_mask: torch.Tensor | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        coords, vis, conf = self._make_initial_state(
            feature_pyramid,
            query_xy,
            coords_init,
            vis_init,
            conf_init,
            init_mask,
        )
        coarse_iters, refine_iters = self._resolve_tracking_iters(int(iters))

        coord_preds, vis_preds, conf_preds = [], [], []
        coarse_coords, coarse_vis, coarse_conf, local_evidence = self.local_head(
            feature_pyramid,
            anchor_pyramid,
            query_xy,
            coords,
            vis,
            conf,
            image_size=image_size,
            iters=coarse_iters,
            local_anchor=local_anchor,
        )
        coord_preds.extend(coarse_coords)
        vis_preds.extend(coarse_vis)
        conf_preds.extend(coarse_conf)
        if coord_preds:
            coords = coord_preds[-1]
            vis = vis_preds[-1].unsqueeze(-1)
            conf = conf_preds[-1].unsqueeze(-1)
        if (
            self.point_refine_use_local_evidence
            and self.point_refine_detach_local_evidence
            and local_evidence is not None
        ):
            local_evidence = local_evidence.detach()

        refined_coords, refined_vis, refined_conf = self.point_head(
            feature_pyramid,
            anchor_pyramid,
            query_xy,
            coords,
            vis,
            conf,
            image_size=image_size,
            iters=refine_iters,
            local_anchor=local_anchor,
            local_evidence=local_evidence,
        )
        coord_preds.extend(refined_coords)
        vis_preds.extend(refined_vis)
        conf_preds.extend(refined_conf)
        if not coord_preds:
            coord_preds.append(coords)
            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])
        return coord_preds, vis_preds, conf_preds

    def _track_sequence(
        self,
        feature_pyramid: List[torch.Tensor],
        anchor_pyramid: List[torch.Tensor],
        query_xy: torch.Tensor,
        image_size: Tuple[int, int],
        iters: int,
        first_window_local_anchor: bool,
        is_train: bool,
    ):
        batch, frames = feature_pyramid[0].shape[:2]
        num_queries = query_xy.shape[1]
        coords_predicted = feature_pyramid[0].new_zeros(batch, frames, num_queries, 2)
        vis_logits_predicted = feature_pyramid[0].new_zeros(batch, frames, num_queries)
        conf_logits_predicted = feature_pyramid[0].new_zeros(batch, frames, num_queries)
        all_coords_predictions, all_vis_predictions, all_conf_predictions = [], [], []

        window_size = min(int(self.window_size), frames)
        step = max(1, int(self.window_size) // 2)
        num_windows = max(1, (max(frames - window_size, 0) + step - 1) // step + 1)
        overlap = max(0, int(self.window_size) - step)

        for window_idx in range(num_windows):
            start = window_idx * step
            if start >= frames:
                break
            end = min(start + int(self.window_size), frames)
            window_frames = end - start
            coords_init = vis_init = conf_init = None

            if window_idx > 0:
                overlap_frames = min(overlap, window_frames)
                new_frames = window_frames - overlap_frames
                coords_overlap = coords_predicted[:, start : start + overlap_frames]
                vis_overlap = vis_logits_predicted[:, start : start + overlap_frames, :, None]
                conf_overlap = conf_logits_predicted[:, start : start + overlap_frames, :, None]
                if overlap_frames > 0:
                    coords_tail = coords_overlap[:, -1:]
                    vis_tail = vis_overlap[:, -1:]
                    conf_tail = conf_overlap[:, -1:]
                else:
                    coords_tail = coords_predicted[:, start - 1 : start]
                    vis_tail = vis_logits_predicted[:, start - 1 : start, :, None]
                    conf_tail = conf_logits_predicted[:, start - 1 : start, :, None]
                if new_frames > 0:
                    coords_init = torch.cat(
                        [coords_overlap, coords_tail.expand(-1, new_frames, -1, -1)],
                        dim=1,
                    )
                    vis_init = torch.cat(
                        [vis_overlap, vis_tail.expand(-1, new_frames, -1, -1)],
                        dim=1,
                    )
                    conf_init = torch.cat(
                        [conf_overlap, conf_tail.expand(-1, new_frames, -1, -1)],
                        dim=1,
                    )
                else:
                    coords_init, vis_init, conf_init = coords_overlap, vis_overlap, conf_overlap

            coord_preds, vis_preds, conf_preds = self._track_from_pyramids(
                [feature[:, start:end] for feature in feature_pyramid],
                anchor_pyramid,
                query_xy,
                image_size=image_size,
                iters=int(iters),
                local_anchor=first_window_local_anchor and window_idx == 0,
                coords_init=coords_init,
                vis_init=vis_init,
                conf_init=conf_init,
            )
            coords_predicted[:, start:end] = coord_preds[-1][:, :window_frames]
            vis_logits_predicted[:, start:end] = vis_preds[-1][:, :window_frames]
            conf_logits_predicted[:, start:end] = conf_preds[-1][:, :window_frames]

            if is_train:
                all_coords_predictions.append([coord[:, :window_frames] for coord in coord_preds])
                all_vis_predictions.append([vis[:, :window_frames] for vis in vis_preds])
                all_conf_predictions.append([conf[:, :window_frames] for conf in conf_preds])

        if is_train:
            valid_mask = query_xy.new_ones(batch, frames, num_queries, dtype=torch.bool)
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_conf_predictions,
                valid_mask,
            )
        else:
            train_data = None
        return (
            coords_predicted,
            torch.sigmoid(vis_logits_predicted),
            torch.sigmoid(conf_logits_predicted),
            train_data,
        )

    def forward(
        self,
        rgbs: torch.Tensor,
        events: torch.Tensor,
        queries: torch.Tensor,
        iters: int | None = None,
        img_ifnew: torch.Tensor | None = None,
        feat_init=None,
        is_train: bool = False,
        reference_rgbs: torch.Tensor | None = None,
        reference_events: torch.Tensor | None = None,
        reference_only_train: bool = False,
    ):
        del feat_init
        if iters is None:
            raise ValueError("TAPFormerPointWarp.forward requires an explicit iters argument.")
        batch, frames, _, height, width = events.shape
        if batch != 1:
            raise AssertionError("TAPFormerPointWarp currently expects batch_size == 1.")
        if frames <= 0:
            raise ValueError("TAPFormerPointWarp requires at least one frame.")
        if (height, width) != self.model_resolution:
            raise ValueError(f"Expected input resolution {self.model_resolution}, got {(height, width)}.")
        if height % self.stride != 0 or width % self.stride != 0:
            raise ValueError("Input height/width must be divisible by model.stride.")
        query_xy = self._validate_queries(queries)

        has_reference = reference_rgbs is not None or reference_events is not None
        if has_reference and (reference_rgbs is None or reference_events is None):
            raise ValueError("reference_rgbs and reference_events must be provided together.")
        if reference_only_train and not has_reference:
            raise ValueError("reference_only_train requires reference inputs.")

        if has_reference:
            if reference_rgbs.shape[1] != 1 or reference_events.shape[1] != 1:
                raise ValueError("Point-warp reference input must contain exactly one frame.")
            self._reset_fusion_state()
            reference_ifnew = torch.ones(
                1,
                device=reference_rgbs.device,
                dtype=reference_rgbs.dtype,
            )
            anchor_pyramid = self._prepare_pyramid(
                self._encode(reference_rgbs, reference_events, reference_ifnew)
            )
            self._reset_fusion_state()
            feature_pyramid = self._prepare_pyramid(self._encode(rgbs, events, img_ifnew))
            first_window_local_anchor = False
        else:
            self._reset_fusion_state()
            feature_pyramid = self._prepare_pyramid(self._encode(rgbs, events, img_ifnew))
            anchor_pyramid = [feature[:, :1] for feature in feature_pyramid]
            first_window_local_anchor = True

        return self._track_sequence(
            feature_pyramid,
            anchor_pyramid,
            query_xy,
            image_size=(height, width),
            iters=int(iters),
            first_window_local_anchor=first_window_local_anchor,
            is_train=is_train,
        )
