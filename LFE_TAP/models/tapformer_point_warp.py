from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from LFE_TAP.models.fusionFormer import Fusionformer
from LFE_TAP.models.point_global_initializer import PointGlobalInitializer
from LFE_TAP.models.point_warp_head import PointWarpTrackingHead


class TAPFormerPointWarp(nn.Module):
    """Native TAPFormer fusion with sparse global initialization and warp refinement."""

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
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.point_corr_levels = int(point_corr_levels)
        if self.point_corr_levels <= 0:
            raise ValueError("point_corr_levels must be positive.")
        self.latent_dim = 128
        self.model_resolution = (384, 512)
        self.fusion_block = Fusionformer(
            image_size=self.model_resolution,
            out_dim=self.latent_dim,
            mlp_dim=512,
            stride=self.stride,
            depth=2,
        )
        self.initializer = PointGlobalInitializer(
            feature_dim=self.latent_dim,
            base_stride=point_cost_base_stride,
            cost_levels=point_cost_levels,
            stride=point_initializer_stride,
            temperature=point_initializer_temperature,
            softargmax_radius=point_initializer_radius,
            chunk_size=point_initializer_chunk_size,
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
            cost_levels=point_cost_levels,
            cost_radius=point_cost_radius,
            cost_dim=point_cost_dim,
            detach_coords=point_detach_coords,
            limit_update=point_limit_update,
            max_update_ratio=point_max_update_ratio,
            max_magnitude_ratio=point_max_magnitude_ratio,
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
        coords, vis, conf, initializer_aux = self.initializer(
            feature_pyramid[0],
            anchor_pyramid[0],
            query_xy,
            image_size=image_size,
            local_anchor=local_anchor,
        )

        init_values = (coords_init, vis_init, conf_init)
        if any(value is not None for value in init_values):
            if not all(value is not None for value in init_values):
                raise ValueError("coords_init, vis_init and conf_init must be provided together.")
            expected_scalar_shape = (*coords.shape[:-1], 1)
            if coords_init.shape != coords.shape:
                raise ValueError(f"coords_init must have shape {tuple(coords.shape)}.")
            if (
                vis_init.shape != expected_scalar_shape
                or conf_init.shape != expected_scalar_shape
            ):
                raise ValueError(f"vis_init/conf_init must have shape {expected_scalar_shape}.")
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

        coord_preds = [coords]
        vis_preds = [vis[..., 0]]
        conf_preds = [conf[..., 0]]
        refined_coords, refined_vis, refined_conf = self.point_head(
            feature_pyramid,
            anchor_pyramid,
            initializer_aux["cost_pyramid"],
            initializer_aux["cost_strides"],
            query_xy,
            coords,
            vis,
            conf,
            image_size=image_size,
            iters=iters,
            local_anchor=local_anchor,
        )
        coord_preds.extend(refined_coords)
        vis_preds.extend(refined_vis)
        conf_preds.extend(refined_conf)
        return coord_preds, vis_preds, conf_preds

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
            local_anchor = False
        else:
            self._reset_fusion_state()
            feature_pyramid = self._prepare_pyramid(self._encode(rgbs, events, img_ifnew))
            anchor_pyramid = [feature[:, :1] for feature in feature_pyramid]
            local_anchor = True

        coord_preds, vis_preds, conf_preds = self._track_from_pyramids(
            feature_pyramid,
            anchor_pyramid,
            query_xy,
            image_size=(height, width),
            iters=int(iters),
            local_anchor=local_anchor,
        )

        coords_predicted = coord_preds[-1]
        vis_predicted = torch.sigmoid(vis_preds[-1])
        conf_predicted = torch.sigmoid(conf_preds[-1])
        if is_train:
            valid_mask = queries[..., 0].long()[:, None] <= torch.arange(
                frames,
                device=queries.device,
            )[None, :, None]
            train_data = ([coord_preds], [vis_preds], [conf_preds], valid_mask)
        else:
            train_data = None
        return coords_predicted, vis_predicted, conf_predicted, train_data
