import torch
import time
import torch.nn.functional as F
import numpy as np

from LFE_TAP.models.tapformer import TAPFormer, posenc
from LFE_TAP.models.tapformer_cow_dense import TAPFormerCowDense
from LFE_TAP.models.tapformer_point_warp import TAPFormerPointWarp
from LFE_TAP.utils.model_utils import get_track_feat, normalize_voxels
from LFE_TAP.models.embeddings import get_1d_sincos_pos_embed_from_grid

torch.manual_seed(0)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

class TAPFormer_online(TAPFormer):
    def __init__(self, trained_model=None, window_size=16, stride=4, corr_radius=3, corr_levels=3, 
                 backbone="basic", num_heads=8, hidden_size=384, space_depth=3, time_depth=3,
                 frontend_type="base"):
        """
        Initialize TAPFormer_online model for memory-efficient inference.
        
        Args:
            trained_model: (optional) A trained TAPFormer model instance. If provided, parameters will be copied from it.
            window_size, stride, corr_radius, corr_levels, backbone, num_heads, hidden_size, 
            space_depth, time_depth: Model configuration parameters. Used when trained_model is None.
        """
        # If trained_model is provided, use it to initialize (backward compatibility)
        if trained_model is not None:
            super(TAPFormer_online, self).__init__(
                window_size=trained_model.window_size,
                stride=trained_model.stride,
                corr_radius=trained_model.corr_radius,
                corr_levels=trained_model.corr_levels,
                backbone=trained_model.backbone,
                hidden_size=trained_model.hidden_size,
                space_depth=trained_model.space_depth,
                time_depth=trained_model.time_depth,
                frontend_type=getattr(trained_model, "frontend_type", "base")
            )
            self.fusion_block = trained_model.fusion_block
            self.updateformer2 = trained_model.updateformer2
            self.corr_mlp = trained_model.corr_mlp
        else:
            # Direct initialization from parameters
            super(TAPFormer_online, self).__init__(
                window_size=window_size,
                stride=stride,
                corr_radius=corr_radius,
                corr_levels=corr_levels,
                backbone=backbone,
                num_heads=num_heads,
                hidden_size=hidden_size,
                space_depth=space_depth,
                time_depth=time_depth,
                frontend_type=frontend_type
            )
        
        time_grid = torch.linspace(0, self.window_size - 1, self.window_size).reshape(1, self.window_size, 1)
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )
        self.corr_pyramid = []
        
    @torch.no_grad()  
    def forward(self, rgbs, events, queries, iters=6, img_ifnew=None, feat_init=None, interp_shape=(384, 512), is_train=False):
        # starter.record()
        if self.backbone == "image":
            self.updateformer2 = self.updateformer
        B, T, C, H, W = events.shape
        _, N, _ = queries.shape
        _, T, C_img, _, _ = rgbs.shape
        S = self.window_size
        step = S // 2
        device = queries.device
        
        queried_frames = queries[:, :, 0].long()
        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride
        
        coords_predicted = torch.zeros((B, T, N, 2), device=device)
        vis_predicted= torch.zeros((B, T, N), device=device)
        conf_predicted = torch.zeros((B, T, N), device=device)
        
        H_stride, W_stride = interp_shape[0] // self.stride, interp_shape[1] // self.stride
        
        coords_init = queries[:, :, 1:].reshape(B, 1, N, 2).repeat(1, self.window_size, 1, 1) / float(self.stride)
        
        vis_init = torch.zeros((B, S, N, 1), device=device).float()
        conf_init = torch.zeros((B, S, N, 1), device=device).float()
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()
        
        num_windows =  (T - S + step - 1) // step + 1
        indices = range(0, step * num_windows, step)
        
        fmaps_fusion = None
        first_window = True
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        
        for ind in indices:
            if ind > 0:
                overlap = S - step
                copy_over = (queried_frames < ind + overlap)[:, None, :, None]  # B 1 N 1
                coords_prev = coords_predicted[:, ind : ind + overlap] / self.stride
                padding_tensor = coords_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                coords_prev = torch.cat([coords_prev, padding_tensor], dim=1)

                vis_prev = vis_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = vis_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                vis_prev = torch.cat([vis_prev, padding_tensor], dim=1)

                conf_prev = conf_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = conf_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                conf_prev = torch.cat([conf_prev, padding_tensor], dim=1)

                coords_init = torch.where(copy_over.expand_as(coords_init), coords_prev, coords_init)
                vis_init = torch.where(copy_over.expand_as(vis_init), vis_prev, vis_init)
                conf_init = torch.where(copy_over.expand_as(conf_init), conf_prev, conf_init)
            
            events_seq = events[:, ind : ind + S]
            rgbs_seq = rgbs[:, ind : ind + S]
            if img_ifnew is not None:
                img_ifnew_seq = img_ifnew[ind : ind + S]
            if not isinstance(events_seq, torch.Tensor):
                events_seq = torch.from_numpy(events_seq).to(device).float()
                rgbs_seq = torch.from_numpy(rgbs_seq).to(device).float()
                events_seq = events_seq.contiguous()
                rgbs_seq = rgbs_seq.contiguous()
            
            if ind + S > T:
                pad = (S - rgbs_seq.shape[1]) % S
                rgbs_seq = rgbs_seq.reshape(B, 1, (S - pad), C_img * H * W)
                events_seq = events_seq.reshape(B, 1, (S - pad), C * H * W)
                padding_tensor = rgbs_seq[:, :, -1:, :].expand(B, 1, pad, C_img * H * W)
                rgbs_seq = torch.cat([rgbs_seq, padding_tensor], dim=2)
                padding_tensor = events_seq[:, :, -1:, :].expand(B, 1, pad, C * H * W)
                events_seq = torch.cat([events_seq, padding_tensor], dim=2)
                if img_ifnew is not None:
                    if isinstance(img_ifnew_seq, torch.Tensor):
                        padding = img_ifnew_seq.new_ones(pad)
                        img_ifnew_seq = torch.cat([img_ifnew_seq, padding], dim=0)
                    else:
                        padding_numpy = np.ones(pad)
                        img_ifnew_seq = np.concatenate((img_ifnew_seq, padding_numpy), axis=0)
                # rgbs_seq = rgbs_seq.reshape(B, -1, C_img, H, W)
                # events_seq = events_seq.reshape(B, -1, C, H, W)
            
            events_seq = events_seq.reshape(B * S, C, H, W)
            rgbs_seq = rgbs_seq.reshape(B * S, C_img, H, W)
            events_seq = F.interpolate(events_seq, tuple(interp_shape), mode='bilinear', align_corners=True)
            rgbs_seq = F.interpolate(rgbs_seq, tuple(interp_shape), mode='bilinear', align_corners=True)
            
            events_seq = 2 * events_seq - 1.0
            rgbs_seq = 2 * (rgbs_seq / 255.0) - 1.0
            
            dtype = rgbs_seq.dtype
            
            if fmaps_fusion is None or self.frontend_type in {"ts_query", "time_surface_query"}:
                fmaps_pyramid = self.fusion_block(rgbs_seq, events_seq, img_ifnew_seq if img_ifnew is not None else None)
                
                if isinstance(fmaps_pyramid, list):
                    for i, fmaps_fusion in enumerate(fmaps_pyramid):
                        fmaps_fusion = fmaps_fusion.permute(0, 2, 3, 1)
                        fmaps_fusion = fmaps_fusion / torch.sqrt(
                            torch.maximum(
                                torch.sum(torch.square(fmaps_fusion), axis=-1, keepdims=True),
                                torch.tensor(1e-12, device=fmaps_fusion.device),
                            )
                        )
                        fmaps_fusion = fmaps_fusion.permute(0, 3, 1, 2).reshape(
                            B, -1, self.latent_dim, int(H_stride / 2**i), int(W_stride / 2**i)
                        )
                        fmaps_fusion = fmaps_fusion.to(dtype)
                        fmaps_pyramid[i] = fmaps_fusion
                else:
                    fmaps_fusion = fmaps_pyramid.permute(0, 2, 3, 1)
                    fmaps_fusion = fmaps_fusion / torch.sqrt(
                        torch.maximum(
                            torch.sum(torch.square(fmaps_fusion), axis=-1, keepdims=True),
                            torch.tensor(1e-12, device=fmaps_fusion.device),
                        )
                    )
                    fmaps_fusion = fmaps_fusion.permute(0, 3, 1, 2).reshape(
                        B, -1, self.latent_dim, H_stride, W_stride
                    )
            else:
                rgbs_, events_ = rgbs_seq[S//2:], events_seq[S//2:]
                fmaps_pyramid_last = self.fusion_block(rgbs_, events_, img_ifnew_seq[S//2:] if img_ifnew is not None else None)
                
                if isinstance(fmaps_pyramid_last, list):
                    for i, fmaps_fusion_last in enumerate(fmaps_pyramid_last):
                        fmaps_fusion_last = fmaps_fusion_last.permute(0, 2, 3, 1)
                        fmaps_fusion_last = fmaps_fusion_last / torch.sqrt(
                            torch.maximum(
                                torch.sum(torch.square(fmaps_fusion_last), axis=-1, keepdims=True),
                                torch.tensor(1e-12, device=fmaps_fusion.device),
                            )
                        )
                        fmaps_fusion_last = fmaps_fusion_last.permute(0, 3, 1, 2).reshape(
                            B, -1, self.latent_dim, int(H_stride / 2**i), int(W_stride / 2**i)
                        )
                        fmaps_fusion = torch.cat([fmaps_pyramid[i][:, S//2:], fmaps_fusion_last], dim=1)
                        fmaps_fusion = fmaps_fusion.to(dtype)
                        fmaps_pyramid[i] = fmaps_fusion
                else:
                    fmaps_fusion_last = fmaps_pyramid_last.permute(0, 2, 3, 1)
                    fmaps_fusion_last = fmaps_fusion_last / torch.sqrt(
                        torch.maximum(
                            torch.sum(torch.square(fmaps_fusion_last), axis=-1, keepdims=True),
                            torch.tensor(1e-12, device=fmaps_fusion_last.device),
                        )
                    )
                    fmaps_fusion_last = fmaps_fusion_last.permute(0, 3, 1, 2).reshape(
                        B, -1, self.latent_dim, H_stride, W_stride
                    )
                    fmaps_fusion = torch.cat([fmaps_pyramid[0][:, S//2:], fmaps_fusion_last], dim=1)
                    fmaps_fusion = fmaps_fusion.to(dtype)
                    fmaps_pyramid = None
            
            if not isinstance(fmaps_pyramid, list):
                fmaps_pyramid = []
                fmaps_pyramid.append(fmaps_fusion)
                for i in range(self.corr_levels - 1):
                    fmaps_ = fmaps_fusion.reshape(B * S, self.latent_dim, fmaps_fusion.shape[-2], fmaps_fusion.shape[-1])
                    fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
                    fmaps_fusion = fmaps_.reshape(B, S, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1])
                    fmaps_pyramid.append(fmaps_fusion)
            if first_window:
                for i in range(self.corr_levels):
                    track_feat, track_feat_support = get_track_feat(fmaps_pyramid[i], queried_frames, queried_coords/2**i, support_radius=self.corr_radius)
                    track_feat_pyramid.append(track_feat.repeat(1, S, 1, 1))
                    track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))
                first_window = False
                
                
            attenstion_mask = (queried_frames < ind + S).reshape(B, 1, N) # B, 1, N
            coords, viss, confs = self.forward_window(
                fmaps_pyramid=[fmap for fmap in fmaps_pyramid],
                coords=coords_init,
                track_feat_support_pyramid=[attenstion_mask[:, None, :, :, None]*tfeat for tfeat in track_feat_support_pyramid],
                corr_map_pyramid=self.corr_pyramid,
                vis=vis_init,
                conf=conf_init,
                attenstion_mask=attenstion_mask.repeat(1, S, 1),
                iters=iters,
            )
            S_trimmed = min(T - ind, S)     # accounts for last window duration
            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = viss[-1][:, :S_trimmed]
            conf_predicted[:, ind : ind + S] = confs[-1][:, :S_trimmed]
            
        vis_predicted = torch.sigmoid(vis_predicted)
        conf_predicted = torch.sigmoid(conf_predicted)
        return coords_predicted, vis_predicted, conf_predicted
    
    def forward_window(self, fmaps_pyramid, coords, track_feat_support_pyramid, corr_map_pyramid, vis, conf, attenstion_mask, iters=6):
        B, S, *_ = fmaps_pyramid[0].shape
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1
        
        coord_preds, vis_preds, conf_preds = [], [], []
        for it in range(iters):
            coords = coords.detach()
            coord_init = coords.view(B * S, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(fmaps_pyramid[i], coord_init / 2 ** i)
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum("btnhwc,bnijc->btnhwij", corr_feat, track_feat_support).reshape(B, S, N, r * r, r * r)
                corr_emb = self.corr_mlp(corr_volume.reshape(B * S * N, r * r * r *r))
                corr_embs.append(corr_emb)      
                
            corr_embs = torch.cat(corr_embs, dim=1)
            corr_embs = corr_embs.view(B, S, N, corr_embs.shape[-1])
            
            transformer_input = [vis, conf, corr_embs]
            
            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]
            
            rel_coords_forward = torch.nn.functional.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
            rel_coords_backward = torch.nn.functional.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))
            
            scale = (torch.tensor([self.model_resolution[1], self.model_resolution[0]], device=coords.device,) / self.stride)
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale
            
            rel_pos_emb_input = posenc(torch.cat([rel_coords_forward, rel_coords_backward], dim=-1), min_deg=0, max_deg=10,)
            transformer_input.append(rel_pos_emb_input)
            
            x = (torch.cat(transformer_input, dim=-1).permute(0, 2, 1, 3).reshape(B*N, S, -1))
            
            x = x + self.interpolate_time_embed(x, S)
            x = x.view(B, N, S, -1)
            delta = self.updateformer2(x)
            
            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)
            
            vis = vis + delta_vis
            conf = conf + delta_conf

            coords = coords + delta_coords
            coord_preds.append(coords[..., :2] * float(self.stride))

            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])
        return coord_preds, vis_preds, conf_preds
            
    
    def load_parameters(self, model):
        # 从训练模型加载参数
        self.load_state_dict(model.state_dict())


class TAPFormerPointWarp_online(TAPFormerPointWarp):
    """Half-window online inference for the sparse point-warp model."""

    def __init__(self, *args, point_online_init_mode="overlap", **kwargs):
        super().__init__(*args, **kwargs)
        self.point_online_init_mode = str(point_online_init_mode).lower().strip()
        if self.point_online_init_mode != "overlap":
            raise ValueError("point_online_init_mode must be overlap for local-update point-warp.")

    @staticmethod
    def _pad_last(tensor, pad):
        if pad <= 0:
            return tensor
        return torch.cat(
            [tensor, tensor[:, -1:].expand(-1, pad, *tensor.shape[2:])],
            dim=1,
        )

    @staticmethod
    def _build_window_init(coords, vis, conf, step):
        overlap_values = [value[:, step:] for value in (coords, vis, conf)]
        new_values = [
            value[:, -1:].expand(-1, step, *value.shape[2:])
            for value in (coords, vis, conf)
        ]
        init_values = tuple(
            torch.cat([overlap, new], dim=1)
            for overlap, new in zip(overlap_values, new_values)
        )
        return (*init_values, None)

    @torch.no_grad()
    def forward(
        self,
        rgbs,
        events,
        queries,
        iters=6,
        img_ifnew=None,
        feat_init=None,
        interp_shape=(384, 512),
        is_train=False,
    ):
        del feat_init
        if is_train:
            raise ValueError("TAPFormerPointWarp_online is inference-only.")
        if iters is None:
            raise ValueError(
                "TAPFormerPointWarp_online.forward requires an explicit iters argument."
            )

        device = queries.device
        if not isinstance(rgbs, torch.Tensor):
            rgbs = torch.as_tensor(np.asarray(rgbs), device=device)
        if not isinstance(events, torch.Tensor):
            events = torch.as_tensor(np.asarray(events), device=device)
        if not torch.is_floating_point(rgbs):
            rgbs = rgbs.float()
        if not torch.is_floating_point(events):
            events = events.float()

        batch, frames, _, height, width = events.shape
        if batch != 1:
            raise AssertionError("TAPFormerPointWarp_online expects batch_size == 1.")
        if frames <= 0:
            raise ValueError("TAPFormerPointWarp_online requires at least one frame.")
        if tuple(interp_shape) != self.model_resolution:
            raise ValueError(
                f"interp_shape must match point-warp model resolution {self.model_resolution}."
            )
        if (
            (height, width) != self.model_resolution
            or rgbs.shape[:2] != (batch, frames)
            or rgbs.shape[-2:] != self.model_resolution
        ):
            raise ValueError(
                f"Expected input resolution {self.model_resolution}, got {(height, width)}."
            )
        query_xy = self._validate_queries(queries)

        if img_ifnew is None:
            img_ifnew = torch.ones(frames, device=device, dtype=rgbs.dtype)
        else:
            img_ifnew = torch.as_tensor(img_ifnew, device=device, dtype=rgbs.dtype)
        if img_ifnew.numel() != frames:
            raise ValueError("img_ifnew must contain one flag per input frame.")

        window_size = int(self.window_size)
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        step = max(1, window_size // 2)
        overlap = window_size - step
        num_windows = max(
            1,
            (max(frames - window_size, 0) + step - 1) // step + 1,
        )
        padded_frames = (num_windows - 1) * step + window_size
        pad = padded_frames - frames
        rgbs = self._pad_last(rgbs, pad)
        events = self._pad_last(events, pad)
        if pad > 0:
            img_ifnew = torch.cat([img_ifnew, img_ifnew.new_zeros(pad)])

        num_queries = queries.shape[1]
        coords_out = queries.new_zeros(batch, padded_frames, num_queries, 2)
        vis_logits_out = queries.new_zeros(batch, padded_frames, num_queries)
        conf_logits_out = queries.new_zeros(batch, padded_frames, num_queries)

        self._reset_fusion_state()
        anchor_pyramid = None
        cached_tail = None
        prev_coords = prev_vis = prev_conf = None
        for window_idx in range(num_windows):
            start = window_idx * step
            end = start + window_size
            if window_idx == 0:
                feature_pyramid = self._prepare_pyramid(
                    self._encode(
                        rgbs[:, start:end],
                        events[:, start:end],
                        img_ifnew[start:end],
                    )
                )
                anchor_pyramid = [feature[:, :1] for feature in feature_pyramid]
                init_values = (None, None, None, None)
            else:
                new_start = start + overlap
                new_pyramid = self._prepare_pyramid(
                    self._encode(
                        rgbs[:, new_start:end],
                        events[:, new_start:end],
                        img_ifnew[new_start:end],
                    )
                )
                if cached_tail is None:
                    raise RuntimeError("cached_tail is required for point-warp online inference.")
                feature_pyramid = [
                    torch.cat([tail, new], dim=1)
                    for tail, new in zip(cached_tail, new_pyramid)
                ]
                init_values = self._build_window_init(
                    prev_coords,
                    prev_vis,
                    prev_conf,
                    step,
                )

            coord_preds, vis_preds, conf_preds = self._track_from_pyramids(
                feature_pyramid,
                anchor_pyramid,
                query_xy,
                image_size=self.model_resolution,
                iters=int(iters),
                local_anchor=window_idx == 0,
                coords_init=init_values[0],
                vis_init=init_values[1],
                conf_init=init_values[2],
                init_mask=init_values[3],
            )
            prev_coords = coord_preds[-1]
            prev_vis = vis_preds[-1].unsqueeze(-1)
            prev_conf = conf_preds[-1].unsqueeze(-1)
            coords_out[:, start:end] = prev_coords
            vis_logits_out[:, start:end] = prev_vis[..., 0]
            conf_logits_out[:, start:end] = prev_conf[..., 0]
            cached_tail = [
                feature[:, -overlap:] if overlap else feature[:, :0]
                for feature in feature_pyramid
            ]

        return (
            coords_out[:, :frames],
            torch.sigmoid(vis_logits_out[:, :frames]),
            torch.sigmoid(conf_logits_out[:, :frames]),
        )


class TAPFormerCowDense_online(TAPFormerCowDense):
    def __init__(
        self,
        trained_model=None,
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
        cow_online_use_window_init=False,
        cow_online_use_global_first_anchor=False,
        cow_online_use_memory_features=False,
        cow_online_num_memory_frames=10,
        cow_window_stride=None,
        cow_window_num_memory_frames=None,
        cow_frontend_type="base",
        cow_anchor_state_mix=0.7,
        cow_anchor_skip_mix=0.7,
    ):
        del cow_window_stride, cow_window_num_memory_frames
        if trained_model is not None:
            super().__init__(
                window_size=trained_model.window_size,
                stride=trained_model.stride,
                corr_radius=corr_radius,
                corr_levels=corr_levels,
                backbone=backbone,
                num_heads=num_heads,
                hidden_size=hidden_size,
                space_depth=space_depth,
                time_depth=time_depth,
                cow_refine_model=cow_refine_model,
                cow_refine_patch_size=cow_refine_patch_size,
                cow_refine_blocks=cow_refine_blocks,
                cow_temporal_interleave_stride=cow_temporal_interleave_stride,
                cow_tracking_down_ratio=cow_tracking_down_ratio,
                cow_limit_flow=cow_limit_flow, 
                cow_max_flow_update_ratio=cow_max_flow_update_ratio,
                cow_max_flow_magnitude_ratio=cow_max_flow_magnitude_ratio,
                cow_refine_checkpoint=cow_refine_checkpoint,
                cow_info_update_mode=getattr(getattr(trained_model, "dense_head", None), "info_update_mode", cow_info_update_mode),
                cow_frontend_type=getattr(trained_model, "cow_frontend_type", cow_frontend_type),
                cow_anchor_state_mix=cow_anchor_state_mix,
                cow_anchor_skip_mix=cow_anchor_skip_mix,
            )
            self.fusion_block = trained_model.fusion_block
            self.dense_head = trained_model.dense_head
            self.cow_frontend_type = getattr(trained_model, "cow_frontend_type", self.cow_frontend_type)
        else:
            super().__init__(
                window_size=window_size,
                stride=stride,
                corr_radius=corr_radius,
                corr_levels=corr_levels,
                backbone=backbone,
                num_heads=num_heads,
                hidden_size=hidden_size,
                space_depth=space_depth,
                time_depth=time_depth,
                cow_refine_model=cow_refine_model,
                cow_refine_patch_size=cow_refine_patch_size,
                cow_refine_blocks=cow_refine_blocks,
                cow_temporal_interleave_stride=cow_temporal_interleave_stride,
                cow_tracking_down_ratio=cow_tracking_down_ratio,
                cow_limit_flow=cow_limit_flow,
                cow_max_flow_update_ratio=cow_max_flow_update_ratio,
                cow_max_flow_magnitude_ratio=cow_max_flow_magnitude_ratio,
                cow_refine_checkpoint=cow_refine_checkpoint,
                cow_info_update_mode=cow_info_update_mode,
                cow_tapir_init=cow_tapir_init,
                cow_tapir_init_stride=cow_tapir_init_stride,
                cow_tapir_init_temperature=cow_tapir_init_temperature,
                cow_tapir_init_radius=cow_tapir_init_radius,
                cow_tapir_init_chunk_size=cow_tapir_init_chunk_size,
                cow_frontend_type=cow_frontend_type,
                cow_anchor_state_mix=cow_anchor_state_mix,
                cow_anchor_skip_mix=cow_anchor_skip_mix,
            )

        self.cow_online_use_window_init = bool(cow_online_use_window_init)
        self.cow_online_use_global_first_anchor = bool(cow_online_use_global_first_anchor)
        self.cow_online_use_memory_features = bool(cow_online_use_memory_features)
        self.cow_online_num_memory_frames = int(cow_online_num_memory_frames)
        if self.cow_online_num_memory_frames < 0:
            raise ValueError("cow_online_num_memory_frames must be non-negative")

    @staticmethod
    def _select_memory_frame_indices(window_idx: int, window_start: int, num_memory_frames: int):
        if num_memory_frames <= 0 or window_idx == 0:
            return []

        memory_indices = [0]
        for offset in [2, 1]:
            idx = window_start - offset
            if idx > 0 and idx not in memory_indices:
                memory_indices.append(idx)

        if window_start > 10:
            mid_start, mid_end = 5, window_start - 3
            step = (mid_end - mid_start) / 6
            for i in range(5):
                idx = int(mid_start + (i + 1) * step)
                if idx not in memory_indices:
                    memory_indices.append(idx)

        if len(memory_indices) > num_memory_frames:
            memory_indices = sorted(memory_indices)[-num_memory_frames:]
        return sorted(memory_indices)

    @staticmethod
    def _gather_memory_features(feature_bank, memory_indices):
        features = [feature_bank[idx] for idx in memory_indices if idx in feature_bank]
        if not features:
            return None
        return torch.cat(features, dim=1)

    def _encode_global_first_anchor(self, rgbs, events):
        img_ifnew = torch.ones(1, device=rgbs.device, dtype=rgbs.dtype)
        return self._encode_window_features(rgbs[:, :1], events[:, :1], img_ifnew=img_ifnew, reset_state=True)

    @staticmethod
    def _neutral_track_init(batch_size: int, total_len: int, height: int, width: int, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([x, y], dim=-1)
        return coords.unsqueeze(0).unsqueeze(0).expand(batch_size, total_len, -1, -1, -1).clone()

    @staticmethod
    def _neutral_prob_init(batch_size: int, total_len: int, height: int, width: int, device, dtype):
        return torch.full((batch_size, total_len, height, width), 0.5, device=device, dtype=dtype)

    @staticmethod
    def _build_window_init(
        dense_track,
        dense_vis,
        dense_conf,
        window_len: int,
        overlap: int,
        memory_len: int,
        height: int,
        width: int,
        device,
        dtype,
        window_prefix_len: int = 0,
        return_valid_mask: bool = False,
    ):
        total_len = int(memory_len) + int(window_len)

        def pack(track, vis, conf, valid_mask):
            if return_valid_mask:
                return track, vis, conf, valid_mask
            return track, vis, conf

        if total_len <= 0:
            return pack(None, None, None, None)

        if memory_len <= 0 and (
            dense_track is None or dense_vis is None or dense_conf is None or overlap <= 0
        ):
            return pack(None, None, None, None)

        init_track = TAPFormerCowDense_online._neutral_track_init(
            1, total_len, height, width, device, dtype
        )
        init_vis = TAPFormerCowDense_online._neutral_prob_init(
            1, total_len, height, width, device, dtype
        )
        init_conf = TAPFormerCowDense_online._neutral_prob_init(
            1, total_len, height, width, device, dtype
        )
        valid_mask = torch.zeros(
            1, total_len, height, width, device=device, dtype=torch.bool
        )

        if dense_track is None or dense_vis is None or dense_conf is None or overlap <= 0:
            return pack(init_track, init_vis, init_conf, valid_mask)

        prefix_len = min(max(int(window_prefix_len), 0), int(window_len))
        copy_len = min(
            int(overlap),
            int(window_len) - prefix_len,
            int(dense_track.shape[1]),
        )
        if copy_len <= 0:
            return pack(init_track, init_vis, init_conf, valid_mask)

        start = int(memory_len) + prefix_len
        end = start + copy_len
        init_track[:, start:end] = dense_track[:, -copy_len:].to(device=device, dtype=dtype)
        init_vis[:, start:end] = dense_vis[:, -copy_len:].to(device=device, dtype=dtype)
        init_conf[:, start:end] = dense_conf[:, -copy_len:].to(device=device, dtype=dtype)
        valid_mask[:, start:end] = True

        pad_len = int(window_len) - prefix_len - copy_len
        if pad_len > 0:
            init_track[:, end : end + pad_len] = init_track[:, end - 1 : end].expand(
                -1, pad_len, -1, -1, -1
            )
            init_vis[:, end : end + pad_len] = init_vis[:, end - 1 : end].expand(
                -1, pad_len, -1, -1
            )
            init_conf[:, end : end + pad_len] = init_conf[:, end - 1 : end].expand(
                -1, pad_len, -1, -1
            )

        return pack(init_track, init_vis, init_conf, valid_mask)

    @staticmethod
    def _slice_dense_debug_window(dense_debug, memory_len: int):
        if dense_debug is None or memory_len <= 0:
            return dense_debug
        return {
            "dense_tracks": dense_debug["dense_tracks"][:, memory_len:].clone(),
            "dense_vis": dense_debug["dense_vis"][:, memory_len:].clone(),
            "dense_conf": dense_debug["dense_conf"][:, memory_len:].clone(),
        }

    @torch.no_grad()
    def forward(
        self,
        rgbs,
        events,
        queries,
        iters=4,
        img_ifnew=None,
        feat_init=None,
        interp_shape=(384, 512),
        is_train=False,
        return_merge_variants=False,
    ):
        device = queries.device
        if not isinstance(rgbs, torch.Tensor):
            rgbs = torch.as_tensor(np.asarray(rgbs), device=device)
        if not isinstance(events, torch.Tensor):
            events = torch.as_tensor(np.asarray(events), device=device)
        if not torch.is_floating_point(rgbs):
            rgbs = rgbs.float()
        if not torch.is_floating_point(events):
            events = events.float()

        B, T, C_event, H, W = events.shape
        _, N, _ = queries.shape
        _, _, C_img, _, _ = rgbs.shape
        if B != 1:
            raise AssertionError("TAPFormerCowDense_online expects batch_size == 1")

        queries = queries.clone()
        rgbs_flat = rgbs.reshape(B * T, C_img, H, W)
        events_flat = events.reshape(B * T, C_event, H, W)
        rgbs_flat = F.interpolate(rgbs_flat, tuple(interp_shape), mode="bilinear", align_corners=True)
        events_flat = F.interpolate(events_flat, tuple(interp_shape), mode="bilinear", align_corners=True)
        rgbs = rgbs_flat.reshape(B, T, C_img, interp_shape[0], interp_shape[1])
        events = events_flat.reshape(B, T, C_event, interp_shape[0], interp_shape[1])

        queries[:, :, 1] *= (interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (interp_shape[0] - 1) / (H - 1)
        query_xy = self._prepare_query_xy(queries)

        if img_ifnew is None:
            img_ifnew = torch.ones(T, device=rgbs.device, dtype=rgbs.dtype)
        elif isinstance(img_ifnew, torch.Tensor):
            img_ifnew = img_ifnew.to(device=rgbs.device, dtype=rgbs.dtype)
        else:
            img_ifnew = torch.as_tensor(np.asarray(img_ifnew), device=rgbs.device).to(dtype=rgbs.dtype)

        coords_predicted = torch.zeros((B, T, N, 2), device=device, dtype=queries.dtype)
        vis_predicted = torch.zeros((B, T, N), device=device, dtype=queries.dtype)
        conf_predicted = torch.zeros((B, T, N), device=device, dtype=queries.dtype)
        coords_predicted_overwrite = None
        vis_predicted_overwrite = None
        conf_predicted_overwrite = None
        if return_merge_variants:
            coords_predicted_overwrite = torch.zeros_like(coords_predicted)
            vis_predicted_overwrite = torch.zeros_like(vis_predicted)
            conf_predicted_overwrite = torch.zeros_like(conf_predicted)

        S = int(self.window_size)
        step = max(1, S // 2)
        num_windows = max(1, (max(T - S, 0) + step - 1) // step + 1)
        prev_end = 0
        cached_tail = None
        prev_dense_track = None
        prev_dense_vis = None
        prev_dense_conf = None
        tapir_init_enabled = bool(getattr(self.dense_head, "tapir_init_enabled", False))
        feature_bank = {} if self.cow_online_use_memory_features else None
        global_first_feature = None
        if self.cow_online_use_global_first_anchor:
            global_first_feature = self._encode_global_first_anchor(rgbs, events)
            if feature_bank is not None:
                feature_bank[0] = global_first_feature[:, :1].clone()

        for window_idx, ind in enumerate(range(0, step * num_windows, step)):
            if ind >= T:
                break

            end = min(ind + S, T)
            overlap = max(0, prev_end - ind)
            init_track = init_vis = init_conf = None
            init_valid_mask = None
            encoded_features = None
            encoded_start = None
            if ind == 0:
                window_query_xy = query_xy
                window_features = self._encode_window_features(
                    rgbs[:, ind:end],
                    events[:, ind:end],
                    img_ifnew=img_ifnew[ind:end],
                    reset_state=True,
                )
                encoded_features = window_features
                encoded_start = ind
            else:
                if self.cow_online_use_global_first_anchor:
                    window_query_xy = query_xy
                else:
                    window_query_xy = coords_predicted[:, ind].clone()
                new_start = ind + overlap
                new_features = self._encode_window_features(
                    rgbs[:, new_start:end],
                    events[:, new_start:end],
                    img_ifnew=img_ifnew[new_start:end],
                    reset_state=False,
                )
                if cached_tail is None:
                    raise RuntimeError("cached_tail is required for incremental cow-dense online inference")
                window_features = torch.cat([cached_tail, new_features], dim=1)
                encoded_features = new_features
                encoded_start = new_start

            if feature_bank is not None and encoded_features is not None and encoded_start is not None:
                for offset in range(encoded_features.shape[1]):
                    global_idx = encoded_start + offset
                    if global_idx >= T:
                        continue
                    if global_idx == 0 and global_first_feature is not None:
                        continue
                    feature_bank[global_idx] = encoded_features[:, offset : offset + 1].clone()

            memory_features = None
            memory_len = 0
            if feature_bank is not None:
                memory_indices = self._select_memory_frame_indices(window_idx, ind, self.cow_online_num_memory_frames)
                if global_first_feature is not None:
                    memory_indices = [idx for idx in memory_indices if idx != 0]
                memory_features = self._gather_memory_features(feature_bank, memory_indices)
                if memory_features is not None:
                    memory_len = int(memory_features.shape[1])

            tracked_features = window_features if memory_features is None else torch.cat([memory_features, window_features], dim=1)

            if self.cow_online_use_window_init:
                init_values = self._build_window_init(
                    prev_dense_track,
                    prev_dense_vis,
                    prev_dense_conf,
                    window_len=window_features.shape[1],
                    overlap=overlap,
                    memory_len=memory_len,
                    height=interp_shape[0],
                    width=interp_shape[1],
                    device=tracked_features.device,
                    dtype=tracked_features.dtype,
                    return_valid_mask=tapir_init_enabled,
                )
                if tapir_init_enabled:
                    init_track, init_vis, init_conf, init_valid_mask = init_values
                else:
                    init_track, init_vis, init_conf = init_values

            window_anchor_features = global_first_feature
            if window_anchor_features is None and memory_len > 0:
                window_anchor_features = window_features[:, :1]

            traj, vis, conf, _, _, _, dense_debug = self._forward_window(
                tracked_features,
                window_query_xy,
                image_size=interp_shape,
                iters=int(iters),
                init_track=init_track,
                init_vis=init_vis,
                init_conf=init_conf,
                init_valid_mask=init_valid_mask,
                first_frame_features=window_anchor_features,
                return_debug=self.cow_online_use_window_init,
            )

            if memory_len > 0:
                traj = traj[:, memory_len:]
                vis = vis[:, memory_len:]
                conf = conf[:, memory_len:]
                dense_debug = self._slice_dense_debug_window(dense_debug, memory_len)

            traj_window = traj[:, : end - ind].clone()
            vis_window = vis[:, : end - ind].clone()
            conf_window = conf[:, : end - ind].clone()
            if return_merge_variants:
                coords_predicted_overwrite[:, ind:end] = traj_window
                vis_predicted_overwrite[:, ind:end] = vis_window
                conf_predicted_overwrite[:, ind:end] = conf_window

            if overlap > 0:
                traj_window[:, :overlap] = coords_predicted[:, ind:ind + overlap]
                vis_window[:, :overlap] = vis_predicted[:, ind:ind + overlap]
                conf_window[:, :overlap] = conf_predicted[:, ind:ind + overlap]

            coords_predicted[:, ind:end] = traj_window
            vis_predicted[:, ind:end] = vis_window
            conf_predicted[:, ind:end] = conf_window
            if self.cow_online_use_window_init:
                if dense_debug is None:
                    raise RuntimeError("dense_debug is required when cow_online_use_window_init is enabled")
                prev_dense_track = dense_debug["dense_tracks"].clone()
                prev_dense_vis = dense_debug["dense_vis"].clone()
                prev_dense_conf = dense_debug["dense_conf"].clone()
            cached_tail = window_features[:, -min(step, window_features.shape[1]):].clone()
            prev_end = end

        if return_merge_variants:
            merge_variants = {
                "keep": (
                    coords_predicted.clone(),
                    vis_predicted.clone(),
                    conf_predicted.clone(),
                ),
                "overwrite": (
                    coords_predicted_overwrite,
                    vis_predicted_overwrite,
                    conf_predicted_overwrite,
                ),
            }
            return coords_predicted, vis_predicted, conf_predicted, merge_variants
        return coords_predicted, vis_predicted, conf_predicted


class TAPFormerCowDense_windowed(TAPFormerCowDense):
    def __init__(
        self,
        trained_model=None,
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
        cow_window_stride=None,
        cow_window_num_memory_frames=None,
        cow_online_use_window_init=False,
        cow_online_use_global_first_anchor=False,
        cow_online_use_memory_features=False,
        cow_online_num_memory_frames=10,
        cow_frontend_type="base",
        cow_anchor_state_mix=0.7,
        cow_anchor_skip_mix=0.7,
    ):
        del cow_online_use_global_first_anchor, cow_online_use_memory_features

        if trained_model is not None:
            super().__init__(
                window_size=trained_model.window_size,
                stride=trained_model.stride,
                corr_radius=corr_radius,
                corr_levels=corr_levels,
                backbone=backbone,
                num_heads=num_heads,
                hidden_size=hidden_size,
                space_depth=space_depth,
                time_depth=time_depth,
                cow_refine_model=cow_refine_model,
                cow_refine_patch_size=cow_refine_patch_size,
                cow_refine_blocks=cow_refine_blocks,
                cow_temporal_interleave_stride=cow_temporal_interleave_stride,
                cow_tracking_down_ratio=cow_tracking_down_ratio,
                cow_limit_flow=cow_limit_flow,
                cow_max_flow_update_ratio=cow_max_flow_update_ratio,
                cow_max_flow_magnitude_ratio=cow_max_flow_magnitude_ratio,
                cow_refine_checkpoint=cow_refine_checkpoint,
                cow_info_update_mode=getattr(getattr(trained_model, "dense_head", None), "info_update_mode", cow_info_update_mode),
                cow_frontend_type=getattr(trained_model, "cow_frontend_type", cow_frontend_type),
                cow_anchor_state_mix=cow_anchor_state_mix,
                cow_anchor_skip_mix=cow_anchor_skip_mix,
            )
            self.fusion_block = trained_model.fusion_block
            self.dense_head = trained_model.dense_head
            self.cow_frontend_type = getattr(trained_model, "cow_frontend_type", self.cow_frontend_type)
        else:
            super().__init__(
                window_size=window_size,
                stride=stride,
                corr_radius=corr_radius,
                corr_levels=corr_levels,
                backbone=backbone,
                num_heads=num_heads,
                hidden_size=hidden_size,
                space_depth=space_depth,
                time_depth=time_depth,
                cow_refine_model=cow_refine_model,
                cow_refine_patch_size=cow_refine_patch_size,
                cow_refine_blocks=cow_refine_blocks,
                cow_temporal_interleave_stride=cow_temporal_interleave_stride,
                cow_tracking_down_ratio=cow_tracking_down_ratio,
                cow_limit_flow=cow_limit_flow,
                cow_max_flow_update_ratio=cow_max_flow_update_ratio,
                cow_max_flow_magnitude_ratio=cow_max_flow_magnitude_ratio,
                cow_refine_checkpoint=cow_refine_checkpoint,
                cow_info_update_mode=cow_info_update_mode,
                cow_tapir_init=cow_tapir_init,
                cow_tapir_init_stride=cow_tapir_init_stride,
                cow_tapir_init_temperature=cow_tapir_init_temperature,
                cow_tapir_init_radius=cow_tapir_init_radius,
                cow_tapir_init_chunk_size=cow_tapir_init_chunk_size,
                cow_frontend_type=cow_frontend_type,
                cow_anchor_state_mix=cow_anchor_state_mix,
                cow_anchor_skip_mix=cow_anchor_skip_mix,
            )

        resolved_stride = cow_window_stride
        if resolved_stride is None:
            resolved_stride = max(1, int(self.window_size) // 2)
        self.cow_window_stride = int(resolved_stride)
        if self.cow_window_stride <= 0:
            raise ValueError("cow_window_stride must be positive")

        resolved_memory = cow_window_num_memory_frames
        if resolved_memory is None:
            resolved_memory = cow_online_num_memory_frames
        self.cow_window_num_memory_frames = int(resolved_memory)
        if self.cow_window_num_memory_frames < 0:
            raise ValueError("cow_window_num_memory_frames must be non-negative")
        self.cow_online_use_window_init = bool(cow_online_use_window_init)

    @staticmethod
    def _compute_windows(total_frames: int, window_len: int, stride: int):
        if total_frames <= window_len:
            return [(0, total_frames)]

        last_start = total_frames - window_len
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)
        return [(start, start + window_len) for start in starts]

    @staticmethod
    def _select_memory_frames(window_idx: int, window_start: int, num_memory_frames: int):
        if num_memory_frames <= 0 or window_idx == 0:
            return []

        recent_window_span = 4
        recent_start = max(0, window_start - recent_window_span)
        recent_indices = list(range(recent_start, window_start))
        if len(recent_indices) > num_memory_frames:
            return recent_indices[:num_memory_frames]

        recent_set = set(recent_indices)
        remaining = num_memory_frames - len(recent_indices)
        long_term_candidates = [
            idx for idx in range(0, recent_start, recent_window_span) if idx not in recent_set
        ]
        if remaining > 0 and long_term_candidates:
            long_term_indices = long_term_candidates[-remaining:]
        else:
            long_term_indices = []

        return long_term_indices + recent_indices

    @staticmethod
    def _find_aligned_start(window_start: int, img_ifnew):
        aligned_start = int(window_start)
        while aligned_start > 0:
            value = img_ifnew[aligned_start]
            if isinstance(value, torch.Tensor):
                value = value.item()
            if float(value) == 1.0:
                break
            aligned_start -= 1
        return aligned_start

    def _gather_window_inputs(self, rgbs, events, img_ifnew, sequence_start: int, end: int):
        packed_indices = list(range(sequence_start, end))
        target_len = end - sequence_start
        if target_len < self.window_size:
            packed_indices.extend([end - 1] * (self.window_size - target_len))

        index = torch.as_tensor(packed_indices, device=rgbs.device, dtype=torch.long)
        return (
            rgbs.index_select(1, index),
            events.index_select(1, index),
            img_ifnew.index_select(0, index),
        )

    @staticmethod
    def _merge_window_predictions(window_start: int, window_end: int, previous_end: int, window_pred, accumulated):
        overlap = max(0, previous_end - window_start)
        accumulated[:, window_start + overlap : window_end] = window_pred[:, overlap : window_end - window_start]

    @torch.no_grad()
    def forward(
        self,
        rgbs,
        events,
        queries,
        iters=4,
        img_ifnew=None,
        feat_init=None,
        interp_shape=(384, 512),
        is_train=False,
        return_merge_variants=False,
    ):
        del feat_init, is_train
        if return_merge_variants:
            raise ValueError(
                "TAPFormerCowDense_windowed does not support return_merge_variants. "
                "Use TAPFormerCowDense_online for dual merge exports."
            )

        device = queries.device
        if not isinstance(rgbs, torch.Tensor):
            rgbs = torch.as_tensor(np.asarray(rgbs), device=device)
        if not isinstance(events, torch.Tensor):
            events = torch.as_tensor(np.asarray(events), device=device)
        if not torch.is_floating_point(rgbs):
            rgbs = rgbs.float()
        if not torch.is_floating_point(events):
            events = events.float()

        B, T, C_event, H, W = events.shape
        _, N, _ = queries.shape
        _, _, C_img, _, _ = rgbs.shape
        if B != 1:
            raise AssertionError("TAPFormerCowDense_windowed expects batch_size == 1")

        queries = queries.clone()
        rgbs_flat = rgbs.reshape(B * T, C_img, H, W)
        events_flat = events.reshape(B * T, C_event, H, W)
        rgbs_flat = F.interpolate(rgbs_flat, tuple(interp_shape), mode="bilinear", align_corners=True)
        events_flat = F.interpolate(events_flat, tuple(interp_shape), mode="bilinear", align_corners=True)
        rgbs = rgbs_flat.reshape(B, T, C_img, interp_shape[0], interp_shape[1])
        events = events_flat.reshape(B, T, C_event, interp_shape[0], interp_shape[1])

        queries[:, :, 1] *= (interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (interp_shape[0] - 1) / (H - 1)
        query_xy = self._prepare_query_xy(queries)

        if img_ifnew is None:
            img_ifnew = torch.ones(T, device=rgbs.device, dtype=rgbs.dtype)
        elif isinstance(img_ifnew, torch.Tensor):
            img_ifnew = img_ifnew.to(device=rgbs.device, dtype=rgbs.dtype)
        else:
            img_ifnew = torch.as_tensor(np.asarray(img_ifnew), device=rgbs.device).to(dtype=rgbs.dtype)

        coords_predicted = torch.zeros((B, T, N, 2), device=device, dtype=queries.dtype)
        vis_predicted = torch.zeros((B, T, N), device=device, dtype=queries.dtype)
        conf_predicted = torch.zeros((B, T, N), device=device, dtype=queries.dtype)
        prev_dense_track = None
        prev_dense_vis = None
        prev_dense_conf = None
        tapir_init_enabled = bool(getattr(self.dense_head, "tapir_init_enabled", False))

        first_frame_features = self._encode_window_features(
            rgbs[:, :1],
            events[:, :1],
            img_ifnew=img_ifnew[:1],
            reset_state=True,
        )
        windows = self._compute_windows(T, int(self.window_size), int(self.cow_window_stride))
        previous_end = 0
        for window_idx, (start, end) in enumerate(windows):
            memory_indices = self._select_memory_frames(window_idx, start, self.cow_window_num_memory_frames)
            aligned_start = self._find_aligned_start(start, img_ifnew)
            long_term_indices = []
            for idx in memory_indices:
                if idx == 0:
                    continue
                if idx >= aligned_start:
                    continue
                value = img_ifnew[idx]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if float(value) == 1.0:
                    long_term_indices.append(idx)

            long_term_features = []
            for idx in long_term_indices:
                long_term_features.append(
                    self._encode_window_features(
                        rgbs[:, idx : idx + 1],
                        events[:, idx : idx + 1],
                        img_ifnew=img_ifnew[idx : idx + 1],
                        reset_state=True,
                    )
                )

            window_rgbs, window_events, window_img_ifnew = self._gather_window_inputs(
                rgbs,
                events,
                img_ifnew,
                aligned_start,
                end,
            )
            short_and_current_features = self._encode_window_features(
                window_rgbs,
                window_events,
                img_ifnew=window_img_ifnew,
                reset_state=True,
            )
            if long_term_features:
                gathered_features = torch.cat(long_term_features + [short_and_current_features], dim=1)
            else:
                gathered_features = short_and_current_features

            init_track = init_vis = init_conf = None
            init_valid_mask = None
            if self.cow_online_use_window_init:
                overlap = max(0, previous_end - start)
                init_values = TAPFormerCowDense_online._build_window_init(
                    prev_dense_track,
                    prev_dense_vis,
                    prev_dense_conf,
                    window_len=short_and_current_features.shape[1],
                    overlap=overlap,
                    memory_len=len(long_term_indices) if tapir_init_enabled else 0,
                    height=interp_shape[0],
                    width=interp_shape[1],
                    device=gathered_features.device,
                    dtype=gathered_features.dtype,
                    window_prefix_len=start - aligned_start if tapir_init_enabled else 0,
                    return_valid_mask=tapir_init_enabled,
                )
                if tapir_init_enabled:
                    init_track, init_vis, init_conf, init_valid_mask = init_values
                else:
                    init_track, init_vis, init_conf = init_values

            traj, vis, conf, _, _, _, dense_debug = self._forward_window(
                gathered_features,
                query_xy,
                image_size=interp_shape,
                iters=int(iters),
                init_track=init_track,
                init_vis=init_vis,
                init_conf=init_conf,
                init_valid_mask=init_valid_mask,
                first_frame_features=first_frame_features,
                return_debug=self.cow_online_use_window_init,
            )
            prefix_len = len(long_term_indices) + (start - aligned_start)
            traj_window = traj[:, prefix_len : prefix_len + end - start].clone()
            vis_window = vis[:, prefix_len : prefix_len + end - start].clone()
            conf_window = conf[:, prefix_len : prefix_len + end - start].clone()

            self._merge_window_predictions(start, end, previous_end, traj_window, coords_predicted)
            self._merge_window_predictions(start, end, previous_end, vis_window, vis_predicted)
            self._merge_window_predictions(start, end, previous_end, conf_window, conf_predicted)
            if self.cow_online_use_window_init:
                if dense_debug is None:
                    raise RuntimeError("dense_debug is required when cow_online_use_window_init is enabled")
                prev_dense_track = dense_debug["dense_tracks"][:, prefix_len:].clone()
                prev_dense_vis = dense_debug["dense_vis"][:, prefix_len:].clone()
                prev_dense_conf = dense_debug["dense_conf"][:, prefix_len:].clone()
            previous_end = max(previous_end, end)

        return coords_predicted, vis_predicted, conf_predicted
