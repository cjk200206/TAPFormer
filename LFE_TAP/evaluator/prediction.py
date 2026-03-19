import torch
import time
import torch.nn.functional as F
import numpy as np

from LFE_TAP.models.tapformer import TAPFormer, posenc
from LFE_TAP.utils.model_utils import get_track_feat, normalize_voxels
from LFE_TAP.models.embeddings import get_1d_sincos_pos_embed_from_grid

torch.manual_seed(0)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

class TAPFormer_online(TAPFormer):
    def __init__(self, trained_model=None, window_size=16, stride=4, corr_radius=3, corr_levels=3, 
                 backbone="basic", num_heads=8, hidden_size=384, space_depth=3, time_depth=3):
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
                time_depth=trained_model.time_depth
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
                time_depth=time_depth
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
            
            if fmaps_fusion is None:
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