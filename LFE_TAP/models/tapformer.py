import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from LFE_TAP.models.blocks import (
    BasicEncoder,
    FusionBlock,
    FusionBlock_basic,
    EfficientUpdateFormer,
    UpdateFormer,
    Mlp,
)
from LFE_TAP.models.fusionFormer import Fusionformer
from LFE_TAP.utils.model_utils import get_track_feat, bilinear_sampler, get_support_points
from LFE_TAP.models.embeddings import get_1d_sincos_pos_embed_from_grid

torch.manual_seed(0)


def posenc(x, min_deg, max_deg):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
    Args:
      x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.
    Returns:
      encoded: torch.Tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )

    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


class TAPFormer(nn.Module):
    def __init__(self, window_size=16, stride=8, corr_radius=3, corr_levels=3, backbone="basic", num_heads=8, hidden_size=384, space_depth=3, time_depth=3):
        super(TAPFormer, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.hidden_size = hidden_size
        self.space_depth = space_depth
        self.time_depth = time_depth
        self.latent_dim = 128
        self.backbone = backbone
        self.model_resolution = (384, 512)
        self.mlp_output_dim = 256
        self.input_dim = 2 + 84 + self.mlp_output_dim * self.corr_levels
        num_virtual_tracks = 32
        

        self.fusion_block = Fusionformer(image_size=self.model_resolution, out_dim=self.latent_dim, mlp_dim=512, stride=self.stride, depth=2)
        self.updateformer2 = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.input_dim,
            hidden_size=hidden_size,
            output_dim=4,
            mlp_ratio=4.0,
            num_virtual_tracks=num_virtual_tracks,
            linear_layer_for_vis_conf=True,
        )
        
        # self.norm = nn.GroupNorm(1, self.latent_dim)
        # self.ffeat_updater = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim),
        #     nn.ReLU(),
        # )
        self.corr_mlp = Mlp(in_features=(2*corr_radius+1) ** 4, hidden_features=384, out_features=256)
        
        time_grid = torch.linspace(0, window_size - 1, window_size).reshape(1, window_size, 1)
        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )
        
    def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)    
        
    def get_correlation_feat(self, fmaps, queried_coords):
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )
        
    def forward_window(self, fmaps_pyramid, coords, track_feat_support_pyramid, vis, conf, attenstion_mask, iters=4):
        B, S, D, *_ = fmaps_pyramid[0].shape
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1
        
        coord_preds, vis_preds, conf_preds = [], [], []
        for it in range(iters):
            coords = coords.detach()
            coord_init = coords.view(B * S, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(fmaps_pyramid[i], coord_init / 2**i)
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum("btnhwc,bnijc->btnhwij", corr_feat, track_feat_support)
                corr_emb = self.corr_mlp(corr_volume.reshape(B * S * N, r * r * r *r))
                # del corr_volume, corr_feat
                # torch.cuda.empty_cache()
                corr_embs.append(corr_emb)        
                
            corr_embs = torch.cat(corr_embs, dim=1)
            corr_embs = corr_embs.view(B, S, N, corr_embs.shape[-1])
            
            transformer_input = [vis, conf, corr_embs]
            
            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]
            
            rel_coords_forward = torch.nn.functional.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
            rel_coords_backward = torch.nn.functional.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))
            
            scale = (torch.tensor([self.model_resolution[1], self.model_resolution[0]], device=coords.device,) / self.stride)
            rel_coords_forward = rel_coords_forward / scale     # 归一化到[-1, 1]
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
    
    def forward(self, rgbs, events, queries, iters=4, img_ifnew=None, feat_init=None, is_train=False):
        B, T, C, H, W = events.shape
        B, N, _ = queries.shape
        _, _, C_img, _, _ = rgbs.shape
        S = self.window_size
        step = S // 2
        device = events.device
        assert H % self.stride == 0 and W % self.stride == 0
        assert B == 1, "batch size should be 1"
        
        queried_frames = queries[:, :, 0].long()
        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride
        
        coords_predicted = torch.zeros((B, T, N, 2), device=device)
        vis_predicted= torch.zeros((B, T, N), device=device)
        conf_predicted = torch.zeros((B, T, N), device=device)
        
        all_coords_predictions, all_vis_predictions, all_confidence_predictions = ([], [], [])
        H_stride, W_stride = H // self.stride, W // self.stride
        
        rgbs = 2 * (rgbs / 255.0) - 1.0
        # events = torch.sigmoid(events)
        events = 2 * events - 1.0
        dtype = rgbs.dtype
        
        # fusion event and image to get fusion feature
        # start = time.time()
        fmaps = self.fusion_block(rgbs.reshape(-1, C_img, H, W), events.reshape(-1, C, H, W), img_ifnew)
        # print(f"fusion block time: {time.time() - start}")
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H_stride, W_stride
        )
        fmaps = fmaps.to(dtype)
        
        # compute queries point feature
        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1])
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1])
            fmaps_pyramid.append(fmaps)
            
        for i in range(self.corr_levels):
            track_feat, track_feat_support = get_track_feat(fmaps_pyramid[i], queried_frames, queried_coords/2**i, support_radius=self.corr_radius)
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))
            
        vis_init = torch.zeros((B, S, N, 1), device=device).float()
        conf_init = torch.zeros((B, S, N, 1), device=device).float()
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()
        
        num_windows =  (T - S + step - 1) // step + 1
        indices = range(0, step * num_windows, step)
        
        for ind in indices:
            if ind > 0:
                overlap = S - step
                copy_over = (queried_frames < ind + overlap)[:, None, :, None]   # B, 1, N, 1
                coords_prev = coords_predicted[:, ind : ind + overlap] / self.stride
                padding_tensor = coords_prev[:, -1:, :, :].expand(-1, step, -1, -1)  # 将上一个时刻的坐标作为待优化坐标的初始值
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
                
            attenstion_mask = (queried_frames < ind + S).reshape(B, 1, N) # B, 1, N
            # start = time.time()
            coords, viss, confs = self.forward_window(
                fmaps_pyramid=[fmap[:, ind : ind +S] for fmap in fmaps_pyramid],
                coords=coords_init,
                track_feat_support_pyramid=[attenstion_mask[:, None, :, :, None]*tfeat for tfeat in track_feat_support_pyramid],
                vis=vis_init,
                conf=conf_init,
                attenstion_mask=attenstion_mask.repeat(1, S, 1),
                iters=iters,
            )
            # print(f"forward window time: {time.time() - start}")
            S_trimmed = min(T - ind, S)     # accounts for last window duration
            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = viss[-1][:, :S_trimmed]
            conf_predicted[:, ind : ind + S] = confs[-1][:, :S_trimmed]
            if is_train:
                all_coords_predictions.append(
                    [coord[:, :S_trimmed] for coord in coords]
                )
                all_vis_predictions.append(
                    [torch.sigmoid(vis[:, :S_trimmed]) for vis in viss]
                )
                all_confidence_predictions.append(
                    [torch.sigmoid(conf[:, :S_trimmed]) for conf in confs]
                )
                
        vis_predicted = torch.sigmoid(vis_predicted)
        conf_predicted = torch.sigmoid(conf_predicted)
        
        if is_train:
            valid_mask = (
                queried_frames[:, None]
                <= torch.arange(0, T, device=device)[None, :, None]
            )
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                valid_mask,
            )
        else:
            train_data = None
            
        return coords_predicted, vis_predicted, conf_predicted, train_data
        
        
