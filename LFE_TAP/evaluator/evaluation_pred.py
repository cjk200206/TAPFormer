import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple

from LFE_TAP.models.tapformer import TAPFormer
from LFE_TAP.utils.model_utils import get_points_on_a_grid, get_sift_sampled_pts, get_uniformly_sampled_pts, normalize_voxels

class EvaluationPredictor(torch.nn.Module):
    def __init__(
        self,
        model: TAPFormer,
        interp_shape: Tuple[int, int] = [384, 512],
        grid_size: int = 5,
        local_grid_size: int = 8,
        single_point: bool = True,
        sift_size: int = 0,
        num_uniformly_sampled_pts: int = 0,
        n_iters: int = 6,
        local_extent: int = 50,
        if_test: bool = False,
        input_mode: str = "fusion",
    ) -> None:
        super(EvaluationPredictor, self).__init__()
        self.model = model
        self.interp_shape = interp_shape
        self.grid_size = grid_size
        self.local_grid_size = local_grid_size
        self.single_point = single_point
        self.sift_size = sift_size
        self.num_uniformly_sampled_pts = num_uniformly_sampled_pts
        self.n_iters = n_iters
        self.local_extent = local_extent
        self.if_test = if_test
        self.input_mode = self._normalize_input_mode(input_mode)
        self.model.eval()

    @staticmethod
    def _normalize_input_mode(input_mode: str) -> str:
        mode = str(input_mode).lower().strip()
        if mode in {"fusion", "fused", "both", "frame_event", "rgb_event"}:
            return "fusion"
        if mode in {"frame", "frames", "rgb", "image", "images"}:
            return "frame"
        if mode in {"event", "events"}:
            return "event"
        raise ValueError(
            f"Unsupported input_mode={input_mode}. "
            "Use one of: fusion, frame, event."
        )

    def _apply_input_mode(self, video, events):
        if self.input_mode == "fusion":
            return video, events
        if self.input_mode == "frame":
            if isinstance(events, torch.Tensor):
                events = torch.full_like(events, 0)
            else:
                events = np.full(events.shape, 0, dtype=np.float32)
            return video, events
        if isinstance(video, torch.Tensor):
            video = torch.full_like(video, 127.5)
        else:
            video = np.full(video.shape, 127.5, dtype=np.float32)
        return video, events
    
    def _trim_extra_predictions(self, traj_e, vis_e, conf_e, num_extra_pts):
        if num_extra_pts <= 0:
            return traj_e, vis_e, conf_e
        sl = slice(None, -num_extra_pts)
        traj_e = traj_e[:, :, sl]
        vis_e = vis_e[:, :, sl]
        if conf_e is not None:
            conf_e = conf_e[:, :, sl]
        return traj_e, vis_e, conf_e

    def _finalize_predictions(self, traj_e, vis_e, conf_e, queries, H, W):
        if self.if_test:
            thr = 0.9
            vis_e = vis_e > thr
            for i in range(len(queries)):
                queries_t = queries[i, : traj_e.size(2), 0].to(torch.int64)
                arange = torch.arange(0, len(queries_t))
                traj_e[i, queries_t, arange] = queries[i, : traj_e.size(2), 1:]
                vis_e[i, queries_t, arange] = True

        traj_e *= traj_e.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return traj_e, vis_e, conf_e

    def forward(self, video, events, queries=None, img_ifnew=None, return_merge_variants=False):
        B, T, C_r, H, W = video.shape
        C_e = events[0].shape[1]
        if queries is None and self.grid_size > 0:
            grid_pts = get_points_on_a_grid(
                self.grid_size, (H, W), device="cuda"
            )
            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * 0, grid_pts],
                dim=2,
            )
            self.grid_size = 0
            
        queries = queries.clone()
        B, N, D = queries.shape
        device = queries.device
        
        assert D == 3
        assert B == 1
        
        interp_shape = self.interp_shape
        merge_variants = None
        
        if isinstance(video, torch.Tensor) and isinstance(events, torch.Tensor):
            video = video.reshape(B * T, C_r, H, W)
            events = events.reshape(B * T, C_e, H, W)
            video = F.interpolate(video, tuple(interp_shape), mode="bilinear", align_corners=True)
            events = F.interpolate(events, tuple(interp_shape), mode="bilinear", align_corners=True)
            video = video.reshape(B, T, C_r, interp_shape[0], interp_shape[1])
            events = events.reshape(B, T, C_e, interp_shape[0], interp_shape[1])

        queries[:, :, 1] *= (interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (interp_shape[0] - 1) / (H - 1)
        
        if self.single_point:
            model_video, model_events = self._apply_input_mode(video, events)
            traj_e = torch.zeros((B, T, N, 2), device=device)
            vis_e = torch.zeros((B, T, N), device=device)
            conf_e = torch.zeros((B, T, N), device=device)
            
            for pind in range((N)):
                querie = queries[:, pind : pind + 1]
                traj_e_pind, vis_e_pind, conf_e_pind = self._process_one_point(model_video[:,:], model_events[:,:], querie, img_ifnew=img_ifnew)
                traj_e[:, :, pind:pind+1] = traj_e_pind[:, :, :1]
                vis_e[:, :, pind:pind+1] = vis_e_pind[:, :, :1]
                conf_e[:, :, pind:pind+1] = conf_e_pind[:, :, :1]
        else:
            if self.grid_size > 0:
                xy = get_points_on_a_grid(self.grid_size, interp_shape)
                xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  
                queries = torch.cat([queries, xy], dim=1)  
                
            if self.num_uniformly_sampled_pts > 0:
                xy = get_uniformly_sampled_pts(
                    self.num_uniformly_sampled_pts,
                    video.shape[1],
                    interp_shape,
                    device=device,
                )
                queries = torch.cat([queries, xy], dim=1)  
                
            sift_size = self.sift_size
            if sift_size > 0:
                xy = get_sift_sampled_pts(video, sift_size, T, [H, W], device=device)
                if xy.shape[1] == sift_size:
                    queries = torch.cat([queries, xy], dim=1)  #
                else:
                    sift_size = 0 
        
            model_video, model_events = self._apply_input_mode(video, events)
            model_kwargs = dict(
                rgbs=model_video,
                events=model_events,
                queries=queries,
                iters=self.n_iters,
                img_ifnew=img_ifnew,
            )
            if return_merge_variants:
                model_kwargs["return_merge_variants"] = True
            preds = self.model(**model_kwargs)
            traj_e, vis_e = preds[0], preds[1]
            conf_e = preds[2]
            merge_variants = preds[3] if return_merge_variants and len(preds) > 3 else None
            num_extra_pts = self.grid_size**2 + sift_size + self.num_uniformly_sampled_pts
            if (
                queries is not None
                and sift_size > 0
                or self.grid_size > 0
                or self.num_uniformly_sampled_pts > 0
            ):
                traj_e, vis_e, conf_e = self._trim_extra_predictions(traj_e, vis_e, conf_e, num_extra_pts)
                if merge_variants is not None:
                    trimmed_variants = {}
                    for name, (traj_v, vis_v, conf_v) in merge_variants.items():
                        trimmed_variants[name] = self._trim_extra_predictions(traj_v, vis_v, conf_v, num_extra_pts)
                    merge_variants = trimmed_variants

        traj_e, vis_e, conf_e = self._finalize_predictions(traj_e, vis_e, conf_e, queries, H, W)
        if merge_variants is None:
            return traj_e, vis_e, conf_e

        finalized_variants = {}
        for name, (traj_v, vis_v, conf_v) in merge_variants.items():
            finalized_variants[name] = self._finalize_predictions(
                traj_v.clone(),
                vis_v.clone(),
                None if conf_v is None else conf_v.clone(),
                queries,
                H,
                W,
            )
        return traj_e, vis_e, conf_e, finalized_variants
                
    def _process_one_point(self, video, events, query, img_ifnew):
        B, T, C, H, W = video.shape
        device = query.device
        if self.local_grid_size > 0:
            xy_target = get_points_on_a_grid(self.local_grid_size, (self.local_extent, self.local_extent), [query[0,0,2].item(), query[0,0,1].item()])
            xy_target = torch.cat([torch.zeros_like(xy_target[:, :, :1]), xy_target], dim=2).to(device)
            query = torch.cat([query, xy_target], dim=1)
            
        if self.grid_size > 0:
            xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  
            query = torch.cat([query, xy], dim=1) 
            
        sift_size = self.sift_size
        if sift_size > 0:
            xy = get_sift_sampled_pts(video, sift_size, T, [H, W], device=device)
            sift_size = xy.shape[1]
            if sift_size > 0:
                query = torch.cat([query, xy], dim=1)  #

            num_uniformly_sampled_pts = self.sift_size - sift_size
            if num_uniformly_sampled_pts > 0:
                xy2 = get_uniformly_sampled_pts(
                    num_uniformly_sampled_pts,
                    video.shape[1],
                    video.shape[3:],
                    device=device,
                )
                query = torch.cat([query, xy2], dim=1)  #

        if self.num_uniformly_sampled_pts > 0:
            xy = get_uniformly_sampled_pts(
                self.num_uniformly_sampled_pts,
                video.shape[1],
                video.shape[3:],
                device=device,
            )
            query = torch.cat([query, xy], dim=1)  
            
        traj_e_pind, vis_e_pind, conf_e_pind = self.model(
            rgbs=video, events=events, queries=query, iters=self.n_iters, img_ifnew=img_ifnew
        )
    
        return traj_e_pind[..., :2], vis_e_pind, conf_e_pind
        
