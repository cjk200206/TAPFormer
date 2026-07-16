import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Callable, Optional, Tuple

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
        if queries is None:
            if not isinstance(video, torch.Tensor):
                raise ValueError("queries can only be omitted when video is a torch.Tensor")
            B, T, C_r, H, W = video.shape
        else:
            device = queries.device
            if not isinstance(video, torch.Tensor):
                video = torch.as_tensor(np.asarray(video), device=device)
            if not isinstance(events, torch.Tensor):
                events = torch.as_tensor(np.asarray(events), device=device)
            if not torch.is_floating_point(video):
                video = video.float()
            if not torch.is_floating_point(events):
                events = events.float()
            B, T, C_r, H, W = video.shape

        C_e = events[0].shape[1]
        if queries is None and self.grid_size > 0:
            grid_pts = get_points_on_a_grid(self.grid_size, (H, W), device="cuda")
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

            for pind in range(N):
                querie = queries[:, pind : pind + 1]
                traj_e_pind, vis_e_pind, conf_e_pind = self._process_one_point(
                    model_video[:, :],
                    model_events[:, :],
                    querie,
                    img_ifnew=img_ifnew,
                )
                traj_e[:, :, pind : pind + 1] = traj_e_pind[:, :, :1]
                vis_e[:, :, pind : pind + 1] = vis_e_pind[:, :, :1]
                conf_e[:, :, pind : pind + 1] = conf_e_pind[:, :, :1]
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
                    queries = torch.cat([queries, xy], dim=1)
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
                traj_e, vis_e, conf_e = self._trim_extra_predictions(
                    traj_e,
                    vis_e,
                    conf_e,
                    num_extra_pts,
                )
                if merge_variants is not None:
                    trimmed_variants = {}
                    for name, (traj_v, vis_v, conf_v) in merge_variants.items():
                        trimmed_variants[name] = self._trim_extra_predictions(
                            traj_v,
                            vis_v,
                            conf_v,
                            num_extra_pts,
                        )
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
            xy_target = get_points_on_a_grid(
                self.local_grid_size,
                (self.local_extent, self.local_extent),
                [query[0, 0, 2].item(), query[0, 0, 1].item()],
            )
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
                query = torch.cat([query, xy], dim=1)

            num_uniformly_sampled_pts = self.sift_size - sift_size
            if num_uniformly_sampled_pts > 0:
                xy2 = get_uniformly_sampled_pts(
                    num_uniformly_sampled_pts,
                    video.shape[1],
                    video.shape[3:],
                    device=device,
                )
                query = torch.cat([query, xy2], dim=1)

        if self.num_uniformly_sampled_pts > 0:
            xy = get_uniformly_sampled_pts(
                self.num_uniformly_sampled_pts,
                video.shape[1],
                video.shape[3:],
                device=device,
            )
            query = torch.cat([query, xy], dim=1)

        traj_e_pind, vis_e_pind, conf_e_pind = self.model(
            rgbs=video,
            events=events,
            queries=query,
            iters=self.n_iters,
            img_ifnew=img_ifnew,
        )

        return traj_e_pind[..., :2], vis_e_pind, conf_e_pind


class ExternalCoTrackerEvaluationPredictor(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "online",
        device: Optional[torch.device] = None,
        backward_tracking: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.mode = str(mode).lower().strip()
        if self.mode not in {"online", "offline"}:
            raise ValueError("External CoTracker mode must be one of: online, offline.")
        self.device = device or next(model.parameters()).device
        self.backward_tracking = bool(backward_tracking)
        self.model.eval()

    def _to_device_tensor(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
        else:
            tensor = torch.as_tensor(np.asarray(value), device=self.device)
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        return tensor

    def _prepare_inputs(self, video, queries):
        if queries is None:
            raise ValueError("ExternalCoTrackerEvaluationPredictor requires explicit query points.")
        video = self._to_device_tensor(video)
        queries = self._to_device_tensor(queries)
        if video.ndim != 5:
            raise ValueError(f"CoTracker expects video with shape [B,T,C,H,W], got {tuple(video.shape)}")
        if queries.ndim != 3 or queries.shape[-1] != 3:
            raise ValueError(f"CoTracker expects queries with shape [B,N,3], got {tuple(queries.shape)}")
        if video.shape[2] != 3:
            raise ValueError(f"CoTracker expects RGB video with 3 channels, got C={video.shape[2]}")
        return video, queries

    @staticmethod
    def _anchor_queries(pred_track, pred_vis, queries):
        query_xy = queries[..., 1:3]
        query_frames = torch.round(queries[..., 0]).to(torch.long)
        point_ids = torch.arange(pred_track.shape[2], device=pred_track.device)
        for batch_idx in range(pred_track.shape[0]):
            frame_ids = query_frames[batch_idx].clamp(0, pred_track.shape[1] - 1)
            pred_track[batch_idx, frame_ids, point_ids] = query_xy[batch_idx]
            pred_vis[batch_idx, frame_ids, point_ids] = True

    @staticmethod
    def _match_time_length(pred_track, pred_vis, target_len):
        pred_len = pred_track.shape[1]
        if pred_len == target_len:
            return pred_track, pred_vis
        if pred_len > target_len:
            return pred_track[:, :target_len], pred_vis[:, :target_len]
        if pred_len <= 0:
            raise ValueError("CoTracker returned an empty prediction.")
        pad_len = target_len - pred_len
        track_pad = pred_track[:, -1:].expand(-1, pad_len, -1, -1)
        vis_pad = pred_vis[:, -1:].expand(-1, pad_len, -1)
        return torch.cat([pred_track, track_pad], dim=1), torch.cat([pred_vis, vis_pad], dim=1)

    def _run_offline(self, video, queries):
        return self.model(
            video,
            queries=queries,
            grid_size=0,
            backward_tracking=self.backward_tracking,
        )

    def _run_online(self, video, queries):
        self.model(
            video,
            is_first_step=True,
            queries=queries,
            grid_size=0,
            add_support_grid=False,
        )
        step = int(getattr(self.model, "step", 0))
        if step <= 0:
            raise ValueError("CoTracker online predictor must expose a positive `step`.")

        target_len = video.shape[1]
        last_tracks, last_visibility = None, None
        for start in range(0, max(target_len - step, 1), step):
            video_chunk = video[:, start : start + step * 2]
            last_tracks, last_visibility = self.model(
                video_chunk,
                is_first_step=False,
                grid_size=0,
                add_support_grid=False,
            )

        if last_tracks is None or last_visibility is None:
            raise ValueError("CoTracker online predictor did not return predictions.")
        return self._match_time_length(last_tracks, last_visibility, target_len)

    def forward(self, video, events, queries=None, img_ifnew=None, return_merge_variants=False):
        del events, img_ifnew, return_merge_variants
        video, queries = self._prepare_inputs(video, queries)
        with torch.no_grad():
            if self.mode == "online":
                pred_track, pred_vis = self._run_online(video, queries)
            else:
                pred_track, pred_vis = self._run_offline(video, queries)

        pred_vis = pred_vis.bool()
        self._anchor_queries(pred_track, pred_vis, queries)
        pred_conf = pred_vis.float()
        return pred_track, pred_vis, pred_conf


class CowTrackerEvaluationPredictor(torch.nn.Module):
    SAFE_INFER_SHAPE_DIVISOR = 112

    def __init__(
        self,
        model: torch.nn.Module,
        sample_dense_predictions: Callable,
        compute_padding_params: Callable,
        apply_padding: Callable,
        remove_padding_and_scale_back: Callable,
        infer_shape: Optional[Tuple[int, int]] = None,
        use_padding: bool = True,
        skip_upscaling: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        model_call_mode: str = "standard",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.sample_dense_predictions = sample_dense_predictions
        self.compute_padding_params = compute_padding_params
        self.apply_padding = apply_padding
        self.remove_padding_and_scale_back = remove_padding_and_scale_back
        self.infer_shape = None if infer_shape is None else tuple(int(v) for v in infer_shape)
        self.use_padding = bool(use_padding)
        self.skip_upscaling = bool(skip_upscaling)
        self.amp_dtype = amp_dtype
        self.model_call_mode = str(model_call_mode).lower().strip()
        self.device = device or next(model.parameters()).device
        self._validate_infer_shape()
        self.model.eval()

    @classmethod
    def _is_safe_infer_dim(cls, size: int) -> bool:
        if size <= 0 or size % 14 != 0:
            return False
        o1 = (size + 2 * 3 - 7) // 2 + 1
        o2 = (o1 + 2 * 1 - 3) // 2 + 1
        o3 = (o2 + 2 * 1 - 3) // 2 + 1
        o4 = (o3 + 2 * 1 - 3) // 2 + 1
        return (o4 * 2 == o3) and (o3 * 2 == o2) and (o2 * 2 == o1)

    @classmethod
    def _nearest_safe_infer_dim(cls, target: int) -> int:
        target = max(int(target), cls.SAFE_INFER_SHAPE_DIVISOR)
        step = cls.SAFE_INFER_SHAPE_DIVISOR
        lower = max(step, (target // step) * step)
        upper = lower if lower >= target else lower + step
        candidates = []
        for size in {lower, upper}:
            if cls._is_safe_infer_dim(size):
                candidates.append(size)
        if not candidates:
            raise ValueError(f"Unable to find a safe CowTracker infer dimension near target={target}")
        return min(candidates, key=lambda size: (abs(size - target), size))

    def _resolve_infer_shape(self, height: int, width: int) -> Tuple[int, int]:
        if self.infer_shape is not None:
            return self.infer_shape
        return (
            self._nearest_safe_infer_dim(height),
            self._nearest_safe_infer_dim(width),
        )

    def _validate_infer_shape(self) -> None:
        if not self.use_padding or self.infer_shape is None:
            return

        infer_h, infer_w = self.infer_shape
        if not self._is_safe_infer_dim(infer_h) or not self._is_safe_infer_dim(infer_w):
            divisor = self.SAFE_INFER_SHAPE_DIVISOR
            raise ValueError(
                "CowTracker bridge requires each static infer_size dimension to satisfy both "
                "VGGT patch_size=14 and side-ResNet deconvolution alignment. "
                f"Got infer_size={self.infer_shape}. Prefer setting `eval_resolution` in the main "
                "real-benchmark config; it overrides the bridge config when present. If you keep a "
                f"static bridge infer_size, use safe values such as [448, 560], [336, 448], or "
                f"fallback to `infer_size: auto`, all based on {divisor}-pixel steps."
            )

    def _to_device_tensor(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
        else:
            tensor = torch.as_tensor(np.asarray(value), device=self.device)
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        return tensor

    def _prepare_video(self, video):
        video = self._to_device_tensor(video)
        if video.ndim != 5:
            raise ValueError(f"CowTracker expects video with shape [B,T,C,H,W], got {tuple(video.shape)}")

        if not self.use_padding:
            return video, None

        b, t, c, h, w = video.shape
        infer_h, infer_w = self._resolve_infer_shape(h, w)
        padding_info = self.compute_padding_params(
            h,
            w,
            infer_h,
            infer_w,
            skip_upscaling=self.skip_upscaling,
        )
        padded = self.apply_padding(video.reshape(b * t, c, h, w), padding_info)
        padded = padded.reshape(b, t, c, padded.shape[-2], padded.shape[-1])
        return padded, padding_info

    def _run_model(self, video):
        with torch.no_grad():
            use_amp = self.amp_dtype is not None and self.device.type == "cuda"
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    if self.model_call_mode == "official_windowed":
                        return self.model(video)
                    return self.model(video, return_all_iters=False)
            if self.model_call_mode == "official_windowed":
                return self.model(video)
            return self.model(video, return_all_iters=False)

    def _restore_dense_predictions(self, predictions, padding_info):
        if padding_info is None:
            return predictions

        tracks, visibilities, confidences = [], [], []
        for batch_idx in range(predictions["track"].shape[0]):
            track_b, vis_b, conf_b = self.remove_padding_and_scale_back(
                predictions["track"][batch_idx],
                predictions["vis"][batch_idx],
                predictions["conf"][batch_idx],
                padding_info,
            )
            tracks.append(track_b)
            visibilities.append(vis_b)
            confidences.append(conf_b)

        predictions["track"] = torch.stack(tracks, dim=0)
        predictions["vis"] = torch.stack(visibilities, dim=0)
        predictions["conf"] = torch.stack(confidences, dim=0)
        return predictions

    @staticmethod
    def _anchor_queries(pred_track, pred_vis, query_xy, query_frames):
        num_points = pred_track.shape[2]
        point_ids = torch.arange(num_points, device=pred_track.device)
        for batch_idx in range(pred_track.shape[0]):
            frame_ids = query_frames[batch_idx].clamp(0, pred_track.shape[1] - 1)
            pred_track[batch_idx, frame_ids, point_ids] = query_xy[batch_idx]
            pred_vis[batch_idx, frame_ids, point_ids] = 1.0

    def forward(self, video, events, queries=None, img_ifnew=None, return_merge_variants=False):
        del events, img_ifnew, return_merge_variants
        if queries is None:
            raise ValueError("CowTrackerEvaluationPredictor requires explicit first-frame query points.")

        video, padding_info = self._prepare_video(video)
        queries = self._to_device_tensor(queries)
        query_xy = queries[..., 1:3].clone()
        query_frames = torch.round(queries[..., 0]).to(torch.long)

        dense_predictions = self._run_model(video)
        dense_predictions = self._restore_dense_predictions(dense_predictions, padding_info)

        pred_track = self.sample_dense_predictions(dense_predictions["track"], query_xy)
        pred_vis = self.sample_dense_predictions(
            dense_predictions["vis"][..., None],
            query_xy,
        ).squeeze(-1)
        pred_conf = self.sample_dense_predictions(
            dense_predictions["conf"][..., None],
            query_xy,
        ).squeeze(-1)

        self._anchor_queries(pred_track, pred_vis, query_xy, query_frames)
        return pred_track, pred_vis, pred_conf
