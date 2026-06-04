"""
Visualize TAPFormer predictions on Kubric/ETAP-style data.

Usage:
    python test/test_kubric.py --config config/config_kubric_vis.yaml
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LFE_TAP.datasets.kubric_movif_dataset import KubricMovifDataset_etap
from LFE_TAP.evaluator.evaluation_pred import EvaluationPredictor
from LFE_TAP.evaluator.prediction import TAPFormerCowDense_online, TAPFormer_online
from LFE_TAP.models.tapformer import TAPFormer
from LFE_TAP.models.tapformer_cow_dense import TAPFormerCowDense
from LFE_TAP.utils.model_utils import get_points_on_a_grid
from LFE_TAP.utils.visualizer import Visualizer

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TAPFormer on Kubric/ETAP dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_kubric_vis.yaml",
        help="Path to configuration YAML file",
    )
    return parser.parse_args()


def build_query_from_first_visible(traj, visibility, valid=None):
    vis = visibility > 0.5
    if valid is not None:
        vis = vis & (valid > 0.5)

    any_vis = vis.any(dim=0)
    first_vis = torch.argmax(vis.int(), dim=0)
    first_vis = torch.where(any_vis, first_vis, torch.zeros_like(first_vis))

    point_idx = torch.arange(traj.shape[1], device=traj.device)
    query_xy = traj[first_vis, point_idx]
    return torch.cat([first_vis.float().unsqueeze(1), query_xy], dim=1).unsqueeze(0)


def build_query_from_clip_start(traj, visibility, valid=None):
    vis0 = visibility[0] > 0.5
    if valid is not None:
        vis0 = vis0 & (valid[0] > 0.5)
    if not vis0.any():
        return None

    traj = traj[:, vis0]
    visibility = visibility[:, vis0]
    valid = None if valid is None else valid[:, vis0]
    query_xy = traj[0]
    query_t = torch.zeros((query_xy.shape[0], 1), device=traj.device, dtype=traj.dtype)
    queries = torch.cat([query_t, query_xy], dim=1).unsqueeze(0)
    return queries, traj, visibility, valid


def build_query_from_grid(image_shape, grid_size, device, dtype):
    grid_xy = get_points_on_a_grid(grid_size, image_shape, device=device).to(dtype=dtype)
    query_t = torch.zeros_like(grid_xy[:, :, :1])
    return torch.cat([query_t, grid_xy], dim=2)


def build_visualization_queries(traj, visibility, valid, image_shape, is_cow_dense, query_mode, grid_size):
    if query_mode == "gt":
        if is_cow_dense:
            return build_query_from_clip_start(traj, visibility, valid=valid)
        return build_query_from_first_visible(traj, visibility, valid=valid), traj, visibility, valid
    if query_mode == "grid":
        queries = build_query_from_grid(image_shape, grid_size, traj.device, traj.dtype)
        return queries, traj, visibility, valid
    raise ValueError(f"Unsupported visualization.query_mode={query_mode}. Use gt or grid.")


def slice_sequence(tensor, start, end):
    if tensor is None:
        return None
    return tensor[start:end]


def compress_events_for_visualization(events: torch.Tensor) -> torch.Tensor:
    if events.shape[2] == 10:
        events_vis = events.clone()
        events_vis[:, :, 1::2] = -events_vis[:, :, 1::2]
        max_abs_indices = torch.abs(events_vis).argmax(dim=2, keepdim=True)
        compressed = (torch.gather(events_vis, dim=2, index=max_abs_indices) + 1) / 2 * 255.0
        return compressed.repeat(1, 1, 3, 1, 1)

    mean_events = events.mean(dim=2, keepdim=True)
    mean_events = mean_events - mean_events.amin(dim=(3, 4), keepdim=True)
    denom = mean_events.amax(dim=(3, 4), keepdim=True).clamp_min(1e-6)
    mean_events = mean_events / denom * 255.0
    return mean_events.repeat(1, 1, 3, 1, 1)


def flow_to_rgb_video(flow: torch.Tensor) -> torch.Tensor:
    flow_np = flow.detach().cpu().numpy()
    magnitude = np.linalg.norm(flow_np, axis=-1)
    scale = float(np.percentile(magnitude, 95))
    if scale <= 1e-6:
        scale = 1.0

    frames = []
    for frame_flow in flow_np:
        dx = frame_flow[..., 0]
        dy = frame_flow[..., 1]
        frame_mag = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx)
        hsv = np.zeros((*dx.shape, 3), dtype=np.uint8)
        hsv[..., 0] = np.mod((angle + np.pi) * 90.0 / np.pi, 180).astype(np.uint8)
        hsv[..., 1] = (np.clip(frame_mag / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
        hsv[..., 2] = hsv[..., 1]
        frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].byte()


def scalar_map_to_rgb_video(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().clamp(0.0, 1.0)
    values = (values * 255.0).to(torch.uint8)
    return values.unsqueeze(2).repeat(1, 1, 3, 1, 1)


def blend_videos(base_video: torch.Tensor, overlay_video: torch.Tensor, alpha: float) -> torch.Tensor:
    blended = (1.0 - alpha) * base_video.float() + alpha * overlay_video.float()
    return blended.clamp(0, 255).to(torch.uint8)


def export_dense_debug_videos(visualizer, seq_name, events, dense_debug, alpha):
    dense_tracks = dense_debug["dense_tracks"]
    _, _, H, W, _ = dense_tracks.shape
    device = dense_tracks.device
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dense_tracks.dtype),
        torch.arange(W, device=device, dtype=dense_tracks.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([x, y], dim=-1).unsqueeze(0).unsqueeze(0)
    dense_flow = dense_tracks - base_grid

    flow_video = flow_to_rgb_video(dense_flow[0])
    vis_video = scalar_map_to_rgb_video(dense_debug["dense_vis"])
    conf_video = scalar_map_to_rgb_video(dense_debug["dense_conf"])
    event_video = compress_events_for_visualization(events.detach().cpu())
    flow_event_video = blend_videos(event_video, flow_video, alpha)

    visualizer.save_video(flow_video, f"{seq_name}_dense_flow")
    visualizer.save_video(vis_video, f"{seq_name}_dense_vis")
    visualizer.save_video(conf_video, f"{seq_name}_dense_conf")
    visualizer.save_video(flow_event_video, f"{seq_name}_dense_flow_event")


def build_model_from_config(model_cfg, inference_mode="online"):
    model_name = str(model_cfg.get("name", "tapformer_online")).lower().strip()
    common_kwargs = dict(
        window_size=int(model_cfg.get("window_size", 16)),
        stride=int(model_cfg.get("stride", 4)),
        corr_radius=int(model_cfg.get("corr_radius", 3)),
        corr_levels=int(model_cfg.get("corr_levels", 3)),
        backbone=model_cfg.get("backbone", "basic"),
        hidden_size=int(model_cfg.get("hidden_size", 384)),
        space_depth=int(model_cfg.get("space_depth", 3)),
        time_depth=int(model_cfg.get("time_depth", 3)),
    )
    if model_name in {"tapformer_cow_dense", "cow_dense"}:
        cow_kwargs = dict(
            cow_refine_model=str(model_cfg.get("cow_refine_model", "vits")),
            cow_refine_patch_size=int(model_cfg.get("cow_refine_patch_size", 4)),
            cow_refine_blocks=model_cfg.get("cow_refine_blocks", None),
            cow_temporal_interleave_stride=int(model_cfg.get("cow_temporal_interleave_stride", 2)),
            cow_tracking_down_ratio=int(model_cfg.get("cow_tracking_down_ratio", 2)),
            cow_limit_flow=bool(model_cfg.get("cow_limit_flow", True)),
            cow_max_flow_update_ratio=float(model_cfg.get("cow_max_flow_update_ratio", 0.15)),
            cow_max_flow_magnitude_ratio=float(model_cfg.get("cow_max_flow_magnitude_ratio", 1.0)),
            cow_refine_checkpoint=bool(model_cfg.get("cow_refine_checkpoint", False)),
            cow_info_update_mode=str(model_cfg.get("cow_info_update_mode", "direct")),
            cow_online_use_window_init=bool(model_cfg.get("cow_online_use_window_init", False)),
            cow_online_use_global_first_anchor=bool(model_cfg.get("cow_online_use_global_first_anchor", False)),
            cow_online_use_memory_features=bool(model_cfg.get("cow_online_use_memory_features", False)),
            cow_online_num_memory_frames=int(model_cfg.get("cow_online_num_memory_frames", 10)),
            cow_frontend_type=str(model_cfg.get("cow_frontend_type", "base")),
            cow_anchor_state_mix=float(model_cfg.get("cow_anchor_state_mix", 0.7)),
            cow_anchor_skip_mix=float(model_cfg.get("cow_anchor_skip_mix", 0.7)),
            **common_kwargs,
        )
        if inference_mode == "offline":
            return TAPFormerCowDense(**cow_kwargs)
        return TAPFormerCowDense_online(**cow_kwargs)
    if inference_mode == "offline":
        return TAPFormer(**common_kwargs)
    return TAPFormer_online(**common_kwargs)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    pred_cfg = cfg.get("predictor", {})
    vis_cfg = cfg.get("visualization", {})
    eval_cfg = cfg.get("eval", {})
    inference_mode = str(pred_cfg.get("inference_mode", "online")).lower().strip()
    if inference_mode not in {"online", "offline"}:
        raise ValueError(f"Unsupported predictor.inference_mode={inference_mode}. Use online or offline.")

    data_root = dataset_cfg["data_root"]
    ckpt_root = cfg["ckpt_root"]

    dataset = KubricMovifDataset_etap(
        root_dir=data_root,
        representation=dataset_cfg.get("representation", "time_surfaces_v2_5"),
        crop_size=(int(dataset_cfg.get("height", 384)), int(dataset_cfg.get("width", 512))),
        seq_len=int(dataset_cfg.get("seq_len", 95)),
        traj_per_sample=int(dataset_cfg.get("traj_per_sample", 256)),
        sample_vis_1st_frame=bool(dataset_cfg.get("sample_vis_1st_frame", False)),
        choose_long_point=bool(dataset_cfg.get("choose_long_point", False)),
        use_augs=bool(dataset_cfg.get("use_augs", False)),
        if_test=bool(dataset_cfg.get("if_test", True)),
    )

    model = build_model_from_config(model_cfg, inference_mode=inference_mode)

    state_dict = torch.load(ckpt_root, map_location=DEFAULT_DEVICE)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    predictor = EvaluationPredictor(
        model=model,
        interp_shape=(int(dataset_cfg.get("height", 384)), int(dataset_cfg.get("width", 512))),
        grid_size=int(pred_cfg.get("grid_size", 0)),
        local_grid_size=int(pred_cfg.get("local_grid_size", 0)),
        single_point=bool(pred_cfg.get("single_point", False)),
        sift_size=int(pred_cfg.get("sift_size", 0)),
        num_uniformly_sampled_pts=int(pred_cfg.get("num_uniformly_sampled_pts", 0)),
        n_iters=int(pred_cfg.get("n_iters", 6)),
        local_extent=int(pred_cfg.get("local_extent", 50)),
        if_test=bool(pred_cfg.get("if_test", True)),
        input_mode=pred_cfg.get("input_mode", "fusion"),
    )

    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    output_dir = vis_cfg.get("output_dir", "output/eval_kubric")
    os.makedirs(output_dir, exist_ok=True)
    visualizer = Visualizer(
        save_dir=output_dir,
        fps=int(vis_cfg.get("fps", 20)),
        linewidth=int(vis_cfg.get("linewidth", 2)),
        mode=vis_cfg.get("mode", "rainbow"),
        tracks_leave_trace=int(vis_cfg.get("tracks_leave_trace", 16)),
    )

    all_seq_names = [os.path.basename(p) for p in dataset.seq_names]
    requested_seq_names = eval_cfg.get("seq_names", [])
    if requested_seq_names:
        seq_indices = [all_seq_names.index(name) for name in requested_seq_names if name in all_seq_names]
    else:
        start_index = int(eval_cfg.get("start_index", 0))
        max_seqs = int(eval_cfg.get("max_seqs", 5))
        seq_indices = list(range(start_index, min(start_index + max_seqs, len(dataset))))

    if len(seq_indices) == 0:
        print("No sequence selected. Please check eval.seq_names or eval.start_index/max_seqs.")
        return

    visibility_threshold = float(vis_cfg.get("visibility_threshold", 0.8))
    use_clear_video = bool(vis_cfg.get("use_clear_video", True))
    use_gt_as_prediction = bool(vis_cfg.get("use_gt_as_prediction", True))
    query_mode = str(vis_cfg.get("query_mode", "gt")).lower().strip()
    query_grid_size = int(vis_cfg.get("grid_size", 5))
    dense_debug = bool(vis_cfg.get("dense_debug", False))
    dense_overlay_alpha = float(vis_cfg.get("dense_overlay_alpha", 0.5))
    online_dual_merge_export = bool(vis_cfg.get("online_dual_merge_export", False))
    video_start = max(0, int(vis_cfg.get("video_start", 0)))
    video_length = int(vis_cfg.get("video_length", 0))
    is_cow_dense = str(model_cfg.get("name", "")).lower().strip() in {"tapformer_cow_dense", "cow_dense"}
    if query_mode not in {"gt", "grid"}:
        raise ValueError(f"Unsupported visualization.query_mode={query_mode}. Use gt or grid.")
    if query_mode == "grid" and query_grid_size <= 0:
        raise ValueError("visualization.grid_size must be positive when visualization.query_mode=grid.")
    if query_mode == "grid" and use_gt_as_prediction:
        print("visualization.query_mode=grid does not support use_gt_as_prediction=true; falling back to model prediction.")
        use_gt_as_prediction = False
    if dense_debug and (not is_cow_dense or inference_mode != "offline"):
        raise ValueError("visualization.dense_debug only supports cow_dense with predictor.inference_mode=offline.")
    if online_dual_merge_export and (not is_cow_dense or inference_mode != "online"):
        raise ValueError("visualization.online_dual_merge_export only supports cow_dense with predictor.inference_mode=online.")
    if not 0.0 <= dense_overlay_alpha <= 1.0:
        raise ValueError("visualization.dense_overlay_alpha must be in [0, 1].")

    print(f"Using device: {DEFAULT_DEVICE}")
    print(f"Inference mode: {inference_mode}")
    print(f"Input mode: {predictor.input_mode}")
    print(f"Query mode: {query_mode}" + (f" (grid_size={query_grid_size})" if query_mode == "grid" else ""))
    print(f"Clip setting: start={video_start}, length={video_length}")
    print(f"Online dual merge export: {online_dual_merge_export}")
    print(f"Selected {len(seq_indices)} sequences for visualization")

    for idx in seq_indices:
        sample, gotit = dataset[idx]
        seq_name = os.path.basename(sample.seq_name)
        if not gotit:
            print(f"Skip {seq_name}: gotit=False")
            continue

        video = sample.clear_video if (use_clear_video and sample.clear_video is not None) else sample.video
        events = sample.events
        traj = sample.trajectory
        visibility = sample.visibility
        valid = sample.valid
        img_ifnew = sample.img_ifnew

        clip_end = video.shape[0] if video_length <= 0 else min(video.shape[0], video_start + video_length)
        if video_start >= clip_end:
            print(f"Skip {seq_name}: empty clip after start={video_start}, end={clip_end}")
            continue

        video = slice_sequence(video, video_start, clip_end)
        events = slice_sequence(events, video_start, clip_end)
        traj = slice_sequence(traj, video_start, clip_end)
        visibility = slice_sequence(visibility, video_start, clip_end)
        valid = slice_sequence(valid, video_start, clip_end)
        img_ifnew = slice_sequence(img_ifnew, video_start, clip_end)

        query_data = build_visualization_queries(
            traj,
            visibility,
            valid,
            video.shape[-2:],
            is_cow_dense,
            query_mode,
            query_grid_size,
        )
        if query_data is None:
            print(f"Skip {seq_name}: no point visible at clip start for cow-dense")
            continue
        queries, traj, visibility, valid = query_data

        video_b = video.unsqueeze(0).to(DEFAULT_DEVICE)
        events_b = events.unsqueeze(0).to(DEFAULT_DEVICE)
        queries_b = queries.to(DEFAULT_DEVICE)
        img_ifnew = img_ifnew.to(DEFAULT_DEVICE) if isinstance(img_ifnew, torch.Tensor) else img_ifnew
        merge_variants = None

        if use_gt_as_prediction:
            pred_traj = traj.unsqueeze(0).to(DEFAULT_DEVICE)
            pred_vis = visibility.unsqueeze(0).to(DEFAULT_DEVICE)
            pred_conf = torch.ones_like(pred_vis, dtype=torch.float32)
            print(f"[{seq_name}] using GT for visualization")
        else:
            merge_variants = None
            with torch.no_grad():
                preds = predictor(
                    video_b,
                    events_b,
                    queries_b,
                    img_ifnew=img_ifnew,
                    return_merge_variants=online_dual_merge_export,
                )
            if online_dual_merge_export:
                pred_traj, pred_vis, pred_conf, merge_variants = preds
            else:
                pred_traj, pred_vis, pred_conf = preds

        print(
            f"[{seq_name}] clip_len={video.shape[0]} traj={tuple(pred_traj.shape)} vis={tuple(pred_vis.shape)} conf={tuple(pred_conf.shape)}"
        )

        if dense_debug:
            with torch.no_grad():
                dense_debug_data = model.forward_dense_debug(
                    rgbs=video_b,
                    events=events_b,
                    iters=int(pred_cfg.get("n_iters", 6)),
                    img_ifnew=img_ifnew,
                )
            export_dense_debug_videos(
                visualizer,
                seq_name,
                events_b,
                dense_debug_data,
                dense_overlay_alpha,
            )

        vis_mask = pred_vis.bool() if use_gt_as_prediction else (pred_vis > visibility_threshold)
        keep_filename = seq_name if not online_dual_merge_export else f"{seq_name}_online_keep"
        visualizer.visualize(
            video_b.detach().cpu(),
            events_b.detach().cpu(),
            pred_traj.detach().cpu(),
            vis_mask.detach().cpu(),
            filename=keep_filename,
            video_model=vis_cfg.get("video_model", "events"),
        )

        if merge_variants is not None:
            overwrite_traj, overwrite_vis, _ = merge_variants["overwrite"]
            overwrite_mask = overwrite_vis > visibility_threshold
            visualizer.visualize(
                video_b.detach().cpu(),
                events_b.detach().cpu(),
                overwrite_traj.detach().cpu(),
                overwrite_mask.detach().cpu(),
                filename=f"{seq_name}_online_overwrite",
                video_model=vis_cfg.get("video_model", "events"),
            )


if __name__ == "__main__":
    main()
