"""
Visualize TAPFormer predictions on Kubric/ETAP-style data.

Usage:
    python test/test_kubric.py --config config/config_kubric_vis.yaml
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LFE_TAP.datasets.kubric_movif_dataset import KubricMovifDataset_etap
from LFE_TAP.evaluator.evaluation_pred import EvaluationPredictor
from LFE_TAP.evaluator.prediction import TAPFormer_online
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


def build_query_from_first_visible(sample):
    # sample.trajectory: [T, N, 2], sample.visibility: [T, N]
    traj = sample.trajectory
    vis = sample.visibility > 0.5
    if sample.valid is not None:
        vis = vis & (sample.valid > 0.5)

    any_vis = vis.any(dim=0)
    first_vis = torch.argmax(vis.int(), dim=0)
    first_vis = torch.where(any_vis, first_vis, torch.zeros_like(first_vis))

    point_idx = torch.arange(traj.shape[1], device=traj.device)
    query_xy = traj[first_vis, point_idx]
    queries = torch.cat([first_vis.float().unsqueeze(1), query_xy], dim=1).unsqueeze(0)  # [1, N, 3]
    return queries


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    pred_cfg = cfg.get("predictor", {})
    vis_cfg = cfg.get("visualization", {})
    eval_cfg = cfg.get("eval", {})

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

    model = TAPFormer_online(
        window_size=int(model_cfg.get("window_size", 16)),
        stride=int(model_cfg.get("stride", 4)),
        corr_radius=int(model_cfg.get("corr_radius", 3)),
        corr_levels=int(model_cfg.get("corr_levels", 3)),
        backbone=model_cfg.get("backbone", "basic"),
        hidden_size=int(model_cfg.get("hidden_size", 384)),
        space_depth=int(model_cfg.get("space_depth", 3)),
        time_depth=int(model_cfg.get("time_depth", 3)),
    )

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

    print(f"Using device: {DEFAULT_DEVICE}")
    print(f"Selected {len(seq_indices)} sequences for visualization")

    for idx in seq_indices:
        sample, gotit = dataset[idx]
        seq_name = os.path.basename(sample.seq_name)
        if not gotit:
            print(f"Skip {seq_name}: gotit=False")
            continue

        video = sample.clear_video if (use_clear_video and sample.clear_video is not None) else sample.video
        events = sample.events
        queries = build_query_from_first_visible(sample)

        video_b = video.unsqueeze(0).to(DEFAULT_DEVICE)
        events_b = events.unsqueeze(0).to(DEFAULT_DEVICE)
        queries_b = queries.to(DEFAULT_DEVICE)
        img_ifnew = sample.img_ifnew

        if use_gt_as_prediction:
            pred_traj = sample.trajectory.unsqueeze(0).to(DEFAULT_DEVICE)
            pred_vis = sample.visibility.unsqueeze(0).to(DEFAULT_DEVICE)
            pred_conf = torch.ones_like(pred_vis, dtype=torch.float32)
            print(f"[{seq_name}] using GT for visualization")
        else:
            with torch.no_grad():
                pred_traj, pred_vis, pred_conf = predictor(video_b, events_b, queries_b, img_ifnew=img_ifnew)

        print(
            f"[{seq_name}] traj={tuple(pred_traj.shape)} vis={tuple(pred_vis.shape)} conf={tuple(pred_conf.shape)}"
        )

        vis_mask = pred_vis.bool() if use_gt_as_prediction else (pred_vis > visibility_threshold)

        visualizer.visualize(
            video_b.detach().cpu(),
            events_b.detach().cpu(),
            pred_traj.detach().cpu(),
            vis_mask.detach().cpu(),
            filename=seq_name,
            video_model=vis_cfg.get("video_model", "events"),
        )


if __name__ == "__main__":
    main()
