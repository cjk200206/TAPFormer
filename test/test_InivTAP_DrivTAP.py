"""
Test script for TAPFormer model on Aedat4 dataset.
This script evaluates the model performance on test sequences.

Usage:
    python test_aedat4.py [--config config_aedat4.yaml]
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import argparse
import yaml
import torch
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LFE_TAP.datasets.TAPFormer_dataset import TAPFormer_dataset
from LFE_TAP.evaluator.model_factory import build_eval_predictor_from_config
from LFE_TAP.utils.visualizer import Visualizer
from LFE_TAP.evaluator.evaluator import compute_tapvid_metrics

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

def load_config(config_path):
    """Load configuration from YAML file."""
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config["__config_path__"] = str(config_path)
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test TAPFormer model on Aedat4 dataset')
    parser.add_argument('--config', type=str, default='config/config_InivTAP_DrivTAP.yaml',
                        help='Path to configuration YAML file')
    return parser.parse_args()


def _to_metric_scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def _compute_dataset_average(entries):
    metric_names = entries[0]["metrics"].keys()
    avg_metrics = {
        metric_name: float(np.mean([entry["metrics"][metric_name] for entry in entries]))
        for metric_name in metric_names
    }
    avg_time = float(np.mean([entry["elapsed_time"] for entry in entries]))
    seq_names = [entry["seq_name"] for entry in entries]
    return avg_metrics, avg_time, seq_names


def _print_dataset_average(dataset_name, avg_metrics, avg_time):
    print(f"\n{dataset_name} Dataset Average Results:")
    for metric_name, metric_value in avg_metrics.items():
        print(f"  Average {metric_name}: {metric_value:.6f}")
    print(f"  Average Time per Frame: {avg_time:.6f} seconds")


def _write_dataset_summary(base_dir, dataset_name, avg_metrics, avg_time, seq_names):
    os.makedirs(base_dir, exist_ok=True)
    summary_path = os.path.join(base_dir, f"{dataset_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"{dataset_name} Dataset Average Results\n")
        for metric_name, metric_value in avg_metrics.items():
            f.write(f"Average {metric_name}: {metric_value:.6f}\n")
        f.write(f"Average Time per Frame: {avg_time:.6f} seconds\n")
        f.write("Sequences: " + ", ".join(seq_names) + "\n")
    print(f"Summary saved to {summary_path}")


def _get_visualization_video_models(vis_cfg):
    video_models = vis_cfg.get('video_models')
    if video_models is None:
        video_models = [vis_cfg.get('video_model', 'rgb')]
    elif isinstance(video_models, str):
        video_models = [video_models]
    return [str(mode).lower().strip() for mode in video_models]


def _to_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.from_numpy(value).float()


def _events_for_visualization(events):
    if isinstance(events, torch.Tensor):
        return events.clone().float()
    return torch.from_numpy(events.copy()).float()


def _predict_sequence(predictor, sample):
    predictor_device = getattr(predictor, "device", torch.device(DEFAULT_DEVICE))
    queries = sample.query_points[np.newaxis, ...].to(predictor_device)
    sample.video = sample.video[np.newaxis, ...]
    sample.events = sample.events[np.newaxis, ...]
    sample.trajectory = sample.trajectory[np.newaxis, ...]
    sample.visibility = sample.visibility[np.newaxis, ...]

    start = time.time()
    pred_tracks = predictor(sample.video, sample.events, queries, img_ifnew=sample.img_ifnew)
    elapsed_time = (time.time() - start) / sample.events.shape[1]
    return pred_tracks, elapsed_time


# Parse arguments and load config
args = parse_args()
config = load_config(args.config)

# Extract configuration
dataset_dir = config['dataset_dir']
EVAL_DATASETS = config['eval_datasets']
representation = config['representation']
dt = config.get('dt', 0.0100)
if isinstance(dt, str) and dt.lower() == "auto":
    dt = None
drivtap_gt_mode = config.get('drivtap_gt_mode', 'current')
grid_size = config.get('grid_size', 0)
n_iters = config.get('n_iters', 5)
vis_cfg = config.get('visualization', {})
output_cfg = config.get('output', {})
enable_visualization = vis_cfg.get('enable', False)
save_results = output_cfg.get('save_results', False)
save_trajectory = output_cfg.get('save_trajectory', False)
base_output_dir = output_cfg.get('base_dir', 'output/eval_InivTAP_DrivTAP_subseq')
input_mode = config.get('input_mode', 'fusion')
inference_mode = str(config.get('inference_mode', 'online')).lower().strip()
eval_backend = str(config.get('eval_model', {}).get('backend', 'tapformer_family')).lower().strip()

visualization_video_models = _get_visualization_video_models(vis_cfg)

# ========== Predictor Initialization ==========
print("Loading evaluation predictor...")
predictor = build_eval_predictor_from_config(
    config,
    grid_size=grid_size,
    local_grid_size=0,
    single_point=False,
    num_uniformly_sampled_pts=0,
    n_iters=n_iters,
    input_mode=input_mode,
)
print("Predictor loaded successfully!")
print(f"Predictor backend: {eval_backend}")
print(f"Input mode: {input_mode}")
print(f"Inference mode: {inference_mode}")

# ========== Evaluation ==========
print("\n" + "="*50)
print("Evaluating on InivTAP and DrivTAP dataset...")
print("="*50)
dataset_entries = {"InivTAP": [], "DrivTAP": []}
datasets_inivtap = TAPFormer_dataset(os.path.join(dataset_dir, "InivTAP"), representation=representation, dt=dt)
datasets_drivtap = TAPFormer_dataset(
    os.path.join(dataset_dir, "DrivTAP"),
    representation=representation,
    dt=dt,
    drivtap_gt_mode=drivtap_gt_mode,
)
for seq_name, dataset_type in EVAL_DATASETS:
    if dataset_type == "InivTAP":
        sample, gotit = datasets_inivtap.get_a_seq(seq_name)
    elif dataset_type == "DrivTAP":
        sample, gotit = datasets_drivtap.get_a_seq(seq_name)
    else:
        continue
    if not gotit:
        continue
    
    # Setup output directory
    output_dir = os.path.join(base_output_dir, seq_name)
    if enable_visualization or save_results or save_trajectory:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer if needed
    vis = None
    if enable_visualization:
        vis = Visualizer(output_dir, fps=vis_cfg.get('fps', 20))

    pred_tracks, elapsed_time = _predict_sequence(predictor, sample)
    print("time per frame:", elapsed_time)

    if isinstance(pred_tracks, tuple):
        pred_trajectory, pred_visibility, _ = pred_tracks
    else:
        pred_visibility = None
    
    if pred_visibility is None:
        pred_visibility = torch.zeros_like(torch.as_tensor(sample.visibility))

    if pred_visibility.dtype != torch.bool:
        pred_visibility = pred_visibility > 0.8
        
    pred_occluded = (
            torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
            .cpu()
            .numpy()
        )
    pred_tracks_np = pred_trajectory.permute(0, 2, 1, 3).detach().cpu().numpy()

    query_points = np.concatenate(
        (
            np.zeros_like(sample.trajectory[:, 0, :, :1]),
            sample.trajectory[:, 0, :, 1:],
        ),
        axis=2,
    )
    
    gt_tracks = np.transpose(sample.trajectory[:, :, :, 1:].copy(), (0, 2, 1, 3))
    gt_occluded = np.transpose(sample.visibility.copy(), (0, 2, 1))

    if pred_trajectory.shape[1] != sample.trajectory.shape[1]:
        pred_trajectory_new = np.zeros_like(gt_tracks)
        pred_occluded_new = np.zeros_like(gt_occluded)
        for i in range(pred_tracks_np.shape[1]):
            pred_traj = pred_tracks_np[:, i, :, :]
            gt_t = sample.trajectory[:, :, i, 0].squeeze()
            est_t = sample.segmentation * 1e-6
            pred_traj_x, pred_traj_y = pred_traj.squeeze().T
            
            pred_traj_x_ = np.interp(gt_t, est_t, pred_traj_x)
            pred_traj_y_ = np.interp(gt_t, est_t, pred_traj_y)
            pred_traj_ = np.stack((pred_traj_x_, pred_traj_y_), axis=1)
            pred_trajectory_new[:, i, :, :] = pred_traj_
            
            pred_occ = pred_occluded[:, i, :].squeeze()
            indices = np.searchsorted(est_t, gt_t, side='left')
            # 处理左边界 (新时间点小于所有原始时间)
            left_mask = indices == 0
            pred_occluded_new[:, i, left_mask] = pred_occ[0]
            
            # 处理右边界 (新时间点大于所有原始时间)
            right_mask = indices == len(pred_occ)
            pred_occluded_new[:, i, right_mask] = pred_occ[-1]
            
            # 处理中间区域 (新时间点在原始时间范围内)
            mid_mask = ~(left_mask | right_mask)
            mid_indices = indices[mid_mask]
            
            # 计算新时间点与左右邻居的距离
            left_dist = gt_t[mid_mask] - est_t[mid_indices - 1]
            right_dist = est_t[mid_indices] - gt_t[mid_mask]
            
            # 选择更近的邻居
            closer_to_left = left_dist < right_dist
            pred_occluded_new[:, i,mid_mask] = np.where(
                closer_to_left, 
                pred_occ[mid_indices - 1], 
                pred_occ[mid_indices]
            )

        pred_tracks_np = pred_trajectory_new.copy()
        pred_occluded = pred_occluded_new.copy()
    out_metrics = compute_tapvid_metrics(
        query_points,
        gt_occluded,
        gt_tracks,
        pred_occluded,
        pred_tracks_np,
        query_mode="first",
    )
    out_metrics = {metric_name: _to_metric_scalar(metric_value) for metric_name, metric_value in out_metrics.items()}
    dataset_entries[dataset_type].append(
        {
            "seq_name": seq_name,
            "metrics": out_metrics,
            "elapsed_time": elapsed_time,
        }
    )
    print("metrics", out_metrics)
    
    # Visualization
    if enable_visualization and vis is not None:
        video_tensor = _to_float_tensor(sample.video)
        for video_model in visualization_video_models:
            event_tensor = _events_for_visualization(sample.events)
            vis.visualize(
                video_tensor,
                event_tensor,
                pred_trajectory,
                pred_visibility > 0.8,
                filename=f"{seq_name}_{video_model}",
                video_model=video_model,
            )
    
    # Save trajectory if requested
    if save_trajectory:
        B, T, N, _ = pred_trajectory.shape
        ind = np.arange(N).reshape(1, 1, N, 1).repeat(T, axis=1)
        t = sample.segmentation.astype(float)
        t *= 1e-6
        t = t.reshape(1, -1, 1, 1).repeat(N, axis=2)
        pred_trajectory_with_time = np.concatenate((ind, t, pred_trajectory.detach().cpu().numpy()), axis=3)
        pred_trajectory_txt = np.transpose(pred_trajectory_with_time, (0, 2, 1, 3)).reshape(-1, 4)
        traj_path = os.path.join(output_dir, "pred_trajectory.txt")
        np.savetxt(traj_path, pred_trajectory_txt)
        print(f"Trajectory saved to {traj_path}")
    
    # Save results if requested
    if save_results:
        result_path = os.path.join(output_dir, "result.txt")
        with open(result_path, 'w') as f:
            f.write("metrics " + str(out_metrics) + "\n")
            f.write(f"time per frame: {elapsed_time}\n")
        print(f"Results saved to {result_path}")

for dataset_name, entries in dataset_entries.items():
    if not entries:
        continue
    avg_metrics, avg_time, seq_names = _compute_dataset_average(entries)
    _print_dataset_average(dataset_name, avg_metrics, avg_time)
    if save_results:
        _write_dataset_summary(base_output_dir, dataset_name, avg_metrics, avg_time, seq_names)
