"""
Test script for TAPFormer model on Aedat4 dataset.
This script evaluates the model performance on test sequences.

Usage:
    python test_aedat4.py [--config config_aedat4.yaml]
"""

import os
import sys
import argparse
import yaml
import torch
import time
import numpy as np

from LFE_TAP.datasets.TAPFormer_dataset import TAPFormer_dataset
from LFE_TAP.evaluator.prediction import TAPFormer_online
from LFE_TAP.evaluator.evaluation_pred import EvaluationPredictor
from LFE_TAP.utils.visualizer import Visualizer
from LFE_TAP.evaluator.evaluator import compute_tapvid_metrics

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test TAPFormer model on Aedat4 dataset')
    parser.add_argument('--config', type=str, default='config/config_InivTAP_DrivTAP.yaml',
                        help='Path to configuration YAML file')
    return parser.parse_args()


# Parse arguments and load config
args = parse_args()
config = load_config(args.config)

# Extract configuration
dataset_dir = config['dataset_dir']
ckpt_root = config['ckpt_root']
EVAL_DATASETS = config['eval_datasets']
model_name = os.path.basename(os.path.dirname(ckpt_root))

representation = config['representation']
stride = config.get('stride')

# Auto-detect backbone
corr_levels = config.get('corr_levels')
backbone = config.get('backbone')
# Model configuration
window_size = config.get('window_size', 16)
corr_radius = config.get('corr_radius', 3)
hidden_size = config.get('hidden_size', 384)
space_depth = config.get('space_depth', 3)
time_depth = config.get('time_depth', 3)

# Evaluation settings
dt = config.get('dt', 0.0100)
grid_size = config.get('grid_size', 0)
n_iters = config.get('n_iters', 5)

# Visualization and output settings
vis_cfg = config.get('visualization', {})
output_cfg = config.get('output', {})
enable_visualization = vis_cfg.get('enable', False)
save_results = output_cfg.get('save_results', False)
save_trajectory = output_cfg.get('save_trajectory', False)
base_output_dir = output_cfg.get('base_dir', 'output/eval_aedat4_subseq')

# ========== Model Initialization ==========
print("Loading model...")
model = TAPFormer_online(
    window_size=window_size,
    stride=stride,
    corr_radius=corr_radius,
    corr_levels=corr_levels,
    backbone=backbone,
    hidden_size=hidden_size,
    space_depth=space_depth,
    time_depth=time_depth
)

# Load checkpoint
state_dict = torch.load(ckpt_root, map_location=DEFAULT_DEVICE)
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded successfully!")

# ========== Evaluation ==========
print("\n" + "="*50)
print("Evaluating on Aedat4 dataset...")
print("="*50)
datasets = TAPFormer_dataset(os.path.join(dataset_dir), representation=representation, dt=dt)
for seq_name in EVAL_DATASETS:
    sample, gotit = datasets.get_a_seq(seq_name)
    if not gotit:
        continue
    
    # Setup output directory
    output_dir = os.path.join(base_output_dir, seq_name, model_name)
    if enable_visualization or save_results or save_trajectory:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer if needed
    vis = None
    if enable_visualization:
        vis = Visualizer(output_dir, fps=vis_cfg.get('fps', 20))
 
    predictor = EvaluationPredictor(
                model,
                grid_size=grid_size,
                local_grid_size=0,
                single_point=False,
                num_uniformly_sampled_pts=0,
                n_iters=n_iters,
            )
    
    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    queries = sample.query_points[np.newaxis, ...]
    queries = queries.to(DEFAULT_DEVICE)
    
    
    sample.video = sample.video[np.newaxis, ...]
    sample.events = sample.events[np.newaxis, ...]
    sample.trajectory = sample.trajectory[np.newaxis, ...]
    sample.visibility = sample.visibility[np.newaxis, ...]
    
    start = time.time()
    pred_tracks = predictor(sample.video, sample.events, queries, img_ifnew=sample.img_ifnew)
    end = time.time()
    elapsed_time = (end-start)/sample.events.shape[1]
    print("time per frame:", elapsed_time)
    
    if isinstance(pred_tracks, tuple):
        pred_trajectory, pred_visibility, _ = pred_tracks
    else:
        pred_visibility = None
    
    if pred_visibility is None:
        pred_visibility = torch.zeros_like(sample.visibility)

    if not pred_visibility.dtype == torch.bool:
        pred_visibility = pred_visibility > 0.8
        
    pred_occluded = (
            torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
            .cpu()
            .numpy()
        )
    pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
    
    query_points = np.concatenate(
        (
            np.zeros_like(sample.trajectory[:, 0, :, :1]),
            sample.trajectory[:, 0, :, 1:],
        ),
        axis=2,
    )
    
    gt_tracks = np.transpose(sample.trajectory[:, :, :, 1:].copy(), (0, 2, 1, 3))
    gt_occluded = np.transpose(sample.visibility.copy(), (0, 2, 1))
    
    def expand_trajectory(traj, vis):
        """
        traj: numpy array of shape (B, N, T, 2)
        return: numpy array of shape (B, N, 2*T, 2)
        """
        B, N, T, _ = traj.shape
        
        # 初始化结果
        expanded = np.zeros((B, N, 2*T, 2), dtype=traj.dtype)
        expanded_vis = np.zeros((B, N, 2*T), dtype=vis.dtype)
        
        expanded_vis[:, :, ::2] = vis
        expanded_vis[:, :, 1::2] = vis
        
        # 原始帧放在偶数位置
        expanded[:, :, ::2, :] = traj
        
        # 计算速度（v_t = p_t - p_{t-1}）
        vel = np.zeros_like(traj)
        vel[:, :, 1:, :] = traj[:, :, 1:, :] - traj[:, :, :-1, :]
        
        # 用上一帧速度估算插值帧位置
        # p_{t+0.5} = p_t + 0.5 * v_t
        expanded[:, :, 1::2, :] = traj + 0.5 * vel
        
        # 对 t=0 的插值帧（expanded[:, :, 1, :]），因为没有前一帧速度，保持不变
        expanded[:, :, 1, :] = traj[:, :, 0, :]

        return expanded, expanded_vis
    
    if pred_trajectory.shape[1] != sample.trajectory.shape[1]:
        pred_trajectory_new = np.zeros_like(gt_tracks)
        pred_occluded_new = np.zeros_like(gt_occluded)
        for i in range(pred_tracks.shape[1]):
            pred_traj = pred_tracks[:, i, :, :]
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
            
        pred_tracks = pred_trajectory_new.copy()
        pred_occluded = pred_occluded_new.copy()
    out_metrics = compute_tapvid_metrics(
        query_points,
        gt_occluded,
        gt_tracks,
        pred_occluded,
        pred_tracks,
        query_mode="first",
    )
    print("metrics", out_metrics)
    
    # Visualization
    if enable_visualization and vis is not None:
        vis.visualize(
            sample.video if isinstance(sample.video, torch.Tensor) else torch.from_numpy(sample.video).float(), 
            sample.events if isinstance(sample.events, torch.Tensor) else torch.from_numpy(sample.events).float(),
            pred_trajectory, 
            pred_visibility > 0.8,
            filename=seq_name,
            video_model="rgb",
        )
    
    # Save trajectory if requested
    if save_trajectory:
        B, T, N, _ = pred_trajectory.shape
        ind = np.arange(N).reshape(1, 1, N, 1).repeat(T, axis=1)
        t = sample.segmentation.astype(float)
        t *= 1e-6
        t = t.reshape(1, -1, 1, 1).repeat(N, axis=2)
        pred_trajectory_with_time = np.concatenate((ind, t, pred_trajectory.cpu().numpy()), axis=3)
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
