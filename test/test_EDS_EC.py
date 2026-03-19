"""
Test script for TAPFormer model on EDS and EC datasets.
This script evaluates the model performance on test sequences.

Usage:
    python test_EDS_EC.py [--config config_eds_ec.yaml]
"""

import os
import sys
import argparse
import yaml
import torch
import time
import numpy as np

from LFE_TAP.evaluator.evaluator import compareTracks
from LFE_TAP.datasets.EDS_dataset import EDS_dataset
from LFE_TAP.datasets.EC_dataset import EC_dataset
from LFE_TAP.evaluator.prediction import TAPFormer_online
from LFE_TAP.evaluator.evaluation_pred import EvaluationPredictor
from LFE_TAP.utils.visualizer import Visualizer

# ========== Configuration ==========
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
    parser = argparse.ArgumentParser(description='Test TAPFormer model on EDS and EC datasets')
    parser.add_argument('--config', type=str, default='config/config_eds_ec.yaml',
                        help='Path to configuration YAML file')
    return parser.parse_args()

# Parse arguments and load config
args = parse_args()
config = load_config(args.config)

# Extract configuration
dataset_dir = config['dataset_dir']
ckpt_root = config['ckpt_root']

EVAL_DATASETS_EDS = config.get('eval_datasets_eds') or []
EVAL_DATASETS_EC = config.get('eval_datasets_ec') or []

# Model configuration
model_cfg = config
representation = model_cfg['representation']
stride = model_cfg['stride']
corr_levels = model_cfg['corr_levels']
backbone = model_cfg['backbone']
window_size = model_cfg.get('window_size', 16)
corr_radius = model_cfg.get('corr_radius', 3)
hidden_size = model_cfg.get('hidden_size', 384)
space_depth = model_cfg.get('space_depth', 3)
time_depth = model_cfg.get('time_depth', 3)

# Evaluation settings
eds_cfg = config['eds']
ec_cfg = config['ec']

# Visualization and output settings
vis_cfg = config.get('visualization', {})
output_cfg = config.get('output', {})
enable_visualization = vis_cfg.get('enable', False)
save_results = output_cfg.get('save_results', False)
save_trajectory = output_cfg.get('save_trajectory', False)

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

# ========== Evaluation on EDS Dataset ==========
print("\n" + "="*50)
print("Evaluating on EDS dataset...")
print("="*50)
fa, efa, t_l = [], [], []
for seq_name in EVAL_DATASETS_EDS:
    datasets = EDS_dataset(os.path.join(dataset_dir, "eds_subseq"), representation=representation, dt=eds_cfg['dt'])
    sample, gotit = datasets.get_a_seq(seq_name)
    if not gotit:
        continue
    
    # Setup output directory
    output_dir = os.path.join(output_cfg.get('eds_dir', 'output/eval_eds_subseq'), seq_name)
    if enable_visualization or save_results or save_trajectory:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer if needed
    vis = None
    if enable_visualization:
        vis = Visualizer(output_dir, fps=vis_cfg.get('fps', 50))
    
    grid_size = eds_cfg['grid_size'] 
 
    predictor = EvaluationPredictor(
                model,
                grid_size=grid_size,
                local_grid_size=0,
                single_point=False,
                num_uniformly_sampled_pts=0,
                n_iters=eds_cfg['n_iters'],
                if_test=True,
            )
    
    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    queries = sample.query_points[np.newaxis, ...]
    queries = queries.to(DEFAULT_DEVICE)
    
    sample.video = sample.video[np.newaxis, ...]
    sample.events = sample.events[np.newaxis, ...]
    
    start = time.time()
    pred_tracks = predictor(sample.video, sample.events, queries, img_ifnew=sample.img_ifnew)
    elapsed_time = (time.time()-start)/sample.events.shape[1]
    t_l.append(elapsed_time)
    print("time per frame:", elapsed_time)
    
    # Visualization
    if enable_visualization and vis is not None:
        vis.visualize(
            sample.video if isinstance(sample.video, torch.Tensor) else torch.from_numpy(sample.video).float(), 
            sample.events if isinstance(sample.events, torch.Tensor) else torch.from_numpy(sample.events).float(), 
            pred_tracks[0], 
            pred_tracks[1] > 0.8,
            filename=seq_name,
            video_model="events",
        )
    
    B, T, N, _ = pred_tracks[0].shape
    ind = np.arange(N).reshape(1, 1, N, 1).repeat(T, axis=1)
    t = sample.segmentation.astype(float)
    t *= 1e-6
    t = t.reshape(1, -1, 1, 1).repeat(N, axis=2)
    pred_trajectory = np.concatenate((ind, t, pred_tracks[0].cpu().numpy()), axis=3)
    pred_trajectory_txt = np.transpose(pred_trajectory, (0, 2, 1, 3)).reshape(-1, 4)
    
    # Save trajectory if requested
    if save_trajectory:
        traj_path = os.path.join(output_dir, "pred_trajectory.txt")
        np.savetxt(traj_path, pred_trajectory_txt, fmt=["%i", "%.9f", "%.2f", "%.2f"], delimiter=" ")
        print(f"Trajectory saved to {traj_path}")
    
    traj = sample.trajectory.copy()
            
    mean_err_avg = []
    for i in range(1, 32):
        error_datas, errors, mean_err = compareTracks(pred_trajectory_txt, traj, i)
        mean_err_avg.append(mean_err)
    mean_err_avg = np.stack(mean_err_avg)
    mean_err_avg = np.mean(mean_err_avg, axis=0)
    print(seq_name, "deep_ev mean error:", mean_err_avg[0], " mean age:", mean_err_avg[1], "expect age:", mean_err_avg[2])
    fa.append(mean_err_avg[1])
    efa.append(mean_err_avg[2])
    
    # Save results if requested
    if save_results:
        result_path = os.path.join(output_dir, "result.txt")
        with open(result_path, 'w') as f:
            f.write(f"{seq_name} deep_ev mean error: {mean_err_avg[0]} mean age: {mean_err_avg[1]} expect age: {mean_err_avg[2]}\n")
            f.write(f"time per frame: {elapsed_time}\n")
        print(f"Results saved to {result_path}")

if len(fa) > 0:
    avg_fa = np.array(fa).sum()/len(fa)
    avg_efa = np.array(efa).sum()/len(efa)
    avg_time = np.array(t_l).sum()/len(t_l)
    print("ave fa:", avg_fa, "efa:", avg_efa, "time:", avg_time)

if len(fa) > 0:
    print(f"\nEDS Dataset Average Results:")
    print(f"  Average FA: {np.array(fa).sum()/len(fa):.4f}")
    print(f"  Average EFA: {np.array(efa).sum()/len(efa):.4f}")
    print(f"  Average Time per Frame: {np.array(t_l).sum()/len(t_l):.4f} seconds")

# ========== Evaluation on EC Dataset ==========
print("\n" + "="*50)
print("Evaluating on EC dataset...")
print("="*50)
fa, efa, t_l = [], [], []
datasets = EC_dataset(os.path.join(dataset_dir, "ec_subseq"), representation=representation, dt=ec_cfg['dt'])
for seq_name in EVAL_DATASETS_EC:
    sample, gotit = datasets.get_a_seq(seq_name)
    if not gotit:
        continue
    
    # Setup output directory
    output_dir = os.path.join(output_cfg.get('ec_dir', 'output/eval_ec_subseq'), seq_name)
    if enable_visualization or save_results or save_trajectory:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer if needed
    vis = None
    if enable_visualization:
        vis = Visualizer(output_dir, fps=vis_cfg.get('fps_ec', 10))
    
    grid_size = ec_cfg['grid_size']
 
    predictor = EvaluationPredictor(
                model,
                grid_size=grid_size,
                local_grid_size=0,
                single_point=False,
                num_uniformly_sampled_pts=0,
                n_iters=ec_cfg['n_iters'],
                if_test=True,
            )
    
    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    queries = sample.query_points[np.newaxis, ...]
    queries = queries.to(DEFAULT_DEVICE)
    
    sample.video = sample.video[np.newaxis, ...]
    sample.events = sample.events[np.newaxis, ...]
    
    start = time.time()
    pred_tracks = predictor(sample.video, sample.events, queries, img_ifnew=sample.img_ifnew)
    elapsed_time = (time.time()-start)/sample.events.shape[1]
    t_l.append(elapsed_time)
    
    # Visualization
    if enable_visualization and vis is not None:
        vis.visualize(
            sample.video if isinstance(sample.video, torch.Tensor) else torch.from_numpy(sample.video).float(), 
            sample.events if isinstance(sample.events, torch.Tensor) else torch.from_numpy(sample.events).float(), 
            pred_tracks[0], 
            pred_tracks[1] > 0.8,
            filename=seq_name,
            video_model='events',
        )
    
    B, T, N, _ = pred_tracks[0].shape
    ind = np.arange(N).reshape(1, 1, N, 1).repeat(T, axis=1)
    t = sample.segmentation.astype(float)
    t *= 1e-6
    t = t.reshape(1, -1, 1, 1).repeat(N, axis=2)
    pred_trajectory = np.concatenate((ind, t, pred_tracks[0].cpu().numpy()), axis=3)
    pred_trajectory_txt = np.transpose(pred_trajectory, (0, 2, 1, 3)).reshape(-1, 4)
    
    # Save trajectory if requested
    if save_trajectory:
        traj_path = os.path.join(output_dir, "pred_trajectory.txt")
        np.savetxt(traj_path, pred_trajectory_txt, fmt=["%i", "%.9f", "%.2f", "%.2f"], delimiter=" ")
        print(f"Trajectory saved to {traj_path}")
    
    traj = sample.trajectory.copy()
            
    mean_err_avg = []
    for i in range(1, 32):
        error_datas, errors, mean_err = compareTracks(pred_trajectory_txt, traj, i)
        mean_err_avg.append(mean_err)
    mean_err_avg = np.stack(mean_err_avg)
    mean_err_avg = np.mean(mean_err_avg, axis=0)
    print(seq_name, "deep_ev mean error:", mean_err_avg[0], " mean age:", mean_err_avg[1], "expect age:", mean_err_avg[2])
    
    fa.append(mean_err_avg[1])
    efa.append(mean_err_avg[2])
    
    # Save results if requested
    if save_results:
        result_path = os.path.join(output_dir, "result.txt")
        with open(result_path, 'w') as f:
            f.write(f"{seq_name} deep_ev mean error: {mean_err_avg[0]} mean age: {mean_err_avg[1]} expect age: {mean_err_avg[2]}\n")
            f.write(f"time per frame: {elapsed_time}\n")
        print(f"Results saved to {result_path}")

if len(fa) > 0:
    avg_fa = np.array(fa).sum()/len(fa)
    avg_efa = np.array(efa).sum()/len(efa)
    avg_time = np.array(t_l).sum()/len(t_l)
    print("ave fa:", avg_fa, "efa:", avg_efa, "time:", avg_time)

if len(fa) > 0:
    print(f"\nEC Dataset Average Results:")
    print(f"  Average FA: {np.array(fa).sum()/len(fa):.4f}")
    print(f"  Average EFA: {np.array(efa).sum()/len(efa):.4f}")
    print(f"  Average Time per Frame: {np.array(t_l).sum()/len(t_l):.4f} seconds")