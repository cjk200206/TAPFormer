import os
import numpy as np
import torch
import logging
from tqdm import tqdm
from typing import Optional, Mapping
from torch.utils.tensorboard import SummaryWriter
from LFE_TAP.models.tapformer import TAPFormer
from LFE_TAP.utils.visualizer import Visualizer
from LFE_TAP.utils.dataset_utils import dataclass_to_cuda_

def get_error(est_data, gt_data):
    # discard gt which happen after last est_data
    # gt_data = gt_data[gt_data[:, 0] <= est_data[-1, 0]]

    est_t, est_x, est_y = est_data.T
    gt_t, gt_x, gt_y = gt_data.T

    
    if len(gt_t) == 0 or len(est_t) == 0:
        return [], [], []

    if len(est_t) < 2:
        return gt_t, np.array([0]), np.array([0])

    # find samples which have dt > threshold
    error_x = np.interp(gt_t, est_t, est_x) - gt_x
    error_y = np.interp(gt_t, est_t, est_y) - gt_y

    return gt_t, error_x, error_y


def compareTracks(tracks_pred, tracks_gt, max_distance=10):
    pred_ids = np.unique(tracks_pred[:, 0])
    gt_ids = np.unique(tracks_gt[:, 0])
    pred_datas = {i: tracks_pred[tracks_pred[:, 0] == i, 1:] for i in pred_ids}
    gt_datas = {i: tracks_gt[tracks_gt[:, 0] == i, 1:] for i in gt_ids}
    
    error_datas = np.zeros(shape=(0, 4))
    errors = np.zeros(shape=(0, 2))
    
    for track_id, pred_data in tqdm(pred_datas.items(), disable=True):
        gt_data = gt_datas[track_id]
        init_time = gt_data[0, 0]
        gt_data[:, 0] -= init_time
        pred_data[:, 0] -= init_time
        pred_data = np.concatenate((gt_data[0, :].reshape(1, -1), pred_data), axis=0)
        
        gt_t, error_x, error_y = get_error(pred_data, gt_data)
        
        if len(gt_t) != 0:
            ids = (track_id * np.ones_like(error_x)).astype(int)
            added_data = np.stack([ids, gt_t, error_x, error_y]).T
            error_euclidean = np.sqrt(added_data[:, 2]**2 + added_data[:, 3]**2)
            error_euclidean[0] = 0
            idxs = np.where(error_euclidean > max_distance)[0]
            if len(idxs) > 0:
                feature_age = (gt_t[int(idxs[0])] - gt_t[0]) / (gt_t[-1] - gt_t[0])
                if int(idxs[0]) <= 1:
                    continue
                error_euclidean = error_euclidean[:idxs[0]]
                error = np.mean(error_euclidean)
            else:
                feature_age = 1
                error = np.mean(error_euclidean)
                
            errors = np.concatenate([errors, [[error, feature_age]]])
            error_datas = np.concatenate([error_datas, added_data])
            
    if errors.shape[0] == 0:
        return [], [], [0, (gt_t[1]-gt_t[0])/(gt_t[-1] - gt_t[0]), 0]
            
    mean_error = np.mean(errors, axis=0)
    ecpect_FA = (errors.shape[0]/len(gt_ids)) * mean_error[1]
    mean_error = np.concatenate((mean_error, [ecpect_FA]))
    # print("abandon ", len(gt_ids) - errors.shape[0], " points, remain ", errors.shape[0], " points")
    # print("Mean Error: {:.4f}, Mean Feature Age: {:.4f}, Ecppect Feature Age: {:.4f}".format(mean_error[0], mean_error[1], mean_error[2]))
    
    return error_datas, errors, mean_error


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics

class Evaluator:
    def __init__(self, output_dir) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def compute_metrics(self, metrics, sample, pred_trajectory, dataset_name):
        if isinstance(pred_trajectory, tuple):
            pred_trajectory, pred_visibility = pred_trajectory
        else:
            pred_visibility = None
            
        if "kubric" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone()
            thr = 0.6

            if pred_visibility is None:
                logging.warning("visibility is NONE")
                pred_visibility = torch.zeros_like(sample.visibility)

            if not pred_visibility.dtype == torch.bool:
                pred_visibility = pred_visibility > thr

            query_points = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).cpu().numpy()

            pred_visibility = pred_visibility[:, :, :N]
            pred_trajectory = pred_trajectory[:, :, :N]

            gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            gt_occluded = (
                torch.logical_not(sample.visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )

            pred_occluded = (
                torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()

            out_metrics = compute_tapvid_metrics(
                query_points,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = np.mean(
                    [v[metric_name] for k, v in metrics.items() if k != "avg"]
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])
        elif "EC" or "EDS" in dataset_name:
            traj = sample.trajectory.copy()
            
            mean_err_avg = []
            for i in range(1, 31):
                error_datas, errors, mean_err = compareTracks(pred_trajectory.cpu().numpy(), traj[0], sample.segmentation, i)
                mean_err_avg.append(mean_err)
            mean_err_avg = np.stack(mean_err_avg)
            mean_err_avg = np.mean(mean_err_avg, axis=0)
            print(sample.seq_name, "deep_ev mean error:", mean_err_avg[0], " mean age:", mean_err_avg[1], "expect age:", mean_err_avg[2])
            
    
    @torch.no_grad()       
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        visualize_every: int = 1,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
    ):
        metrics = {}
        
        vis = Visualizer(self.output_dir, fps=10 if "kubric" in dataset_name else 50)
        
        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print(f"Skipping sample {ind} because gotit is {gotit}")
                    continue
                
            if torch.cuda.is_available():
                if "kubric" in dataset_name:
                    dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
            if "kubric" in dataset_name:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).to(device)
            elif "EC" or "EDS" in dataset_name:
                queries = sample.query_points
                queries = queries.to(device)
            
            pred_tracks = model(sample.video, sample.events, queries)
            
            if dataset_name == "EC" or dataset_name == "EDS":
                seq_name = sample.seq_name[0]
            else:
                seq_name = str(ind)
            if ind % visualize_every == 0:
                vis.visualize(
                    sample.video if isinstance(sample.video, torch.Tensor) else torch.from_numpy(sample.video).float(), 
                    pred_tracks[0], 
                    pred_tracks[1] > 0.8,
                    filename=dataset_name + "_" + seq_name,
                    writer=writer,
                    step=step,
                )
            self.compute_metrics(metrics, sample, pred_tracks, dataset_name)
        return metrics
                
            
                
            
                
            
