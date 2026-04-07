import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


class TAPFormerLoss(nn.Module):
    """Training loss for TAPFormer.

    This consumes the model's train_data tuple:
      (all_coords_predictions, all_vis_predictions, all_conf_predictions, causal_valid_mask)
    where each predictions list is windowed and contains per-iteration tensors.
    """

    def __init__(
        self,
        coord_weight: float = 1.0,
        visibility_weight: float = 1.0,
        confidence_weight: float = 1.0,
        iter_gamma: float = 0.8,
        confidence_threshold: float = 8.0,
    ) -> None:
        super().__init__()
        self.coord_weight = float(coord_weight)
        self.visibility_weight = float(visibility_weight)
        self.confidence_weight = float(confidence_weight)
        self.iter_gamma = float(iter_gamma)
        self.confidence_threshold = float(confidence_threshold)

    def forward(
        self,
        train_data,
        gt_trajectory: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_valid: torch.Tensor,
    ):
        all_coords_predictions, all_vis_predictions, all_conf_predictions, causal_valid_mask = train_data

        if len(all_coords_predictions) == 0:
            zero = gt_trajectory.new_zeros(())
            return {
                "loss": zero,
                "coord_loss": zero,
                "visibility_loss": zero,
                "confidence_loss": zero,
            }

        gt_visibility = gt_visibility.float()
        gt_valid = gt_valid.float()
        causal_valid_mask = causal_valid_mask.float()

        window_size = int(all_coords_predictions[0][0].shape[1])
        step = max(1, window_size // 2)
        window_starts = [i * step for i in range(len(all_coords_predictions))]

        coord_loss_total = gt_trajectory.new_zeros(())
        vis_loss_total = gt_trajectory.new_zeros(())
        conf_loss_total = gt_trajectory.new_zeros(())

        for w_idx, start in enumerate(window_starts):
            iter_coord_preds = all_coords_predictions[w_idx]
            iter_vis_preds = all_vis_predictions[w_idx]
            iter_conf_preds = all_conf_predictions[w_idx]

            n_iters = len(iter_coord_preds)
            if n_iters == 0:
                continue

            s_trim = int(iter_coord_preds[-1].shape[1])
            gt_traj_w = gt_trajectory[:, start : start + s_trim]
            gt_vis_w = gt_visibility[:, start : start + s_trim]
            gt_valid_w = gt_valid[:, start : start + s_trim] * causal_valid_mask[:, start : start + s_trim]
            coord_mask_w = gt_valid_w * gt_vis_w

            for it in range(n_iters):
                weight = self.iter_gamma ** (n_iters - 1 - it)
                pred_coords = iter_coord_preds[it]
                pred_vis = iter_vis_preds[it].clamp(1e-4, 1.0 - 1e-4)
                pred_conf = iter_conf_preds[it].clamp(1e-4, 1.0 - 1e-4)

                coord_l2 = torch.linalg.norm(pred_coords - gt_traj_w, dim=-1)
                coord_loss_total = coord_loss_total + weight * _masked_mean(coord_l2, coord_mask_w)

                vis_bce = F.binary_cross_entropy(pred_vis, gt_vis_w, reduction="none")
                vis_loss_total = vis_loss_total + weight * _masked_mean(vis_bce, gt_valid_w)

                conf_target = ((coord_l2.detach() < self.confidence_threshold).float() * gt_vis_w)
                conf_bce = F.binary_cross_entropy(pred_conf, conf_target, reduction="none")
                conf_loss_total = conf_loss_total + weight * _masked_mean(conf_bce, gt_valid_w)

        norm = max(1, len(all_coords_predictions))
        coord_loss_total = coord_loss_total / norm
        vis_loss_total = vis_loss_total / norm
        conf_loss_total = conf_loss_total / norm

        total_loss = (
            self.coord_weight * coord_loss_total
            + self.visibility_weight * vis_loss_total
            + self.confidence_weight * conf_loss_total
        )

        return {
            "loss": total_loss,
            "coord_loss": coord_loss_total,
            "visibility_loss": vis_loss_total,
            "confidence_loss": conf_loss_total,
        }
