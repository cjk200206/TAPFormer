import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(x.dtype)
    return flag * 0.5 * diff**2 + (1.0 - flag) * delta * (abs_diff - 0.5 * delta)


class TAPFormerLoss(nn.Module):
    """CoTracker-style sequence losses adapted to TAPFormer train_data.

    train_data format:
      (all_coords_predictions, all_vis_predictions, all_conf_predictions, causal_valid_mask)

    Notes:
    - Coordinate loss follows CoTracker sequence_loss style (L1 or optional Huber).
    - Visibility / confidence BCE keep TAPFormer bf16-safe semantics by using logits
      with binary_cross_entropy_with_logits.
    """

    def __init__(
        self,
        coord_weight: float = 1.0,
        visibility_weight: float = 1.0,
        confidence_weight: float = 1.0,
        iter_gamma: float = 0.8,
        confidence_threshold: float = 8.0,
        add_huber_loss: bool = False,
        huber_delta: float = 6.0,
        loss_only_for_visible: bool = False,
    ) -> None:
        super().__init__()
        self.coord_weight = float(coord_weight)
        self.visibility_weight = float(visibility_weight)
        self.confidence_weight = float(confidence_weight)
        self.iter_gamma = float(iter_gamma)
        self.confidence_threshold = float(confidence_threshold)
        self.add_huber_loss = bool(add_huber_loss)
        self.huber_delta = float(huber_delta)
        self.loss_only_for_visible = bool(loss_only_for_visible)

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

            n_predictions = len(iter_coord_preds)
            if n_predictions == 0:
                continue

            s_trim = int(iter_coord_preds[-1].shape[1])
            gt_traj_w = gt_trajectory[:, start : start + s_trim]
            gt_vis_w = gt_visibility[:, start : start + s_trim]
            gt_valid_w = gt_valid[:, start : start + s_trim] * causal_valid_mask[:, start : start + s_trim]

            flow_valid = gt_valid_w.clone()
            if self.loss_only_for_visible:
                flow_valid = flow_valid * gt_vis_w

            coord_loss_w = gt_trajectory.new_zeros(())
            vis_loss_w = gt_trajectory.new_zeros(())
            conf_loss_w = gt_trajectory.new_zeros(())

            for i in range(n_predictions):
                i_weight = self.iter_gamma ** (n_predictions - i - 1)
                pred_coords = iter_coord_preds[i]
                pred_vis_logits = iter_vis_preds[i]
                pred_conf_logits = iter_conf_preds[i]

                if self.add_huber_loss:
                    flow_elem = _huber_loss(pred_coords, gt_traj_w, delta=self.huber_delta)
                else:
                    flow_elem = (pred_coords - gt_traj_w).abs()
                flow_elem = flow_elem.mean(dim=-1)
                coord_loss_w = coord_loss_w + i_weight * _masked_mean(flow_elem, flow_valid)

                vis_bce = F.binary_cross_entropy_with_logits(
                    pred_vis_logits,
                    gt_vis_w,
                    reduction="none",
                )
                vis_loss_w = vis_loss_w + _masked_mean(vis_bce, gt_valid_w)

                err_sq = torch.sum((pred_coords.detach() - gt_traj_w) ** 2, dim=-1)
                conf_target = (err_sq <= self.confidence_threshold**2).float()
                conf_bce = F.binary_cross_entropy_with_logits(
                    pred_conf_logits,
                    conf_target,
                    reduction="none",
                )
                conf_mask = gt_vis_w * gt_valid_w
                conf_loss_w = conf_loss_w + _masked_mean(conf_bce, conf_mask)

            coord_loss_total = coord_loss_total + (coord_loss_w / n_predictions)
            vis_loss_total = vis_loss_total + (vis_loss_w / n_predictions)
            conf_loss_total = conf_loss_total + (conf_loss_w / n_predictions)

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
