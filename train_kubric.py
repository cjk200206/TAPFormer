import os
import argparse


def _configure_visible_devices_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default=None)
    known_args, _ = parser.parse_known_args()
    if known_args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = known_args.gpus


_configure_visible_devices_from_argv()

import json
import math
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from LFE_TAP.datasets.kubric_movif_dataset import KubricMovifDataset_etap
from LFE_TAP.models.tapformer import TAPFormer
from LFE_TAP.models.tapformer_ablation import TAPFormerAblation
from LFE_TAP.models.tapformer_cow_dense import TAPFormerCowDense
from LFE_TAP.models.tapformer_point_warp import TAPFormerPointWarp
from LFE_TAP.models.losses import TAPFormerLoss
from LFE_TAP.utils.dataset_utils import FrameEventData


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TAPFormer on Kubric/ETAP-format data")
    parser.add_argument("--config", type=str, default="config/config_kubric_train.yaml", help="Path to training YAML config")
    parser.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES value, for example '0' or '0,1'")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_collate(batch):
    samples, gotits = zip(*batch)

    def stack_or_none(attr):
        vals = [getattr(s, attr, None) for s in samples]
        if any(v is None for v in vals):
            return None
        if isinstance(vals[0], torch.Tensor):
            return torch.stack(vals, dim=0)
        return vals

    stacked = FrameEventData(
        video=torch.stack([s.video for s in samples], dim=0),
        events=torch.stack([s.events for s in samples], dim=0),
        segmentation=torch.stack([s.segmentation for s in samples], dim=0),
        trajectory=torch.stack([s.trajectory for s in samples], dim=0),
        visibility=torch.stack([s.visibility for s in samples], dim=0),
        img_ifnew=stack_or_none("img_ifnew"),
        clear_video=stack_or_none("clear_video"),
        valid=stack_or_none("valid"),
        seq_name=[s.seq_name for s in samples],
        query_points=stack_or_none("query_points"),
        reference_video=stack_or_none("reference_video"),
        reference_events=stack_or_none("reference_events"),
    )
    gotit = torch.tensor(gotits, dtype=torch.bool)
    return stacked, gotit


def build_queries(trajectory: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    # trajectory: [B, T, N, 2], visibility: [B, T, N]
    B, _, N, _ = trajectory.shape
    vis_bool = visibility > 0.5
    first_vis = torch.argmax(vis_bool.int(), dim=1)  # [B, N]

    batch_idx = torch.arange(B, device=trajectory.device)[:, None]
    point_idx = torch.arange(N, device=trajectory.device)[None, :]
    query_xy = trajectory[batch_idx, first_vis, point_idx]
    return torch.cat([first_vis.float().unsqueeze(-1), query_xy], dim=-1)


def move_batch_to_device(sample: FrameEventData, device: torch.device) -> FrameEventData:
    sample.video = sample.video.to(device, non_blocking=True)
    sample.events = sample.events.to(device, non_blocking=True)
    sample.segmentation = sample.segmentation.to(device, non_blocking=True)
    sample.trajectory = sample.trajectory.to(device, non_blocking=True)
    sample.visibility = sample.visibility.to(device, non_blocking=True)
    if sample.valid is not None and isinstance(sample.valid, torch.Tensor):
        sample.valid = sample.valid.to(device, non_blocking=True)
    if sample.img_ifnew is not None and isinstance(sample.img_ifnew, torch.Tensor):
        sample.img_ifnew = sample.img_ifnew.to(device, non_blocking=True)
    if sample.query_points is not None and isinstance(sample.query_points, torch.Tensor):
        sample.query_points = sample.query_points.to(device, non_blocking=True)
    if sample.reference_video is not None and isinstance(sample.reference_video, torch.Tensor):
        sample.reference_video = sample.reference_video.to(device, non_blocking=True)
    if sample.reference_events is not None and isinstance(sample.reference_events, torch.Tensor):
        sample.reference_events = sample.reference_events.to(device, non_blocking=True)
    return sample


def build_model_from_config(model_cfg):
    common_kwargs = dict(
        window_size=int(model_cfg.get("window_size", 8)),
        stride=int(model_cfg.get("stride", 4)),
        corr_radius=int(model_cfg.get("corr_radius", 3)),
        corr_levels=int(model_cfg.get("corr_levels", 3)),
        hidden_size=int(model_cfg.get("hidden_size", 384)),
        space_depth=int(model_cfg.get("space_depth", 3)),
        time_depth=int(model_cfg.get("time_depth", 3)),
    )
    model_name = str(model_cfg.get("name", "tapformer")).lower().strip()
    if model_name == "tapformer":
        return TAPFormer(**common_kwargs)
    if model_name in {"tapformer_ablation", "ablation"}:
        return TAPFormerAblation(
            feature_mode=model_cfg.get("feature_mode", "fusion"),
            **common_kwargs,
        )
    if model_name in {"tapformer_cow_dense", "cow_dense"}:
        return TAPFormerCowDense(
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
            cow_tapir_init=bool(model_cfg.get("cow_tapir_init", False)),
            cow_tapir_init_stride=int(model_cfg.get("cow_tapir_init_stride", 16)),
            cow_tapir_init_temperature=float(model_cfg.get("cow_tapir_init_temperature", 20.0)),
            cow_tapir_init_radius=int(model_cfg.get("cow_tapir_init_radius", 5)),
            cow_tapir_init_chunk_size=int(model_cfg.get("cow_tapir_init_chunk_size", 64)),
            cow_frontend_type=str(model_cfg.get("cow_frontend_type", "base")),
            cow_anchor_state_mix=float(model_cfg.get("cow_anchor_state_mix", 0.7)),
            cow_anchor_skip_mix=float(model_cfg.get("cow_anchor_skip_mix", 0.7)),
            **common_kwargs,
        )
    if model_name in {"tapformer_point_warp", "point_warp"}:
        return TAPFormerPointWarp(
            point_state_dim=int(model_cfg.get("point_state_dim", 128)),
            point_warp_dim=int(model_cfg.get("point_warp_dim", 256)),
            point_corr_levels=int(
                model_cfg.get("point_corr_levels", model_cfg.get("corr_levels", 3))
            ),
            point_patch_radius=int(model_cfg.get("point_patch_radius", 2)),
            point_local_corr_radius=int(model_cfg.get("point_local_corr_radius", 3)),
            point_coarse_iters=model_cfg.get("point_coarse_iters", None),
            point_refine_iters=model_cfg.get("point_refine_iters", None),
            point_refine_use_local_evidence=bool(model_cfg.get("point_refine_use_local_evidence", True)),
            point_refine_detach_local_evidence=bool(model_cfg.get("point_refine_detach_local_evidence", True)),
            point_use_global_init=bool(model_cfg.get("point_use_global_init", False)),
            point_cost_base_stride=int(model_cfg.get("point_cost_base_stride", 8)),
            point_cost_levels=int(model_cfg.get("point_cost_levels", 3)),
            point_cost_radius=int(model_cfg.get("point_cost_radius", 3)),
            point_cost_dim=int(model_cfg.get("point_cost_dim", 256)),
            point_initializer_stride=int(model_cfg.get("point_initializer_stride", 16)),
            point_initializer_temperature=float(model_cfg.get("point_initializer_temperature", 20.0)),
            point_initializer_radius=int(model_cfg.get("point_initializer_radius", 5)),
            point_initializer_chunk_size=int(model_cfg.get("point_initializer_chunk_size", 64)),
            point_detach_coords=bool(model_cfg.get("point_detach_coords", True)),
            point_limit_update=bool(model_cfg.get("point_limit_update", True)),
            point_max_update_ratio=float(model_cfg.get("point_max_update_ratio", 0.15)),
            point_max_magnitude_ratio=float(model_cfg.get("point_max_magnitude_ratio", 1.0)),
            point_support_mode=str(model_cfg.get("point_support_mode", "none")),
            **common_kwargs,
        )
    raise ValueError(
        f"Unsupported model.name={model_name}. "
        "Use one of: tapformer, tapformer_ablation, tapformer_cow_dense, tapformer_point_warp."
    )


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Checkpoint format is not supported: expected a state_dict-like dict.")


def normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_point_warp_pretrained(model, source_state):
    """Load compatible fusion/attention weights without changing legacy loaders."""
    target_state = model.state_dict()
    attention_prefixes = (
        "time_blocks.",
        "space_virtual_blocks.",
        "space_point2virtual_blocks.",
        "space_virtual2point_blocks.",
        "virual_tracks",
    )
    compatible = {}
    shape_mismatch = []
    for source_key, value in source_state.items():
        target_key = source_key
        if source_key.startswith("updateformer2."):
            suffix = source_key[len("updateformer2.") :]
            if not suffix.startswith(attention_prefixes):
                continue
            target_key = f"point_head.updateformer2.{suffix}"
        elif not source_key.startswith("fusion_block.") and not source_key.startswith(
            ("initializer.", "point_head.")
        ):
            continue

        if target_key not in target_state:
            continue
        if target_state[target_key].shape != value.shape:
            shape_mismatch.append(target_key)
            continue
        compatible[target_key] = value

    incompatible = model.load_state_dict(compatible, strict=False)
    print(
        "Point-warp pretrained load: "
        f"loaded={len(compatible)}, missing={len(incompatible.missing_keys)}, "
        f"shape_mismatch={len(shape_mismatch)}",
        flush=True,
    )
    return incompatible


def metric_sign(mode: str):
    mode_norm = str(mode).lower().strip()
    if mode_norm == "min":
        return -1.0
    if mode_norm == "max":
        return 1.0
    raise ValueError(f"Unsupported best_mode={mode}. Use one of: min, max.")


def build_scheduler(optimizer, scheduler_cfg, total_steps: int):
    if not scheduler_cfg or not bool(scheduler_cfg.get("enabled", False)):
        return None

    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower().strip()
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported scheduler.type={scheduler_type}. Use: cosine")

    step_on = str(scheduler_cfg.get("step_on", "step")).lower().strip()
    if step_on != "step":
        raise ValueError(f"Unsupported scheduler.step_on={step_on}. Use: step")

    total_steps = max(1, int(total_steps))
    if "warmup_steps" in scheduler_cfg:
        warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    else:
        warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.0))
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    cosine_steps = max(1, total_steps - warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr,
    )

    if warmup_steps == 0:
        return cosine_scheduler

    start_factor = 1.0 / float(warmup_steps) if warmup_steps > 1 else 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


def get_current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]

    set_seed(train_cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_first_train_prob = float(dataset_cfg.get("global_first_train_prob", 0.0))
    reference_only_train = (
        bool(dataset_cfg.get("packed_window_train", False))
        and int(dataset_cfg.get("num_first_frames", 0)) == 0
        and int(dataset_cfg.get("num_memory_frames", 0)) == 0
    )
    model_name = str(model_cfg.get("name", "tapformer")).lower().strip()
    is_point_warp = model_name in {"tapformer_point_warp", "point_warp"}
    reference_model_names = {"tapformer_cow_dense", "cow_dense", "tapformer_point_warp", "point_warp"}
    if global_first_train_prob > 0.0 and model_name not in reference_model_names:
        raise ValueError(
            "dataset.global_first_train_prob > 0 only supports CowDense or PointWarp models."
        )

    batch_size = int(train_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("TAPFormer forward currently asserts batch_size == 1. Please set train.batch_size: 1")

    save_root = Path(train_cfg.get("save_dir", "output/train_kubric"))
    save_root.mkdir(parents=True, exist_ok=True)
    run_dir = save_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    best_ckpt_path = run_dir / str(train_cfg.get("best_filename", "best.pth"))
    print(f"run_dir={run_dir}", flush=True)
    print(f"tensorboard_dir={tensorboard_dir}", flush=True)
    print(f"best_ckpt_path={best_ckpt_path}", flush=True)
    with open(run_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    dataset = KubricMovifDataset_etap(
        root_dir=dataset_cfg["data_root"],
        representation=dataset_cfg.get("representation", "time_surfaces_v2_5"),
        crop_size=(int(dataset_cfg.get("height", 384)), int(dataset_cfg.get("width", 512))),
        seq_len=int(dataset_cfg.get("seq_len", 24)),
        traj_per_sample=int(dataset_cfg.get("traj_per_sample", 128)),
        sample_vis_1st_frame=bool(dataset_cfg.get("sample_vis_1st_frame", False)),
        choose_long_point=bool(dataset_cfg.get("choose_long_point", False)),
        use_augs=bool(dataset_cfg.get("use_augs", False)),
        if_test=bool(dataset_cfg.get("if_test", False)),
        resample_max_tries=int(dataset_cfg.get("resample_max_tries", 8)),
        global_first_train_prob=global_first_train_prob,
        packed_window_train=bool(dataset_cfg.get("packed_window_train", False)),
        num_first_frames=int(dataset_cfg.get("num_first_frames", 0)),
        num_memory_frames=int(dataset_cfg.get("num_memory_frames", 0)),
        num_current_frames=int(dataset_cfg.get("num_current_frames", 0)),
        window_gap_bin_edges=dataset_cfg.get("window_gap_bin_edges"),
        window_gap_probs_start=dataset_cfg.get("window_gap_probs_start"),
        window_gap_probs_end=dataset_cfg.get("window_gap_probs_end"),
        window_gap_curriculum_epochs=dataset_cfg.get("window_gap_curriculum_epochs"),
        paired_frame_event_prob=float(dataset_cfg.get("paired_frame_event_prob", 0.0)),
        paired_temporal_strides=dataset_cfg.get("paired_temporal_strides", [1]),
        paired_temporal_stride_probs=dataset_cfg.get("paired_temporal_stride_probs", [1.0]),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train_cfg.get("shuffle", True)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
        collate_fn=train_collate,
        drop_last=bool(train_cfg.get("drop_last", True)),
    )

    model = build_model_from_config(model_cfg).to(device)
    print(
        f"model.name={model_cfg.get('name', 'tapformer')} "
        f"feature_mode={model_cfg.get('feature_mode', 'fusion')}",
        flush=True,
    )

    criterion = TAPFormerLoss(
        coord_weight=float(loss_cfg.get("coord_weight", 0.1)),
        invisible_coord_weight=float(loss_cfg.get("invisible_coord_weight", 0.0)),
        visibility_weight=float(loss_cfg.get("visibility_weight", 1.0)),
        confidence_weight=float(loss_cfg.get("confidence_weight", 1.0)),
        iter_gamma=float(loss_cfg.get("iter_gamma", 0.8)),
        confidence_threshold=float(loss_cfg.get("confidence_threshold", 8.0)),
        add_huber_loss=bool(loss_cfg.get("add_huber_loss", False)),
        huber_delta=float(loss_cfg.get("huber_delta", 6.0)),
        loss_only_for_visible=bool(loss_cfg.get("loss_only_for_visible", False)),
    ).to(device)

    base_lr = float(train_cfg.get("lr", 5e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 40))
    scheduler_cfg = train_cfg.get("scheduler", {})
    total_steps = epochs * len(loader)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_cfg=scheduler_cfg,
        total_steps=total_steps,
    )

    precision = train_cfg.get("precision", "bf16")
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    pretrained_path = train_cfg.get("pretrained", None)
    pretrained_strict = bool(train_cfg.get("pretrained_strict", False))
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location=device)
        model_state = normalize_state_dict_keys(extract_state_dict(ckpt))
        if model_name in {"tapformer_point_warp", "point_warp"} and not pretrained_strict:
            incompatible = load_point_warp_pretrained(model, model_state)
        else:
            incompatible = model.load_state_dict(model_state, strict=pretrained_strict)
        print(
            f"Loaded pretrained weights from {pretrained_path} "
            f"(strict={pretrained_strict}, missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)})",
            flush=True,
        )

    resume_path = train_cfg.get("resume", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model_state = normalize_state_dict_keys(extract_state_dict(ckpt))
        model.load_state_dict(model_state, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(
            f"Resumed training from {resume_path} at epoch {start_epoch} "
            f"(global_step={global_step})",
            flush=True,
        )

    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    model.train()
    iters = int(train_cfg.get("iters", 4))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    log_every = int(train_cfg.get("log_every", 10))
    save_every_epochs = int(train_cfg.get("save_every_epochs", 1))
    save_last = bool(train_cfg.get("save_last", True))
    save_best = bool(train_cfg.get("save_best", True))
    best_metric_name = str(train_cfg.get("best_metric", "loss"))
    best_mode = str(train_cfg.get("best_mode", "min")).lower().strip()
    best_filename = best_ckpt_path.name
    best_sign = metric_sign(best_mode)
    best_metric = math.inf if best_mode == "min" else -math.inf
    best_epoch = -1

    if resume_path:
        best_metric = float(ckpt.get("best_metric", best_metric))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        if "best_metric_name" in ckpt:
            best_metric_name = str(ckpt["best_metric_name"])
        if "best_mode" in ckpt:
            best_mode = str(ckpt["best_mode"]).lower().strip()
            best_sign = metric_sign(best_mode)

    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        running = {"loss": 0.0, "coord_loss": 0.0, "invisible_coord_loss": 0.0, "visibility_loss": 0.0, "confidence_loss": 0.0}
        n_steps = 0

        for sample, gotit in loader:
            if not gotit.any():
                continue

            sample = move_batch_to_device(sample, device)
            queries = sample.query_points if sample.query_points is not None else build_queries(sample.trajectory, sample.visibility)
            valid = sample.valid if sample.valid is not None else torch.ones_like(sample.visibility, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            use_amp = (autocast_dtype is not None and device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=autocast_dtype if autocast_dtype is not None else torch.float32, enabled=use_amp):
                model_kwargs = dict(
                    iters=iters,
                    img_ifnew=sample.img_ifnew[0],
                    is_train=True,
                )
                if model_name in reference_model_names:
                    model_kwargs.update(
                        reference_rgbs=sample.reference_video,
                        reference_events=sample.reference_events,
                        reference_only_train=reference_only_train,
                    )
                _, _, _, train_data = model(
                    sample.video,
                    sample.events,
                    queries,
                    **model_kwargs,
                )
                loss_dict = criterion(
                    train_data=train_data,
                    gt_trajectory=sample.trajectory,
                    gt_visibility=sample.visibility,
                    gt_valid=valid,
                )
                loss = loss_dict["loss"]

            if precision == "fp16" and device.type == "cuda":
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            n_steps += 1
            global_step += 1
            for k in running:
                running[k] += float(loss_dict[k].detach().item())
            writer.add_scalar("train/loss", float(loss_dict["loss"].detach().item()), global_step)
            writer.add_scalar("train/coord_loss", float(loss_dict["coord_loss"].detach().item()), global_step)
            writer.add_scalar(
                "train/invisible_coord_loss",
                float(loss_dict["invisible_coord_loss"].detach().item()),
                global_step,
            )
            writer.add_scalar("train/visibility_loss", float(loss_dict["visibility_loss"].detach().item()), global_step)
            writer.add_scalar("train/confidence_loss", float(loss_dict["confidence_loss"].detach().item()), global_step)
            writer.add_scalar("train/lr", get_current_lr(optimizer), global_step)

            if global_step % log_every == 0:
                msg = (
                    f"epoch={epoch} step={global_step} "
                    f"lr={get_current_lr(optimizer):.6e} "
                    f"loss={running['loss']/max(1,n_steps):.4f} "
                    f"coord={running['coord_loss']/max(1,n_steps):.4f} "
                    f"inv_coord={running['invisible_coord_loss']/max(1,n_steps):.4f} "
                    f"vis={running['visibility_loss']/max(1,n_steps):.4f} "
                    f"conf={running['confidence_loss']/max(1,n_steps):.4f}"
                )
                print(msg, flush=True)

        epoch_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_metric_name": best_metric_name,
            "best_mode": best_mode,
        }
        if save_every_epochs > 0 and ((epoch + 1) % save_every_epochs == 0):
            torch.save(epoch_ckpt, run_dir / f"epoch_{epoch:03d}.pth")

        writer.add_scalar("epoch/valid_steps", n_steps, epoch)

        if n_steps > 0:
            epoch_metrics = {k: (running[k] / n_steps) for k in running}
            writer.add_scalar("epoch/loss", epoch_metrics["loss"], epoch)
            writer.add_scalar("epoch/coord_loss", epoch_metrics["coord_loss"], epoch)
            writer.add_scalar("epoch/invisible_coord_loss", epoch_metrics["invisible_coord_loss"], epoch)
            writer.add_scalar("epoch/visibility_loss", epoch_metrics["visibility_loss"], epoch)
            writer.add_scalar("epoch/confidence_loss", epoch_metrics["confidence_loss"], epoch)
            if best_metric_name not in epoch_metrics:
                raise ValueError(
                    f"best_metric={best_metric_name} is not available. "
                    f"Choose one of: {', '.join(epoch_metrics.keys())}"
                )
            current_metric = float(epoch_metrics[best_metric_name])
            is_better = (current_metric - best_metric) * best_sign > 0
            if save_best and is_better:
                best_metric = current_metric
                best_epoch = epoch
                best_ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "config": cfg,
                    "global_step": global_step,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "best_metric_name": best_metric_name,
                    "best_mode": best_mode,
                }
                torch.save(best_ckpt, run_dir / best_filename)
                print(
                    f"[Best] epoch={epoch} {best_metric_name}={best_metric:.4f} "
                    f"saved to {run_dir / best_filename}",
                    flush=True,
                )
            print(
                f"[Epoch {epoch}] loss={running['loss']/n_steps:.4f}, "
                f"coord={running['coord_loss']/n_steps:.4f}, "
                f"inv_coord={running['invisible_coord_loss']/n_steps:.4f}, "
                f"vis={running['visibility_loss']/n_steps:.4f}, "
                f"conf={running['confidence_loss']/n_steps:.4f}",
                flush=True,
            )

        if save_last:
            last_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "config": cfg,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_metric_name": best_metric_name,
                "best_mode": best_mode,
            }
            torch.save(last_ckpt, run_dir / "last.pth")

    writer.close()


if __name__ == "__main__":
    main()
