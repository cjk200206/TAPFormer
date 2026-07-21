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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from LFE_TAP.models.losses import TAPFormerLoss
from train_kubric import (
    build_kubric_dataset,
    build_model_from_config,
    build_scheduler,
    compute_loss,
    extract_state_dict,
    get_current_lr,
    get_scheduler_total_steps,
    get_scheduler_type,
    load_config,
    load_point_warp_pretrained,
    log_scheduler_info,
    metric_sign,
    normalize_state_dict_keys,
    set_seed,
    train_collate,
)


METRIC_KEYS = (
    "loss",
    "coord_loss",
    "invisible_coord_loss",
    "visibility_loss",
    "confidence_loss",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TAPFormer on Kubric/ETAP-format data with DDP")
    parser.add_argument("--config", type=str, default="config/config_kubric_train.yaml", help="Path to training YAML config")
    parser.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES value, for example '0,1,2,3'")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA. Please launch with GPUs available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def rank0_print(rank, *args, **kwargs):
    if is_main_process(rank):
        print(*args, **kwargs)


def broadcast_run_dir(run_dir, distributed):
    values = [str(run_dir) if run_dir is not None else None]
    if distributed:
        dist.broadcast_object_list(values, src=0)
    return Path(values[0])


def all_ranks_have_valid_batch(gotit, device, distributed):
    has_valid = bool(gotit.any().item())
    flag = torch.tensor([1 if has_valid else 0], dtype=torch.int32, device=device)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def reduce_metric_values(values, device, distributed, average=True):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= dist.get_world_size()
    return tensor.cpu().tolist()


def reduce_loss_dict(loss_dict, device, distributed):
    values = [float(loss_dict[key].detach().item()) for key in METRIC_KEYS]
    reduced = reduce_metric_values(values, device, distributed, average=True)
    return dict(zip(METRIC_KEYS, reduced))


def build_checkpoint(
    raw_model,
    optimizer,
    scheduler,
    scheduler_type,
    scheduler_total_steps,
    cfg,
    epoch,
    global_step,
    best_metric,
    best_epoch,
    best_metric_name,
    best_mode,
):
    return {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scheduler_type": scheduler_type,
        "scheduler_total_steps": scheduler_total_steps,
        "config": cfg,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_mode": best_mode,
    }


def log_train_scalars(writer, metrics, lr, global_step):
    writer.add_scalar("train/loss", metrics["loss"], global_step)
    writer.add_scalar("train/coord_loss", metrics["coord_loss"], global_step)
    writer.add_scalar("train/invisible_coord_loss", metrics["invisible_coord_loss"], global_step)
    writer.add_scalar("train/visibility_loss", metrics["visibility_loss"], global_step)
    writer.add_scalar("train/confidence_loss", metrics["confidence_loss"], global_step)
    writer.add_scalar("train/lr", lr, global_step)


def validate_on_rank0(
    raw_model,
    criterion,
    val_loader,
    device,
    iters,
    autocast_dtype,
    model_name,
    reference_model_names,
):
    raw_model.eval()
    val_running = {key: 0.0 for key in METRIC_KEYS}
    val_steps = 0
    with torch.no_grad():
        for sample, gotit in val_loader:
            if not gotit.any():
                continue
            loss_dict = compute_loss(
                model=raw_model,
                criterion=criterion,
                sample=sample,
                device=device,
                iters=iters,
                autocast_dtype=autocast_dtype,
                model_name=model_name,
                reference_model_names=reference_model_names,
                reference_only_train=False,
            )
            val_steps += 1
            for key in METRIC_KEYS:
                val_running[key] += float(loss_dict[key].detach().item())
    raw_model.train()

    if val_steps == 0:
        raise ValueError("Validation produced no valid batches. Please check dataset.val_root.")
    return {f"val/{key}": (val_running[key] / val_steps) for key in METRIC_KEYS}, val_steps


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = init_distributed()

    try:
        cfg = load_config(args.config)
        dataset_cfg = cfg["dataset"]
        train_cfg = cfg["train"]
        model_cfg = cfg["model"]
        loss_cfg = cfg["loss"]

        set_seed(int(train_cfg.get("seed", 0)))

        global_first_train_prob = float(dataset_cfg.get("global_first_train_prob", 0.0))
        reference_only_train = (
            bool(dataset_cfg.get("packed_window_train", False))
            and int(dataset_cfg.get("num_first_frames", 0)) == 0
            and int(dataset_cfg.get("num_memory_frames", 0)) == 0
        )
        model_name = str(model_cfg.get("name", "tapformer")).lower().strip()
        reference_model_names = {"tapformer_cow_dense", "cow_dense", "tapformer_point_warp", "point_warp"}
        if global_first_train_prob > 0.0 and model_name not in reference_model_names:
            raise ValueError(
                "dataset.global_first_train_prob > 0 only supports CowDense or PointWarp models."
            )

        batch_size = int(train_cfg.get("batch_size", 1))
        if batch_size != 1:
            raise ValueError("TAPFormer forward currently asserts batch_size == 1. Please set train.batch_size: 1")

        best_metric_name = str(train_cfg.get("best_metric", "loss"))
        val_root = dataset_cfg.get("val_root", None)
        if best_metric_name.startswith("val/") and not val_root:
            raise ValueError("train.best_metric uses val/*, but dataset.val_root is not set.")

        if is_main_process(rank):
            save_root = Path(train_cfg.get("save_dir", "output/train_kubric"))
            save_root.mkdir(parents=True, exist_ok=True)
            run_dir = save_root / datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=False)
        else:
            run_dir = None
        run_dir = broadcast_run_dir(run_dir, distributed)

        tensorboard_dir = run_dir / "tensorboard"
        best_ckpt_path = run_dir / str(train_cfg.get("best_filename", "best.pth"))
        writer = SummaryWriter(log_dir=str(tensorboard_dir)) if is_main_process(rank) else None
        rank0_print(rank, f"run_dir={run_dir}", flush=True)
        rank0_print(rank, f"tensorboard_dir={tensorboard_dir}", flush=True)
        rank0_print(rank, f"best_ckpt_path={best_ckpt_path}", flush=True)
        rank0_print(
            rank,
            f"ddp.enabled={distributed} rank={rank} local_rank={local_rank} world_size={world_size} "
            f"per_gpu_batch_size={batch_size}",
            flush=True,
        )
        if is_main_process(rank):
            with open(run_dir / "train_args.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

        dataset = build_kubric_dataset(dataset_cfg, root_key="data_root")
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(train_cfg.get("shuffle", True)),
            drop_last=bool(train_cfg.get("drop_last", True)),
        ) if distributed else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(train_cfg.get("shuffle", True)) if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
            collate_fn=train_collate,
            drop_last=bool(train_cfg.get("drop_last", True)) if train_sampler is None else False,
        )

        val_loader = None
        if is_main_process(rank) and val_root:
            if not Path(val_root).is_dir():
                raise ValueError(f"dataset.val_root does not exist or is not a directory: {val_root}")
            val_dataset = build_kubric_dataset(
                dataset_cfg,
                root_key="val_root",
                overrides={
                    "use_augs": False,
                    "if_test": False,
                    "global_first_train_prob": 0.0,
                    "packed_window_train": False,
                },
            )
            if len(val_dataset.seq_names) == 0:
                raise ValueError(f"dataset.val_root has no sequence directories: {val_root}")
            val_num_workers = int(train_cfg.get("val_num_workers", train_cfg.get("num_workers", 4)))
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=val_num_workers,
                pin_memory=bool(train_cfg.get("pin_memory", True)),
                persistent_workers=val_num_workers > 0,
                collate_fn=train_collate,
                drop_last=False,
            )

        raw_model = build_model_from_config(model_cfg).to(device)
        rank0_print(
            rank,
            f"model.name={model_cfg.get('name', 'tapformer')} "
            f"feature_mode={model_cfg.get('feature_mode', 'fusion')} "
            f"frontend_type={model_cfg.get('frontend_type', 'base')}",
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
            raw_model.parameters(),
            lr=base_lr,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        epochs = int(train_cfg.get("epochs", 40))
        scheduler_cfg = train_cfg.get("scheduler", {})
        max_steps_cfg = train_cfg.get("max_steps", None)
        total_steps = int(max_steps_cfg) if max_steps_cfg is not None else epochs * len(loader)
        total_steps = max(1, total_steps)
        scheduler_type = get_scheduler_type(scheduler_cfg)
        scheduler_total_steps = get_scheduler_total_steps(scheduler_cfg, total_steps)
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_cfg=scheduler_cfg,
            total_steps=total_steps,
            max_lr=base_lr,
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
                incompatible = load_point_warp_pretrained(raw_model, model_state)
            else:
                incompatible = raw_model.load_state_dict(model_state, strict=pretrained_strict)
            rank0_print(
                rank,
                f"Loaded pretrained weights from {pretrained_path} "
                f"(strict={pretrained_strict}, missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)})",
                flush=True,
            )

        resume_path = train_cfg.get("resume", None)
        resume_ckpt = None
        if resume_path:
            resume_ckpt = torch.load(resume_path, map_location=device)
            model_state = normalize_state_dict_keys(extract_state_dict(resume_ckpt))
            raw_model.load_state_dict(model_state, strict=False)
            if "optimizer" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            ckpt_scheduler_type = resume_ckpt.get("scheduler_type", None)
            if scheduler is not None:
                if "scheduler" in resume_ckpt and resume_ckpt["scheduler"] is not None:
                    if ckpt_scheduler_type is None:
                        rank0_print(rank, "Warning: checkpoint has scheduler state but no scheduler_type metadata.", flush=True)
                    elif str(ckpt_scheduler_type).lower().strip() != scheduler_type:
                        raise ValueError(
                            "Checkpoint scheduler_type does not match current config: "
                            f"checkpoint={ckpt_scheduler_type}, current={scheduler_type}"
                        )
                    scheduler.load_state_dict(resume_ckpt["scheduler"])
                else:
                    rank0_print(rank, "Warning: checkpoint has no scheduler state; scheduler will start from current config.", flush=True)
            elif "scheduler" in resume_ckpt and resume_ckpt["scheduler"] is not None:
                rank0_print(rank, "Warning: checkpoint has scheduler state but scheduler is disabled in current config.", flush=True)
            start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
            global_step = int(resume_ckpt.get("global_step", 0))
            rank0_print(
                rank,
                f"Resumed training from {resume_path} at epoch {start_epoch} "
                f"(global_step={global_step})",
                flush=True,
            )

        find_unused_parameters = bool(train_cfg.get("ddp_find_unused_parameters", True))
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        ) if distributed else raw_model

        if is_main_process(rank):
            log_scheduler_info(scheduler_cfg, total_steps, base_lr, get_current_lr(optimizer), global_step)
        if global_step >= total_steps:
            rank0_print(
                rank,
                f"Training already completed: global_step={global_step} >= max_steps={total_steps}",
                flush=True,
            )
            if writer is not None:
                writer.close()
            return

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
        best_mode = str(train_cfg.get("best_mode", "min")).lower().strip()
        best_filename = best_ckpt_path.name
        best_sign = metric_sign(best_mode)
        best_metric = math.inf if best_mode == "min" else -math.inf
        best_epoch = -1

        if resume_ckpt is not None:
            best_metric = float(resume_ckpt.get("best_metric", best_metric))
            best_epoch = int(resume_ckpt.get("best_epoch", best_epoch))
            if "best_metric_name" in resume_ckpt:
                best_metric_name = str(resume_ckpt["best_metric_name"])
            if "best_mode" in resume_ckpt:
                best_mode = str(resume_ckpt["best_mode"]).lower().strip()
                best_sign = metric_sign(best_mode)

        if best_metric_name.startswith("val/") and is_main_process(rank) and val_loader is None:
            raise ValueError("best_metric uses val/*, but no validation loader is available.")

        should_stop = False
        for epoch in range(start_epoch, epochs):
            dataset.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running = {key: 0.0 for key in METRIC_KEYS}
            n_steps = 0
            skipped_steps = 0

            for sample, gotit in loader:
                if global_step >= total_steps:
                    should_stop = True
                    break
                if not all_ranks_have_valid_batch(gotit, device, distributed):
                    skipped_steps += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss_dict = compute_loss(
                    model=model,
                    criterion=criterion,
                    sample=sample,
                    device=device,
                    iters=iters,
                    autocast_dtype=autocast_dtype,
                    model_name=model_name,
                    reference_model_names=reference_model_names,
                    reference_only_train=reference_only_train,
                )
                loss = loss_dict["loss"]

                optimizer_updated = True
                if precision == "fp16" and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip_norm)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_updated = scaler.get_scale() >= old_scale
                else:
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()
                if not optimizer_updated:
                    continue
                if scheduler is not None:
                    scheduler.step()

                step_metrics = reduce_loss_dict(loss_dict, device, distributed)
                n_steps += 1
                global_step += 1
                for key in METRIC_KEYS:
                    running[key] += step_metrics[key]

                if is_main_process(rank):
                    log_train_scalars(writer, step_metrics, get_current_lr(optimizer), global_step)

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
                if global_step >= total_steps:
                    should_stop = True
                    break

            val_metrics = {}
            if distributed:
                dist.barrier()
            if is_main_process(rank) and val_loader is not None:
                val_metrics, val_steps = validate_on_rank0(
                    raw_model=raw_model,
                    criterion=criterion,
                    val_loader=val_loader,
                    device=device,
                    iters=iters,
                    autocast_dtype=autocast_dtype,
                    model_name=model_name,
                    reference_model_names=reference_model_names,
                )
                writer.add_scalar("epoch/val_steps", val_steps, epoch)
                for key, value in val_metrics.items():
                    writer.add_scalar(key, value, epoch)
                print(
                    f"[Val {epoch}] loss={val_metrics['val/loss']:.4f}, "
                    f"coord={val_metrics['val/coord_loss']:.4f}, "
                    f"inv_coord={val_metrics['val/invisible_coord_loss']:.4f}, "
                    f"vis={val_metrics['val/visibility_loss']:.4f}, "
                    f"conf={val_metrics['val/confidence_loss']:.4f}",
                    flush=True,
                )
            if distributed:
                dist.barrier()

            if is_main_process(rank):
                epoch_ckpt = build_checkpoint(
                    raw_model,
                    optimizer,
                    scheduler,
                    scheduler_type,
                    scheduler_total_steps,
                    cfg,
                    epoch,
                    global_step,
                    best_metric,
                    best_epoch,
                    best_metric_name,
                    best_mode,
                )
                if save_every_epochs > 0 and ((epoch + 1) % save_every_epochs == 0):
                    torch.save(epoch_ckpt, run_dir / f"epoch_{epoch:03d}.pth")

                writer.add_scalar("epoch/valid_steps", n_steps, epoch)
                writer.add_scalar("epoch/skipped_steps", skipped_steps, epoch)

                if n_steps > 0:
                    epoch_metrics = {key: (running[key] / n_steps) for key in METRIC_KEYS}
                    epoch_metrics.update(val_metrics)
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
                        best_ckpt = build_checkpoint(
                            raw_model,
                            optimizer,
                            scheduler,
                            scheduler_type,
                            scheduler_total_steps,
                            cfg,
                            epoch,
                            global_step,
                            best_metric,
                            best_epoch,
                            best_metric_name,
                            best_mode,
                        )
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
                        f"conf={running['confidence_loss']/n_steps:.4f}, "
                        f"skipped={skipped_steps}",
                        flush=True,
                    )

                if save_last:
                    last_ckpt = build_checkpoint(
                        raw_model,
                        optimizer,
                        scheduler,
                        scheduler_type,
                        scheduler_total_steps,
                        cfg,
                        epoch,
                        global_step,
                        best_metric,
                        best_epoch,
                        best_metric_name,
                        best_mode,
                    )
                    torch.save(last_ckpt, run_dir / "last.pth")

                if should_stop:
                    print(f"Reached max_steps={total_steps}; stopping training.", flush=True)

            if distributed:
                dist.barrier()
            if should_stop:
                break

        if is_main_process(rank) and global_step < total_steps:
            print(
                f"Warning: training ended at global_step={global_step}, below max_steps={total_steps}. "
                "Increase train.epochs if this was not intended.",
                flush=True,
            )

        if writer is not None:
            writer.close()
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
