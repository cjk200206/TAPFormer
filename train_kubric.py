import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from LFE_TAP.datasets.kubric_movif_dataset import KubricMovifDataset_etap
from LFE_TAP.models.tapformer import TAPFormer
from LFE_TAP.models.losses import TAPFormerLoss
from LFE_TAP.utils.dataset_utils import FrameEventData


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TAPFormer on Kubric/ETAP-format data")
    parser.add_argument("--config", type=str, default="config/config_kubric_train.yaml", help="Path to training YAML config")
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
    return sample


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]

    set_seed(train_cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(train_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("TAPFormer forward currently asserts batch_size == 1. Please set train.batch_size: 1")

    save_dir = Path(train_cfg.get("save_dir", "output/train_kubric"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_args.json", "w", encoding="utf-8") as f:
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

    model = TAPFormer(
        window_size=int(model_cfg.get("window_size", 8)),
        stride=int(model_cfg.get("stride", 4)),
        corr_radius=int(model_cfg.get("corr_radius", 3)),
        corr_levels=int(model_cfg.get("corr_levels", 3)),
        hidden_size=int(model_cfg.get("hidden_size", 384)),
        space_depth=int(model_cfg.get("space_depth", 3)),
        time_depth=int(model_cfg.get("time_depth", 3)),
    ).to(device)

    criterion = TAPFormerLoss(
        coord_weight=float(loss_cfg.get("coord_weight", 0.1)),
        visibility_weight=float(loss_cfg.get("visibility_weight", 1.0)),
        confidence_weight=float(loss_cfg.get("confidence_weight", 1.0)),
        iter_gamma=float(loss_cfg.get("iter_gamma", 0.8)),
        confidence_threshold=float(loss_cfg.get("confidence_threshold", 8.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    precision = train_cfg.get("precision", "bf16")
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))

    start_epoch = 0
    resume_path = train_cfg.get("resume", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    model.train()
    global_step = 0
    epochs = int(train_cfg.get("epochs", 40))
    iters = int(train_cfg.get("iters", 4))
    log_every = int(train_cfg.get("log_every", 10))
    save_every_epochs = int(train_cfg.get("save_every_epochs", 1))
    save_last = bool(train_cfg.get("save_last", True))

    for epoch in range(start_epoch, epochs):
        running = {"loss": 0.0, "coord_loss": 0.0, "visibility_loss": 0.0, "confidence_loss": 0.0}
        n_steps = 0

        for sample, gotit in loader:
            if not gotit.any():
                continue

            sample = move_batch_to_device(sample, device)
            queries = build_queries(sample.trajectory, sample.visibility)
            valid = sample.valid if sample.valid is not None else torch.ones_like(sample.visibility, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            use_amp = (autocast_dtype is not None and device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=autocast_dtype if autocast_dtype is not None else torch.float32, enabled=use_amp):
                _, _, _, train_data = model(
                    sample.video,
                    sample.events,
                    queries,
                    iters=iters,
                    img_ifnew=sample.img_ifnew[0],
                    is_train=True,
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
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            n_steps += 1
            global_step += 1
            for k in running:
                running[k] += float(loss_dict[k].detach().item())

            if global_step % log_every == 0:
                msg = (
                    f"epoch={epoch} step={global_step} "
                    f"loss={running['loss']/max(1,n_steps):.4f} "
                    f"coord={running['coord_loss']/max(1,n_steps):.4f} "
                    f"vis={running['visibility_loss']/max(1,n_steps):.4f} "
                    f"conf={running['confidence_loss']/max(1,n_steps):.4f}"
                )
                print(msg, flush=True)

        epoch_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "global_step": global_step,
        }
        if save_every_epochs > 0 and ((epoch + 1) % save_every_epochs == 0):
            torch.save(epoch_ckpt, save_dir / f"epoch_{epoch:03d}.pth")

        if n_steps > 0:
            print(
                f"[Epoch {epoch}] loss={running['loss']/n_steps:.4f}, "
                f"coord={running['coord_loss']/n_steps:.4f}, "
                f"vis={running['visibility_loss']/n_steps:.4f}, "
                f"conf={running['confidence_loss']/n_steps:.4f}",
                flush=True,
            )

    if save_last:
        final_epoch = max(start_epoch, epochs) - 1
        final_ckpt = {
            "epoch": final_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "global_step": global_step,
        }
        torch.save(final_ckpt, save_dir / "final.pth")


if __name__ == "__main__":
    main()
