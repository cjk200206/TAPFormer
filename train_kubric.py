import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from LFE_TAP.datasets.kubric_movif_dataset import KubricMovifDataset_new
from LFE_TAP.models.tapformer import TAPFormer
from LFE_TAP.models.losses import TAPFormerLoss
from LFE_TAP.utils.dataset_utils import FrameEventData


def parse_args():
    parser = argparse.ArgumentParser(description="Train TAPFormer on Kubric-format data")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Kubric root_dir expected by KubricMovifDataset_new")
    parser.add_argument("--data_root_fast", type=str, default=None, help="Optional root_dir_fast_dataset")
    parser.add_argument("--save_dir", type=str, default="output/train_kubric")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1, help="TAPFormer currently supports batch_size=1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--iters", type=int, default=4, help="Refinement iterations in model forward")
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--representation", type=str, default="time_surfaces_v2_5")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--traj_per_sample", type=int, default=128)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--corr_levels", type=int, default=3)
    parser.add_argument("--corr_radius", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--space_depth", type=int, default=3)
    parser.add_argument("--time_depth", type=int, default=3)

    parser.add_argument("--coord_weight", type=float, default=0.1)
    parser.add_argument("--visibility_weight", type=float, default=1.0)
    parser.add_argument("--confidence_weight", type=float, default=1.0)
    parser.add_argument("--iter_gamma", type=float, default=0.8)
    parser.add_argument("--confidence_threshold", type=float, default=8.0)

    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
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
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.batch_size != 1:
        raise ValueError("TAPFormer forward currently asserts batch_size == 1. Please set --batch_size 1.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    dataset = KubricMovifDataset_new(
        root_dir=args.data_root,
        root_dir_fast_dataset=args.data_root_fast,
        representation=args.representation,
        crop_size=(args.height, args.width),
        seq_len=args.seq_len,
        traj_per_sample=args.traj_per_sample,
        sample_vis_1st_frame=False,
        choose_long_point=False,
        use_augs=False,
        if_test=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=train_collate,
        drop_last=True,
    )

    model = TAPFormer(
        window_size=args.window_size,
        stride=args.stride,
        corr_radius=args.corr_radius,
        corr_levels=args.corr_levels,
        hidden_size=args.hidden_size,
        space_depth=args.space_depth,
        time_depth=args.time_depth,
    ).to(device)

    criterion = TAPFormerLoss(
        coord_weight=args.coord_weight,
        visibility_weight=args.visibility_weight,
        confidence_weight=args.confidence_weight,
        iter_gamma=args.iter_gamma,
        confidence_threshold=args.confidence_threshold,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    if args.precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    model.train()
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
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
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                _, _, _, train_data = model(
                    sample.video,
                    sample.events,
                    queries,
                    iters=args.iters,
                    img_ifnew=sample.img_ifnew,
                    is_train=True,
                )
                loss_dict = criterion(
                    train_data=train_data,
                    gt_trajectory=sample.trajectory,
                    gt_visibility=sample.visibility,
                    gt_valid=valid,
                )
                loss = loss_dict["loss"]

            if args.precision == "fp16" and device.type == "cuda":
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

            if global_step % args.log_every == 0:
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
            "args": vars(args),
        }
        torch.save(epoch_ckpt, save_dir / f"epoch_{epoch:03d}.pth")

        if n_steps > 0:
            print(
                f"[Epoch {epoch}] loss={running['loss']/n_steps:.4f}, "
                f"coord={running['coord_loss']/n_steps:.4f}, "
                f"vis={running['visibility_loss']/n_steps:.4f}, "
                f"conf={running['confidence_loss']/n_steps:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
