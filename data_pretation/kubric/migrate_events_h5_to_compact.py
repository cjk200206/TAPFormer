#!/usr/bin/env python3
"""Migrate legacy Kubric events.h5 files to the compact split-format layout."""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(PROJECT_ROOT))

from LFE_TAP.utils.event.utils import (
    COMPACT_EVENTS_H5_FORMAT,
    COMPACT_EVENTS_H5_TIME_UNIT,
    blosc_opts,
    is_compact_events_h5,
)

CHUNK_ROWS = 1_000_000


def _extract_sequence_index(seq_name: str) -> Optional[int]:
    m = re.search(r"\d+", seq_name)
    if m is None:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _discover_sequence_dirs(dataset_root: Path, start_index: Optional[int], end_index: Optional[int]) -> List[Path]:
    seq_candidates = [child for child in sorted(dataset_root.iterdir()) if child.is_dir()]
    if start_index is None and end_index is None:
        return seq_candidates

    seq_dirs: List[Path] = []
    for pos_idx, seq_dir in enumerate(seq_candidates):
        seq_idx = _extract_sequence_index(seq_dir.name)
        idx = seq_idx if seq_idx is not None else pos_idx
        if start_index is not None and idx < start_index:
            continue
        if end_index is not None and idx >= end_index:
            continue
        seq_dirs.append(seq_dir)
    return seq_dirs


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def _estimate_compact_size(n_rows: int) -> int:
    return n_rows * (4 + 2 + 2 + 1)


def _validate_legacy_chunk(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if chunk.ndim != 2 or chunk.shape[1] != 4:
        raise ValueError(f"legacy events chunk must have shape (N,4), got {chunk.shape}")

    t = np.asarray(chunk[:, 0], dtype=np.int64)
    x = np.asarray(chunk[:, 1], dtype=np.int64)
    y = np.asarray(chunk[:, 2], dtype=np.int64)
    p = np.asarray(chunk[:, 3], dtype=np.int64)

    if t.size > 0:
        if np.min(t) < 0 or np.max(t) > np.iinfo(np.uint32).max:
            raise ValueError("event timestamp range is outside uint32")
        if np.min(x) < 0 or np.max(x) > np.iinfo(np.uint16).max:
            raise ValueError("event x range is outside uint16")
        if np.min(y) < 0 or np.max(y) > np.iinfo(np.uint16).max:
            raise ValueError("event y range is outside uint16")
        if not np.all((p == 0) | (p == 1)):
            raise ValueError("event polarity must be 0 or 1")

    return (
        t.astype(np.uint32, copy=False),
        x.astype(np.uint16, copy=False),
        y.astype(np.uint16, copy=False),
        p.astype(np.uint8, copy=False),
    )


def _create_compact_datasets(h5f: h5py.File, n_rows: int):
    h5f.attrs["format"] = COMPACT_EVENTS_H5_FORMAT
    h5f.attrs["time_unit"] = COMPACT_EVENTS_H5_TIME_UNIT
    chunk_rows = min(CHUNK_ROWS, max(1, n_rows))
    return {
        "t": h5f.create_dataset("t", shape=(n_rows,), dtype=np.uint32, chunks=(chunk_rows,), **blosc_opts(complevel=1, shuffle="byte")),
        "x": h5f.create_dataset("x", shape=(n_rows,), dtype=np.uint16, chunks=(chunk_rows,), **blosc_opts(complevel=1, shuffle="byte")),
        "y": h5f.create_dataset("y", shape=(n_rows,), dtype=np.uint16, chunks=(chunk_rows,), **blosc_opts(complevel=1, shuffle="byte")),
        "p": h5f.create_dataset("p", shape=(n_rows,), dtype=np.uint8, chunks=(chunk_rows,), **blosc_opts(complevel=1, shuffle="byte")),
    }


def _convert_legacy_events_file(input_path: Path, temp_path: Path) -> Tuple[int, int]:
    with h5py.File(str(input_path), "r") as src_h5:
        if "events" not in src_h5:
            raise KeyError("legacy file is missing dataset 'events'")
        events_ds = src_h5["events"]
        if events_ds.ndim != 2 or events_ds.shape[1] != 4:
            raise ValueError(f"legacy events dataset must have shape (N,4), got {events_ds.shape}")
        n_rows = int(events_ds.shape[0])

        with h5py.File(str(temp_path), "w") as dst_h5:
            dst_ds = _create_compact_datasets(dst_h5, n_rows)
            for start in range(0, n_rows, CHUNK_ROWS):
                end = min(start + CHUNK_ROWS, n_rows)
                chunk = events_ds[start:end]
                t_u32, x_u16, y_u16, p_u8 = _validate_legacy_chunk(chunk)
                dst_ds["t"][start:end] = t_u32
                dst_ds["x"][start:end] = x_u16
                dst_ds["y"][start:end] = y_u16
                dst_ds["p"][start:end] = p_u8

    return n_rows, temp_path.stat().st_size


def _verify_converted_file(input_path: Path, temp_path: Path) -> None:
    with h5py.File(str(input_path), "r") as src_h5, h5py.File(str(temp_path), "r") as dst_h5:
        if "events" not in src_h5:
            raise KeyError("legacy file is missing dataset 'events'")
        if not is_compact_events_h5(dst_h5):
            raise ValueError("converted file is missing split datasets {t,x,y,p}")
        if dst_h5.attrs.get("format", "") != COMPACT_EVENTS_H5_FORMAT:
            raise ValueError("converted file has unexpected format attr")
        if dst_h5.attrs.get("time_unit", "") != COMPACT_EVENTS_H5_TIME_UNIT:
            raise ValueError("converted file has unexpected time_unit attr")

        src_ds = src_h5["events"]
        n_rows = int(src_ds.shape[0])
        if any(int(dst_h5[name].shape[0]) != n_rows for name in ("t", "x", "y", "p")):
            raise ValueError("converted file length mismatch")

        for start in range(0, n_rows, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, n_rows)
            chunk = src_ds[start:end]
            t_u32, x_u16, y_u16, p_u8 = _validate_legacy_chunk(chunk)
            if not np.array_equal(dst_h5["t"][start:end], t_u32):
                raise ValueError("converted file t mismatch")
            if not np.array_equal(dst_h5["x"][start:end], x_u16):
                raise ValueError("converted file x mismatch")
            if not np.array_equal(dst_h5["y"][start:end], y_u16):
                raise ValueError("converted file y mismatch")
            if not np.array_equal(dst_h5["p"][start:end], p_u8):
                raise ValueError("converted file p mismatch")


def _replace_original(input_path: Path, temp_path: Path, keep_backup: bool) -> None:
    if keep_backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        if backup_path.exists():
            raise FileExistsError(f"backup already exists: {backup_path}")
        input_path.replace(backup_path)
    temp_path.replace(input_path)


def _inspect_file(input_path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {
        "path": input_path,
        "exists": input_path.is_file(),
        "already_compact": False,
        "valid_legacy": False,
        "rows": 0,
        "old_size": input_path.stat().st_size if input_path.is_file() else 0,
        "estimated_new_size": 0,
    }
    if not input_path.is_file():
        return info

    with h5py.File(str(input_path), "r") as h5f:
        if is_compact_events_h5(h5f):
            info["already_compact"] = True
            info["rows"] = int(h5f["t"].shape[0])
            return info
        if "events" not in h5f:
            return info
        ds = h5f["events"]
        if ds.ndim != 2 or ds.shape[1] != 4:
            return info
        info["valid_legacy"] = True
        info["rows"] = int(ds.shape[0])
        info["estimated_new_size"] = _estimate_compact_size(int(ds.shape[0]))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate Kubric legacy events.h5 files to compact split-format events.h5.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory containing sequence folders.")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive) for sequence selection.")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive) for sequence selection.")
    parser.add_argument("--input-name", type=str, default="events.h5", help="Input events filename under each sequence directory.")
    parser.add_argument("--apply", action="store_true", help="Actually migrate files in-place. Without this flag, run in dry-run mode.")
    parser.add_argument("--keep-backup", action="store_true", help="When applying, rename the original file to .bak before replacing it.")
    parser.add_argument("--verbose", action="store_true", help="Print per-sequence details.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    seq_dirs = _discover_sequence_dirs(dataset_root, args.start_index, args.end_index)
    print(
        f"[INFO] Mode={'apply' if args.apply else 'dry-run'} dataset_root={dataset_root} "
        f"range=[{args.start_index}, {args.end_index}) input_name={args.input_name}"
    )
    print(f"[INFO] Selected {len(seq_dirs)} sequence(s).")

    inspected = 0
    eligible = 0
    already_compact = 0
    missing = 0
    invalid = 0
    migrated = 0
    planned_old_size = 0
    planned_estimated_new_size = 0
    actual_new_size = 0

    iterator = tqdm(seq_dirs, desc="Migrating events.h5") if args.apply else seq_dirs
    for seq_dir in iterator:
        inspected += 1
        input_path = seq_dir / args.input_name
        info = _inspect_file(input_path)
        seq_name = seq_dir.name

        if not info["exists"]:
            missing += 1
            if args.verbose:
                print(f"[WARN] {seq_name}: missing {args.input_name}")
            continue
        if info["already_compact"]:
            already_compact += 1
            if args.verbose:
                print(f"[OK] {seq_name}: already compact")
            continue
        if not info["valid_legacy"]:
            invalid += 1
            print(f"[WARN] {seq_name}: unsupported or invalid legacy events.h5 format")
            continue

        eligible += 1
        planned_old_size += int(info["old_size"])
        planned_estimated_new_size += int(info["estimated_new_size"])
        if args.verbose or not args.apply:
            print(
                f"[PLAN] {seq_name}: rows={info['rows']} old={_format_bytes(int(info['old_size']))} "
                f"estimated_new={_format_bytes(int(info['estimated_new_size']))}"
            )
        if not args.apply:
            continue

        temp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()
        try:
            _, new_size = _convert_legacy_events_file(input_path, temp_path)
            _verify_converted_file(input_path, temp_path)
            _replace_original(input_path, temp_path, keep_backup=args.keep_backup)
            migrated += 1
            actual_new_size += new_size
            if args.verbose:
                print(f"[DONE] {seq_name}: new_size={_format_bytes(new_size)}")
        except Exception as exc:
            print(f"[ERROR] {seq_name}: {exc}")
            if temp_path.exists():
                temp_path.unlink()

    print("[SUMMARY]")
    print(f"  inspected={inspected}")
    print(f"  eligible_legacy={eligible}")
    print(f"  already_compact={already_compact}")
    print(f"  missing={missing}")
    print(f"  invalid={invalid}")
    print(f"  migrated={migrated}")
    print(f"  planned_old_size={_format_bytes(planned_old_size)}")
    print(f"  planned_estimated_new_size={_format_bytes(planned_estimated_new_size)}")
    if args.apply:
        print(f"  actual_new_size={_format_bytes(actual_new_size)}")


if __name__ == "__main__":
    main()
