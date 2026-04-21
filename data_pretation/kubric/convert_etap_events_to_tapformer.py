#!/usr/bin/env python3
"""
Convert ETAP-generated raw event packets (.npz) into TAPFormer-compatible events.h5.

Expected ETAP layout (per sequence):
    <sequence_dir>/events/*.npz

Each .npz should contain event arrays with keys: x, y, t, p.

Output (per sequence):
    <sequence_dir>/events.h5
      - dataset: "events"
      - shape: (N, 4), columns: [t_us, x, y, p]
      - p is normalized to {0, 1}

Optional:
    Trigger TAPFormer representation generation after conversion.
"""

import argparse
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
from tqdm import tqdm


def _extract_sequence_index(seq_name: str) -> Optional[int]:
    """Try to parse an integer sequence index from sequence folder name."""
    m = re.search(r"\d+", seq_name)
    if m is None:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _discover_sequence_dirs(
    dataset_root: Path,
    events_subdir: str,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> List[Path]:
    """
    Discover sequence folders that contain events_subdir/*.npz.

    Index range semantics: [start_index, end_index), where end_index is exclusive.
    If sequence folder name contains digits (e.g., 00000123), that numeric id is used.
    Otherwise, fallback to its positional index in sorted order.
    """
    seq_candidates: List[Path] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        event_dir = child / events_subdir
        if event_dir.is_dir() and len(glob.glob(str(event_dir / "*.npz"))) > 0:
            seq_candidates.append(child)

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


def _normalize_polarity_to_zero_one(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float64, copy=False)
    p_min, p_max = float(np.min(p)), float(np.max(p))
    if p_min >= 0.0 and p_max <= 1.0:
        return p.astype(np.uint8)
    if p_min >= -1.0 and p_max <= 1.0:
        return (p > 0).astype(np.uint8)
    # Fallback for unexpected polarity range: positive -> 1, else 0
    return (p > 0).astype(np.uint8)


def _convert_time_to_us(t: np.ndarray, time_unit: str) -> np.ndarray:
    t = t.astype(np.float64, copy=False)
    if time_unit == "ns":
        return t / 1e3
    if time_unit == "us":
        return t
    if time_unit == "ms":
        return t * 1e3
    if time_unit == "s":
        return t * 1e6

    # auto mode heuristic
    t_abs_max = float(np.max(np.abs(t))) if t.size > 0 else 0.0
    # Typical ETAP ESIM runs are in ns (~1e9 for 1 second). us are usually <= 1e7.
    if t_abs_max > 1e8:
        return t / 1e3  # treat as ns
    return t  # treat as us


def _load_npz_event_packet(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    required_keys = ("x", "y", "t", "p")
    for k in required_keys:
        if k not in data:
            raise KeyError(f"{npz_path} missing required key '{k}'")
    return data["x"], data["y"], data["t"], data["p"]


def _merge_event_packets(npz_paths: Iterable[Path], time_unit: str) -> np.ndarray:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for npz_path in npz_paths:
        x, y, t, p = _load_npz_event_packet(npz_path)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        ps.append(p)

    if len(ts) == 0:
        return np.zeros((0, 4), dtype=np.int64)

    x_all = np.concatenate(xs, axis=0).astype(np.int64, copy=False)
    y_all = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    t_all = _convert_time_to_us(np.concatenate(ts, axis=0), time_unit=time_unit)
    p_all = _normalize_polarity_to_zero_one(np.concatenate(ps, axis=0))

    # Ensure temporal order
    order = np.argsort(t_all)
    t_all = t_all[order]
    x_all = x_all[order]
    y_all = y_all[order]
    p_all = p_all[order]

    # TAPFormer data_pretation expects integer-like values for indexing.
    t_us_i64 = np.rint(t_all).astype(np.int64, copy=False)
    events = np.stack([t_us_i64, x_all, y_all, p_all.astype(np.int64)], axis=1)
    return events


def _write_events_h5(events: np.ndarray, output_h5: Path, overwrite: bool) -> None:
    if output_h5.exists() and not overwrite:
        return
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w", locking=False) as h5f:
        h5f.create_dataset("events", data=events, dtype=np.int64)


def _convert_one_sequence(
    seq_dir: Path,
    events_subdir: str,
    output_name: str,
    time_unit: str,
    overwrite: bool,
) -> Tuple[str, bool, str]:
    """Convert one sequence; returns (seq_name, ok, error_msg)."""
    try:
        event_dir = seq_dir / events_subdir
        npz_paths = sorted(Path(p) for p in glob.glob(str(event_dir / "*.npz")))
        if len(npz_paths) == 0:
            return seq_dir.name, False, "no npz files"

        events = _merge_event_packets(npz_paths, time_unit=time_unit)
        out_h5 = seq_dir / output_name
        _write_events_h5(events, out_h5, overwrite=overwrite)
        return seq_dir.name, True, ""
    except Exception as exc:
        return seq_dir.name, False, str(exc)


def convert_dataset(
    dataset_root: Path,
    events_subdir: str,
    output_name: str,
    time_unit: str,
    overwrite: bool,
    start_index: Optional[int],
    end_index: Optional[int],
    num_workers: int,
) -> int:
    seq_dirs = _discover_sequence_dirs(
        dataset_root,
        events_subdir,
        start_index=start_index,
        end_index=end_index,
    )
    if len(seq_dirs) == 0:
        print(
            f"[WARN] No sequence with '{events_subdir}/*.npz' found under {dataset_root} "
            f"for range [{start_index}, {end_index})."
        )
        return 0

    print(
        f"[INFO] Converting {len(seq_dirs)} sequence(s) in range [{start_index}, {end_index}) "
        f"with num_workers={num_workers}."
    )

    converted = 0
    if num_workers <= 1:
        for seq_dir in tqdm(seq_dirs, desc="Converting sequences"):
            _, ok, err = _convert_one_sequence(
                seq_dir=seq_dir,
                events_subdir=events_subdir,
                output_name=output_name,
                time_unit=time_unit,
                overwrite=overwrite,
            )
            if ok:
                converted += 1
            else:
                print(f"[ERROR] {seq_dir.name}: {err}")
        return converted

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _convert_one_sequence,
                seq_dir,
                events_subdir,
                output_name,
                time_unit,
                overwrite,
            )
            for seq_dir in seq_dirs
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting sequences"):
            seq_name, ok, err = fut.result()
            if ok:
                converted += 1
            else:
                print(f"[ERROR] {seq_name}: {err}")

    return converted


def _maybe_generate_representations(
    do_generate: bool,
    dataset_root: Path,
    output_root: Path,
    representation_type: str,
    n_bins: int,
    dt: float,
) -> None:
    if not do_generate:
        return
    from data_pretation.kubric.generate_event_representations import generate

    print(
        f"[INFO] Generating TAPFormer representations: "
        f"type={representation_type}, n_bins={n_bins}, dt={dt}"
    )
    generate(
        input_dir=str(dataset_root),
        output_dir=str(output_root),
        representation_type=representation_type,
        n_bins=n_bins,
        dts=(dt,),
        visualize=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ETAP raw event packets (.npz) to TAPFormer events.h5 format."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing sequence folders (e.g., 00000000, 00000001, ...).",
    )
    parser.add_argument(
        "--events-subdir",
        type=str,
        default="events",
        help="Relative subdirectory inside each sequence that stores ETAP npz packets.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index (inclusive) for selecting sequence ids. If omitted, no lower bound.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index (exclusive) for selecting sequence ids. If omitted, no upper bound.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for conversion. Use 1 for single-process mode.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="events.h5",
        help="Output h5 filename written under each sequence directory.",
    )
    parser.add_argument(
        "--time-unit",
        type=str,
        default="auto",
        choices=("auto", "ns", "us", "ms", "s"),
        help="Input time unit of ETAP npz packet timestamps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing events.h5 if present.",
    )
    parser.add_argument(
        "--generate-representations",
        action="store_true",
        help="After conversion, run TAPFormer representation generation.",
    )
    parser.add_argument(
        "--representation-type",
        type=str,
        default="time_surface",
        choices=(
            "time_surface",
            "time_surface_accumulate",
            "voxel_grid",
            "event_stack",
            "event_count",
            "time_order_surface",
        ),
        help="Representation type for TAPFormer generation (if enabled).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of bins/channels base parameter for representation generation.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Delta time (seconds) for representation generation.",
    )
    parser.add_argument(
        "--representation-output-root",
        type=str,
        default=None,
        help="Root folder to write generated representations. Default: dataset-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = (
        Path(args.representation_output_root).expanduser().resolve()
        if args.representation_output_root is not None
        else dataset_root
    )

    converted = convert_dataset(
        dataset_root=dataset_root,
        events_subdir=args.events_subdir,
        output_name=args.output_name,
        time_unit=args.time_unit,
        overwrite=args.overwrite,
        start_index=args.start_index,
        end_index=args.end_index,
        num_workers=max(1, int(args.num_workers)),
    )
    print(f"[INFO] Converted {converted} sequence(s) to TAPFormer events.h5 format.")

    _maybe_generate_representations(
        do_generate=args.generate_representations,
        dataset_root=dataset_root,
        output_root=output_root,
        representation_type=args.representation_type,
        n_bins=args.n_bins,
        dt=args.dt,
    )


if __name__ == "__main__":
    main()
