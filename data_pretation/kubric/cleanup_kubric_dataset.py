#!/usr/bin/env python3
"""
Inspect and clean Kubric ETAP-style TAPFormer data down to a minimal keep list.

Minimal keep list per sequence:
    <seq>/
      - annotations.npy
      - events.h5
      - raw/rgba_*.png
      - raw/rgba_blur_*.png
      - events/<representation>/*.h5

Default mode is dry-run. Pass --apply to actually delete redundant files.
"""

import argparse
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class DeleteGroup:
    label: str
    paths: List[Path] = field(default_factory=list)
    file_count: int = 0
    bytes_total: int = 0


@dataclass
class SequencePlan:
    seq_dir: Path
    seq_index: int
    valid: bool
    missing_items: List[str] = field(default_factory=list)
    delete_groups: List[DeleteGroup] = field(default_factory=list)
    delete_file_count: int = 0
    delete_bytes_total: int = 0
    kept_counts: Dict[str, int] = field(default_factory=dict)


def _extract_sequence_index(seq_name: str) -> Optional[int]:
    match = re.search(r"\d+", seq_name)
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _discover_sequence_dirs(
    dataset_root: Path,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> List[Tuple[int, Path]]:
    seq_candidates = [child for child in sorted(dataset_root.iterdir()) if child.is_dir()]
    selected: List[Tuple[int, Path]] = []
    for pos_idx, seq_dir in enumerate(seq_candidates):
        seq_idx = _extract_sequence_index(seq_dir.name)
        idx = seq_idx if seq_idx is not None else pos_idx
        if start_index is not None and idx < start_index:
            continue
        if end_index is not None and idx >= end_index:
            continue
        selected.append((idx, seq_dir))
    return selected


def _count_path_stats(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size

    file_count = 0
    byte_count = 0
    for child in path.rglob("*"):
        if child.is_file():
            file_count += 1
            byte_count += child.stat().st_size
    return file_count, byte_count


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def _make_group_label(parent_name: str, path: Path) -> str:
    if path.is_dir():
        return f"{parent_name}/{path.name}/"
    if parent_name == "raw" and "_" in path.name:
        prefix = re.sub(r"_\d+$", "", path.stem)
        return f"raw/{prefix}_*{path.suffix}"
    if parent_name == "events" and path.suffix:
        return f"events/*{path.suffix}"
    return f"{parent_name}/{path.name}"


def _add_delete_candidate(groups: Dict[str, DeleteGroup], label: str, path: Path) -> None:
    file_count, byte_count = _count_path_stats(path)
    group = groups.setdefault(label, DeleteGroup(label=label))
    group.paths.append(path)
    group.file_count += file_count
    group.bytes_total += byte_count


def _list_matching_files(directory: Path, pattern: str) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def _list_raw_rgba_files(raw_dir: Path) -> Tuple[List[Path], List[Path]]:
    if not raw_dir.is_dir():
        return [], []

    rgba_files: List[Path] = []
    rgba_blur_files: List[Path] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix != ".png":
            continue
        if path.name.startswith("rgba_blur_"):
            rgba_blur_files.append(path)
        elif path.name.startswith("rgba_"):
            rgba_files.append(path)
    return rgba_files, rgba_blur_files


def _build_sequence_plan(seq_dir: Path, seq_index: int, representation: str) -> SequencePlan:
    annotations_path = seq_dir / "annotations.npy"
    events_h5_path = seq_dir / "events.h5"
    raw_dir = seq_dir / "raw"
    events_dir = seq_dir / "events"
    representation_dir = events_dir / representation

    rgba_files, rgba_blur_files = _list_raw_rgba_files(raw_dir)
    representation_files = _list_matching_files(representation_dir, "*.h5")

    kept_counts = {
        "raw/rgba_*.png": len(rgba_files),
        "raw/rgba_blur_*.png": len(rgba_blur_files),
        f"events/{representation}/*.h5": len(representation_files),
    }

    missing_items: List[str] = []
    if not annotations_path.is_file():
        missing_items.append("annotations.npy")
    if not events_h5_path.is_file():
        missing_items.append("events.h5")
    if len(rgba_files) == 0:
        missing_items.append("raw/rgba_*.png")
    if len(rgba_blur_files) == 0:
        missing_items.append("raw/rgba_blur_*.png")
    if len(representation_files) == 0:
        missing_items.append(f"events/{representation}/*.h5")

    if missing_items:
        return SequencePlan(
            seq_dir=seq_dir,
            seq_index=seq_index,
            valid=False,
            missing_items=missing_items,
            kept_counts=kept_counts,
        )

    delete_groups: Dict[str, DeleteGroup] = {}
    keep_root_names = {"annotations.npy", "events.h5", "raw", "events"}

    for child in sorted(seq_dir.iterdir()):
        if child.name in keep_root_names:
            continue
        _add_delete_candidate(delete_groups, child.name + ("/" if child.is_dir() else ""), child)

    keep_raw_names = {path.name for path in rgba_files}
    keep_raw_names.update(path.name for path in rgba_blur_files)
    for child in sorted(raw_dir.iterdir()):
        if child.name in keep_raw_names:
            continue
        label = _make_group_label("raw", child)
        _add_delete_candidate(delete_groups, label, child)

    for child in sorted(events_dir.iterdir()):
        if child.name == representation:
            continue
        label = _make_group_label("events", child)
        _add_delete_candidate(delete_groups, label, child)

    delete_group_list = sorted(delete_groups.values(), key=lambda group: group.label)
    delete_file_count = sum(group.file_count for group in delete_group_list)
    delete_bytes_total = sum(group.bytes_total for group in delete_group_list)
    return SequencePlan(
        seq_dir=seq_dir,
        seq_index=seq_index,
        valid=True,
        delete_groups=delete_group_list,
        delete_file_count=delete_file_count,
        delete_bytes_total=delete_bytes_total,
        kept_counts=kept_counts,
    )


def _cleanup_empty_parents(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    while current != stop_dir and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _flatten_delete_paths(delete_groups: Sequence[DeleteGroup]) -> List[Path]:
    paths: List[Path] = []
    for group in delete_groups:
        paths.extend(group.paths)
    return sorted(paths, key=lambda path: (len(path.parts), str(path)), reverse=True)


def _apply_sequence_plan(plan: SequencePlan) -> Tuple[int, int]:
    removed_files = 0
    removed_bytes = 0
    for path in _flatten_delete_paths(plan.delete_groups):
        file_count, byte_count = _count_path_stats(path)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed_files += file_count
        removed_bytes += byte_count
        _cleanup_empty_parents(path.parent, stop_dir=plan.seq_dir)
    return removed_files, removed_bytes


def _print_verbose_plan(plan: SequencePlan, representation: str) -> None:
    print(f"  keep: annotations.npy, events.h5, raw/rgba_*.png, raw/rgba_blur_*.png, events/{representation}/")
    print(
        "  keep-counts: "
        f"rgba={plan.kept_counts.get('raw/rgba_*.png', 0)}, "
        f"rgba_blur={plan.kept_counts.get('raw/rgba_blur_*.png', 0)}, "
        f"{representation}={plan.kept_counts.get(f'events/{representation}/*.h5', 0)}"
    )
    if len(plan.delete_groups) == 0:
        print("  delete: none")
        return
    print("  delete:")
    for group in plan.delete_groups:
        print(
            f"    - {group.label} "
            f"({group.file_count} files, {_format_bytes(group.bytes_total)})"
        )


def _print_summary(
    *,
    scanned: int,
    valid: int,
    invalid: int,
    cleaned: int,
    skipped: int,
    planned_files: int,
    planned_bytes: int,
    removed_files: int,
    removed_bytes: int,
    apply: bool,
) -> None:
    print("[SUMMARY]")
    print(f"  scanned={scanned}")
    print(f"  valid={valid}")
    print(f"  invalid={invalid}")
    print(f"  cleaned={cleaned}")
    print(f"  skipped={skipped}")
    print(f"  planned_delete_files={planned_files}")
    print(f"  planned_delete_bytes={_format_bytes(planned_bytes)}")
    if apply:
        print(f"  actual_deleted_files={removed_files}")
        print(f"  actual_deleted_bytes={_format_bytes(removed_bytes)}")
    else:
        print("  actual_deleted_files=0")
        print("  actual_deleted_bytes=0.0B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and clean Kubric ETAP-style TAPFormer data to a minimal keep list."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/event_kubric/test",
        help="Root directory containing Kubric sequence folders.",
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
        "--representation",
        type=str,
        default="time_surfaces_v2_5",
        help="Representation directory to keep under each sequence's events/ folder.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete redundant files. Without this flag, run in dry-run mode.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sequence keep and delete details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    seq_items = _discover_sequence_dirs(
        dataset_root,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    print(
        f"[INFO] Mode={'apply' if args.apply else 'dry-run'} "
        f"dataset_root={dataset_root} representation={args.representation}"
    )
    print(
        f"[INFO] Selected {len(seq_items)} sequence(s) in range "
        f"[{args.start_index}, {args.end_index})."
    )

    scanned = len(seq_items)
    valid = 0
    invalid = 0
    cleaned = 0
    skipped = 0
    planned_files = 0
    planned_bytes = 0
    removed_files = 0
    removed_bytes = 0

    for seq_index, seq_dir in seq_items:
        plan = _build_sequence_plan(seq_dir, seq_index, args.representation)
        seq_name = seq_dir.name

        if not plan.valid:
            invalid += 1
            skipped += 1
            print(f"[WARN] Skip {seq_name}: missing {', '.join(plan.missing_items)}")
            continue

        valid += 1
        planned_files += plan.delete_file_count
        planned_bytes += plan.delete_bytes_total

        if plan.delete_file_count == 0:
            skipped += 1
            print(f"[OK] {seq_name}: already matches the minimal keep list.")
            if args.verbose:
                _print_verbose_plan(plan, args.representation)
            continue

        cleaned += 1
        action = "CLEAN" if args.apply else "DRY-RUN"
        print(
            f"[{action}] {seq_name}: "
            f"{plan.delete_file_count} files, {_format_bytes(plan.delete_bytes_total)}"
        )
        if args.verbose:
            _print_verbose_plan(plan, args.representation)

        if args.apply:
            seq_removed_files, seq_removed_bytes = _apply_sequence_plan(plan)
            removed_files += seq_removed_files
            removed_bytes += seq_removed_bytes

    _print_summary(
        scanned=scanned,
        valid=valid,
        invalid=invalid,
        cleaned=cleaned,
        skipped=skipped,
        planned_files=planned_files,
        planned_bytes=planned_bytes,
        removed_files=removed_files,
        removed_bytes=removed_bytes,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
