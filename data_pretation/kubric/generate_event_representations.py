import argparse
import multiprocessing
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

try:
    import cv2
except ImportError:
    cv2 = None
# import fire
import h5py
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TAPFormer 根目录
sys.path.append(str(PROJECT_ROOT))

from LFE_TAP.utils.event.representations import VoxelGrid, TimeOrderSurface, events_to_voxel_grid
from LFE_TAP.utils.event.utils import load_events_h5_columns
# from utils.event.utils import blosc_opts

IMG_H = 512
IMG_W = 512
VOXEL_GRID_CONSTRUCTOR = VoxelGrid((5, 512, 512), True)

def blosc_opts(complevel=1, complib="blosc:zstd", shuffle="byte"):
    if hdf5plugin is None:
        args = {
            "compression": "gzip",
            "compression_opts": max(1, min(int(complevel), 9)),
        }
        if shuffle in ("bit", "byte"):
            args["shuffle"] = True
        return args

    # Inspired by: https://github.com/h5py/h5py/issues/611#issuecomment-353694301
    # More info on options: https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L55-L79
    shuffle = 2 if shuffle == "bit" else 1 if shuffle == "byte" else 0
    compressors = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
    complib = ["blosc:" + c for c in compressors].index(complib)
    args = {
        "compression": 32001,
        "compression_opts": (0, 0, 0, 0, complevel, shuffle, complib),
    }
    if shuffle > 0:
        # Do not use h5py shuffle if blosc shuffle is enabled.
        args["shuffle"] = False
    return args


def _load_sorted_events(events_h5_path):
    time, x, y, p = load_events_h5_columns(events_h5_path)
    idxs_sorted = np.argsort(time)
    return x[idxs_sorted], y[idxs_sorted], p[idxs_sorted], time[idxs_sorted]


def check_number_of_files(directory_path, expected_count=95):
    directory = Path(directory_path)
    # 计算目录中文件的数量
    file_count = len([f for f in directory.iterdir() if f.is_file()])
    return file_count == expected_count

def generate_event_count_images_single(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.01:0.9], generate the event count image for a Multiflow sequence
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    output_dir = Path(output_dir)
    output_seq_dir = output_dir / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"count_images_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)

    dt_us = 1 / 48 * 1e6
    dt_us_bin = dt_us / n_bins
    x, y, p, time = _load_sorted_events(input_seq_dir / "events.h5")

    for i in range(1, 96, 1):
        output_path = output_dir / f"{i:03}.h5"
        count_images = np.zeros((IMG_H, IMG_W, n_bins), dtype=np.int32)
        t1 = i * dt_us
        t0 = t1 - dt_us

        for i_bin in range(n_bins):
            t0_bin = t0
            t1_bin = t0_bin + (i_bin + 1) * dt_us_bin
            mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
            x_bin, y_bin, p_bin = x[mask_t], y[mask_t], p[mask_t]
            n_events = len(x_bin)
            p_bin = p_bin.astype(np.int64) * 2 - 1
            for j in range(n_events):
                count_images[y_bin[j], x_bin[j], i_bin] += p_bin[j]

        with h5py.File(output_path, "w") as h5f_out:
            h5f_out.create_dataset(
                "count_images",
                data=count_images,
                shape=count_images.shape,
                dtype=np.float32,
                **blosc_opts(complevel=1, shuffle="byte"),
            )

        if visualize:
            for i_bin in range(n_bins):
                img_vis = np.interp(
                    count_images[:, :, i_bin],
                    (count_images[:, :, i_bin].min(), count_images[:, :, i_bin].max()),
                    (0, 255),
                ).astype(np.uint8)
                cv2.imshow("Count Image", img_vis)
                cv2.waitKey(0)

def generate_sbt_single(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"{dt:.4f}" / f"event_stacks_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins
    x, y, p, time = _load_sorted_events(input_seq_dir / "events" / "events.h5")

    for t1 in np.arange(400000, 900000 + dt_us, dt_us):
        output_path = output_dir / f"0{t1}.h5"
        if output_path.exists():
            continue

        time_surface = np.zeros((IMG_H, IMG_W, n_bins), dtype=np.int64)
        t0 = t1 - dt_us

        for i_bin in range(n_bins):
            t0_bin = t0 + i_bin * dt_us_bin
            t1_bin = t0_bin + dt_us_bin
            idx0 = np.searchsorted(time, t0_bin, side="left")
            idx1 = np.searchsorted(time, t1_bin, side="right")
            x_bin = x[idx0:idx1]
            y_bin = y[idx0:idx1]
            p_bin = p[idx0:idx1].astype(int) * 2 - 1

            n_events = len(x_bin)
            for j in range(n_events):
                time_surface[y_bin[j], x_bin[j], i_bin] += p_bin[j]

        with h5py.File(output_path, "w") as h5f_out:
            h5f_out.create_dataset(
                "event_stack",
                data=time_surface,
                shape=time_surface.shape,
                dtype=np.float32,
                **blosc_opts(complevel=1, shuffle="byte"),
            )

        if visualize:
            for i_bin in range(n_bins):
                cv2.imshow(
                    f"Time Surface Bin {i_bin}",
                    (time_surface[:, :, i_bin] * 255).astype(np.uint8),
                )
                cv2.waitKey(0)

def generate_time_surface_single(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    output_dir = Path(output_dir)
    # split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / input_seq_dir.stem
    if not output_seq_dir.exists():
        return
    output_dir = output_seq_dir / "events" / f"time_surfaces_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    if check_number_of_files(output_dir, 95):
        return
    dt_us = 1/48 * 1e6
    dt_us_bin = dt_us / n_bins

    x, y, p, time = _load_sorted_events(input_seq_dir / "events.h5")

    for i in range(1 , 96, 1):
            output_path = output_dir / f"{i:03}.h5"
            # if output_path.exists() and os.path.getsize(output_path) > 20 * 1024:
            #     continue
            # print(f"Processing {output_path}")

            time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
            t1 = i * dt_us
            t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                t0_bin = t0 + i_bin * dt_us_bin
                t1_bin = t0_bin + dt_us_bin
                mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                x_bin, y_bin, p_bin, t_bin = (
                    x[mask_t],
                    y[mask_t],
                    p[mask_t],
                    time[mask_t],
                )
                n_events = len(x_bin)
                for i in range(n_events):
                    time_surface[y_bin[i], x_bin[i], 2 * i_bin + int(p_bin[i])] = (
                        t_bin[i] - t0
                    )
            time_surface = np.divide(time_surface, dt_us)

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "time_surface",
                    data=time_surface,
                    shape=time_surface.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )
            # Visualize
            if visualize:
                for i in range(n_bins):
                    cv2.imshow(
                        f"Time Surface Bin {i}",
                        (time_surface[:, :, i] * 255).astype(np.uint8),
                    )
                    cv2.waitKey(0)
                    
                    
def generate_time_surface_accumulate(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    output_dir = Path(output_dir)
    output_seq_dir = output_dir / input_seq_dir.stem
    if not output_seq_dir.exists():
        return
    output_dir = output_seq_dir / "events" / f"time_surfaces_accumulate_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = 1 / 50 * 1e6

    x, y, p, time = _load_sorted_events(input_seq_dir / "events.h5")

    for i in range(1, 24, 1):
        output_path = output_dir / f"{i:03}.h5"
        time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
        t1 = i * dt_us
        t0 = t1 - (((i - 1) % 3) + 1) * dt_us
        dt_us_bin = (t1 - t0) / n_bins

        for i_bin in range(n_bins):
            t0_bin = t0 + i_bin * dt_us_bin
            t1_bin = t0_bin + dt_us_bin
            mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
            x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
            n_events = len(x_bin)
            for j in range(n_events):
                time_surface[y_bin[j], x_bin[j], 2 * i_bin + int(p_bin[j])] = t_bin[j] - t0
        time_surface = np.divide(time_surface, t1 - t0)

        with h5py.File(output_path, "w") as h5f_out:
            h5f_out.create_dataset(
                "time_surface_accumulate",
                data=time_surface,
                shape=time_surface.shape,
                dtype=np.float32,
                **blosc_opts(complevel=1, shuffle="byte"),
            )

        if visualize:
            for i_bin in range(n_bins):
                cv2.imshow(
                    f"Time Surface Bin {i_bin}",
                    (time_surface[:, :, i_bin] * 255).astype(np.uint8),
                )
                cv2.waitKey(0)

sae = np.zeros((2, IMG_H, IMG_W))
sae_latest = np.zeros((2, IMG_H, IMG_W))

def on_board(x, y, p, t):
    if x>=3 and x<IMG_W-3 and y>=3 and y<IMG_H-3:
        # pol = 1 if p else 0
        # pol_inv = 0 if p else 1
        # if ((t > sae_latest[pol][y][x] + 20000) or
        #         (sae_latest[pol_inv][y][x] > sae_latest[pol][y][x])):
        #     sae_latest[pol][y][x] = t
        #     sae[pol][y][x] = t
        #     return True
        # else:
        #     sae_latest[pol][y][x] = t
        #     return False
        return True
    else:
        return False


def generate_time_order_surface_single1(input_seq_dir, output_dir, visualize=True, n_bins=5, dt=0.01, **kwargs):
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events/0.0100" / f"time_order_surfaces_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins
    tos_bin = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint8)
    tos = np.zeros((IMG_H, IMG_W, 2), dtype=np.uint8)

    x, y, p, time = _load_sorted_events(input_seq_dir / "events" / "events.h5")
    mask_t = np.logical_and(time > 380000, time <= 900000)
    x, y, p, time = x[mask_t], y[mask_t], p[mask_t], time[mask_t]

    index = 0
    for t1 in np.arange(400000, 900000 + dt_us, dt_us):
        t0 = t1 - dt_us
        for i_bin in range(n_bins):
            t1_bin = t0 + (i_bin + 1) * dt_us_bin

            while index < len(time) and time[index] <= t1_bin:
                if on_board(x[index], y[index], p[index], time[index]):
                    for x0 in range(x[index] - 3, x[index] + 4):
                        for y0 in range(y[index] - 3, y[index] + 4):
                            if tos[y0, x0, int(p[index])] != 0:
                                tos[y0, x0, int(p[index])] -= 1
                            if tos[y0, x0, int(p[index])] < 241:
                                tos[y0, x0, int(p[index])] = 0
                    tos[y[index], x[index], int(p[index])] = 255
                index += 1

            tos_bin[:, :, 2 * i_bin] = tos[:, :, 0]
            tos_bin[:, :, 2 * i_bin + 1] = tos[:, :, 1]

        output_path = output_dir / f"0{t1}.h5"
        with h5py.File(output_path, "w") as h5f_out:
            h5f_out.create_dataset(
                "time_order_surface",
                data=tos_bin,
                shape=tos_bin.shape,
                dtype=np.float32,
                **blosc_opts(complevel=1, shuffle="byte"),
            )

        if visualize:
            cv2.imshow("p_tos", tos[:, :, 0])
            cv2.imshow("n_tos", tos[:, :, 1])
            cv2.waitKey(1)

def generate_voxel_grid_single(input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"voxel_grids_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    if check_number_of_files(output_dir, 95):
        return

    dt_us = 1 / 48 * 1e6
    x, y, p, time = _load_sorted_events(input_seq_dir / "events.h5")

    for i in range(1, 96, 1):
        output_path = output_dir / f"{i:03}.h5"
        t1 = i * dt_us
        t0 = t1 - dt_us
        mask_t = np.logical_and(time > t0, time <= t1)
        x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
        curr_voxel_grid = events_to_voxel_grid(
            VOXEL_GRID_CONSTRUCTOR, p_bin, t_bin, x_bin, y_bin
        )
        curr_voxel_grid = curr_voxel_grid.numpy()
        curr_voxel_grid = np.transpose(curr_voxel_grid, (1, 2, 0))

        with h5py.File(output_path, "w") as h5f_out:
            h5f_out.create_dataset(
                "voxel_grid",
                data=curr_voxel_grid,
                shape=curr_voxel_grid.shape,
                dtype=np.float32,
                **blosc_opts(complevel=1, shuffle="byte"),
            )

        if visualize:
            for i_bin in range(n_bins):
                cv2.imshow(
                    f"curr_voxel_grid Bin {i_bin}",
                    (curr_voxel_grid[:, :, i_bin] * 255).astype(np.uint8),
                )
                cv2.waitKey(0)

def generate_representations(generation_function, input_seq_dir, output_dir, visualize, n_bins, dt):
    # keep this helper for backward compatibility; no per-worker tqdm here
    for input_seq in input_seq_dir:
        try:
            generation_function(input_seq, output_dir, visualize=visualize, n_bins=n_bins, dt=dt)
        except Exception as e:
            print(f"Error occurred with input sequence directory: {input_seq}. Error: {e}")


def _generate_one_sequence(args):
    generation_function, input_seq, output_dir, visualize, n_bins, dt = args
    try:
        generation_function(input_seq, output_dir, visualize=visualize, n_bins=n_bins, dt=dt)
        return input_seq, True, ""
    except Exception as e:
        return input_seq, False, str(e)


def repeat_process(
    input_path,
    output_path,
    num_repeats,
    generation_function,
    visualize=False,
    n_bins=5,
    dt=0.01,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    seq_paths = [
        str(input_path / fname)
        for fname in sorted(os.listdir(str(input_path)))
        if (input_path / fname).is_dir()
    ]
    if len(seq_paths) == 0:
        print(f"[WARN] No sequence directory found in {input_path}")
        return

    num_workers = max(1, min(int(num_repeats), len(seq_paths)))
    if visualize and num_workers > 1:
        print("[WARN] visualize=True is incompatible with multi-worker progress display; forcing num_workers=1.")
        num_workers = 1

    tasks = [
        (generation_function, seq_path, output_path, visualize, n_bins, dt)
        for seq_path in seq_paths
    ]

    if num_workers == 1:
        for task in tqdm(tasks, total=len(tasks), desc="Generating representations"):
            _, ok, err = _generate_one_sequence(task)
            if not ok:
                print(f"Error occurred with input sequence directory: {task[1]}. Error: {err}")
        return

    with Pool(processes=num_workers) as pool:
        for input_seq, ok, err in tqdm(
            pool.imap_unordered(_generate_one_sequence, tasks),
            total=len(tasks),
            desc="Generating representations",
        ):
            if not ok:
                print(f"Error occurred with input sequence directory: {input_seq}. Error: {err}")


def generate(
    input_dir,
    output_dir,
    representation_type,
    dts=(0.01,),
    n_bins=5,
    visualize=False,
    num_repeats=10,
    **kwargs,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if representation_type == "time_surface":
        generation_function = generate_time_surface_single
    elif representation_type == "time_surface_accumulate":
        generation_function = generate_time_surface_accumulate
    elif representation_type == "voxel_grid":
        generation_function = generate_voxel_grid_single
    elif representation_type == "event_stack":
        generation_function = generate_sbt_single
    elif representation_type == "event_count":
        generation_function = generate_event_count_images_single
    elif representation_type == "time_order_surface":
        generation_function = generate_time_order_surface_single1
    else:
        raise NotImplementedError(f"No generation function for {representation_type}")
    
    if isinstance(dts, (list, tuple)):
        dt = float(dts[0]) if len(dts) > 0 else 0.01
    else:
        dt = float(dts)

    repeat_process(
        input_dir,
        output_dir,
        num_repeats,
        generation_function,
        visualize=visualize,
        n_bins=n_bins,
        dt=dt,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate event representations for TAPFormer Kubric-style data."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing per-sequence folders.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write generated representations.")
    parser.add_argument(
        "--representation-type",
        type=str,
        required=True,
        choices=(
            "time_surface",
            "time_surface_accumulate",
            "voxel_grid",
            "event_stack",
            "event_count",
            "time_order_surface",
        ),
        help="Representation type to generate.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time interval in seconds.")
    parser.add_argument("--n-bins", type=int, default=5, help="Number of temporal bins.")
    parser.add_argument("--num-repeats", type=int, default=10, help="Number of worker processes.")
    parser.add_argument("--visualize", action="store_true", help="Enable OpenCV visualization while generating.")
    return parser.parse_args()


if __name__ == "__main__":
    # generate("input_dir", "output_dir", "time_surface")
    args = parse_args()
    generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        representation_type=args.representation_type,
        dts=(args.dt,),
        n_bins=args.n_bins,
        visualize=args.visualize,
        num_repeats=args.num_repeats,
    )
