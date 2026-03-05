import multiprocessing
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import cv2
# import fire
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('TAPFormer/LFE_TAP')

from LFE_TAP.utils.event.representations import VoxelGrid, TimeOrderSurface, events_to_voxel_grid
# from utils.event.utils import blosc_opts

IMG_H = 512
IMG_W = 512
VOXEL_GRID_CONSTRUCTOR = VoxelGrid((5, 512, 512), True)

def blosc_opts(complevel=1, complib="blosc:zstd", shuffle="byte"):
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
    # split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"count_images_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    # if check_folder_filenum(output_dir, 95):
    #     return
    # if not output_seq_dir.exists():
    #     return
    
    dt_us = 1/48 * 1e6
    dt_us_bin = dt_us / n_bins

    with h5py.File(str(input_seq_dir / "events.h5"), "r") as h5f:
        h5f = np.array(h5f["events"])
        time = h5f[:, 0]
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f[:, 1])[idxs_sorted],
            np.asarray(h5f[:, 2])[idxs_sorted],
            np.asarray(h5f[:, 3])[idxs_sorted],
            np.asarray(h5f[:, 0])[idxs_sorted],
        )

        for i in range(1 , 96, 1):
            output_path = output_dir / f"{i:03}.h5"
            # if output_path.exists():
            #     continue

            count_images = np.zeros((IMG_H, IMG_W, n_bins), dtype=np.int32)
            t1 = i * dt_us
            t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                t0_bin = t0
                t1_bin = t0_bin + (i_bin+1) * dt_us_bin
                mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                x_bin, y_bin, p_bin, t_bin = (
                    x[mask_t],
                    y[mask_t],
                    p[mask_t],
                    time[mask_t],
                )
                n_events = len(x_bin)
                p_bin = p_bin.astype(np.int64) * 2 - 1
                for i in range(n_events):
                    count_images[y_bin[i], x_bin[i], i_bin] += p_bin[i]

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "count_images",
                    data=count_images,
                    shape=count_images.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )
            # Visualize
            if visualize:
                for i in range(n_bins):
                    img_vis = np.interp(count_images[:,:,i], (count_images[:,:,i].min(), count_images[:,:,i].max()), (0, 255)).astype(np.uint8)
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

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        x, y, p, time = (
            np.asarray(h5f["x"]),
            np.asarray(h5f["y"]),
            np.asarray(h5f["p"]),
            np.asarray(h5f["t"]),
        )

        # dt of labels
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{t1}.h5"
            if output_path.exists():
                continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins), dtype=np.int64)
            t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                t0_bin = t0 + i_bin * dt_us_bin
                t1_bin = t0_bin + dt_us_bin
                idx0 = np.searchsorted(time, t0_bin, side="left")
                idx1 = np.searchsorted(time, t1_bin, side="right")
                x_bin = x[idx0:idx1]
                y_bin = y[idx0:idx1]
                p_bin = p[idx0:idx1].astype(int) * 2 - 1

                n_events = len(x_bin)
                for i in range(n_events):
                    time_surface[y_bin[i], x_bin[i], i_bin] += p_bin[i]

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "event_stack",
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

    with h5py.File(str(input_seq_dir / "events.h5"), "r") as h5f:
        h5f = np.array(h5f["events"])
        time = h5f[:, 0]
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f[:, 1])[idxs_sorted],
            np.asarray(h5f[:, 2])[idxs_sorted],
            np.asarray(h5f[:, 3])[idxs_sorted],
            np.asarray(h5f[:, 0])[idxs_sorted],
        )

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
    # split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / input_seq_dir.stem
    if not output_seq_dir.exists():
        return
    output_dir = output_seq_dir / "events" / f"time_surfaces_accumulate_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = 1/50 * 1e6

    with h5py.File(str(input_seq_dir / "events.h5"), "r") as h5f:
        h5f = np.array(h5f["events"])
        time = h5f[:, 0]
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f[:, 1])[idxs_sorted],
            np.asarray(h5f[:, 2])[idxs_sorted],
            np.asarray(h5f[:, 3])[idxs_sorted],
            np.asarray(h5f[:, 0])[idxs_sorted],
        )

        for i in range(1 , 24, 1):
            output_path = output_dir / f"{i:03}.h5"
            # if output_path.exists():
            #     continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
            t1 = i * dt_us
            t0 = t1 - (((i-1)%3)+1) * dt_us

            dt_us_bin = (t1 - t0) / n_bins

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
            time_surface = np.divide(time_surface, t1-t0)

            # Write to disk
            with h5py.File("f:\\datasets\\kubric_movi_f\\0000\events\\template\sobel\\000.h5", "w") as h5f_out:
                h5f_out.create_dataset(
                    "time_surface_accumulate",
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

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f["x"])[idxs_sorted],
            np.asarray(h5f["y"])[idxs_sorted],
            np.asarray(h5f["p"])[idxs_sorted],
            np.asarray(h5f["t"])[idxs_sorted],
        )

        mask_t = np.logical_and(time > 380000, time <= 900000)
        x, y, p, time = x[mask_t], y[mask_t], p[mask_t], time[mask_t]

        index = 0
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            t0 = t1 - dt_us
            for i_bin in range(n_bins):
                t1_bin = t0 + (i_bin + 1) * dt_us_bin
                
                while time[index] <= t1_bin:
                    if on_board(x[index], y[index], p[index], time[index]):
                        for x0 in range(x[index] - 3, x[index] + 4):
                            for y0 in range(y[index] - 3, y[index] + 4):
                                if tos[y0, x0, int(p[index])] != 0:
                                    tos[y0, x0, int(p[index])] -= 1
                                if tos[y0, x0, int(p[index])] < 241:
                                    tos[y0, x0, int(p[index])] = 0
                        tos[y[index], x[index], int(p[index])] = 255
                    # index = min(index+1, len(time)-1)
                    index += 1
                    if index >= len(time):
                        break

                tos_bin[:, :, 2 * i_bin] = tos[:, :, 0]
                tos_bin[:, :, 2 * i_bin + 1] = tos[:, :, 1]


            output_path = output_dir / f"0{t1}.h5"
            # if output_path.exists():
            #     continue
            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "time_order_surface",
                    data=tos_bin,
                    shape=tos_bin.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )

            # Visualize
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
    
    dt_us = 1/48 * 1e6

    with h5py.File(str(input_seq_dir / "events.h5"), "r") as h5f:
        h5f = np.array(h5f["events"])
        time = h5f[:, 0]
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f[:, 1])[idxs_sorted],
            np.asarray(h5f[:, 2])[idxs_sorted],
            np.asarray(h5f[:, 3])[idxs_sorted],
            np.asarray(h5f[:, 0])[idxs_sorted],
        )

        # dt of labels
        for i in range(1 , 96, 1):
            output_path = output_dir / f"{i:03}.h5"
            # if output_path.exists() and os.path.getsize(output_path) > 10 * 1024:
            #     continue
            # print(f"Processing {output_path}")

            t1 = i * dt_us
            t0 = t1 - dt_us
            mask_t = np.logical_and(time > t0, time <= t1)
            x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
            curr_voxel_grid = events_to_voxel_grid(
                VOXEL_GRID_CONSTRUCTOR, p_bin, t_bin, x_bin, y_bin
            )
            curr_voxel_grid = curr_voxel_grid.numpy()
            curr_voxel_grid = np.transpose(curr_voxel_grid, (1, 2, 0))

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "voxel_grid",
                    data=curr_voxel_grid,
                    shape=curr_voxel_grid.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )
                
            # Visualize
            if visualize:
                for i in range(n_bins):
                    cv2.imshow(
                        f"curr_voxel_grid Bin {i}",
                        (curr_voxel_grid[:, :, i] * 255).astype(np.uint8),
                    )
                    cv2.waitKey(0)


def generate_representations(generation_function, input_seq_dir, output_dir, visualize, n_bins, dt):
    # for d in dt:
    for input_seq in tqdm(input_seq_dir, total=len(input_seq_dir)):
        try:
            generation_function(input_seq, output_dir, visualize=visualize, n_bins=n_bins, dt=dt)
        except Exception as e:
            print(f"Error occurred with input sequence directory: {input_seq}. Error: {e}")

def repeat_process(input_path, output_path, num_repeats, generation_function):
    # for split in ["train", "test0"]:
    #     split_dir = input_path / split
    #     n_seqs = len(os.listdir(str(split_dir)))
    #     num = n_seqs // num_repeats
    #     file_lists = [ [] for _ in range(num_repeats) ]
    #     for idx, file in enumerate(os.listdir(str(split_dir))):
    #         file_lists[idx % num_repeats].append(os.path.join(split_dir, file))
    #     pool = Pool(processes=num_repeats)
    #     for i in range(num_repeats):
    #         pool.apply_async(generate_representations, args=(generation_function, file_lists[i], output_path, False, 5, 0.01))
    #     pool.close()
    #     pool.join()        
        
    n_seqs = len(os.listdir(str(input_path)))
    num = n_seqs // num_repeats
    file_lists = [ [] for _ in range(num_repeats) ]
    for idx, file in enumerate(sorted(os.listdir(str(input_path)))):
        file_lists[idx % num_repeats].append(os.path.join(input_path, file))
    pool = Pool(processes=num_repeats)
    for i in range(num_repeats):
        pool.apply_async(generate_representations, args=(generation_function, file_lists[i], output_path, False, 5, 0.01))
    pool.close()
    pool.join()    
            
def generate(
    input_dir,
    output_dir,
    representation_type,
    dts=(0.01,),
    n_bins=5,
    visualize=False,
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
    
    repeat_process(input_dir, output_dir, 10, generation_function)


if __name__ == "__main__":
    generate("input_dir", "output_dir", "time_surface")
