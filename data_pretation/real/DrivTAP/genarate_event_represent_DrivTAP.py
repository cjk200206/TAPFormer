import os
import cv2
import numpy as np
from pathlib import Path
import h5py
import hdf5plugin

from data_pretation.real.aedat4.read_aedat4_2_dataset import blosc_opts

def generate_time_surface_single(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, center_time=False, fix_num=None, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    output_dir = Path(output_dir)
    if fix_num is not None:
        output_dir = output_dir/ "events" / f"time_surfaces_v2_{n_bins}" / f"fix_num_{fix_num}"
    else:
        output_dir = output_dir / "events" / f"time_surfaces_v2_{n_bins}" / f"{dt:.4f}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins
    
    timestamps = np.loadtxt(input_seq_dir / "image_timestamps_full.txt")
    
    IMG_W, IMG_H = (int(cv2.VideoCapture(str(input_seq_dir / f"{os.path.basename(input_seq_dir)}.mp4")).get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    
    new_timestamps = [timestamps[0]]
    for i in range(1, len(timestamps)):
        prev_time = timestamps[i-1]
        next_time = timestamps[i]
        
        current = prev_time + dt_us
        
        # 在当前间隔内插入时间点（直到超过next_time）
        while current < next_time:
            new_timestamps.append(current)
            current += dt_us
        
        # 添加下一个原始时间点
        new_timestamps.append(next_time)

    with h5py.File(str(input_seq_dir / "events_undistorted.h5"), "r") as h5f:
        h5f = np.array(h5f["events"])
        time = h5f[:, 0]
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f[:, 1])[idxs_sorted],
            np.asarray(h5f[:, 2])[idxs_sorted],
            np.asarray(h5f[:, 3])[idxs_sorted],
            np.asarray(h5f[:, 0])[idxs_sorted],
        )

        for t1 in timestamps:
            output_path = output_dir / f"{int(t1)}.h5"
            if output_path.exists():
                continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
            if center_time:
                t1 = t1 + (dt_us / 2)
            if fix_num is not None:
                idx_t1 = np.searchsorted(time, t1, side="right")
                idx_t0 = max(0, idx_t1 - fix_num + 1)
                t0 = time[idx_t0]
                dt_us = t1 - t0
                t0_bin_ = t0
            else:
                t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                if fix_num is not None:
                    t0_bin = t0_bin_
                    idx_t1 = max(0, idx_t0 + fix_num // n_bins)
                    t1_bin = time[idx_t1]
                    t0_bin_ = t1_bin
                    idx_t0 = idx_t1
                else:
                    t0_bin = t0 + i_bin * dt_us_bin
                    t1_bin = t0_bin + dt_us_bin
                mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                x_bin, y_bin, p_bin, t_bin = (
                    x[mask_t],
                    y[mask_t],
                    p[mask_t],
                    time[mask_t],
                )
                
                # # 扁平化索引
                # flat_idx = (y_bin * IMG_W + x_bin) * (n_bins * 2) + (2 * i_bin + p_bin)

                # # 更新最大值
                # np.maximum.at(time_surface.ravel(), flat_idx, t_bin.astype(np.float32) - t0)
                
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
                        (time_surface[:, :, i*2] * 177.5 + time_surface[:, :, i*2+1] * 177.5).astype(np.uint8),
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
    output_dir = output_dir / "events" / f"time_surfaces_accumulate_v2_{n_bins}" / f"{dt:.4f}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins
    
    timestamps = np.loadtxt(input_seq_dir / "image_timestamps.txt")
    image_timestamps = timestamps.copy()
    
    IMG_W, IMG_H = (int(cv2.VideoCapture(str(input_seq_dir / f"{os.path.basename(input_seq_dir)}.mp4")).get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    
    new_timestamps = [timestamps[0]]
    for i in range(1, len(timestamps)):
        prev_time = timestamps[i-1]
        next_time = timestamps[i]
        
        current = prev_time + dt_us
        
        # 在当前间隔内插入时间点（直到超过next_time）
        while current < next_time:
            new_timestamps.append(current)
            current += dt_us
        
        # 添加下一个原始时间点
        new_timestamps.append(next_time)

    with h5py.File(str(input_seq_dir / "events_undistorted.h5"), "r") as h5f:
        h5f = np.array(h5f["events"])
        time = h5f[:, 0]
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f[:, 1])[idxs_sorted],
            np.asarray(h5f[:, 2])[idxs_sorted],
            np.asarray(h5f[:, 3])[idxs_sorted],
            np.asarray(h5f[:, 0])[idxs_sorted],
        )

        img_ind = 0
        for t1 in new_timestamps:
            output_path = output_dir / f"{int(t1)}.h5"
            # if output_path.exists():
            #     continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
            if img_ind == 0:
                t0 = t1 - dt_us
                img_ind += 1
            elif t1 > image_timestamps[img_ind]:
                t0 = image_timestamps[img_ind]
            else:
                t0 = image_timestamps[img_ind-1]

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
            # with h5py.File(output_path, "w") as h5f_out:
            #     h5f_out.create_dataset(
            #         "time_surface_accumulate",
            #         data=time_surface,
            #         shape=time_surface.shape,
            #         dtype=np.float32,
            #         **blosc_opts(complevel=1, shuffle="byte"),
            #     )
            # Visualize
            if visualize:
                for i in range(n_bins):
                    cv2.imshow(
                        f"Time Surface Accumulate Bin {i}",
                        (time_surface[:, :, i*2] * 177.5 + time_surface[:, :, i*2+1] * 177.5).astype(np.uint8),
                    )
                    cv2.waitKey(0)
                    
                    
if __name__ == "__main__":
    representation_type = "time_surface"
    
    
    input_folder = ""
    for dt in [0.003]:
        generate_time_surface_single(input_folder, output_dir=input_folder, visualize=False, n_bins=5, dt=dt, center_time=False)
        