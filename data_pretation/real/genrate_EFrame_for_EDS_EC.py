import numpy as np
from pathlib import Path
import os
import cv2
import h5py
import hdf5plugin
from LFE_TAP.utils.event.representations import VoxelGrid, events_to_voxel_grid


def blosc_opts(complevel=1, complib="blosc:zstd", shuffle="byte"):
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


def generate_time_surface_single(
    input_dir, output_dir, img_shape, visualize=False, n_bins=5, dt=0.01, type="EDS", center_time=False, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    IMG_W, IMG_H = img_shape
    data_root_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir = output_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins
    
    if type == "EDS":
        timestamps = np.loadtxt(os.path.join(data_root_dir, "images_timestamps.txt"))
        event_path = os.path.join(data_root_dir, "events_corrected.h5")
    elif type == "EC":
        timestamps = np.loadtxt(os.path.join(data_root_dir, "images.txt")) *1e6
        event_path = os.path.join(data_root_dir, "events_corrected.txt")
    
    new_timestamps = [timestamps[0]]
    for i in range(1, len(timestamps)):
        prev_time = timestamps[i-1]
        next_time = timestamps[i]
        
        current = prev_time + dt_us
        
        while current < next_time:
            new_timestamps.append(current)
            current += dt_us
        
        new_timestamps.append(next_time)
    
    if type == "EDS":
        with h5py.File(str(event_path), "r") as h5f:
            time = np.asarray(h5f["t"])
            idxs_sorted = np.argsort(time)
            x, y, p, time = (
                np.asarray(np.asarray(h5f["x"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["y"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["p"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["t"]))[idxs_sorted],
            )
    elif type == "EC":
        events = np.loadtxt(event_path)
        time = events[:, 0]
        time *= 1e6
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(events[:, 1])[idxs_sorted],
            np.asarray(events[:, 2])[idxs_sorted],
            np.asarray(events[:, 3])[idxs_sorted],
            np.asarray(time)[idxs_sorted],
        )
    
    for t1 in new_timestamps:
        if type == "EDS":
            output_path = output_dir / f"{str(int(t1)).zfill(16)}.h5"
        elif type == "EC":
            output_path = output_dir / f"{str(int(t1)).zfill(8)}.h5"
        # if output_path.exists():
        #     continue

        time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
        if center_time:
            t1 = t1 + (dt_us/2)
        t0 = t1 - dt_us

        # iterate over bins
        for i_bin in range(n_bins):
            t0_bin = t0 + i_bin * dt_us_bin
            t1_bin = t0_bin + dt_us_bin
            mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
            x_bin, y_bin, p_bin, t_bin = (
                np.rint(x[mask_t]).astype(int),
                np.rint(y[mask_t]).astype(int),
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
                    

def generate_gray_img(
    input_dir, output_dir, img_shape, visualize=False, n_bins=None, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    IMG_W, IMG_H = img_shape
    data_root_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir = output_dir / "events" / f"{dt:.4f}" / "gray_img"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    
    timestamps = np.loadtxt(os.path.join(data_root_dir, "images_timestamps.txt"))
    event_path = os.path.join(data_root_dir, "events_corrected.h5")
    
    with h5py.File(str(event_path), "r") as h5f:
        time = np.asarray(h5f["t"])
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(np.asarray(h5f["x"]))[idxs_sorted],
            np.asarray(np.asarray(h5f["y"]))[idxs_sorted],
            np.asarray(np.asarray(h5f["p"]))[idxs_sorted],
            np.asarray(np.asarray(h5f["t"]))[idxs_sorted],
        )
    
        for t1 in np.arange(timestamps[0] + dt_us, timestamps[-1] + dt_us, dt_us):
            output_path = output_dir / f"{str(int(t1)).zfill(16)}.png" 
            
            # if output_path.exists():
            #     continue

            gray_img = np.zeros((IMG_H, IMG_W), dtype=np.uint64)
            t0 = t1 - dt_us
            
            mask_t = np.logical_and(time > t0, time <= t1)
            x_bin, y_bin, p_bin, t_bin = (
                np.rint(x[mask_t]).astype(int),
                np.rint(y[mask_t]).astype(int),
                p[mask_t],
                time[mask_t],
            )
            n_events = len(x_bin)
            for i in range(n_events):
                gray_img[y_bin[i], x_bin[i]] = ((t_bin[i] - t0) * p_bin[i])
            
            gray_img = np.divide(gray_img + dt_us, 2 * dt_us)
            
            cv2.imwrite(str(output_path), (gray_img * 255).astype(np.uint8))
            
            # Visualize
            if visualize:
                cv2.imshow(f"gray_img",(gray_img))
                cv2.waitKey(0)
                
                
def generate_voxel_grid_single(input_dir, output_dir,  visualize=False, n_bins=5, dt=0.01, type="EDS", **kwargs):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    data_root_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir = output_dir / "events" / f"{dt:.4f}" / f"voxel_grids_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    # dt_us_bin = dt_us / n_bins
    
    if type == "EDS":
        timestamps = np.loadtxt(os.path.join(data_root_dir, "images_timestamps.txt"))
        event_path = os.path.join(data_root_dir, "events_corrected.h5")
    elif type == "EC":
        timestamps = np.loadtxt(os.path.join(data_root_dir, "images.txt")) *1e6
        event_path = os.path.join(data_root_dir, "events_corrected.txt")
        
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
        
    if type == "EDS":
        with h5py.File(str(event_path), "r") as h5f:
            time = np.asarray(h5f["t"])
            idxs_sorted = np.argsort(time)
            x, y, p, time = (
                np.asarray(np.asarray(h5f["x"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["y"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["p"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["t"]))[idxs_sorted],
            )
            VOXEL_GRID_CONSTRUCTOR = VoxelGrid((5, 480, 640), True)
    elif type == "EC":
        events = np.loadtxt(event_path)
        time = events[:, 0]
        time *= 1e6
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(events[:, 1])[idxs_sorted],
            np.asarray(events[:, 2])[idxs_sorted],
            np.asarray(events[:, 3])[idxs_sorted],
            np.asarray(time)[idxs_sorted],
        )
        VOXEL_GRID_CONSTRUCTOR = VoxelGrid((5, 180, 240), True)
    
    for t1 in new_timestamps:
        if type == "EDS":
            output_path = output_dir / f"{str(int(t1)).zfill(16)}.h5"
        elif type == "EC":
            output_path = output_dir / f"{str(int(t1)).zfill(8)}.h5"
        # if output_path.exists():
        #     continue

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
        
        if visualize:
            for i in range(n_bins):
                cv2.imshow(
                    f"curr_voxel_grid Bin {i}",
                    (curr_voxel_grid[:, :, i] * 255).astype(np.uint8),
                )
                cv2.waitKey(0)
                

def generate_count_image_v1_single(
    input_dir, output_dir, img_shape, visualize=False, n_bins=5, dt=0.01, type="EDS", mode="nearest_neighbor", **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    IMG_W, IMG_H = img_shape
    data_root_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir = output_dir / "events" / f"{dt:.4f}" / f"count_images_v1_{n_bins}_np"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins
    
    if type == "EDS":
        timestamps = np.loadtxt(os.path.join(data_root_dir, "images_timestamps.txt"))
        event_path = os.path.join(data_root_dir, "events_corrected.h5")
    elif type == "EC":
        timestamps = np.loadtxt(os.path.join(data_root_dir, "images.txt")) *1e6
        event_path = os.path.join(data_root_dir, "events_corrected.txt")
        
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
        
    if type == "EDS":
        with h5py.File(str(event_path), "r") as h5f:
            time = np.asarray(h5f["t"])
            idxs_sorted = np.argsort(time)
            x, y, p, time = (
                np.asarray(np.asarray(h5f["x"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["y"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["p"]))[idxs_sorted],
                np.asarray(np.asarray(h5f["t"]))[idxs_sorted],
            )
    elif type == "EC":
        events = np.loadtxt(event_path)
        time = events[:, 0]
        time *= 1e6
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(events[:, 1])[idxs_sorted],
            np.asarray(events[:, 2])[idxs_sorted],
            np.asarray(events[:, 3])[idxs_sorted],
            np.asarray(time)[idxs_sorted],
        )
    
    for t1 in new_timestamps:
        if type == "EDS":
            output_path = output_dir / f"{str(int(t1)).zfill(16)}.h5"
        elif type == "EC":
            output_path = output_dir / f"{str(int(t1)).zfill(8)}.h5"
            # if output_path.exists():
            #     continue

        count_images = np.zeros((IMG_H, IMG_W, n_bins), dtype=np.int32)
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
            p_bin = p_bin.astype(np.int32) * 2 - 1
            
            if mode == "nearest_neighbor":
                x_bin, y_bin = np.rint(x_bin).astype(int), np.rint(y_bin).astype(int)
                for i in range(n_events):
                    count_images[y_bin[i], x_bin[i], i_bin] += p_bin[i]
            elif mode == "bilinear":
                count_images_ = np.zeros((IMG_H * IMG_W), dtype=np.float64)
                floor_x, floor_y = np.floor(x_bin + 1e-8), np.floor(y_bin + 1e-8)
                floor_to_x, floor_to_y = x_bin - floor_x, y_bin - floor_y
                inds = np.concatenate([
                    floor_x + floor_y*IMG_W,
                    floor_x + (floor_y+1)*IMG_W,
                    (floor_x+1) + floor_y*IMG_W,
                    (floor_x+1) + (floor_y+1)*IMG_W,
                ], axis=-1,)
                inds_mask = np.concatenate([
                    (0<=floor_x) * (floor_x<IMG_W) * (0<=floor_y) * (floor_y<IMG_H),
                    (0<=floor_x) * (floor_x<IMG_W) * (0<=floor_y+1) * (floor_y+1<IMG_H),
                    (0<=floor_x+1) * (floor_x+1<IMG_W) * (0<=floor_y) * (floor_y<IMG_H),
                    (0<=floor_x+1) * (floor_x+1<IMG_W) * (0<=floor_y+1) * (floor_y+1<IMG_H),
                ], axis=-1,)
                w_pos0 = (1-floor_to_x) * (1-floor_to_y) * p_bin
                w_pos1 = floor_to_x * (1-floor_to_y) * p_bin
                w_pos2 = (1-floor_to_x) * floor_to_y * p_bin
                w_pos3 = floor_to_x * floor_to_y * p_bin
                vals = np.concatenate([w_pos0, w_pos1, w_pos2, w_pos3], axis=-1)
                inds = (inds * inds_mask).astype(np.int64)
                vals = vals * inds_mask
                np.add.at(count_images_, inds, vals)
                count_images[:,:,i_bin] = count_images_.reshape((IMG_H, IMG_W))

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

if __name__ == "__main__":
    input_folder_dir = "/media/ljx/ljx"
    representation_type = "voxel_grid"
    
    EVAL_DATASETS_EDS = [
        # ("peanuts_light_160_386", "EDS"),
        # ("rocket_earth_light_338_438", "EDS"),
        # ("ziggy_in_the_arena_1350_1650", "EDS"),
        # ("peanuts_running_2360_2460", "EDS"),
        ("boxes_rotation_198_278", "EC"),
        ("boxes_translation_330_410", "EC"),
        ("shapes_6dof_485_565", "EC"),
        ("shapes_rotation_165_245", "EC"),
        ("shapes_translation_8_88", "EC"),
    ]
    
    for dataset, type in EVAL_DATASETS_EDS:
        if type == "EDS":
            input_folder = os.path.join(input_folder_dir, "eds_subseq", dataset)
            output_folder = os.path.join(input_folder_dir, "eds_test", dataset)
            image_shape = (640, 480)
        elif type == "EC":
            input_folder = os.path.join(input_folder_dir, "ec_subseq", dataset)
            output_folder = input_folder
            image_shape = (240, 180)
    
        if representation_type == "time_surface":
            generate_time_surface_single(input_folder, output_dir=output_folder, img_shape=image_shape, visualize=False, n_bins=5, dt=0.0350, type=type, center_time=False)
        elif representation_type == "gray_img":
            generate_gray_img(input_folder, output_dir=input_folder, img_shape=image_shape, visualize=False, dt=0.005)
        elif representation_type == "voxel_grid":
            generate_voxel_grid_single(input_folder, output_dir=output_folder, visualize=False, n_bins=5, dt=0.010, type=type)
        elif representation_type == "count_images":
            generate_count_image_v1_single(input_folder, output_dir=output_folder, img_shape=image_shape, visualize=False, n_bins=5, dt=0.010, type=type)
            
        print(f"Finish generating {dataset} {representation_type} representation.")
    
    