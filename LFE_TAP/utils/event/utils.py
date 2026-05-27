import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from multiprocessing import Pool

import h5py
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None
import numpy as np
import pandas as pd
# import dv_processing as dv
from pathlib import Path
try:
    import cv2
    from cv2 import IMREAD_GRAYSCALE, imread
except ImportError:
    cv2 = None
    IMREAD_GRAYSCALE = None

    def imread(*args, **kwargs):
        raise ImportError("opencv-python is required for grayscale image reading")
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm


COMPACT_EVENTS_H5_FORMAT = "split_v1"
COMPACT_EVENTS_H5_TIME_UNIT = "us"
COMPACT_EVENTS_H5_DTYPES = {
    "t": np.uint32,
    "x": np.uint16,
    "y": np.uint16,
    "p": np.uint8,
}


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


def is_compact_events_h5(h5f):
    return all(name in h5f for name in ("t", "x", "y", "p"))


def _load_events_h5_columns_from_open_file(h5f):
    if is_compact_events_h5(h5f):
        t = np.asarray(h5f["t"], dtype=np.int64)
        x = np.asarray(h5f["x"], dtype=np.int64)
        y = np.asarray(h5f["y"], dtype=np.int64)
        p = np.asarray(h5f["p"], dtype=np.int64)
        return t, x, y, p

    if "events" not in h5f:
        raise KeyError("events.h5 must contain either split datasets {t,x,y,p} or legacy dataset 'events'.")

    events = np.asarray(h5f["events"])
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"legacy events dataset must have shape (N,4), got {events.shape}")
    return (
        np.asarray(events[:, 0], dtype=np.int64),
        np.asarray(events[:, 1], dtype=np.int64),
        np.asarray(events[:, 2], dtype=np.int64),
        np.asarray(events[:, 3], dtype=np.int64),
    )


def load_events_h5_columns(path_or_h5):
    if isinstance(path_or_h5, (str, Path)):
        with h5py.File(str(path_or_h5), "r") as h5f:
            return _load_events_h5_columns_from_open_file(h5f)
    return _load_events_h5_columns_from_open_file(path_or_h5)


def _validate_compact_events_columns(t, x, y, p):
    arrays = {
        "t": np.asarray(t),
        "x": np.asarray(x),
        "y": np.asarray(y),
        "p": np.asarray(p),
    }
    lengths = {name: arr.shape[0] for name, arr in arrays.items()}
    if any(arr.ndim != 1 for arr in arrays.values()):
        raise ValueError("events columns must all be 1D arrays")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"events columns must have equal length, got lengths={lengths}")

    if arrays["t"].size == 0:
        return (
            arrays["t"].astype(np.uint32, copy=False),
            arrays["x"].astype(np.uint16, copy=False),
            arrays["y"].astype(np.uint16, copy=False),
            arrays["p"].astype(np.uint8, copy=False),
        )

    t_i64 = np.asarray(arrays["t"], dtype=np.int64)
    x_i64 = np.asarray(arrays["x"], dtype=np.int64)
    y_i64 = np.asarray(arrays["y"], dtype=np.int64)
    p_i64 = np.asarray(arrays["p"], dtype=np.int64)

    if np.min(t_i64) < 0 or np.max(t_i64) > np.iinfo(np.uint32).max:
        raise ValueError("event timestamp range is outside uint32")
    if np.min(x_i64) < 0 or np.max(x_i64) > np.iinfo(np.uint16).max:
        raise ValueError("event x range is outside uint16")
    if np.min(y_i64) < 0 or np.max(y_i64) > np.iinfo(np.uint16).max:
        raise ValueError("event y range is outside uint16")
    if not np.all((p_i64 == 0) | (p_i64 == 1)):
        raise ValueError("event polarity must be 0 or 1")

    return (
        t_i64.astype(np.uint32, copy=False),
        x_i64.astype(np.uint16, copy=False),
        y_i64.astype(np.uint16, copy=False),
        p_i64.astype(np.uint8, copy=False),
    )


def write_compact_events_h5(output_h5, t, x, y, p):
    t_u32, x_u16, y_u16, p_u8 = _validate_compact_events_columns(t, x, y, p)

    with h5py.File(str(output_h5), "w") as h5f:
        h5f.attrs["format"] = COMPACT_EVENTS_H5_FORMAT
        h5f.attrs["time_unit"] = COMPACT_EVENTS_H5_TIME_UNIT
        h5f.create_dataset("t", data=t_u32, dtype=np.uint32, **blosc_opts(complevel=1, shuffle="byte"))
        h5f.create_dataset("x", data=x_u16, dtype=np.uint16, **blosc_opts(complevel=1, shuffle="byte"))
        h5f.create_dataset("y", data=y_u16, dtype=np.uint16, **blosc_opts(complevel=1, shuffle="byte"))
        h5f.create_dataset("p", data=p_u8, dtype=np.uint8, **blosc_opts(complevel=1, shuffle="byte"))


def query_events(events_h5, events_t, t0, t1):
    """
    Return a numpy array of events in temporal range [t0, t1)
    :param events_h5: h5 object with events. {x, y, p, t} as keys.
    :param events_t: np array of the uncompressed event times
    :param t0: start time of slice in us
    :param t1: terminal time of slice in us
    :return: (-1, 4) np array
    """
    first_idx = np.searchsorted(events_t, t0, side="left")
    last_idx_p1 = np.searchsorted(events_t, t1, side="right")
    x = np.asarray(events_h5["x"][first_idx:last_idx_p1])
    y = np.asarray(events_h5["y"][first_idx:last_idx_p1])
    p = np.asarray(events_h5["p"][first_idx:last_idx_p1])
    t = np.asarray(events_h5["t"][first_idx:last_idx_p1])
    return {"x": x, "y": y, "p": p, "t": t, "n_events": len(x)}


def events2time_surface(events_h5, events_t, t0, t1, resolution):
    """
    Build a timesurface from events in temporal range [t0, t1)
    :param events_h5: h5 object with events. {x, y, p, t} as keys.
    :param events_t: np array of the uncompressed event times
    :param t0: start time of slice in us
    :param t1: terminal time of slice in us
    :param resolution: 2-element tuple (W, H)
    :return: (H, W) np array
    """
    time_surface = np.zeros((resolution[1], resolution[0]), dtype=np.float64)
    events_dict = query_events(events_h5, events_t, t0, t1)

    for i in range(events_dict["n_events"]):
        x = int(np.rint(events_dict["x"][i]))
        y = int(np.rint(events_dict["y"][i]))

        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
            time_surface[y, x] = (events_dict["t"][i] - t0) / (t1 - t0)

    return time_surface


def read_input(input_path, representation):
    input_path = str(input_path)

    assert os.path.exists(input_path), f"Path to input file {input_path} doesn't exist."
    
    if "time_surfaces_accumulate" in representation:
        return h5py.File(input_path, "r")["time_surface_accumulate"]
    
    elif "event_stack_v1_5" in representation:
        return h5py.File(input_path, "r")["time_surface"]
    
    elif "count_images" in representation:
        return h5py.File(input_path, "r")["count_images"]
    
    elif "time_surface" in representation:
        return h5py.File(input_path, "r")["time_surface"]

    elif "voxel" in representation:
        return h5py.File(input_path, "r")["voxel_grid"]

    elif "event_stack" in representation:
        return h5py.File(input_path, "r")["event_stack"]
    
    elif "time_order_surface" in representation:
        return h5py.File(input_path, "r")["time_order_surface"]
    
    elif "time_surface" in representation:
        return h5py.File(input_path, "r")["time_surface"]
    
    elif "sobel" in representation:
        return h5py.File(input_path, "r")["sobel"]

    elif "grayscale" in representation:
        return imread(input_path, IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    else:
        print("Unsupported representation")
        exit()


def propagate_keys(cfg, testing=False):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.data.representation = cfg.representation
        cfg.data.track_name = cfg.track_name
        cfg.data.patch_size = cfg.patch_size

        cfg.model.representation = cfg.representation
        cfg.data.patch_size = cfg.patch_size

        if not testing:
            cfg.model.n_vis = cfg.n_vis
            cfg.model.init_unrolls = cfg.init_unrolls
            cfg.model.max_unrolls = cfg.max_unrolls
            cfg.model.debug = cfg.debug

        cfg.model.pose_mode = cfg.data.name == "pose"


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def read_events(input_seq_path, start_time = 0.2, end_time = 0.5, generate_video = True):
    """
    Read events from a sequence file.
    param input_seq_path: Path to the sequence file.
    param start_time: Start time(s) of the events to read.
    param end_time: End time(s) of the events to read.
    return: A tuple of (x, y, p, t) arrays.
    """
    global IMG_H, IMG_W
    input_seq_path = str(input_seq_path)
    file_extension = input_seq_path.split(".")[-1]
    file_name = os.path.splitext(os.path.basename(input_seq_path))[0]
    
    if generate_video:
        video_frames = []
        frame_timestamps = []
    
    if file_extension == "hdf5" or file_extension == "h5":
        
        with h5py.File(str(input_seq_path), "r") as h5f:
            if "CD" in h5f: 
                h5f = h5f["CD"]["events"]
            x = np.asarray(h5f["x"])
            y = np.asarray(h5f["y"])
            p = np.asarray(h5f["p"])
            t = np.asarray(h5f["t"])
            t_end = t[-1]
            if start_time > 0:
                start_time *= 1e6
                start_idx = np.searchsorted(t, start_time, side="left")
                x = x[start_idx:]
                y = y[start_idx:]
                p = p[start_idx:]
                t = t[start_idx:]
            if end_time is not None:
                end_time *= 1e6
                end_idx = np.searchsorted(t, t_end - end_time, side="right")
                x = x[:end_idx]
                y = y[:end_idx]
                p = p[:end_idx]
                t = t[:end_idx]
            assert len(x) > 0, "have no event"
            idxs_sorted = np.argsort(t)
            x, y, p, time = (
                np.asarray(x)[idxs_sorted],
                np.asarray(y)[idxs_sorted],
                np.asarray(p)[idxs_sorted],
                np.asarray(t)[idxs_sorted],
            )
            return x, y, p, time
    elif file_extension == "aedat4":
        events_packets = []
        start_time *= 1e6
        end_time *= 1e6
        reader = dv.io.MonoCameraRecording(input_seq_path)
        
        # Check if event stream and frame stream is available
        if reader.isEventStreamAvailable() and reader.isFrameStreamAvailable():
            # Check the resolution of event stream
            resolution = reader.getEventResolution()
            IMG_H = resolution[1]
            IMG_W = resolution[0]
            get_firstime = False
            time_start = 0
            video_frames = []
            image_timestamp = []
            
            # Read events and frame
            while True:
                events = reader.getNextEventBatch()
                frame = reader.getNextFrame()
                if not get_firstime:
                    time_start = events.numpy()[0][0]
                    get_firstime = True
                if events is None and frame is None:
                    break
                if events is not None:
                    events_packets.append(pd.DataFrame(events.numpy()))
                if frame is not None:
                    if frame.timestamp - time_start > start_time:
                        if generate_video:
                            video_frames.append(frame.image)
                            frame_timestamps.append((frame.timestamp - time_start)*1e-6)
                        
                        image_path = str(os.path.dirname(input_seq_path) / Path(file_name) / "images"
                                        / ("%.6f" % ((frame.timestamp - time_start)*1e-6) + ".png"))
                        cv2.imwrite(image_path, frame.image)
                        image_timestamp.append("%.6f" % ((frame.timestamp - time_start)*1e-6))

            if generate_video and len(video_frames) > 0:
                fps = int(round(1 / (frame_timestamps[1] - frame_timestamps[0])))
                video_writer = cv2.VideoWriter(str(os.path.dirname(input_seq_path) / Path(file_name) / "video.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
                
                for frame in video_frames:
                    video_writer.write(frame)
                video_writer.release()
            
            np.savetxt(str(os.path.dirname(input_seq_path) / Path(file_name) / "images.txt"), np.array(image_timestamp), fmt='%s')
            events_all = np.array(pd.concat(events_packets))
            x = events_all[:, 1]
            y = events_all[:, 2]
            p = events_all[:, 3]
            t = events_all[:, 0]
            t -= t[0]
            t_end = t[-1]
            start_time = float(image_timestamp[0])*1e6
            end_time = float(image_timestamp[-1])*1e6
            if start_time > 0:
                start_idx = np.searchsorted(t, start_time, side="left")
                x = x[start_idx:]
                y = y[start_idx:]
                p = p[start_idx:]
                t = t[start_idx:]
            if end_time is not None:
                end_idx = np.searchsorted(t, end_time, side="right")
                x = x[:end_idx]
                y = y[:end_idx]
                p = p[:end_idx]
                t = t[:end_idx]
            assert len(x) > 0, "have no event"
            idxs_sorted = np.argsort(t)
            x, y, p, time = (
                np.asarray(x)[idxs_sorted],
                np.asarray(y)[idxs_sorted],
                np.asarray(p)[idxs_sorted],
                np.asarray(t)[idxs_sorted],
            )
            time[0] = round(start_time)
            return x, y, p, time
    elif file_extension == "txt":
        events = np.loadtxt(input_seq_path)
        t = events[:, 0]
        x = events[:, 1]
        y = events[:, 2]
        p = events[:, 3]
        t *= 1e6
        t_end = t[-1]
        if start_time > 0:
            start_time *= 1e6
            start_idx = np.searchsorted(t, start_time, side="left")
            x = x[start_idx:]
            y = y[start_idx:]
            p = p[start_idx:]
            t = t[start_idx:]
        if end_time is not None:
            end_time *= 1e6
            end_idx = np.searchsorted(t, t_end - end_time, side="right")
            x = x[:end_idx]
            y = y[:end_idx]
            p = p[:end_idx]
            t = t[:end_idx]
        assert len(x) > 0, "have no event"
        idxs_sorted = np.argsort(t)
        x, y, p, time = (
            np.asarray(x)[idxs_sorted],
            np.asarray(y)[idxs_sorted],
            np.asarray(p)[idxs_sorted],
            np.asarray(t)[idxs_sorted],
        )
        return x, y, p, time
    else:
        raise NotImplementedError(f"No read function for {file_extension}")
