import os
import cv2
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import dv_processing as dv

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

if __name__ == '__main__':
    # Define the path to the AEDAT4.2 file
    base_dir = '/home/ljx/code/FE-TAP2/data/ori_data'
    file_name = 'indoor_fruit_dark'
    aedat_file = os.path.join(base_dir, file_name, f"{file_name}.aedat4")
    
    # Define the output folder
    output_folder = Path(os.path.join(base_dir, file_name))
    os.makedirs(output_folder, exist_ok=True)
    frame_timestamps = []
    
    events_packets = []
    events_undistorted_packets = []
    reader = dv.io.MonoCameraRecording(aedat_file)
    
    # Check if event stream and frame stream is available
    if reader.isEventStreamAvailable() and reader.isFrameStreamAvailable():
        # Check the resolution of event stream
        resolution = reader.getEventResolution()
        IMG_H = resolution[1]
        IMG_W = resolution[0]
        get_firstime = False
        time_start = 0
        video_frames = []
        video_undistorted_frames = []
        image_timestamp = []
        image_timestamp_undistorted = []
        j1 = 0
        j2 = 0   
        
        # Read events and frame
        while True:
            events = reader.getNextEventBatch('events')
            frame = reader.getNextFrame('frames')
            events_undistorted = reader.getNextEventBatch('undistortedEvents')
            frame_undistorted = reader.getNextFrame('undistortedFrames')
            if not get_firstime:
                time_start = events.numpy()[0][0]
                get_firstime = True
            if events is None and frame is None and events_undistorted is None and frame_undistorted is None:
                break
            if events is not None:
                events_packets.append(pd.DataFrame(events.numpy()))
            if events_undistorted is not None:
                events_undistorted_packets.append(pd.DataFrame(events_undistorted.numpy()))
        #     if frame is not None:
        #         if frame.timestamp - time_start >= 0:
        #             j1 += 1
        #             video_frames.append(frame.image)
        #             frame_timestamps.append(frame.timestamp - time_start)
        #             os.makedirs(output_folder / "images", exist_ok=True)
        #             image_path = str(output_folder / "images"
        #                             / ("%04d" % (j1) + ".png"))
        #             cv2.imwrite(image_path, frame.image)
        #             image_timestamp.append("%d" % (frame.timestamp - time_start))
        #     if frame_undistorted is not None:
        #         if frame_undistorted.timestamp - time_start >= 0:
        #             j2 += 1
        #             video_undistorted_frames.append(frame_undistorted.image)
        #             os.makedirs(output_folder / "undistorted_images", exist_ok=True)
        #             image_path = str(output_folder / "undistorted_images"
        #                             / ("%04d" % (j2) + ".png"))
        #             cv2.imwrite(image_path, frame_undistorted.image)
        #             image_timestamp_undistorted.append("%d" % (frame_undistorted.timestamp - time_start))
            
        # if len(video_frames) > 0:
        #     fps = int(round(1 / ((frame_timestamps[1] - frame_timestamps[0])*1e-6)))
        #     video_writer = cv2.VideoWriter(str(output_folder / "video.mp4"),
        #                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
            
        #     for frame in video_frames:
        #         video_writer.write(frame)
        #     video_writer.release()
            
        # if len(video_undistorted_frames) > 0:
        #     fps = int(round(1 / ((frame_timestamps[1] - frame_timestamps[0])*1e-6)))
        #     video_writer = cv2.VideoWriter(str(output_folder / "undistorted_video.mp4"),
        #                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
            
        #     for frame in video_undistorted_frames:
        #         video_writer.write(frame)
        #     video_writer.release()
        
        # np.savetxt(str(output_folder / "image_timestamps.txt"), np.array(image_timestamp), fmt='%s')
        # np.savetxt(str(output_folder / "image_timestamps_undistorted.txt"), np.array(image_timestamp_undistorted), fmt='%s')
        
        frame_timestamps = np.loadtxt(str(output_folder / "image_timestamps.txt"), dtype=int)
        events_all = np.array(pd.concat(events_packets))
        events_all[:, 0] = events_all[:, 0] - time_start - 20000
        # Write to disk
        with h5py.File(output_folder / "events.h5", "w") as h5f_out:
            h5f_out.create_dataset(
                "events",
                data=events_all,
                shape=events_all.shape,
                dtype=np.int64,
                **blosc_opts(complevel=1, shuffle="byte"),
            )
        
        events_all_undistorted = np.array(pd.concat(events_undistorted_packets))
        events_all_undistorted[:, 0] = events_all_undistorted[:, 0] - time_start -20000
        # Write to disk
        with h5py.File(output_folder / "events_undistorted.h5", "w") as h5f_out:
            h5f_out.create_dataset(
                "events",
                data=events_all_undistorted,
                shape=events_all_undistorted.shape,
                dtype=np.int64,
                **blosc_opts(complevel=1, shuffle="byte"),
            )
        
        event_frames = []
        for t1 in np.arange(events_all[0, 0] + 25000, events_all[-1, 0] + 25000, 25000):
            event_frame = np.zeros((IMG_H, IMG_W), dtype=np.uint64)
            mask_t = np.logical_and(events_all[:, 0] > t1 - 25000, events_all[:, 0] <= t1)
            events_ = events_all[mask_t, :]
            
            if len(events_) < 100:
                continue
            for i in range(len(events_)):
                event_frame[events_[i, 2], events_[i, 1]] = (events_[i, 3]*2 - 1)*events_[i, 0]
            normalized_data = cv2.normalize(event_frame, None, 0, 255, cv2.NORM_MINMAX)
            normalized_data = normalized_data.astype(np.uint8)  # 转换为uint8类型
            # cv2.imshow("event_frame", normalized_data)
            # cv2.waitKey(1)
            event_frames.append(normalized_data)
        
        if len(event_frames) > 0:
            fps = int(round(1 / ((frame_timestamps[1] - frame_timestamps[0])*1e-6)))
            video_writer = cv2.VideoWriter(str(output_folder / "event_frame.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
            
            for frame in event_frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                video_writer.write(frame)
            video_writer.release()
        
        event_frames = []
        for t1 in np.arange(events_all_undistorted[0, 0] + 25000, events_all_undistorted[-1, 0] + 25000, 25000):
            event_frame = np.zeros((IMG_H, IMG_W), dtype=np.uint64)
            mask_t = np.logical_and(events_all_undistorted[:, 0] > t1 - 25000, events_all_undistorted[:, 0] <= t1)
            events_ = events_all_undistorted[mask_t, :]
            
            if len(events_) < 100:
                continue
            for i in range(len(events_)):
                event_frame[events_[i, 2], events_[i, 1]] = (events_[i, 3]*2 - 1)*events_[i, 0]
            normalized_data = cv2.normalize(event_frame, None, 0, 255, cv2.NORM_MINMAX)
            normalized_data = normalized_data.astype(np.uint8)  # 转换为uint8类型
            # cv2.imshow("event_frame", normalized_data)
            # cv2.waitKey(1)
            event_frames.append(normalized_data)
        
        if len(event_frames) > 0:
            fps = int(round(1 / ((frame_timestamps[1] - frame_timestamps[0])*1e-6)))
            video_writer = cv2.VideoWriter(str(output_folder / "undistorted_event_frame.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
            
            for frame in event_frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                video_writer.write(frame)
            video_writer.release()
            
            
        # Generate debug frames
    debug = True
    if debug:
        debug_dir = output_folder / "debug_frames"
        debug_dir.mkdir(exist_ok=True)
        dt = 5000
        n_frames_debug = 50
        image_timestamp_undistorted = np.loadtxt(str(output_folder / "image_timestamps_undistorted.txt"), dtype=int)
        for i in range(370, 470):
            # Events
            t1 = image_timestamp_undistorted[i-1]
            t0 = t1 - dt
            time_mask = np.logical_and(events_all_undistorted[:, 0] >= t0, events_all_undistorted[:, 0] < t1)
            events_slice = events_all_undistorted[time_mask, :]

            on_mask = events_slice[:, 3] == 1
            off_mask = events_slice[:, 3] == 0
            events_slice_on = events_slice[on_mask, :]
            events_slice_off = events_slice[off_mask, :]

            # Image
            img = cv2.imread(str(os.path.join(output_folder, 'undistorted_images', "%04i" % i + ".png")))

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(img, cmap="gray")
            ax.scatter(events_slice_on[:, 1], events_slice_on[:, 2], s=5, c="green")
            ax.scatter(events_slice_off[:, 1], events_slice_off[:, 2], s=5, c="red")
            # plt.show()
            fig.savefig(str(os.path.join(debug_dir , "%04i" % i + ".png")))
            plt.close()
            
        print("Done!")