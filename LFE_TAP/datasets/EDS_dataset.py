import os
import torch
import imageio
import re
import numpy as np
import torch.nn.functional as F
from LFE_TAP.utils.event.utils import read_input
from LFE_TAP.utils.dataset_utils import FrameEventData, FrameEventData_test

class EDS_dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, dt=0.0050, representation="time_surfaces_v2_5"):
        self.data_root = data_root
        self.dt = dt
        self.representation = representation
        self.seq_names = [
            fname
            for fname in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, fname))
        ]

    def __getitem__(self, index):
        gotit =False
        event_dir_path = os.path.join(str(self.data_root), self.seq_names[index], "events", f"{self.dt:.4f}", self.representation)
        rgb_path = os.path.join(str(self.data_root), self.seq_names[index], "images_corrected")
        track_point_path = os.path.join(f"gt_tracks/{self.seq_names[index]}.gt.txt")

        img_paths = sorted(os.listdir(rgb_path))
        # img_paths = img_paths[::8]
        event_paths = sorted([f for f in os.listdir(event_dir_path) if f.endswith('.h5')])
        rgb_imgs = []
        rgb_ifnew = []
        rgb_times = []
        rgb_imgs_plus = []
        rgb_ind = 0
        event_imgs = []
        event_time = []
        time_emmbed = []
        for i, img_path in enumerate(img_paths):
            try:
                # gray
                img = imageio.v2.imread(os.path.join(rgb_path, img_path)).reshape(480, 640, 1)
                img = img.repeat(3, axis=2)
                # rgb
                # img = imageio.v2.imread(os.path.join(rgb_path, img_path))
                
                rgb_imgs.append(img)
                rgb_times.append(int(re.match(r"\d+", img_path).group()))
            except Exception as e:
                print(f"error reading image at path:{img_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False
        for i, event_path in enumerate(event_paths):
            try:
                event_imgs.append(read_input(os.path.join(event_dir_path, event_path), self.representation))
                rgb_time = rgb_times[rgb_ind] if rgb_ind < len(rgb_times) else float('inf')
                if int(event_path.split('.')[0]) >= rgb_time:
                    # event_imgs.append(read_input(os.path.join(event_dir_path, event_path), self.representation))
                    event_time.append(rgb_times[rgb_ind])
                    rgb_imgs_plus.append(rgb_imgs[rgb_ind])
                    time_emmbed.append(round((int(event_path.split('.')[0]) - rgb_times[rgb_ind])*1e-4, 1))
                    rgb_ind += 1
                    rgb_ifnew.append(1)
                else:
                    # continue
                    event_time.append(int(event_path.split('.')[0]))
                    rgb_imgs_plus.append(rgb_imgs[max(rgb_ind-1,0)])
                    time_emmbed.append(round((int(event_path.split('.')[0]) - rgb_times[max(rgb_ind-1,0)])*1e-4, 1))
                    rgb_ifnew.append(0)
            except Exception as e:
                print(f"error reading event at path:{event_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False

        rgb_imgs = np.stack(rgb_imgs_plus).transpose(0, 3, 1, 2)
        event_imgs = np.stack(event_imgs)
        if self.representation != "event_stack":
            event_imgs = event_imgs.transpose(0, 3, 1, 2)
        traj_data = np.genfromtxt(track_point_path, delimiter=" ")       # id, t, x, y
        track_num, track_ind = np.unique(traj_data[:, 0], return_index=True)   
        query_points = traj_data[track_ind, 2:]
        
        T, C, _, _ = event_imgs.shape

        segs = np.array(event_time)
        rgb_timestamp = np.array(rgb_times)
        img_ifnew = np.array(rgb_ifnew)
        query_points = torch.from_numpy(query_points).float()
        query_points = torch.cat([torch.zeros_like(query_points[:, :1]), query_points], dim=1)
        gotit = True

        sample = FrameEventData_test(
            rgb_imgs,
            event_imgs,
            segs,
            traj_data,
            seq_name=self.seq_names[index],
            query_points=query_points,
            img_ifnew=img_ifnew,
            rgb_timestamp=rgb_timestamp,
        )

        return sample, gotit
    
    def get_a_seq(self, seq_name):
        for i in range(len(self.seq_names)):
            if self.seq_names[i] == seq_name:
                sample, gotit = self.__getitem__(i)
                return sample, gotit
        
        print("WARNNING: did not find the sequence", seq_name)
        return [], False

    def __len__(self):
        return len(self.seq_names)