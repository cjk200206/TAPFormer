import os
import torch
import imageio
import re
import numpy as np
from LFE_TAP.utils.event.utils import read_input
from LFE_TAP.utils.dataset_utils import FrameEventData, FrameEventData_test

class TAPFormer_dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, dt=0.020, representation="time_surfaces_v2_5", fix_num=None, with_gt=True):
        self.data_root = data_root
        self.dt = dt
        self.representation = representation
        self.fix_num = fix_num
        self.with_gt = with_gt
        self.seq_names = [
            fname
            for fname in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, fname))
        ]
        print("found %d unique seqences in %s" % (len(self.seq_names), self.data_root))

    def __getitem__(self, index):
        gotit =False
        if self.fix_num is not None:
            event_dir_path = os.path.join(str(self.data_root), self.seq_names[index], "events", self.representation, f"fix_num_{self.fix_num}")
        else:
            event_dir_path = os.path.join(str(self.data_root), self.seq_names[index], "events", self.representation, f"{self.dt:.4f}")
        rgb_path = os.path.join(str(self.data_root), self.seq_names[index], "images_corrected")
        track_point_path = os.path.join(str(self.data_root), self.seq_names[index], "annotations.npy")
        img_time_full = np.stack(np.loadtxt(os.path.join(str(self.data_root), self.seq_names[index], "image_timestamps.txt")))

        img_paths = sorted(os.listdir(rgb_path), key=lambda x: int(re.search(r'\d+', x).group()))
        event_paths = sorted(os.listdir(event_dir_path), key=lambda x: int(re.search(r'\d+', x).group()))
        rgb_imgs = []
        rgb_ifnew = []
        rgb_ifnew_full = []
        rgb_times = []
        rgb_imgs_plus = []
        rgb_ind = 0
        rgb_ind_full = 0
        event_imgs = []
        event_time = []
        time_emmbed = []
        for i, img_path in enumerate(img_paths):
            try:
                img = imageio.v2.imread(os.path.join(rgb_path, img_path))
                if len(img.shape) == 2:
                    img = img[:,:,np.newaxis]
                    rgb_imgs.append(img.repeat(3, axis=2))
                else:
                    rgb_imgs.append(img)
                rgb_times.append(int(re.match(r"\d+", img_path).group()))
            except Exception as e:
                print(f"error reading image at path:{img_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False
        # rgb_imgs_plus.append(rgb_imgs[0])
        # event_time.append(rgb_times[0])
        # event_imgs.append(read_input(os.path.join(self.data_root, self.seq_names[index], "events", "template", self.event_template, str(event_time[0])+".h5"), self.event_template))
        # rgb_ifnew.append(1)
        # rgb_ind += 1
        for i, event_path in enumerate(event_paths):
            try:
                if int(event_path.split('.')[0]) < rgb_times[0]:
                    continue
                event_imgs.append(read_input(os.path.join(event_dir_path, event_path), self.representation))
                # event_time.append(int(event_path.split('.')[0]))
                rgb_time = rgb_times[min(rgb_ind, len(rgb_times)-1)]   
                if int(event_path.split('.')[0]) >= rgb_time:
                    rgb_imgs_plus.append(rgb_imgs[min(rgb_ind, len(rgb_times)-1)])
                    event_time.append(rgb_times[min(rgb_ind, len(rgb_times)-1)])
                    # time_emmbed.append(round((int(event_path.split('.')[0]) - rgb_times[min(rgb_ind, len(rgb_times)-1)])*1e-4, 1))
                    rgb_ind += 1
                    rgb_ifnew.append(1)
                else:
                    rgb_imgs_plus.append(rgb_imgs[max(rgb_ind-1,0)])
                    event_time.append(int(event_path.split('.')[0]))
                    # time_emmbed.append(round((int(event_path.split('.')[0]) - rgb_times[max(rgb_ind-1,0)])*1e-4, 1))
                    rgb_ifnew.append(0)
                
                    
                rgb_time_full = img_time_full[rgb_ind_full]
                if int(event_path.split('.')[0]) >= rgb_time_full:
                    rgb_ind_full += 1
                    rgb_ifnew_full.append(1)
                else:
                    rgb_ifnew_full.append(0)
                    
            except Exception as e:
                print(f"error reading event at path:{event_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False

        rgb_imgs = np.stack(rgb_imgs_plus).transpose(0, 3, 1, 2)
        event_imgs = np.asarray(event_imgs).transpose(0, 3, 1, 2)
        # event_imgs = np.stack(event_imgs).transpose(0, 3, 1, 2)
        if self.with_gt:
            annot_dict = np.load(track_point_path, allow_pickle=True).item()
            traj_2d = annot_dict["coords"]              # N, T, 2
            visibility = annot_dict["visibility"]       # N, T， 1
        
            traj_data = np.transpose(traj_2d, (1, 0, 2))                      # N, T, 2 -> T, N, 2
            visibility = np.transpose(np.logical_not(np.squeeze(visibility)), (1, 0))   # N, T -> T, N
            query_points = traj_data[0]

        T, H, W, C = event_imgs.shape
        
        segs = np.array(event_time)
        rgb_timestamp = np.array(rgb_times)
        img_ifnew = np.array(rgb_ifnew)
        img_ifnew_full = np.array(rgb_ifnew_full)
        rgb_times = np.array(rgb_times) * 1e-6
        rgb_time_full = np.array(img_time_full) * 1e-6
        if self.with_gt:
            traj_data = np.concatenate([rgb_time_full[:, np.newaxis, np.newaxis].repeat(len(query_points), axis=1), traj_data], axis=2)
            
            query_points = torch.from_numpy(query_points).float()
            query_points = torch.cat([torch.zeros_like(query_points[:, :1]), query_points], dim=1)
        else:
            query_points = None
            traj_data = None
            visibility = None
        gotit = True

        sample = FrameEventData_test(
            rgb_imgs,
            event_imgs,
            segs,
            traj_data,
            visibility=visibility,
            seq_name=self.seq_names[index],
            query_points=query_points,
            img_ifnew=img_ifnew,
            img_ifnew_full=img_ifnew_full,
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