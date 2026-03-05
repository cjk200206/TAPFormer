import os
import torch
import imageio
import cv2
from PIL import Image
import numpy as np
from torchvision.transforms import ColorJitter, GaussianBlur
from LFE_TAP.utils.dataset_utils import FrameEventData
from LFE_TAP.utils.event.utils import *


class FETAPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, crop_size=(512, 512), seq_len=24, traj_per_sample=512, use_augs=False, **kwargs,):
        super(FETAPDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.use_augs = use_augs
        
        # photometric augmentation for rgb images
        self.photo_aug = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14
        )
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        
        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25
        
        # photometric augmentation for event images

        
        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

    def getitem_helper(self, index):
        return NotImplementedError

    def __getitem__(self, index):
        gotit = False

        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print("warning: sampling failed")
            # fake sample, so we can still collate
            sample = FrameEventData(
                video=torch.zeros(
                    (self.seq_len, 3, self.crop_size[0], self.crop_size[1])
                ),
                events=torch.zeros((self.seq_len, 10, self.crop_size[0], self.crop_size[1])),
                segmentation=torch.zeros(
                    (self.seq_len, 1, self.crop_size[0], self.crop_size[1])
                ),
                trajectory=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
                visibility=torch.zeros((self.seq_len, self.traj_per_sample)),
                valid=torch.zeros((self.seq_len, self.traj_per_sample)),
            )

        return sample, gotit
        
    def add_photometric_augs(self, rgbs, events, trajs, visibles, eraser=True, replace=True):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T
        
        # 事件图像增加椒盐噪声

        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            events = [event.astype(np.float32) for event in events]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):  # number of times to occlude

                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        dy = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color_rgb = np.mean(
                            rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0
                        )
                        rgbs[i][y0:y1, x0:x1, :] = mean_color_rgb
                        
                        mean_value_event = np.mean(
                            events[i][y0:y1, x0:x1, :].reshape(-1, 10), axis=0
                        )
                        events[i][y0:y1, x0:x1, :] = mean_value_event

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
            events = [event.astype(np.uint8) for event in events]

        if replace:

            rgbs_alt = [
                np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]
            events_alt = [
                np.array(self.blur_aug(torch.from_numpy(event)), dtype=np.uint8)
                for event in events
            ]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            events = [event.astype(np.float32) for event in events]
            events_alt = [event.astype(np.float32) for event in events_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        dy = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep_rgb = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep_rgb
                        rep_event = events_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        events[i][y0:y1, x0:x1, :] = rep_event

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
            events = [event.astype(np.uint8) for event in events]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [
                np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]
            events = [
                np.array(self.blur_aug(torch.from_numpy(event)), dtype=np.uint8)
                for event in events
            ]

        return rgbs, events, trajs, visibles

    def add_spatial_augs(self, rgbs, events, trajs, visibles):
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = events[0].shape[:2]
        assert S == T

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        events = [event.astype(np.float32) for event in events]

        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [
            np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs
        ]
        events = [
            np.pad(event, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for event in events
        ]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        events_scaled = []
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = (
                    scale_delta_x * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
                scale_delta_y = (
                    scale_delta_y * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0] + 10, None)
            W_new = np.clip(W_new, self.crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = (W_new - 1) / float(W - 1)
            scale_y = (H_new - 1) / float(H - 1)

            rgbs_scaled.append(
                cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            )
            events_scaled.append(
                cv2.resize(events[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            )
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y
        rgbs = rgbs_scaled
        events = events_scaled
        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_x = self.crop_size[0]
            mid_y = self.crop_size[1]

        x0 = int(mid_x - self.crop_size[1] // 2)
        y0 = int(mid_y - self.crop_size[0] // 2)

        offset_x = 0
        offset_y = 0

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(
                    -self.max_crop_offset, self.max_crop_offset
                )
                offset_y = np.random.randint(
                    -self.max_crop_offset, self.max_crop_offset
                )
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    * 0.2
                )
                offset_y = int(
                    offset_y * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

            if W_new == self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            events[s] = events[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
                events = [event[:, ::-1] for event in events]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                events = [event[::-1] for event in events]
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]

        return rgbs, events, trajs

    def crop(self, rgbs, events, trajs, clear_rgbs=None):
        T, N, _ = trajs.shape

        S = len(events)
        H, W = events[0].shape[:2]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else (H_new - self.crop_size[0]) // 2
        # np.random.randint(0,
        x0 = 0 if self.crop_size[1] >= W_new else np.random.randint(0, W_new - self.crop_size[1])
        rgbs = [
            rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            for rgb in rgbs
        ]
        events = [
            event[y0: y0 + self.crop_size[0], x0: x0 + self.crop_size[1]]
            for event in events
        ]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0
        if clear_rgbs is not None:
            clear_rgbs = [
                clear_rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
                for clear_rgb in clear_rgbs
            ]
            return rgbs, events, trajs, clear_rgbs

        return rgbs, events, trajs
    
    
class KubricMovifDataset(FETAPDataset):
    def __init__(
        self,
        root_dir,
        representation="time_surfaces_v2_5",
        event_template="sobel",
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=512,
        sample_vis_1st_frame=False,
        choose_long_point=False,
        use_augs=False,
    ):
        super(KubricMovifDataset, self).__init__(
            root_dir=root_dir,
            representation=representation,
            event_template=event_template,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            choose_long_point=choose_long_point,
            use_augs=use_augs,
        )
        self.representation = representation
        self.event_template = event_template
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.choose_long_point = choose_long_point
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.seq_names = [
            fname
            for fname in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, fname))
        ]
        print("found %d unique seqences in %s" % (len(self.seq_names), self.root_dir))
        
    def getitem_helper(self, index):
        gotit = True
        seq_name = self.seq_names[index]
        
        npy_path = os.path.join(self.root_dir, seq_name, seq_name + ".npy")
        rgb_dir_path = os.path.join(self.root_dir, seq_name, "frames")
        event_dir_path = os.path.join(self.root_dir, seq_name, "events", self.representation)
        
        rgb_files = sorted(os.listdir(rgb_dir_path))
        event_files = sorted(os.listdir(event_dir_path))
        rgb_imgs = []
        img_ifnew = []
        event_imgs = []
        event_imgs.append(read_input(os.path.join(self.root_dir, seq_name, "events", "template", self.event_template, "000.h5"), self.event_template))
        
        for i, img_path in enumerate(rgb_files):
            try:
                if i % 3 == 0:
                    rgb_imgs.append(imageio.v2.imread(os.path.join(rgb_dir_path, img_path)))
                    img_ifnew.append(1)
                else:
                    rgb_imgs.append(rgb_imgs[-1])
                    img_ifnew.append(0)
            except Exception as e:
                print(f"error reading image at path:{img_path}_{rgb_dir_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False
                return [], gotit
            
        for i, event_path in enumerate(event_files):
            try:
                event_imgs.append(read_input(os.path.join(event_dir_path, event_path), self.representation))
            except Exception as e:
                print(f"error reading event at path:{event_path}_{event_dir_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False
                return [], gotit
            
        rgbs = np.stack(rgb_imgs)
        events = np.stack(event_imgs)
        annot_dict = np.load(npy_path, allow_pickle=True).item()
        traj_2d = annot_dict["coords"]
        visibility = annot_dict["visibility"]
        
        assert self.seq_len == len(rgbs)
        
        traj_2d = np.transpose(traj_2d, (1, 0, 2))                      # N, T, 2 -> T, N, 2
        visibility = np.transpose(np.logical_not(visibility), (1, 0))   # N, T -> T, N
        if self.use_augs:  
            # rgbs, events, traj_2d, visibility = self.add_photometric_augs(rgbs, events, traj_2d, visibility)      
            rgbs, events, traj_2d = self.add_spatial_augs(rgbs, events, traj_2d, visibility)
        else:
            rgbs, events, traj_2d = self.crop(rgbs, events, traj_2d)
            
        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False
        
        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)
        
        crop_tensor = torch.tensor(self.crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(traj_2d[..., :2] - crop_tensor, dim=-1) < 1000.0,
            dim=0,
        )
        traj_2d = traj_2d[:, close_pts_inds]
        visibility = visibility[:, close_pts_inds]
        
        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)
        
        if self.sample_vis_1st_frame:
            visiblile_pts_inds = visibile_pts_first_frame_inds
        else:
            visiblile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(as_tuple=False)
            visiblile_pts_inds = torch.cat((visibile_pts_first_frame_inds, visiblile_pts_mid_frame_inds), dim=0)
            
        point_inds = torch.randperm(len(visiblile_pts_inds))[:self.traj_per_sample]
        if len(point_inds) < self.traj_per_sample:
            gotit = False
        
        if self.choose_long_point:
            distance = np.linalg.norm(traj_2d[-1, visiblile_pts_inds, :] - traj_2d[0, visiblile_pts_inds, :], axis=-1)[:,0]
            weight = distance / np.sum(distance)
            point_inds = torch.tensor(np.random.choice(len(distance), size=self.traj_per_sample, p=weight))
            visible_inds_sampled = visiblile_pts_inds[point_inds]
            # visible_inds_sampled = visiblile_pts_inds[np.argsort(distance, axis=0)[-self.traj_per_sample:,0]]
        else:
            visible_inds_sampled = visiblile_pts_inds[point_inds]
        
        if len(visible_inds_sampled.shape) == 2:
            visible_inds_sampled = visible_inds_sampled.squeeze(1)
        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valid = torch.ones((self.seq_len, self.traj_per_sample))
        
        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()
        events = torch.from_numpy(np.stack(events)).permute(0, 3, 1, 2).float()
        seqs = torch.ones((self.seq_len, 1, self.crop_size[0], self.crop_size[1]))
        img_ifnew = np.array(img_ifnew)
        sample = FrameEventData(
            video = rgbs,
            events = events,
            segmentation=seqs,
            trajectory=trajs,
            visibility=visibles,
            valid=valid,
            seq_name=seq_name,
            img_ifnew=img_ifnew,
        )
        return sample, gotit
    
    def __len__(self):
        return len(self.seq_names)
    
    
class KubricMovifDataset_new(FETAPDataset):
    def __init__(
        self,
        root_dir,
        root_dir_fast_dataset=None,
        representation="time_surfaces_v2_5",
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=256,
        sample_vis_1st_frame=False,
        choose_long_point=False,
        use_augs=False,
        if_test=False,
    ):
        super(KubricMovifDataset_new, self).__init__(
            root_dir=root_dir,
            root_dir_fast_dataset=root_dir_fast_dataset,
            representation=representation,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            choose_long_point=choose_long_point,
            use_augs=use_augs,
            if_test=if_test,
        )
        self.root_dir1 = os.path.join(self.root_dir, "kubric_ori_dataset1")
        self.root_dir2 = os.path.join(self.root_dir, "kubric_ori_dataset2")
        self.representation = representation
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.choose_long_point = choose_long_point
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.if_test = if_test
        if root_dir_fast_dataset is not None:
            root_dir_fast_dataset = Path(root_dir_fast_dataset)
            seq_names1 = [
                os.path.join(self.root_dir1, fname)
                for fname in os.listdir(self.root_dir1) if os.path.isdir(os.path.join(self.root_dir1, fname))
            ]
            seq_names2 = [
                os.path.join(self.root_dir2, fname)
                for fname in os.listdir(self.root_dir2) if os.path.isdir(os.path.join(self.root_dir2, fname))
            ]
            
            seq_names_fast = [
                os.path.join(root_dir_fast_dataset, fname)
                for fname in os.listdir(root_dir_fast_dataset) if os.path.isdir(os.path.join(root_dir_fast_dataset, fname))
            ]
            self.seq_names = seq_names1 + seq_names2 + seq_names_fast
        else:
            self.seq_names = [
                os.path.join(self.root_dir, fname)
                for fname in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, fname))
            ]
        print("found %d unique seqences in %s" % (len(self.seq_names), self.root_dir))
        
    def getitem_helper(self, index):
        gotit = True
        seq_name = self.seq_names[index]
        
        npy_path = os.path.join(seq_name, os.path.basename(seq_name) + ".npy")
        if not os.path.exists(npy_path):
            print(npy_path)
        rgb_dir_path = os.path.join(seq_name, "blur_frames")
        clear_rgb_dir_path = os.path.join(seq_name, "frames")
        event_dir_path = os.path.join(seq_name, "events", self.representation)
        
        rgb_files = sorted(os.listdir(rgb_dir_path))
        clear_rgb_files = sorted(os.listdir(clear_rgb_dir_path))
        event_files = sorted(os.listdir(event_dir_path))
        rgb_imgs = []
        img_ifnew = []
        clear_rgb_imgs = []
        event_imgs = []
        if not self.if_test:
            random_id = np.random.randint(0, 95-24)
            rgb_files = rgb_files[random_id: random_id+24]
            event_files = event_files[random_id: random_id+24]
            clear_rgb_files = clear_rgb_files[random_id: random_id+24]
        else:
            rgb_files = rgb_files[:-1]
            clear_rgb_files = clear_rgb_files[:-1]
        
        next_insert = 0  # 记录下一个需要插入新图像的索引
        for i, img_path in enumerate(rgb_files):
            try:
                if i == next_insert:
                    # 读取新图像并添加到列表
                    img = imageio.v2.imread(os.path.join(rgb_dir_path, img_path))
                    rgb_imgs.append(img)
                    img_ifnew.append(1)
                    # 随机选择下一个间隔（3或4）
                    if not self.if_test:
                        next_insert += np.random.choice([3, 4])
                    else:
                        next_insert += 4
                else:
                    rgb_imgs.append(rgb_imgs[-1])
                    img_ifnew.append(0)
            except Exception as e:
                print(f"error reading image at path:{img_path}_{rgb_dir_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False
                return [], gotit
            
        for i, img_path in enumerate(clear_rgb_files):
            try:
                clear_rgb_imgs.append(imageio.v2.imread(os.path.join(clear_rgb_dir_path, img_path)))
            except Exception as e:
                print(f"error reading image at path:{img_path}_{clear_rgb_dir_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False
                return [], gotit
            
        for i, event_path in enumerate(event_files):
            try:
                event_imgs.append(read_input(os.path.join(event_dir_path, event_path), self.representation))
            except Exception as e:
                print(f"error reading event at path:{event_path}_{event_dir_path}")
                print(f"error mrssage:{str(e)}")
                gotit = False
                return [], gotit
            
        rgbs = np.stack(rgb_imgs)
        clear_rgbs = np.stack(clear_rgb_imgs)
        events = np.stack(event_imgs)
        annot_dict = np.load(npy_path, allow_pickle=True).item()
        if not self.if_test:
            traj_2d = annot_dict["coords"][:,random_id: random_id+24]
            visibility = annot_dict["visibility"][:,random_id: random_id+24]
        else:
            traj_2d = annot_dict["coords"][:,:-1]
            visibility = annot_dict["visibility"][:,:-1]
        
        assert self.seq_len == len(rgbs) == len(clear_rgbs)
        
        traj_2d = np.transpose(traj_2d, (1, 0, 2))                      # N, T, 2 -> T, N, 2
        visibility = np.transpose(np.logical_not(visibility), (1, 0))   # N, T -> T, N
        if self.use_augs: 
            print("new kubric dataset can't use augs!!!") 
            # rgbs, events, traj_2d, visibility = self.add_photometric_augs(rgbs, events, traj_2d, visibility)      
            rgbs, events, traj_2d = self.add_spatial_augs(rgbs, events, traj_2d, visibility)
        else:
            rgbs, events, traj_2d, clear_rgbs = self.crop(rgbs, events, traj_2d, clear_rgbs=clear_rgbs)
            
        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False 
        
        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)
        
        crop_tensor = torch.tensor(self.crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(traj_2d[..., :2] - crop_tensor, dim=-1) < 1000.0,
            dim=0,
        )
        traj_2d = traj_2d[:, close_pts_inds]
        visibility = visibility[:, close_pts_inds]
        
        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)
        
        if self.sample_vis_1st_frame:
            visiblile_pts_inds = visibile_pts_first_frame_inds
        else:
            visiblile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(as_tuple=False)
            visibile_pts_last_frame_inds = (visibility[self.seq_len - 1]).nonzero(as_tuple=False)
            visiblile_pts_inds = torch.cat((visibile_pts_first_frame_inds, visiblile_pts_mid_frame_inds, visibile_pts_last_frame_inds), dim=0)
            
        point_inds = torch.randperm(len(visiblile_pts_inds))[:self.traj_per_sample]
        if len(point_inds) < self.traj_per_sample and not self.if_test:
            print(seq_name, "get point num ", len(point_inds), "less than ", self.traj_per_sample, " random_id is", random_id)
            gotit = False
            # shutil.rmtree(seq_name)
        
        if self.choose_long_point:
            distance = np.linalg.norm(traj_2d[-1, visiblile_pts_inds, :] - traj_2d[0, visiblile_pts_inds, :], axis=-1)[:,0]
            weight = distance / np.sum(distance)
            point_inds = torch.tensor(np.random.choice(len(distance), size=self.traj_per_sample, p=weight))
            visible_inds_sampled = visiblile_pts_inds[point_inds]
            # visible_inds_sampled = visiblile_pts_inds[np.argsort(distance, axis=0)[-self.traj_per_sample:,0]]
        else:
            visible_inds_sampled = visiblile_pts_inds[point_inds]
        
        if len(visible_inds_sampled.shape) == 2:
            visible_inds_sampled = visible_inds_sampled.squeeze(1)
        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valid = torch.ones((self.seq_len, self.traj_per_sample))
        
        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()
        clear_rgbs = torch.from_numpy(np.stack(clear_rgbs)).permute(0, 3, 1, 2).float()
        events = torch.from_numpy(np.stack(events)).float()
        if "event_stack" not in self.representation:
            events = events.permute(0, 3, 1, 2)
        # events = torch.from_numpy(np.stack(events)).permute(0, 3, 1, 2).float()
        seqs = torch.ones((self.seq_len, 1, self.crop_size[0], self.crop_size[1]))
        img_ifnew = torch.Tensor(img_ifnew)
        sample = FrameEventData(
            video = rgbs,
            events = events,
            segmentation=seqs,
            trajectory=trajs,
            visibility=visibles,
            valid=valid,
            seq_name=seq_name,
            img_ifnew=img_ifnew,
            clear_video = clear_rgbs,
        )
        assert sample.img_ifnew is not None, f"发现空值样本"
        return sample, gotit
    
    def __len__(self):
        return len(self.seq_names)