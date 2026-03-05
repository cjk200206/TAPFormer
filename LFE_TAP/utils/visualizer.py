import os
import numpy as np
import cv2
import imageio
import torch
import flow_vis

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 12,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
        w: float = 2.0
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps
        self.w = w

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        events: torch.Tensor,  # (B,T,C_e,H,W)
        tracks: torch.Tensor,  # (B,N,T,2)
        visibility: torch.Tensor = None,  # (B, N, T, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,N,T,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        video_model: str = "rgb",
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame: int = 0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        if video_model == "events":
            B, T, C_e, H, W = events.shape
            if C_e == 10:
                events[:, :, 1::2, :, :] = -events[:, :, 1::2, :, :]
                max_abs_indices = torch.abs(events).argmax(dim=2, keepdim=True)
                compressed_events = (torch.gather(events, dim=2, index=max_abs_indices) + 1) / 2 * 255.
                # 将第三个维度变为1
                video = compressed_events.repeat(1, 1, 3, 1, 1)
        
        video = F.pad(
                video,
                (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
                "constant",
                255,
            )
        
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
        )
        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

            # Write the video file
            save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame: int = 0,
        compensate_for_camera_motion=False,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        
        # 图像模糊
        # for i in range(1,len(res_video)-2):
        #     res_video[i] = cv2.addWeighted(res_video[i], 0.5, res_video[i+1], 0.5, 0)
        #     res_video[i] = cv2.addWeighted(res_video[i], 0.5, res_video[i-1], 0.5, 0)
        #     res_video[i] = cv2.GaussianBlur(res_video[i], (5, 5), 0)

        vector_colors = np.zeros((T, N, 3))
        if self.mode == "optical_flow":
            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                y_min, y_max = (
                    tracks[query_frame, :, 1].min(),
                    tracks[query_frame, :, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    color = self.color_map(norm(tracks[query_frame, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (
                    tracks[0, segm_mask > 0, 1].min(),
                    tracks[0, segm_mask > 0, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        # color_pred = np.array((0.0, 143.206, 255.0))
        color_pred = np.array((255.0, 0.0, 0))
        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )
                    
                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                    color_pred,
                )
        
        #  draw points
        for t in range(T):
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):

                        cv2.circle(
                            res_video[t],
                            coord,
                            int(self.linewidth * 2),
                            vector_colors[t, i].tolist(),
                            # color_pred,
                            thickness=-1 if visibile else 2
                            -1,
                        )

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        color_pred: list,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape
        W, H, _ = rgb.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** self.w
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] > 0 and coord_y[1] > 0 and coord_x[0] > 0 and coord_x[1] > 0 and coord_y[0] < H and coord_y[1] < W and coord_x[0] < H and coord_x[1] < W:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        # color_pred,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
                # else:
                #     print("out of range", s, "--", i)
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((97.297, 255.0, 0.0))
        # color = np.array((31.0, 165.0, 16.0))
        W, H, _ = rgb.shape


        for t in range(T - 1):
            original = rgb.copy()
            alpha = (t / T) ** self.w
            for i in range(N):
                gt_track = gt_tracks[t][i]
                #  draw a red cross
                if gt_track[0] > 0 and gt_track[1] > 0:
                    coord_y = (int(gt_tracks[t, i, 0]), int(gt_tracks[t, i, 1]))
                    coord_x = (int(gt_tracks[t + 1, i, 0]), int(gt_tracks[t + 1, i, 1]))
                    
                    if coord_y[0] > 0 and coord_y[1] > 0 and coord_x[0] > 0 and coord_x[1] > 0 and coord_y[0] < H and coord_y[1] < W and coord_x[0] < H and coord_x[1] < W:
                        cv2.line(
                            rgb,
                            coord_y,
                            coord_x,
                            color,
                            self.linewidth,
                            cv2.LINE_AA,
                        )
                    
                    # length = int(self.linewidth * 1.5)
                    # coord_y = (int(gt_track[0]) + length, int(gt_track[1]) + length)
                    # coord_x = (int(gt_track[0]) - length, int(gt_track[1]) - length)
                    # cv2.line(
                    #     rgb,
                    #     coord_y,
                    #     coord_x,
                    #     color,
                    #     self.linewidth,
                    #     cv2.LINE_AA,
                    # )
                    # coord_y = (int(gt_track[0]) - length, int(gt_track[1]) + length)
                    # coord_x = (int(gt_track[0]) + length, int(gt_track[1]) - length)
                    # cv2.line(
                    #     rgb,
                    #     coord_y,
                    #     coord_x,
                    #     color,
                    #     self.linewidth,
                    #     cv2.LINE_AA,
                    # )
                    
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb