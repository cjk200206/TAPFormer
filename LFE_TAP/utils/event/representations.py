import torch
import numpy as np
import cv2
from enum import Enum, auto

import hdf5plugin
import h5py


class EventRepresentationTypes(Enum):
    time_surface = 0
    voxel_grid = 1
    event_stack = 2


class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError


class TimeSurface(EventRepresentation):
    def __init__(self, input_size: tuple, p, t, x, y):
        assert len(input_size) == 3
        t = t.astype('float32')
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        self.input_size = input_size
        self.time_surface = torch.zeros(input_size, dtype=torch.float, requires_grad=False)
        self.n_bins = input_size[0] // 2

    def convert(self, events):
        _, H, W = self.time_surface.shape
        with torch.no_grad():
            self.time_surface = torch.zeros(self.input_size, dtype=torch.float, requires_grad=False,
                                            device=events['p'].device)
            time_surface = self.time_surface.clone()

            t = events['t'].cpu().numpy()
            dt_bin = 1. / self.n_bins
            x0 = events['x'].int()
            y0 = events['y'].int()
            p0 = events['p'].int()
            t0 = events['t']

            # iterate over bins
            for i_bin in range(self.n_bins):
                t0_bin = i_bin * dt_bin
                t1_bin = t0_bin + dt_bin

                # mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                # x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
                idx0 = np.searchsorted(t, t0_bin, side='left')
                idx1 = np.searchsorted(t, t1_bin, side='right')
                x_bin = x0[idx0:idx1]
                y_bin = y0[idx0:idx1]
                p_bin = p0[idx0:idx1]
                t_bin = t0[idx0:idx1]

                n_events = len(x_bin)
                for i in range(n_events):
                    if 0 <= x_bin[i] < W and 0 <= y_bin[i] < H:
                        time_surface[2*i_bin+p_bin[i], y_bin[i], x_bin[i]] = t_bin[i]

        return time_surface
    
    
class TimeOrderSurface():
    def __init__(self, input_size: tuple, x, y, p, t):
        assert len(input_size) == 3
        H, W, C = input_size
        self.t = torch.from_numpy(t.astype('int32'))
        self.x = torch.from_numpy(x.astype('int32'))
        self.y = torch.from_numpy(y.astype('int32'))
        self.pol = torch.from_numpy(p.astype('int32'))
        
        mask = (self.x >= 3) & (self.x < W - 3) & (self.y >= 3) & (self.y < H - 3) & (self.t >= 370000) & (self.t <= 900000)
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.pol = self.pol[mask]
        self.t = self.t[mask]
        
        self.index = torch.tensor(0 , device=self.t.device)
        self.input_size = input_size
        self.n_bins = input_size[2] // 2
        self.tos = torch.zeros((input_size[0], input_size[1], 2), dtype=torch.float, requires_grad=False, device=self.t.device)
        self.sae = torch.zeros((input_size[0], input_size[1], 2), dtype=torch.float, requires_grad=False, device=self.t.device)
        self.sae_latest = torch.zeros((input_size[0], input_size[1], 2), dtype=torch.float, requires_grad=False, device=self.t.device)
        self.TOS_bins_001 = torch.zeros(self.input_size, dtype=torch.float, requires_grad=False,
                                        device=self.t.device)
        self.TOS_bins_002 = torch.zeros(self.input_size, dtype=torch.float, requires_grad=False,
                                        device=self.t.device)
        
    def convert(self, time, n_bins, type):
        assert n_bins < self.n_bins
        with torch.no_grad():
            # TOS_bins_001 = self.TOS_bins_001.clone()
            # TOS_bins_002 = self.TOS_bins_002.clone()
            
            while self.index < len(self.t) and self.t[self.index] <= time:
                pol = 1 if self.pol[self.index] else 0
                pol_inv = 0 if self.pol[self.index] else 1
                if ((self.t[self.index] > self.sae_latest[self.y[self.index]][self.x[self.index]][pol] + 20000) or
                        (self.sae_latest[self.y[self.index]][self.x[self.index]][pol_inv] > self.sae_latest[self.y[self.index]][self.x[self.index]][pol])):
                    self.sae_latest[self.y[self.index]][self.x[self.index]][pol] = self.t[self.index]
                    self.sae[self.y[self.index]][self.x[self.index]][pol] = self.t[self.index]
                    self.tos[self.y[self.index] - 3:self.y[self.index] + 4,
                    self.x[self.index] - 3:self.x[self.index] + 4, self.pol[self.index]] -= 1
                    self.tos[self.y[self.index] - 3:self.y[self.index] + 4, self.x[self.index] - 3:self.x[self.index] + 4, self.pol[self.index]][
                        self.tos[self.y[self.index] - 3:self.y[self.index] + 4, self.x[self.index] - 3:self.x[self.index] + 4, self.pol[self.index]] < 241] = 0
                    self.tos[self.y[self.index], self.x[self.index], self.pol[self.index]] = 255
                else:
                    self.sae_latest[self.y[self.index]][self.x[self.index]][pol] = self.t[self.index]

                self.index += 1
            
            # while self.t[self.index] <= 900000 and self.t[self.index] <= time:
            #     for x0 in range(self.x[self.index]-3, self.x[self.index]+4):
            #         for y0 in range(self.y[self.index]-3, self.y[self.index]+4):
            #             if self.tos[y0, x0, self.pol[self.index]] != 0:
            #                 self.tos[y0, x0, self.pol[self.index]] -= 1
            #                 if self.tos[y0, x0, self.pol[self.index]] < 241:
            #                     self.tos[y0, x0, self.pol[self.index]] = 0
            #                 # self.tos[y0, x0, self.pol[self.index]].clamp_(min=0, max=240)  # 使用clamp函数限制值的范围
            #     self.tos[self.y[self.index], self.x[self.index], self.pol[self.index]] = 255
            #     self.index += 1
            
            # time_threshold = 900000
            #
            # # 使用torch.where函数实现条件操作
            # while torch.any(torch.logical_and(self.t[self.index] <= time_threshold, self.t[self.index] <= time)):
            #     y_slice = slice(self.y[self.index]-3, self.y[self.index]+4)
            #     x_slice = slice(self.x[self.index]-3, self.x[self.index]+4)
            #     mask = (self.tos[y_slice, x_slice, self.pol[self.index]] != 0)
            #     self.tos[y_slice, x_slice, self.pol[self.index]].masked_scatter_(mask, self.tos[y_slice, x_slice, self.pol[self.index]] - 1)
            #     self.tos[y_slice, x_slice, self.pol[self.index]] = torch.where(self.tos[y_slice, x_slice, self.pol[self.index]] < 241,
            #                                                                    torch.tensor(0), self.tos[y_slice, x_slice, self.pol[self.index]])
            #     self.tos[self.y[self.index], self.x[self.index], self.pol[self.index]] = 255
            #     self.index += 1

            if type == 0:
                self.TOS_bins_001[:, :, 2*n_bins] = self.tos[:, :, 0]
                self.TOS_bins_001[:, :, 2*n_bins+1] = self.tos[:, :, 1]
            elif type == 1:
                self.TOS_bins_002[:, :, 2*n_bins] = self.tos[:, :, 0]
                self.TOS_bins_002[:, :, 2*n_bins+1] = self.tos[:, :, 1]
                
    def get_time_order_surface(self, type):
        # tos = self.tos.numpy()
        # cv2.imshow("p_tos", tos[:, :, 0])
        # cv2.imshow("n_tos", tos[:, :, 1])
        # cv2.waitKey(0)
        if type == 0:
            return self.TOS_bins_001
        elif type == 1:
            return self.TOS_bins_002
                         

class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.input_size = input_size
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = torch.zeros((self.input_size), dtype=torch.float, requires_grad=False,
                                          device=events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            value = 2*events['p']-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-events['x']).abs()) * (1 - (ylim-events['y']).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


class EventStack(EventRepresentation):
    def __init__(self, input_size: tuple):
        """
        :param input_size: (C, H, W)
        """
        assert len(input_size) == 3
        self.input_size = input_size
        self.event_stack = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert(self, events):
        C, H, W = self.event_stack.shape
        with torch.no_grad():
            self.event_stack = torch.zeros((self.input_size), dtype=torch.float, requires_grad=False,
                                           device=events['p'].device)
            event_stack = self.event_stack.clone()

            t = events['t'].cpu().numpy()
            dt_bin = 1. / self.nb_channels
            x0 = events['x'].int()
            y0 = events['y'].int()
            p0 = 2*events['p'].int()-1
            t0 = events['t']

            # iterate over bins
            for i_bin in range(self.nb_channels):
                t0_bin = i_bin * dt_bin
                t1_bin = t0_bin + dt_bin

                # mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                # x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
                idx0 = np.searchsorted(t, t0_bin, side='left')
                idx1 = np.searchsorted(t, t1_bin, side='right')
                x_bin = x0[idx0:idx1]
                y_bin = y0[idx0:idx1]
                p_bin = p0[idx0:idx1]

                n_events = len(x_bin)
                for i in range(n_events):
                    if 0 <= x_bin[i] < W and 0 <= y_bin[i] < H:
                        event_stack[i_bin, y_bin[i], x_bin[i]] += p_bin[i]

        return event_stack


def events_to_time_surface(time_surface, p, t, x, y):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
    }
    return time_surface.convert(event_data_torch)

def events_to_time_order_surface(time_order_surface, p, t, x, y, dt):
    t = t.astype('float32')
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
    }
    
    time_order_surface.convert(event_data_torch, dt)

def events_to_event_stack(event_stack, p, t, x, y, dt):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
    }
    return event_stack.convert(event_data_torch)


def events_to_voxel_grid(voxel_grid, p, t, x, y):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
    }
    return voxel_grid.convert(event_data_torch)
