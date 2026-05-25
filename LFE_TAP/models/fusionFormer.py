import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from LFE_TAP.models.blocks import ResidualBlock, CrossAttnBlock, AttnBlock2, BasicEncoder
from timm.models.vision_transformer import Mlp


class ST_Transformer(nn.Module):
    def __init__(self, dim, heads, mlp_dim=512, dropout=0., mlp_ratio=4.0):
        super().__init__()
        self.event_t = CrossAttnBlock(128, 128, num_heads=8, mlp_ratio=4.0, dim_head=16)
        self.fe_space = CrossAttnBlock(128, 128, num_heads=8, mlp_ratio=4.0, dim_head=16)
        
    def forward(self, x_i, x_e, x_e_pre):
        x_q = self.event_t(x_e, x_e_pre)
        x = self.fe_space(x_q, x_i)
        
        return x, x_q
        
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.layer(x)
    

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cov = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=True)
        x_ = self.layer(x1)
        if x2 is not None:
            x_ = torch.cat([x2, x_], dim=1)
            x_ = self.cov(x_)
        else:
            x_ = self.cov(torch.cat([x_, x_], dim=1))
        return x_
        

class Fusionformer(nn.Module):
    def __init__(self, image_size=(384, 512), out_dim=128, mlp_dim=512, depth=6, stride=8, dropout=0.):
        super().__init__()
        img_h, img_w = image_size
        self.stride = stride
        self.in_planes = 32

        self.fnet_img = BasicEncoder(
            output_dim=128, norm_fn="instance", dropout=0, stride=stride, shallow=True, in_planes=32
        )
        self.fnet_event = BasicEncoder(
            input_dim=10, output_dim=128, norm_fn="instance", dropout=0, stride=stride, shallow=True, in_planes=32, dilation=1
        )
        
        self.transunet = CLWF(128, out_dim, image_size, stride, mlp_dim, depth, dropout)
        
        # self.resnet = ResidualBlock(128, out_dim, stride=1)
        
    def forward(self, x_i, x_e, img_ifnew=None, feature_teacher=None):
        _, _, H, W = x_i.size()
        
        x_i = self.fnet_img(x_i)
        x_e = self.fnet_event(x_e)
        
        x_out = self.transunet(x_i, x_e, img_ifnew, feature_teacher)
        return x_out


class FusionformerFrameAnchor(Fusionformer):
    def __init__(
        self,
        image_size=(384, 512),
        out_dim=128,
        mlp_dim=512,
        depth=6,
        stride=8,
        dropout=0.0,
        anchor_state_mix=0.7,
        anchor_skip_mix=0.7,
    ):
        super().__init__(
            image_size=image_size,
            out_dim=out_dim,
            mlp_dim=mlp_dim,
            depth=depth,
            stride=stride,
            dropout=dropout,
        )
        self.transunet = CLWFFrameAnchor(
            128,
            out_dim,
            image_size=image_size,
            stride=stride,
            mlp_dim=mlp_dim,
            depth=depth,
            dropout=dropout,
            anchor_state_mix=anchor_state_mix,
            anchor_skip_mix=anchor_skip_mix,
        )
        

class CLWF(nn.Module):
    def __init__(self, in_channels, out_channels, image_size=(384, 512), stride=8, mlp_dim=512, depth=3, dropout=0.):
        super().__init__()
        img_h, img_w = image_size
        self.patches_resolution = (img_h // (stride * 4), img_w // (stride * 4))
        num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.down1 = downsample(in_channels, 192)
        self.down2 = downsample(192, 256)

        self.up1 = upsample(256, 192)
        self.up2 = upsample(192, out_channels)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, 256))
        self.dropout_i = nn.Dropout(dropout)
        self.dropout_e = nn.Dropout(dropout)
        
        self.xe_history = []
        self.x_e_pre = None
        self.x_out_ = None
        # self.x_i_last = None
        
        self.cov_out1 = nn.Conv2d(256, 128, kernel_size=1)
        self.cov_out2 = nn.Conv2d(192, 128, kernel_size=1)
        
        self.layers = nn.ModuleList(
            [
                CrossAttnBlock(256, 256, num_heads=8, mlp_ratio=4.0, dim_head=32)
                for _ in range(depth)
            ]
        )
        self.TemporalAdapter = nn.ModuleList(
            [
                AttnBlock2(256, num_heads=8, mlp_ratio=4.0, dim_head=32)
                for _ in range(depth)
            ]
        )
        
    def forward(self, x_i, x_e, img_ifnew=None, feature_teacher=None):
        x1_i = self.down1(x_i)
        x2_i = self.down2(x1_i)
        
        x1_e = self.down1(x_e)
        x2_e = self.down2(x1_e)
        
        x2_i_ = x2_i.clone()
        x2_e_ = x2_e.clone()
        x2_i_ = rearrange(x2_i_, 'b c h w -> b (h w) c')
        x2_e_ = rearrange(x2_e_, 'b c h w -> b (h w) c')
        
        b, n, _ = x2_i_.shape
        
        x2_i_ = self.dropout_i(x2_i_)
        x2_e_ = self.dropout_e(x2_e_)
        
        x_out = []
        for time in range(len(x_i)):
            x_i_t = x2_i_[time:time+1]
            x_e_t = x2_e_[time:time+1]

            if img_ifnew[time] == 1:
                for st_attn in self.layers:
                    x_i_t = st_attn(x_i_t, x_e_t)
                self.x_out_ = x_i_t
            else:
                x_i_t = self.x_out_
                for st_attn in self.layers:
                    x_i_t = st_attn(x_i_t, x_e_t)
                self.x_out_ = x_i_t

            x_out.append(rearrange(self.x_out_, 'b (h w) out_dim -> b out_dim h w', h=self.patches_resolution[0], w=self.patches_resolution[1]))
        
        x_out = torch.cat(x_out, dim=0)
        x_out = rearrange(x_out, 't c h w -> (h w) t c')
        for time_attn in self.TemporalAdapter:
            x_out = time_attn(x_out)
        x_out1 = rearrange(x_out, '(h w) t c -> t c h w', h=self.patches_resolution[0], w=self.patches_resolution[1])
        img_ifnew_mask = torch.as_tensor(img_ifnew, device=x1_e.device, dtype=x1_e.dtype)[:, None, None, None]
        img_ifnew_inv_mask = 1.0 - img_ifnew_mask
        x_out2 = self.up1(x_out1, img_ifnew_mask * x1_i + img_ifnew_inv_mask * x1_e)
        x_out3 = self.up2(x_out2, img_ifnew_mask * x_i + img_ifnew_inv_mask * x_e)
        x_out1 = self.cov_out1(x_out1)
        x_out2 = self.cov_out2(x_out2)
        x_out_pyramid = [x_out3, x_out2, x_out1]
        return x_out_pyramid
    

class CLWFFrameAnchor(CLWF):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_size=(384, 512),
        stride=8,
        mlp_dim=512,
        depth=3,
        dropout=0.0,
        anchor_state_mix=0.7,
        anchor_skip_mix=0.7,
    ):
        super().__init__(
            in_channels,
            out_channels,
            image_size=image_size,
            stride=stride,
            mlp_dim=mlp_dim,
            depth=depth,
            dropout=dropout,
        )
        self.anchor_state_mix = float(anchor_state_mix)
        self.anchor_skip_mix = float(anchor_skip_mix)
        self.x_anchor_ = None
        self.x1_anchor_ = None
        self.x0_anchor_ = None

    @staticmethod
    def _to_img_flags(img_ifnew, length, device, dtype):
        if img_ifnew is None:
            return torch.ones(length, device=device, dtype=dtype)
        if isinstance(img_ifnew, torch.Tensor):
            return img_ifnew.to(device=device, dtype=dtype)
        return torch.as_tensor(img_ifnew, device=device, dtype=dtype)

    @staticmethod
    def _apply_cross_attn(layers, x_base, x_event):
        x = x_base
        for st_attn in layers:
            x = st_attn(x, x_event)
        return x

    def _mix_state(self, dynamic_state, anchor_state):
        if dynamic_state is None:
            return anchor_state
        if anchor_state is None:
            return dynamic_state
        mix = self.anchor_state_mix
        return mix * dynamic_state + (1.0 - mix) * anchor_state

    def _mix_skip(self, frame_anchor, event_feat):
        if frame_anchor is None:
            return event_feat
        mix = self.anchor_skip_mix
        return mix * frame_anchor + (1.0 - mix) * event_feat

    def forward(self, x_i, x_e, img_ifnew=None, feature_teacher=None):
        x1_i = self.down1(x_i)
        x2_i = self.down2(x1_i)

        x1_e = self.down1(x_e)
        x2_e = self.down2(x1_e)

        x2_i_ = rearrange(x2_i.clone(), 'b c h w -> b (h w) c')
        x2_e_ = rearrange(x2_e.clone(), 'b c h w -> b (h w) c')

        x2_i_ = self.dropout_i(x2_i_)
        x2_e_ = self.dropout_e(x2_e_)

        img_flags = self._to_img_flags(img_ifnew, len(x_i), x1_e.device, x1_e.dtype)
        x_out = []
        skip1 = []
        skip0 = []
        for time in range(len(x_i)):
            frame_tokens = x2_i_[time:time + 1]
            event_tokens = x2_e_[time:time + 1]
            if img_flags[time] >= 0.5:
                self.x_anchor_ = frame_tokens
                self.x1_anchor_ = x1_i[time:time + 1]
                self.x0_anchor_ = x_i[time:time + 1]
                self.x_out_ = self._apply_cross_attn(self.layers, frame_tokens, event_tokens)
                skip1.append(x1_i[time])
                skip0.append(x_i[time])
            else:
                x_base = self._mix_state(self.x_out_, self.x_anchor_)
                self.x_out_ = self._apply_cross_attn(self.layers, x_base, event_tokens)
                skip1_anchor = None if self.x1_anchor_ is None else self.x1_anchor_[0]
                skip0_anchor = None if self.x0_anchor_ is None else self.x0_anchor_[0]
                skip1.append(self._mix_skip(skip1_anchor, x1_e[time]))
                skip0.append(self._mix_skip(skip0_anchor, x_e[time]))

            x_out.append(
                rearrange(
                    self.x_out_,
                    'b (h w) out_dim -> b out_dim h w',
                    h=self.patches_resolution[0],
                    w=self.patches_resolution[1],
                )
            )

        x_out = torch.cat(x_out, dim=0)
        x_out = rearrange(x_out, 't c h w -> (h w) t c')
        for time_attn in self.TemporalAdapter:
            x_out = time_attn(x_out)
        x_out1 = rearrange(x_out, '(h w) t c -> t c h w', h=self.patches_resolution[0], w=self.patches_resolution[1])
        x_out2 = self.up1(x_out1, torch.stack(skip1, dim=0))
        x_out3 = self.up2(x_out2, torch.stack(skip0, dim=0))
        x_out1 = self.cov_out1(x_out1)
        x_out2 = self.cov_out2(x_out2)
        return [x_out3, x_out2, x_out1]

class Unet_Transformer(nn.Module):
    def __init__(self, input_dim=3, image_size=(384, 512), out_dim=128, mlp_dim=512, depth=6, stride=8, dropout=0.):
        super().__init__()
        img_h, img_w = image_size
        self.stride = stride
        self.in_planes = 32

        self.fnet = BasicEncoder(
            input_dim=input_dim, output_dim=128, norm_fn="instance", dropout=0, stride=stride, shallow=True, in_planes=32
        )
        
        self.transunet = TransUnet_pyramid_onemod(128, out_dim, image_size, stride, mlp_dim, depth, dropout)
        
        # self.resnet = ResidualBlock(128, out_dim, stride=1)
        
    def forward(self, x, feature_teacher=None):
        _, _, H, W = x.size()
        
        x = self.fnet(x)
        
        x_out = self.transunet(x, feature_teacher)
        
        # x_out = self.resnet(x_out)
            
        return x_out
    

class TransUnet_pyramid_onemod(nn.Module):
    def __init__(self, in_channels, out_channels, image_size=(384, 512), stride=8, mlp_dim=512, depth=3, dropout=0.):
        super().__init__()
        img_h, img_w = image_size
        self.patches_resolution = (img_h // (stride * 4), img_w // (stride * 4))
        num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.down1 = downsample(in_channels, 192)
        self.down2 = downsample(192, 256)

        self.up1 = upsample(256, 192)
        self.up2 = upsample(192, out_channels)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, 256))
        self.dropout_i = nn.Dropout(dropout)
        self.dropout_e = nn.Dropout(dropout)
        
        self.xe_history = []
        self.x_e_pre = None
        self.x_out_ = None
        
        self.cov_out1 = nn.Conv2d(256, 128, kernel_size=1)
        self.cov_out2 = nn.Conv2d(192, 128, kernel_size=1)
        

        self.TemporalAdapter = nn.ModuleList(
            [
                AttnBlock2(256, num_heads=8, mlp_ratio=4.0, dim_head=32)
                for _ in range(depth)
            ]
        )
        
    def forward(self, x, feature_teacher=None):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        
        x2_ = x2.clone()
        x2_ = rearrange(x2_, 'b c h w -> b (h w) c')
        
        b, n, _ = x2_.shape
        
        x2_ = self.dropout_i(x2_)
        
        x2_ = x2_.permute(1,0,2)
        
        for time_attn in self.TemporalAdapter:
            x_out = time_attn(x2_)
        x_out1 = rearrange(x_out, '(h w) t c -> t c h w', h=self.patches_resolution[0], w=self.patches_resolution[1])
        x_out2 = self.up1(x_out1, x1)
        x_out3 = self.up2(x_out2, x)
        x_out1 = self.cov_out1(x_out1)
        x_out2 = self.cov_out2(x_out2)
        x_out_pyramid = [x_out3, x_out2, x_out1]
        
        
        return x_out_pyramid