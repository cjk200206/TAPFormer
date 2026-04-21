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