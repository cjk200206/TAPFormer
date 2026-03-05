import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from typing import Callable
from itertools import repeat
from functools import partial
from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp
from LFE_TAP.utils.model_utils import combine_tokens, recover_tokens, bilinear_sampler


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

to_2tuple = _ntuple(2)
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=dilation, dilation=dilation, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (in_planes == planes and stride == 1):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (in_planes == planes and stride == 1):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (in_planes == planes and stride == 1):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (in_planes == planes and stride == 1):
                self.norm3 = nn.Sequential()

        if in_planes == planes and stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
    

class BasicEncoder(nn.Module):
    def __init__(
        self, input_dim=3, output_dim=128, stride=8, norm_fn="instance", dropout=0.0, shallow=False, in_planes=64, dilation=1,
    ):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = in_planes

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = shallow
        if self.shallow:
            # self.layer1 = ResidualBlock(self.in_planes, 64, norm_fn=self.norm_fn, stride=1)
            # self.layer2 = ResidualBlock(64, 96, self.norm_fn, stride=2)
            # self.layer3 = ResidualBlock(96, 128, self.norm_fn, stride=2)
            self.layer1 = self._make_layer(64, stride=1, dilation=dilation)
            self.layer2 = self._make_layer(96, stride=2, dilation=dilation)
            self.layer3 = self._make_layer(128, stride=2, dilation=dilation)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)

            self.conv2 = nn.Conv2d(
                128 + 128 + 96 + 64,
                output_dim * 2,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            )
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1, dilation=dilation)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            d = F.interpolate(
                d,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c, d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x


class FusionBlock_basic(nn.Module):
    def __init__(self, img_in_dim=3, event_in_dim=10, output_dim=128, stride=8, norm_fn="instance", dropout=0.0):
        super().__init__()
        self.imgnet = BasicEncoder(input_dim=img_in_dim, output_dim=output_dim, stride=stride, norm_fn=norm_fn, dropout=dropout)
        self.eventnet = BasicEncoder(input_dim=event_in_dim, output_dim=output_dim, stride=stride, norm_fn=norm_fn, dropout=dropout)
        self.conv1 = nn.Conv2d(output_dim, 192, 1, padding=0)
        self.conv2 = nn.Conv2d(output_dim, 192, 1, padding=0)
        self.convo = nn.Conv2d(192*2, output_dim, 3, padding=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                
    def forward(self, x_i, x_e, _):
        x_i = self.imgnet(x_i)
        x_e = self.eventnet(x_e)
        c1 = F.relu(self.conv1(x_i))
        c2 = F.relu(self.conv2(x_e))
        out = torch.cat([c1, c2], dim=1)
        out = F.relu(self.convo(out))
        return x_i + out
    
class FusionBlock(nn.Module):
    def __init__(self, img_in_dim=3, event_in_dim=10, output_dim=128, stride=8, norm_fn="instance", dropout=0.0):
        super(FusionBlock, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = 32
        
        if self.norm_fn == "group":
            self.norm1_i = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm1_e = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)

        elif self.norm_fn == "batch":
            self.norm1_i = nn.BatchNorm2d(self.in_planes)
            self.norm1_e = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)

        elif self.norm_fn == "instance":
            self.norm1_i = nn.InstanceNorm2d(self.in_planes)
            self.norm1_e = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == "none":
            self.norm1_i = nn.Sequential()
            self.norm1_e = nn.Sequential()
            
        self.conv1_i = nn.Conv2d(img_in_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode="zeros")
        self.conv1_e = nn.Conv2d(event_in_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode="zeros")
        self.relu1 = nn.ReLU(inplace=True)
        
        self.shallow = False
        if self.shallow:
            self.layer1_e = self._make_layer(self.in_planes, 64, stride=1)
            self.layer2_e = self._make_layer(64, 96, stride=2)
            self.layer3_e = self._make_layer(96, 128, stride=2)
            # self.conv_half1 = self._half_conv(self.in_planes*2)
            # self.conv_half2 = self._half_conv(64*2)
            # self.conv_half3 = self._half_conv(96*2)
            self.layer1_i = self._make_layer(self.in_planes*2, 64, stride=1)
            self.layer2_i = self._make_layer(64*2, 96, stride=2)
            self.layer3_i = self._make_layer(96*2, 128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            self.layer1_e = self._make_layer(self.in_planes, 64, stride=1)
            self.layer2_e = self._make_layer(64, 96, stride=2)
            self.layer3_e = self._make_layer(96, 128, stride=2)
            self.layer4_e = self._make_layer(128, 128, stride=2)
            # self.conv_half1 = self._half_conv(self.in_planes*2)
            # self.conv_half2 = self._half_conv(64*2)
            # self.conv_half3 = self._half_conv(96*2)
            # self.conv_half4 = self._half_conv(128*2)
            self.layer1_i = self._make_layer(self.in_planes*2, 64, stride=1)
            self.layer2_i = self._make_layer(64*2, 96, stride=2)
            self.layer3_i = self._make_layer(96*2, 128, stride=2)
            self.layer4_i = self._make_layer(128*2, 128, stride=2)
            self.conv2 = nn.Conv2d(128 + 128 + 96 + 64, output_dim, kernel_size=3, padding=1, padding_mode="zeros")
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim , output_dim, kernel_size=1)
            
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
            
    def _half_conv(self, in_planes):
        conv = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)
        norm = nn.BatchNorm2d(in_planes // 2)
        relu = nn.ReLU(inplace=True)
        layers = (conv, norm, relu)
        return nn.Sequential(*layers)
        
    def _make_layer(self, input_dim, output_dim, stride=1):
        layer1 = ResidualBlock(input_dim, output_dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(output_dim, output_dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)
    
    def forward(self, x_i, x_e):
        _, _, H, W = x_i.size()
        
        x_i = self.conv1_i(x_i)
        x_e = self.conv1_e(x_e)
        x_i = self.relu1(self.norm1_i(x_i))
        x_e = self.relu1(self.norm1_e(x_e))
        
        if self.shallow:
            x_e_a = self.layer1_e(x_e)
            x_e_b = self.layer2_e(x_e_a)
            x_e_c = self.layer3_e(x_e_b)
            # x_i_a = self.layer1_i(self.conv_half1(torch.cat((x_i, x_e), dim=1)))
            # x_i_b = self.layer2_i(self.conv_half2(torch.cat((x_i_a, x_e_a), dim=1)))
            # x_i_c = self.layer3_i(self.conv_half3(torch.cat((x_i_b, x_e_b), dim=1)))
            x_i_a = self.layer1_i(torch.cat((x_i, x_e), dim=1))
            x_i_b = self.layer2_i(torch.cat((x_i_a, x_e_a), dim=1))
            x_i_c = self.layer3_i(torch.cat((x_i_b, x_e_b), dim=1))
            x_i_a = F.interpolate(x_i_a, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x_i_b = F.interpolate(x_i_b, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x_i_c = F.interpolate(x_i_c, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat((x_i_a, x_i_b, x_i_c), dim=1))
        else:
            x_e_a = self.layer1_e(x_e)
            x_e_b = self.layer2_e(x_e_a)
            x_e_c = self.layer3_e(x_e_b)
            # x_e_d = self.layer4_e(x_e_c)
            # x_i_a = self.layer1_i(self.conv_half1(torch.cat((x_i, x_e), dim=1)))
            # x_i_b = self.layer2_i(self.conv_half2(torch.cat((x_i_a, x_e_a), dim=1)))
            # x_i_c = self.layer3_i(self.conv_half3(torch.cat((x_i_b, x_e_b), dim=1)))
            # x_i_d = self.layer4_i(self.conv_half4(torch.cat((x_i_c, x_e_c), dim=1)))
            x_i_a = self.layer1_i(torch.cat((x_i, x_e), dim=1))
            x_i_b = self.layer2_i(torch.cat((x_i_a, x_e_a), dim=1))
            x_i_c = self.layer3_i(torch.cat((x_i_b, x_e_b), dim=1))
            x_i_d = self.layer4_i(torch.cat((x_i_c, x_e_c), dim=1))
            x_i_a = F.interpolate(x_i_a, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x_i_b = F.interpolate(x_i_b, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x_i_c = F.interpolate(x_i_c, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x_i_d = F.interpolate(x_i_d, size=(H // self.stride, W // self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat((x_i_a, x_i_b, x_i_c, x_i_d), dim=1))
            x = self.relu2(self.norm2(x))
            x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
            
        return x


class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class UpdateFormer(nn.Module):
    def __init__(self, space_depth=12, time_depth=12, input_dim=320, hidden_size=384, num_heads=8, output_dim=130, mlp_ratio=4.0):
        super(UpdateFormer, self).__init__()
        self.hidden_size = hidden_size
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(time_depth)
            ]
        )
        
        self.space_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(space_depth)
            ]
        )
        assert len(self.time_blocks) >= len(self.space_blocks)
        self.initialize_weights()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        
    def forward(self, x):
        x = self.input_transform(x)
        j = 0
        for i in range(len(self.time_blocks)):
            B, N, T, _ = x.shape
            x_time = rearrange(x, "b n t c -> (b n) t c", b=B, t=T, n=N)
            x_time = self.time_blocks[i](x_time)
            
            x = rearrange(x_time, "(b n) t c -> b n t c", b=B, t=T, n=N)
            if self.add_space_attn and (i % (len(self.time_blocks) // len(self.space_blocks)) == 0):
                x_space = rearrange(x, "b n t c -> (b n) t c", b=B, t=T, n=N)
                x_space = self.space_blocks[j](x_space)
                x = rearrange(x_space, "(b n) t c -> b n t c", b=B, t=T, n=N)
                j += 1
                
        flow = self.flow_head(x)
        return flow


class Attention2(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        sim = (q @ k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            sim = sim + attn_bias
        attn = sim.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)

 
class CrossAttnBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention2(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x


class AttnBlock2(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = Attention2,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.num_heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x

    
class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=32,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock2(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention2,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock2(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention2,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True):
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        layers = []
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )

                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow
    
    
class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 16

        self.pos_embed_x = None

        self.return_inter = False

    def finetune_track(self, img_size):
        new_patch_size = 16

        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = img_size[0] // self.patch_size, img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = img_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        img_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        img_patch_pos_embed = img_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_x = nn.Parameter(img_patch_pos_embed)
        
        # if self.return_inter:
        #     for i_layer in self.fpn_stage:
        #         if i_layer != 11:
        #             norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #             layer = norm_layer(self.embed_dim)
        #             layer_name = f'norm{i_layer}'
        #             self.add_module(layer_name, layer)


    def forward(self, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic HiViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        B = x.shape[0]
    
        x = self.patch_embed(x)
    
        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)

        x = x[..., 0, 0, :]

        x += self.pos_embed_x       # 添加位置编码
        
        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = blk(x)

        aux_dict = {"attn": None}
        x = self.norm_(x)

        return x, aux_dict


class CorrBlock:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        multiple_track_feats=False,
        padding_mode="zeros",
        appearance_fact_flow_dim=None,
    ):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats
        self.appearance_fact_flow_dim = appearance_fact_flow_dim

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out

    def corr(self, targets, coords=None, gt_flow=None, use_gt_flow=False, use_flow_tokens=False,
             use_af_high_dim=False, interaction_network=None):
        assert sum([coords is not None,
            use_gt_flow,
            use_flow_tokens,
            use_af_high_dim,
            interaction_network is not None]) <= 1, \
            "Exactly one of coords, use_gt_flow, use_flow_tokens, use_af_high_dim, or interaction_network must be specified."
        assert not use_flow_tokens or self.appearance_fact_flow_dim is not None

        # Appearance factorization
        if coords is not None:
            B, S, N, D2 = targets.shape
            targets = targets.reshape(B, S, N, 2, D2 // 2)
            if use_gt_flow:
                flow = gt_flow
            else:
                flow = coords[:, 1:] - coords[:, :-1]
                flow = torch.cat([flow[:, 0:1], flow], dim=1)
            flow = gt_flow if use_gt_flow else flow
            targets = flow[..., 0:1] * targets[..., 0, :] + flow[..., 1:2] * targets[..., 1, :]

            ###### DEBUG ########
            # flow: [B, S, N, 2]
            # targets: [B, S, N, 2, feat_dim], remember feat_dim == latent_dim / 2

        # Appearane factorization with flow tokens
        if use_flow_tokens:
            fdim = self.appearance_fact_flow_dim
            flow = targets[..., -fdim:]
            targets = targets[..., :-fdim]
            B, S, N, D2 = targets.shape
            targets = targets.reshape(B, S, N, fdim, D2 // fdim)
            #targets = flow[..., 0:1] * targets[..., 0, :] + flow[..., 1:2] * targets[..., 1, :]
            targets = torch.einsum('btnji,btnj->btni', targets, flow)

        if use_af_high_dim:
            fdim = self.appearance_fact_flow_dim
            flow = targets[..., -fdim:]
            targets = targets[..., :-fdim]
            targets = flow * targets

        #if interaction_network is not None:
        #    flow_feat = targets[...,  :self.appearance_fact_flow_dim]
        #    targets = targets[..., self.appearance_fact_flow_dim:]

        # Correlation
        B, S, N, C = targets.shape
        #assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for _, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)

    def sample_fmap(self, coords):
        '''Sample at coords directly from scaled feature map.
        '''
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):

            fmap = self.fmaps_pyramid[i]  # B, S, C, H, W
            B, S, C, H, W = fmap.shape

            coords_lvl = coords / 2**i

            fmap = fmap.reshape(B * S, C, H, W)
            coords_lvl = coords_lvl.reshape(B * S, N, 2)

            coords_normalized = coords_lvl.clone()
            coords_normalized[..., 0] = coords_normalized[..., 0] / (W - 1) * 2 - 1
            coords_normalized[..., 1] = coords_normalized[..., 1] / (H - 1) * 2 - 1
            coords_normalized = coords_normalized.unsqueeze(1)
            feature_at_coords = F.grid_sample(fmap, coords_normalized,
                                         mode='bilinear', align_corners=True)
            feature_at_coords = feature_at_coords.permute(0, 2, 3, 1).view(B, S, N, C)
            out_pyramid.append(feature_at_coords)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        return out