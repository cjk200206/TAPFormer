import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from LFE_TAP.models.embeddings import get_1d_sincos_pos_embed_from_grid


MODEL_CONFIGS = {
    "vitt": {"encoder": "vit_tiny_patch16_224", "features": 32},
    "vits": {"encoder": "vit_small_patch16_224", "features": 64},
    "vitb": {"encoder": "vit_base_patch16_224", "features": 128},
    "vitl": {"encoder": "vit_large_patch16_224", "features": 256},
}


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C], attention is applied along time for every spatial patch.
        B, T, N, C = x.shape
        if T <= 1:
            return x
        xt = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        attn_out, _ = self.attn(self.norm1(xt), self.norm1(xt), self.norm1(xt), need_weights=False)
        xt = xt + attn_out
        xt = xt + self.mlp(self.norm2(xt))
        return xt.reshape(B, N, T, C).permute(0, 2, 1, 3)


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class FeatureFusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.res1 = ResidualConvUnit(channels)
        self.res2 = ResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, size=None) -> torch.Tensor:
        if skip is not None:
            x = x + self.res1(skip)
        x = self.res2(x)
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTRefineHead(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.projects = nn.ModuleList([nn.Conv2d(embed_dim, out_dim, 1) for _ in range(4)])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.refine = nn.ModuleList([FeatureFusionBlock(out_dim) for _ in range(4)])
        self.output_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1)

    def _tokens_to_map(self, tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

    def forward(self, outputs, patch_h: int, patch_w: int) -> torch.Tensor:
        while len(outputs) < 4:
            outputs.append(outputs[-1])
        outputs = outputs[:4]

        maps = []
        for i, tokens in enumerate(outputs):
            fmap = self._tokens_to_map(tokens, patch_h, patch_w)
            fmap = self.projects[i](fmap)
            maps.append(self.resize_layers[i](fmap))

        # CoW/WAFT-style DPT top-down fusion.
        path4 = self.refine[3](maps[3], size=maps[2].shape[-2:])
        path3 = self.refine[2](path4, maps[2], size=maps[1].shape[-2:])
        path2 = self.refine[1](path3, maps[1], size=maps[0].shape[-2:])
        path1 = self.refine[0](path2, maps[0], size=maps[0].shape[-2:])
        return self.output_conv(path1)


class VideoRefineTransformer(nn.Module):
    """CoWTracker-style refine module: spatial ViT + interleaved temporal attention + DPT head."""

    def __init__(
        self,
        model_name: str = "vits",
        input_dim: int = 64,
        patch_size: int = 4,
        temporal_interleave_stride: int = 2,
        max_frames: int = 256,
        mlp_ratio: float = 4.0,
        num_blocks: Optional[int] = None,
    ):
        super().__init__()
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported refine model: {model_name}")
        base = timm.create_model(MODEL_CONFIGS[model_name]["encoder"], pretrained=False, num_classes=0)
        self.blocks = base.blocks if num_blocks is None else base.blocks[: int(num_blocks)]
        self.embed_dim = base.embed_dim
        self.output_dim = MODEL_CONFIGS[model_name]["features"]
        self.patch_size = int(patch_size)
        self.temporal_interleave_stride = max(1, int(temporal_interleave_stride))

        self.patch_embed = PatchEmbed(input_dim, self.embed_dim, self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        time_grid = torch.arange(max_frames, dtype=torch.float32)
        self.register_buffer(
            "time_emb",
            get_1d_sincos_pos_embed_from_grid(self.embed_dim, time_grid),
            persistent=False,
        )

        num_heads = getattr(base.blocks[0].attn, "num_heads", 8)
        n_temporal = sum(1 for i in range(len(self.blocks)) if (i + 1) % self.temporal_interleave_stride == 0)
        self.temporal_blocks = nn.ModuleList(
            [TemporalSelfAttentionBlock(self.embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(n_temporal)]
        )
        self.dpt_head = DPTRefineHead(self.embed_dim, self.output_dim)

        idx_map = {
            "vitt": [2, 5, 8, 11],
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }
        self.intermediate_idx = [i for i in idx_map[model_name] if i < len(self.blocks)]

    def interpolate_time_embed(self, dtype: torch.dtype, frames: int) -> torch.Tensor:
        if frames == self.time_emb.shape[1]:
            return self.time_emb.to(dtype)
        emb = F.interpolate(
            self.time_emb.float().permute(0, 2, 1),
            size=frames,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        return emb.to(dtype)

    def interpolate_pos_encoding(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        dtype = x.dtype
        n_patches = x.shape[1]
        if n_patches == self.pos_embed.shape[1]:
            return self.pos_embed.to(dtype)
        dim = x.shape[-1]
        base = int(math.sqrt(self.pos_embed.shape[1]))
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        pos = F.interpolate(
            self.pos_embed.float().reshape(1, base, base, dim).permute(0, 3, 1, 2),
            size=(patch_h, patch_w),
            mode="bicubic",
            align_corners=False,
        )
        return pos.permute(0, 2, 3, 1).reshape(1, patch_h * patch_w, dim).to(dtype)

    def forward(self, x: torch.Tensor):
        # x: [B, T, C, H, W]
        B, T, _, H, W = x.shape
        x = x.reshape(B * T, *x.shape[2:])
        x = self.patch_embed(x)
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        x = x.view(B, T, x.shape[1], x.shape[2])
        x = x + self.interpolate_time_embed(x.dtype, T).unsqueeze(2)
        x = x.view(B * T, x.shape[2], x.shape[3])
        x = x + self.interpolate_pos_encoding(x, H, W)

        outputs = []
        temporal_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % self.temporal_interleave_stride == 0:
                x = x.view(B, T, x.shape[1], x.shape[2])
                x = self.temporal_blocks[temporal_idx](x)
                temporal_idx += 1
                x = x.view(B * T, x.shape[2], x.shape[3])
            if i in self.intermediate_idx:
                outputs.append(x)

        if not outputs:
            outputs.append(x)
        out = self.dpt_head(outputs, patch_h, patch_w)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        return {"out": out.view(B, T, *out.shape[1:])}
