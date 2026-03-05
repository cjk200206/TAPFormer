import numpy as np
import random
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

EPS = 1e-6


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """
    for (a, b) in zip(input.size(), mask.size()):
        assert a == b
    # mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )
    

def sample_features5d(input, coords):
    r"""Sample spatio-temporal features

    `sample_features5d(input, coords)` works in the same way as
    :func:`sample_features4d` but for spatio-temporal features and points:
    :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
    a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
    x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

    Args:
        input (Tensor): spatio-temporal features.
        coords (Tensor): spatio-temporal points.

    Returns:
        Tensor: sampled features.
    """

    B, T, _, _, _ = input.shape

    # B T C H W -> B C T H W
    input = input.permute(0, 2, 1, 3, 4)

    # B R1 R2 3 -> B R1 R2 1 3
    coords = coords.unsqueeze(3)

    # B C R1 R2 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 3, 1, 4).view(
        B, feats.shape[2], feats.shape[3], feats.shape[1]
    )  # B C R1 R2 1 -> B R1 R2 C


def sample_features4d(input: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    2)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C


def get_support_points(coords, r, reshape_back=True):
    B, _, N, _ = coords.shape
    device = coords.device
    centroid_lvl = coords.reshape(B, N, 1, 1, 3)

    dx = torch.linspace(-r, r, 2 * r + 1, device=device)
    dy = torch.linspace(-r, r, 2 * r + 1, device=device)

    xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
    zgrid = torch.zeros_like(xgrid, device=device)
    delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
    delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
    coords_lvl = centroid_lvl + delta_lvl

    if reshape_back:
        return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
    else:
        return coords_lvl


def get_track_feat(fmaps, queried_frames, queried_coords, support_radius=0):
    sample_frames = queried_frames[:, None, :, None]
    sample_coords = torch.cat([sample_frames, queried_coords[:, None]], dim=-1)
    
    support_points = get_support_points(sample_coords, support_radius)
    support_track_feats = sample_features5d(fmaps, support_points)
    return (
        support_track_feats[:, None, support_track_feats.shape[1] // 2],
        support_track_feats,
    )
    
    
def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def get_uniformly_sampled_pts(
    size: int,
    num_frames: int,
    extent: Tuple[float, ...],
    device: Optional[torch.device] = torch.device("cpu"),
):
    time_points = torch.randint(low=0, high=num_frames, size=(size, 1), device=device)
    space_points = torch.rand(size, 2, device=device) * torch.tensor(
        [extent[1], extent[0]], device=device
    )
    points = torch.cat((time_points, space_points), dim=1)
    return points[None]


def get_sift_sampled_pts(
    video,
    size: int,
    num_frames: int,
    extent: Tuple[float, ...],
    device: Optional[torch.device] = torch.device("cpu"),
    num_sampled_frames: int = 8,
    sampling_length_percent: float = 0.25,
):
    import cv2
    # assert size == 384, "hardcoded for experiment"
    sift = cv2.SIFT_create(nfeatures=size // num_sampled_frames)
    points = list()
    for _ in range(num_sampled_frames):
        frame_num = random.randint(0, int(num_frames * sampling_length_percent))
        key_points, _ = sift.detectAndCompute(
            video[0, frame_num, :, :, :]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            .astype(np.uint8),
            None,
        )
        for kp in key_points:
            points.append([frame_num, int(kp.pt[0]), int(kp.pt[1])])
    return torch.tensor(points[:size], device=device)[None]


def combine_tokens(template_tokens, search_tokens, mode='direct', return_res=False):
    # [B, HW, C]
    len_t = template_tokens.shape[1]
    len_s = search_tokens.shape[1]

    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
    elif mode == 'template_central':
        central_pivot = len_s // 2
        first_half = search_tokens[:, :central_pivot, :]
        second_half = search_tokens[:, central_pivot:, :]
        merged_feature = torch.cat((first_half, template_tokens, second_half), dim=1)
    elif mode == 'partition':
        feat_size_s = int(math.sqrt(len_s))
        feat_size_t = int(math.sqrt(len_t))
        window_size = math.ceil(feat_size_t / 2.)
        # pad feature maps to multiples of window size
        B, _, C = template_tokens.shape
        H = W = feat_size_t
        template_tokens = template_tokens.view(B, H, W, C)
        pad_l = pad_b = pad_r = 0
        # pad_r = (window_size - W % window_size) % window_size
        pad_t = (window_size - H % window_size) % window_size
        template_tokens = F.pad(template_tokens, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, Hp // window_size, window_size, W, C)
        template_tokens = torch.cat([template_tokens[:, 0, ...], template_tokens[:, 1, ...]], dim=2)
        _, Hc, Wc, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, -1, C)
        merged_feature = torch.cat([template_tokens, search_tokens], dim=1)

        # calculate new h and w, which may be useful for SwinT or others
        merged_h, merged_w = feat_size_s + Hc, feat_size_s
        if return_res:
            return merged_feature, merged_h, merged_w

    else:
        raise NotImplementedError

    return merged_feature


def recover_tokens(merged_tokens, len_template_token, len_search_token, mode='direct'):
    if mode == 'direct':
        recovered_tokens = merged_tokens
    elif mode == 'template_central':
        central_pivot = len_search_token // 2
        len_remain = len_search_token - central_pivot
        len_half_and_t = central_pivot + len_template_token

        first_half = merged_tokens[:, :central_pivot, :]
        second_half = merged_tokens[:, -len_remain:, :]
        template_tokens = merged_tokens[:, central_pivot:len_half_and_t, :]

        recovered_tokens = torch.cat((template_tokens, first_half, second_half), dim=1)
    elif mode == 'partition':
        recovered_tokens = merged_tokens
    else:
        raise NotImplementedError

    return recovered_tokens

def normalize_voxels(voxels, use_mask=True):
    """Perform channelwise std-mean normalization on voxels."""
    if use_mask:
        mask = voxels != 0
    else:
        mask = torch.ones_like(voxels, dtype=torch.bool)
    mean = voxels.sum(dim=(0, 2, 3), keepdim=True) / mask.sum(dim=(0, 2, 3), keepdim=True)
    var = ((voxels - mean)**2 * mask).sum(dim=(0, 2, 3), keepdim=True) / mask.sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(var + 1e-8)
    return torch.where(mask, (voxels - mean) / std, voxels)