import torch
import dataclasses
import numpy as np
import torch.nn.functional as F
from typing import Optional, Any, Union

@dataclasses.dataclass(eq=False)
class FrameEventData:
    """ Data class for frame, event, tracks data. """
    video: torch.Tensor         # (B, S, C_i, H, W)
    events: torch.Tensor        # (B, S, C_e, H, W)
    segmentation: torch.Tensor  # (B, S, 1, H, W)
    trajectory: torch.Tensor    # (B, S, N, 2)
    visibility: torch.Tensor    # (B, S, N)
    img_ifnew: torch.Tensor = None 
    # optional daa
    clear_video: Optional[torch.Tensor] = None         # (B, S, C_i, H, W)
    valid: Optional[torch.Tensor] = None  # (B, S, N)
    seq_name: Optional[torch.Tensor] = None 
    
    
@dataclasses.dataclass(eq=False)
class FrameEventData_test:
    video: np.array
    events: Union[np.array, list]
    segmentation: np.array
    trajectory: np.array
    query_points: torch.Tensor
    # optional daa
    visibility: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None  # (B, S, N)
    seq_name: Optional[torch.Tensor] = None  
    img_ifnew: Optional[np.array] = None
    img_ifnew_full: Optional[np.array] = None
    rgb_timestamp: Optional[np.array] = None
    
    
def collate_fn(batch):
    """ Collate function for frame, event, tracks data. """
    video = torch.stack([b.video for b, in batch], dim=0)
    events = torch.stack([b.events for b in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b in batch], dim=0)
    visibility = torch.stack([b.visibility for b in batch], dim=0)
    
    seq_name = [b.seq_name for b in batch]
    
    return FrameEventData(video, events, segmentation, trajectory, visibility, seq_name=seq_name)
    
    
def collate_fn_train(batch):
    """ Collate function for training data. """
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    events = torch.stack([b.events for b, _ in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b, _ in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b, _ in batch], dim=0)
    visibility = torch.stack([b.visibility for b, _ in batch], dim=0)
    clear_video = torch.stack([b.clear_video for b, _ in batch], dim=0)
    valid = torch.stack([b.valid for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    for b, _ in batch:
        if b.img_ifnew is None:
            gotit = [False]
            img_ifnew = []
        else:
            img_ifnew = torch.stack([b.img_ifnew for b, _ in batch], dim=0)
    return (FrameEventData(video, events, segmentation, trajectory, visibility, clear_video=clear_video, valid=valid, seq_name=seq_name, img_ifnew=img_ifnew), gotit)


def collate_fn_EDS(batch):
    video = np.stack([b.video for b, _ in batch], axis=0)
    events = np.stack([b.events for b, _ in batch], axis=0)
    segmentation = torch.stack([b.segmentation for b, _ in batch], axis=0)
    trajectory = np.stack([b.trajectory for b, _ in batch], axis=0)
    query_points = None
    if batch[0][0].query_points is not None:
        query_points = torch.stack([b.query_points for b, _ in batch], dim=0)
    
    seq_name = [b.seq_name for b, _ in batch]
    rgb_timestamp = [b.rgb_timestamp for b, _ in batch]
    
    return FrameEventData_test(video, events, segmentation, trajectory, query_points, seq_name=seq_name, rgb_timestamp=rgb_timestamp)


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj