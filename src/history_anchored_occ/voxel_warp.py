"""Voxel-grid alignment shared by the token and feature copy heads."""

import torch
from torch.nn import functional as F

GRID_SHAPE = (200, 200, 16)
GRID_MIN = (-40.0, -40.0, -1.0)
VOXEL = 0.4
FREE = 17


def voxel_centers(device):
    axes = [
        torch.arange(n, device=device, dtype=torch.float64) * VOXEL + low + VOXEL / 2
        for n, low in zip(GRID_SHAPE, GRID_MIN)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)


def warp_occupancy(source, source_from_target, centers):
    """Target-grid voxels sampled from a source frame by floor index, half-open domain.

    Out-of-domain targets get free only so the frozen encoder can run; they never
    qualify as token candidates (token_candidate_mask).
    """
    if source.shape != GRID_SHAPE or source_from_target.shape != (4, 4):
        raise ValueError("Single frame and transform required")
    transform = source_from_target.to(centers)
    point = centers @ transform[:3, :3].T + transform[:3, 3]
    index = torch.floor((point - centers.new_tensor(GRID_MIN)) / VOXEL + 1e-9).long()
    inside = ((index >= 0) & (index < index.new_tensor(GRID_SHAPE))).all(-1)
    safe = torch.where(inside[..., None], index, torch.zeros_like(index))
    warped = source.to(centers.device)[safe[..., 0], safe[..., 1], safe[..., 2]]
    return torch.where(inside, warped, torch.full_like(warped, FREE)), inside


def token_candidate_mask(inside):
    """n,X,Y,Z -> n,50,50: every XY column of the 4x4 block maps inside the source domain.

    Horizontal only: small ego z/pitch offsets push top or bottom voxels out of the 6.4 m
    range on every column, which must not disqualify the whole frame.
    """
    return (-F.max_pool2d(-inside.any(-1).float()[:, None], 4, 4))[:, 0] > 0.5
