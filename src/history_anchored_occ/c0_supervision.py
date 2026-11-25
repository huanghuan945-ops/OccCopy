"""Independent frozen-window GT target construction. No formal losses here."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from .c0_model import encode_classes, validate_classes, validate_rigid


@dataclass(frozen=True)
class EgoTargets:
    mode_index: torch.Tensor       # B,6 q0..q5; -1 when no valid GT mode
    valid_mode: torch.Tensor       # B,6; supervision only
    delta_xy: torch.Tensor         # B,6,2; all six in-window links retained
    valid_delta: torch.Tensor      # B,6 independent of mode validity
    invalid_reasons: tuple


@dataclass(frozen=True)
class FutureGT:
    occupancy: torch.Tensor | None
    raw_masks: dict                # no selected future loss/evaluation mask
    world_from_lidar: torch.Tensor # B,7,4,4 at t0..t6; targets ONLY
    timestamps_us: torch.Tensor    # B,7
    frame_valid: torch.Tensor      # B,7
    adjacent_valid: torch.Tensor   # B,6
    sample_tokens: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class C0Supervision:
    future_occupancy: torch.Tensor | None
    future_token_labels: torch.Tensor | None
    raw_masks: dict
    ego: EgoTargets
    sample_tokens: tuple
    future_loss_domain: None = None
    formal_regression_branch_rule: None = None


def build_ego_targets(world_from_lidar, timestamps_us, frame_valid, adjacent_valid):
    """Fixed t0-axis endpoint class; local-source displacements, not a loss.

    Invalid global mode does NOT erase otherwise valid adjacent delta targets.
    No branch-selection defaults are produced.
    """
    b = len(world_from_lidar)
    if (world_from_lidar.shape != (b, 7, 4, 4) or timestamps_us.shape != (b, 7)
            or frame_valid.shape != (b, 7) or adjacent_valid.shape != (b, 6)
            or timestamps_us.dtype != torch.int64 or frame_valid.dtype != torch.bool
            or adjacent_valid.dtype != torch.bool):
        raise ValueError('Expected seven frozen ego frames and six adjacency flags')
    poses = world_from_lidar.detach().double()
    valid = frame_valid.clone()
    reasons = []
    for batch in range(b):
        reason = []
        for k in range(7):
            if not valid[batch, k]:
                reason.append(f'frame_{k}_unavailable')
                continue
            try:
                validate_rigid(poses[batch, k])
            except ValueError:
                valid[batch, k] = False
                reason.append(f'frame_{k}_invalid_pose')
        offsets = timestamps_us[batch] - timestamps_us[batch, 0]
        expected = offsets.new_tensor([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000])
        tolerance_valid = torch.abs(offsets - expected) <= 100000
        valid[batch] &= tolerance_valid
        if not tolerance_valid.all():
            reason.append('future_timestamp_outside_frozen_tolerance')
        if not adjacent_valid[batch].all():
            reason.append('sample_chain_not_directly_adjacent')
        if not (timestamps_us[batch, 1:] > timestamps_us[batch, :-1]).all():
            reason.append('timestamps_not_strictly_increasing')
        reasons.append(tuple(reason))
    direct = adjacent_valid & (timestamps_us[:, 1:] > timestamps_us[:, :-1])
    valid_delta = valid[:, :-1] & valid[:, 1:] & direct
    valid_mode = valid.all(1) & direct.all(1)
    safe = torch.where(valid[..., None, None], poses, torch.eye(4, dtype=poses.dtype, device=poses.device))
    origin, rotation = safe[..., :3, 3], safe[..., :3, :3]
    delta = torch.einsum('bkji,bkj->bki', rotation[:, :-1], origin[:, 1:] - origin[:, :-1])[..., :2]
    # Numerical placeholders are masked targets, never a default prediction.
    delta = torch.where(valid_delta[..., None], delta, torch.zeros_like(delta))
    endpoint = torch.einsum('bji,bj->bi', rotation[:, 0], origin[:, 6] - origin[:, 0])[:, 0]
    mode = torch.where(endpoint >= 2, 0, torch.where(endpoint <= -2, 1, 2)).long()
    mode = torch.where(valid_mode, mode, mode.new_full(mode.shape, -1))
    return EgoTargets(mode[:, None].expand(-1, 6).clone(), valid_mode[:, None].expand(-1, 6).clone(),
        delta, valid_delta, tuple(reasons))


def build_future_supervision(frozen_vae, future: FutureGT):
    """Only explicit target caller can request future GT VQ token labels."""
    if not isinstance(future, FutureGT):
        raise TypeError('Expected separate FutureGT target payload')
    if any(p.requires_grad for p in frozen_vae.parameters()):
        raise ValueError('Future GT encoding requires the shared frozen VAE')
    ego = build_ego_targets(future.world_from_lidar, future.timestamps_us,
                           future.frame_valid, future.adjacent_valid)
    labels = None
    if future.occupancy is not None:
        validate_classes(future.occupancy, 6)
        device = next(frozen_vae.parameters()).device
        _, labels, _ = encode_classes(frozen_vae, future.occupancy.to(device))
    return C0Supervision(future.occupancy, labels, dict(future.raw_masks), ego, future.sample_tokens)
