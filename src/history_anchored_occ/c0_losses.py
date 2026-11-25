"""Confirmed token/mode/local-delta objectives; never part of the student API."""
import torch
from torch.nn import functional as F


def target_counts(target):
    labels = target.future_token_labels
    if labels is None or labels.dtype != torch.int64 or labels.ndim != 4 or labels.shape[1] != 6:
        raise ValueError('Six independent GT token grids required')
    if torch.any((labels < 0) | (labels >= 512)):
        raise ValueError('GT token targets must be in 0..511, never ignore/default placeholders')
    ego = target.ego
    b = labels.shape[0]
    if (ego.mode_index.shape != (b, 6) or ego.mode_index.dtype != torch.int64
            or ego.valid_mode.shape != (b, 6) or ego.valid_mode.dtype != torch.bool
            or ego.delta_xy.shape != (b, 6, 2) or ego.valid_delta.shape != (b, 6)
            or ego.valid_delta.dtype != torch.bool):
        raise ValueError('Expected q0..q5 mode and independent displacement eligibility')
    if torch.any((ego.mode_index[ego.valid_mode] < 0) | (ego.mode_index[ego.valid_mode] >= 3)):
        raise ValueError('Valid GT modes require real branch indices 0..2')
    if not torch.isfinite(ego.delta_xy[ego.valid_delta]).all():
        raise ValueError('Nonfinite valid displacement target')
    return (labels.numel(), int(ego.valid_delta.sum()), int(ego.valid_mode.sum()))


def objective_sums(output, target):
    counts = target_counts(target)
    logits, mode, candidates = (output[k] for k in ('token_logits', 'mode_logits', 'motion_candidates'))
    b, steps, classes, height, width = logits.shape
    if (steps != 6 or classes != 512 or target.future_token_labels.shape != (b, 6, height, width)
            or mode.shape != (b, 6, 3) or candidates.shape != (b, 6, 3, 2)):
        raise ValueError('Wrong six-step training output layout')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in (logits, mode, candidates)):
        raise ValueError('Finite FP32 student logits/candidates required')
    device = logits.device
    labels = target.future_token_labels.to(device)
    valid_mode, valid_delta = target.ego.valid_mode.to(device), target.ego.valid_delta.to(device)
    gt_mode = target.ego.mode_index.to(device)
    gt_delta = target.ego.delta_xy.to(device, dtype=torch.float32)
    token_sum = F.cross_entropy(logits.reshape(b * 6, 512, height, width),
                               labels.reshape(b * 6, height, width), reduction='sum', label_smoothing=0.)
    mode_sum = (F.cross_entropy(mode[valid_mode], gt_mode[valid_mode], reduction='sum', label_smoothing=0.)
                if counts[2] else mode.sum() * 0.)
    predicted_branch = mode.detach().argmax(-1)
    # Only valid labels replace a predicted loss-branch index. Invalid -1 is never gathered.
    loss_branch = predicted_branch.clone()
    loss_branch[valid_mode] = gt_mode[valid_mode]
    if counts[1]:
        selected = candidates[valid_delta].gather(1, loss_branch[valid_delta][:, None, None].expand(-1, 1, 2)).squeeze(1)
        error = selected - gt_delta[valid_delta]
        delta_sum = (torch.sqrt(error.square().sum(-1) + 1e-12) - 1e-6).sum()
    else:
        delta_sum = candidates.sum() * 0.
    sums = (token_sum, delta_sum, mode_sum)
    if not all(torch.isfinite(s).item() for s in sums):
        raise FloatingPointError('Nonfinite C0 objective')
    branch_counts = [int((loss_branch[valid_delta] == k).sum()) for k in range(3)]
    return dict(sums=sums, counts=counts, loss_branch=loss_branch.detach(),
                valid_delta_without_mode=int((valid_delta & ~valid_mode).sum()), branch_counts=branch_counts)


def normalized_loss(sums, effective_counts):
    if len(effective_counts) != 3 or any(type(n) is not int or n < 0 for n in effective_counts):
        raise ValueError('Expected three independent effective-batch denominators')
    if effective_counts[0] <= 0 or any(n > total for n, total in zip(sums['counts'], effective_counts)):
        raise ValueError('Inconsistent target counts')
    weights = (1., .1, .1)
    return sum(w * value / n if n else value * 0.
               for w, value, n in zip(weights, sums['sums'], effective_counts))
