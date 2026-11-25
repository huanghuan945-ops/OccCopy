"""Full-grid losses used to train the copy head."""
import torch
from torch.nn import functional as F
from history_anchored_occ.c0_losses import target_counts,normalized_loss


def dense_cross_entropy(logits,labels,*,reduction='mean',label_smoothing=0.,class_dim=1):
    if reduction not in ('sum','mean','none') or label_smoothing!=0.:raise ValueError('Original unweighted, unsmoothed CE only')
    axis=class_dim%logits.ndim
    expected=tuple(logits.shape[:axis])+tuple(logits.shape[axis+1:])
    if logits.ndim<2 or tuple(labels.shape)!=expected or labels.dtype!=torch.int64 or labels.numel()==0:
        raise ValueError('Aligned nonempty int64 full-domain labels required')
    if logits.dtype not in (torch.float32,torch.float64) or not torch.isfinite(logits).all():raise ValueError('Finite floating logits required')
    labels=labels.to(logits.device)
    classes=logits.shape[axis]
    if ((labels<0)|(labels>=classes)).any():raise ValueError('No ignored/default labels allowed')
    shape=[1]*logits.ndim;shape[axis]=classes
    select=(torch.arange(classes,device=logits.device).reshape(shape)==labels.unsqueeze(axis))
    loss=-(F.log_softmax(logits,dim=axis)*select.to(logits.dtype)).sum(dim=axis)
    return loss.sum() if reduction=='sum' else (loss.mean() if reduction=='mean' else loss)


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
    token_sum = dense_cross_entropy(logits.reshape(b * 6, 512, height, width),
                               labels.reshape(b * 6, height, width), reduction='sum', label_smoothing=0.)
    mode_sum = (dense_cross_entropy(mode[valid_mode], gt_mode[valid_mode], reduction='sum', label_smoothing=0.)
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


def shared_feedback_ce(feedback_logits,future_gt_tokens):
    if feedback_logits.ndim!=5 or feedback_logits.shape[1:3]!=(5,512):raise ValueError('B,5,512,X,Y feedback required')
    b,_,_,x,y=feedback_logits.shape
    if future_gt_tokens.shape!=(b,6,x,y) or future_gt_tokens.dtype!=torch.int64:raise ValueError('Separate six-slot GT labels required')
    if ((future_gt_tokens<0)|(future_gt_tokens>=512)).any():raise ValueError('Illegal token targets')
    return dense_cross_entropy(feedback_logits.reshape(b*5,512,x,y),future_gt_tokens[:,:5].reshape(b*5,x,y),reduction='mean')


def semantic_ce_sum(semantic_logits,GT):
    if semantic_logits.ndim<3 or semantic_logits.shape[-1]!=18 or semantic_logits.dtype!=torch.float32:raise ValueError('FP32 class-last18 logits required')
    return dense_cross_entropy(semantic_logits,GT,reduction='sum',class_dim=-1),GT.numel()
