"""Isolated auxiliary-loss algebra; not imported by production train/rollout.

The hard-forward straight-through proxy is biased. It changes only a proposed
same-step auxiliary branch; the real detached discrete feedback remains intact.
Temperature is mandatory and no training coefficient is fixed here.
"""
import math
import torch
from torch.nn import functional as F


def auxiliary_embedding(token_logits, codebook, *, temperature):
    if token_logits.ndim != 4 or token_logits.shape[1] != 512 or codebook.shape != (512, 128):
        raise ValueError("Native B,512,X,Y logits and 512,128 codebook required")
    if (
        token_logits.dtype != torch.float32
        or codebook.dtype != torch.float32
        or token_logits.device != codebook.device
    ):
        raise ValueError("Same-device FP32 prototype required")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Explicit finite positive temperature required")
    if codebook.requires_grad or codebook.grad is not None:
        raise ValueError("Codebook must stay frozen")
    if not torch.isfinite(token_logits).all() or not torch.isfinite(codebook).all():
        raise FloatingPointError("Nonfinite auxiliary input")
    ids = token_logits.detach().argmax(1)
    hard = F.embedding(ids, codebook).permute(0, 3, 1, 2).contiguous().detach()
    probabilities = (token_logits / temperature).softmax(1)
    soft = torch.einsum("bnhw,nc->bchw", probabilities, codebook)
    proxy = hard + (soft - soft.detach())
    if not torch.equal(proxy.detach(), hard):
        raise AssertionError("Auxiliary hard forward differs")
    return proxy, hard, ids


def semantic_ce_sum(semantic_logits, GT):
    """Proposed full-label auxiliary sum; caller handles effective-batch division.

    Native decoder layout B,1,200,200,16,18 in production. The algebraic helper
    accepts other spatial sizes for explicitly synthetic unit tests only.
    No mask, class weighting, ignore index, or smoothing is selected implicitly.
    """
    if (
        semantic_logits.ndim < 3
        or semantic_logits.shape[-1] != 18
        or semantic_logits.dtype != torch.float32
    ):
        raise ValueError("FP32 semantic logits with final18-class axis required")
    if GT.dtype != torch.int64 or tuple(GT.shape) != tuple(semantic_logits.shape[:-1]):
        raise ValueError("Aligned int64 semantic GT required")
    if GT.numel() == 0 or torch.any((GT < 0) | (GT > 17)):
        raise ValueError("Complete legal0..17 labels required; no ignored/free substitute")
    if not torch.isfinite(semantic_logits).all():
        raise FloatingPointError("Nonfinite semantic logits")
    return (
        F.cross_entropy(
            semantic_logits.reshape(-1, 18),
            GT.to(semantic_logits.device).reshape(-1),
            reduction="sum",
        ),
        GT.numel(),
    )
