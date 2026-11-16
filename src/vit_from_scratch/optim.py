"""Optimization utilities: AdamW, warmup + cosine, label smoothing, gradient clipping (S3-3.2)."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, Optimizer


def split_params_for_weight_decay(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into "decay" and "no decay" groups.

    Rule: parameters with ndim <= 1 (biases, LayerNorm gamma/beta) get no weight
    decay; all others (Linear / Conv weight matrices) do.

    Returns:
        (decay_params, no_decay_params)
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """Build AdamW with weight decay disabled for biases and LayerNorm.

    Args:
        model: Model to optimize.
        lr: Base learning rate (same for all parameter groups).
        weight_decay: Decoupled weight-decay coefficient, applied to weights with ndim >= 2.
        betas / eps: AdamW momentum and numerical-stability terms.

    Returns:
        torch.optim.AdamW with 1-2 parameter groups.
    """
    decay, no_decay = split_params_for_weight_decay(model)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return AdamW(groups, lr=lr, betas=betas, eps=eps)


def warmup_cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Learning rate at step `step` (0-based).

    For the first warmup_steps steps: lr = base_lr * (step + 1) / warmup_steps;
    afterwards a cosine decay from base_lr to min_lr.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"warmup_steps must be in [0, total_steps), got {warmup_steps}"
        )
    if base_lr <= 0:
        raise ValueError(f"base_lr must be positive, got {base_lr}")
    if not 0.0 <= min_lr <= base_lr:
        raise ValueError(f"min_lr must be in [0, base_lr], got {min_lr}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Step-wise warmup + cosine schedule.

    Training-loop order: call scheduler.step() after optimizer.step().
    The optimizer's initial lr must equal base_lr (guaranteed by build_optimizer).
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr must be positive, got {base_lr}")

    def lr_lambda(step: int) -> float:
        return warmup_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def label_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Label-smoothed cross-entropy; equivalent to nn.CrossEntropyLoss(label_smoothing=smoothing).

    Target distribution: (1 - smoothing) * one_hot + smoothing / num_classes
    loss = -(1 - smoothing) * log p_y - smoothing * mean_k(log p_k)
    """
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()


def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_norm: float = 1.0,
) -> torch.Tensor:
    """Clip gradients by global L2 norm and return the pre-clip norm.

    Must be called after loss.backward() and before optimizer.step();
    with AMP, unscale the gradients first.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
