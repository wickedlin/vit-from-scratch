"""Mixed-precision utilities: AMP dtype selection, GradScaler, micro-batch backward, optimizer step (S3-3.3).

Usage in a training loop with gradient accumulation:

    accum = 4
    for micro_idx, (x, y) in enumerate(loader):
        backward_microbatch(
            model, x, y, criterion, scaler, amp_dtype, loss_weight=1.0 / accum
        )
        if (micro_idx + 1) % accum == 0 or (micro_idx + 1) == len(loader):
            optimizer_step(model, optimizer, scheduler, scaler, max_norm=1.0)
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer

from .optim import clip_gradients

_AMP_MODES = ("off", "auto", "bf16", "fp16")


def resolve_amp_dtype(device: torch.device, mode: str) -> torch.dtype | None:
    """Resolve a CLI mode into an autocast dtype; None means no mixed precision."""
    if mode not in _AMP_MODES:
        raise ValueError(f"mode must be one of {_AMP_MODES}, got {mode!r}")
    if mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 AMP requires CUDA; use bf16 or fp32 elsewhere")
        return torch.float16
    # mode == "auto": disabled on CPU/MPS; prefer bf16 on CUDA.
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_grad_scaler(device: torch.device, amp_dtype: torch.dtype | None) -> torch.amp.GradScaler:
    """Build a GradScaler; loss scaling is enabled only for CUDA + fp16."""
    if amp_dtype is torch.float16:
        if device.type != "cuda":
            raise ValueError("GradScaler for fp16 requires CUDA")
        return torch.amp.GradScaler("cuda", enabled=True)
    # bf16 and fp32 need no scaling; use a disabled scaler to keep one call path.
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(scaler_device, enabled=False)


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    """Return an autocast context; amp_dtype=None returns an empty context (stay in fp32)."""
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def backward_microbatch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    """Forward + backward for one micro-batch; gradients are scaled by loss_weight and accumulated.

    Returns the **unscaled** loss (detached from the graph) for training logs.
    """
    if loss_weight <= 0:
        raise ValueError(f"loss_weight must be positive, got {loss_weight}")

    with autocast_context(x.device, amp_dtype):
        logits = model(x)
        loss = criterion(logits, y)

    scaler.scale(loss * loss_weight).backward()
    return loss.detach()


def optimizer_step(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """One parameter update: unscale -> clip -> scaler.step -> update -> scheduler.step -> zero_grad.

    Returns the pre-clip global gradient norm (a 0-D tensor).
    """
    scaler.unscale_(optimizer)
    norm = clip_gradients(model.parameters(), max_norm=max_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return norm
