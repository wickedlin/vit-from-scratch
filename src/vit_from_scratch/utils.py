"""Shared utilities for the from-scratch ViT course."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return cuda if available, then mps, then cpu."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
