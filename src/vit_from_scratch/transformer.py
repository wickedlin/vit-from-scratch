"""Transformer building blocks implemented from scratch."""

from __future__ import annotations

import torch
from torch import nn

from vit_from_scratch.attention import MultiHeadSelfAttention


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension (per token)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim). Statistics are computed per token over features.
        mean = x.mean(dim=-1, keepdim=True)
        # Biased variance (divide by D), matching nn.LayerNorm.
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class MLP(nn.Module):
    """Per-token feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, dim: int, mlp_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if mlp_dim is None:
            mlp_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropPath(nn.Module):
    """Stochastic depth: drop the residual branch per sample and rescale by 1/keep_prob."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: an attention branch and an MLP branch, each with a residual."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int | None = None,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim=mlp_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop_path(self.attn(x, mask=mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x
