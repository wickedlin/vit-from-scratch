"""Scaled dot-product attention implemented from scratch."""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Args:
        q: (..., N, d_k) query.
        k: (..., M, d_k) key.
        v: (..., M, d_v) value.
        mask: optional bool tensor broadcastable to (..., N, M).
              True means "keep", False means "mask out".
        is_causal: if True, query i can only attend to keys j <= i.

    Returns:
        (..., N, d_v) attention output.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q and k must share the last dim, got {q.shape[-1]} and {k.shape[-1]}"
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"k and v must share the sequence length, got {k.shape[-2]} and {v.shape[-2]}"
        )
    if is_causal and mask is not None:
        raise ValueError("pass either mask or is_causal, not both")

    d_k = q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    # (..., N, M)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if mask is not None:
        # True = keep, so fill the positions where mask is False with -inf
        scores = scores.masked_fill(~mask, float("-inf"))

    if is_causal:
        n, m = q.shape[-2], k.shape[-2]
        if n != m:
            raise ValueError("is_causal requires N == M")
        # (N, M) bool lower-triangular matrix, diagonal included
        causal = torch.arange(m, device=q.device).unsqueeze(0) <= torch.arange(
            n, device=q.device
        ).unsqueeze(1)
        scores = scores.masked_fill(~causal, float("-inf"))

    # (..., N, M), each row sums to 1
    weight = torch.softmax(scores, dim=-1)

    # (..., N, d_v)
    return torch.matmul(weight, v)


def scaled_dot_product_attention_naive(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
) -> torch.Tensor:
    """Loop-based reference implementation (no mask / causal support)."""
    d_k = q.shape[-1]
    d_v = v.shape[-1]
    n = q.shape[-2]
    m = k.shape[-2]
    scale = 1.0 / math.sqrt(d_k)

    # Flatten all leading dims (batch, heads, ...) into one loop dimension.
    q_ = q.reshape(-1, n, d_k)
    k_ = k.reshape(-1, m, d_k)
    v_ = v.reshape(-1, m, d_v)

    out = torch.zeros(q_.shape[0], n, d_v, dtype=q.dtype, device=q.device)

    for b in range(q_.shape[0]):
        for i in range(n):
            # (m,) dot product between query i and every key
            scores = torch.matmul(q_[b, i], k_[b].transpose(-2, -1)) * scale
            # Manual softmax; subtract the max for numerical stability.
            scores = scores - scores.max()
            weights = torch.exp(scores)
            weights = weights / weights.sum()
            # (d_v,)
            out[b, i] = torch.matmul(weights, v_[b])

    return out.reshape(*q.shape[:-1], d_v)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention, implemented with plain PyTorch."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if dim_head is None:
            if dim % heads != 0:
                raise ValueError(f"dim={dim} must be divisible by heads={heads}")
            dim_head = dim // heads

        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        # One fused projection: rows are laid out as [Q block | K block | V block].
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=qkv_bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, dim) -> (B, N, dim). mask: broadcastable to (B, heads, N, M)."""
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head

        # (B, N, 3*h*d) -> three tensors of (B, N, h*d)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # (B, N, h*d) -> (B, N, h, d) -> (B, h, N, d)
        # Split h next to d first, then move the head axis before N.
        q = q.reshape(b, n, h, d).transpose(1, 2)
        k = k.reshape(b, n, h, d).transpose(1, 2)
        v = v.reshape(b, n, h, d).transpose(1, 2)

        # (B, h, N, d); our SDPA scales by sqrt(d_head) internally.
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # (B, h, N, d) -> (B, N, h, d) -> (B, N, h*d)
        out = out.transpose(1, 2).reshape(b, n, h * d)
        return self.to_out(out)

