"""Vision Transformer assembled from scratch (S2)."""

from __future__ import annotations

import math

import torch
from torch import nn

from vit_from_scratch.transformer import LayerNorm, TransformerBlock


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and project each patch to a dim-dimensional token.

    Args:
        image_size: Side length of the square input image, e.g. 32.
        patch_size: Patch side length P.
        in_channels: Number of input channels C.
        dim: Channel width D of each token.
        use_conv: If True, use Conv2d(kernel_size=P, stride=P);
            if False, use reshape + Linear (used to verify that both are equivalent).

    Input:  (B, C, H, W), with H == W == image_size.
    Output: (B, N, dim), where N = (H // P) * (W // P).
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 192,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.use_conv = use_conv

        gh = gw = image_size // patch_size
        self.grid_size = (gh, gw)
        self.num_patches = gh * gw

        patch_dim = in_channels * patch_size ** 2
        if use_conv:
            # Each output position is a linear projection of one patch; all positions share the weights.
            self.proj = nn.Conv2d(
                in_channels, dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            self.proj = nn.Linear(patch_dim, dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, C*P*P) without convolution."""
        b, c, h, w = x.shape
        p = self.patch_size
        gh, gw = h // p, w // p

        # (B, C, gh, P, gw, P): split H/W into (grid index, offset inside the patch).
        x = x.reshape(b, c, gh, p, gw, p)
        # (B, gh, gw, C, P, P): move both patch indices to the front and patch
        # content to the end; the content order (C, P, P) matches Conv2d weights.
        x = x.permute(0, 2, 4, 1, 3, 5)
        # (B, gh*gw, C*P*P)
        return x.reshape(b, gh * gw, c * p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if (h, w) != (self.image_size, self.image_size):
            raise ValueError(
                f"expected input of size {self.image_size}x{self.image_size}, "
                f"got {h}x{w}"
            )

        if self.use_conv:
            x = self.proj(x)                # (B, D, gh, gw)
            x = x.flatten(2)                # (B, D, N)
            return x.transpose(1, 2)        # (B, N, D)

        patches = self._patchify(x)         # (B, N, C*P*P)
        return self.proj(patches)           # (B, N, D)


class ClassToken(nn.Module):
    """Prepend a learnable cls token to the token sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N+1, D); token 0 is the cls token."""
        cls = self.token.expand(x.shape[0], -1, -1)     # (B, 1, D), one shared vector
        return torch.cat([cls, x], dim=1)


class LearnedPositionEmbedding(nn.Module):
    """Learnable absolute position embedding: one D-dimensional vector per position, added element-wise."""

    def __init__(self, num_tokens: int, dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.shape[1], x.shape[2]) != (self.pos_embed.shape[1], self.pos_embed.shape[2]):
            raise ValueError(
                f"expected {self.pos_embed.shape[1]} tokens, got {x.shape[1]}"
                f"expected {self.pos_embed.shape[2]} token dims, got {x.shape[2]}"
            )

        return x + self.pos_embed


def build_sincos_pos_embed(
    num_tokens: int,
    dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Fixed 1-D sinusoidal position embedding, returned as (num_tokens, dim).

    PE[pos, 2i]   = sin(pos / base ** (2i / dim))
    PE[pos, 2i+1] = cos(pos / base ** (2i / dim))
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    pos = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)    # (N, 1)
    freq = torch.exp(
        -math.log(base) * torch.arange(0, dim, 2, dtype=torch.float32) / dim
    )   # (D/2,)

    pe = torch.zeros(num_tokens, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class SinusoidalPositionEmbedding(nn.Module):
    """Module wrapper for fixed sinusoidal position embeddings; cached as a non-trainable buffer."""

    def __init__(self, num_tokens: int, dim: int) -> None:
        super().__init__()
        pe = build_sincos_pos_embed(num_tokens, dim).unsqueeze(0)   # (1, N, D)
        # persistent=False: a deterministic constant, so it is not saved in state_dict.
        self.register_buffer("pos_embed", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.shape[1], x.shape[2]) != (self.pos_embed.shape[1], self.pos_embed.shape[2]):
            raise ValueError(
                f"expected {self.pos_embed.shape[1]} tokens, got {x.shape[1]}"
                f"expected {self.pos_embed.shape[2]} token dims, got {x.shape[2]}"
            )
        return x + self.pos_embed


def pool_tokens(x: torch.Tensor, mode: str = "cls") -> torch.Tensor:
    """(B, N, D) -> (B, D).

    mode="cls"  : take token 0 (requires a prepended cls token).
    mode="mean" : average all tokens (including cls, if present).
    """
    if mode == "cls":
        return x[:, 0]
    if mode == "mean":
        return x.mean(dim=1)
    raise ValueError(f"unknown pool mode: {mode!r}")


class ViT(nn.Module):
    """Vision Transformer classifier: input (B, C, H, W), output (B, num_classes)."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        drop_path: float = 0.0,
        pool: str = "cls",
        pos_embed: str = "learned",
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"unknown pool mode: {pool!r}")
        if pos_embed not in ("learned", "sincos", "none"):
            raise ValueError(f"unknown pos_embed: {pos_embed!r}")

        self.pool = pool
        self.use_cls = pool == "cls"
        self.pre_norm = pre_norm

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = self.num_patches + (1 if self.use_cls else 0)

        self.cls_token = ClassToken(dim) if self.use_cls else nn.Identity()
        if pos_embed == "learned":
            self.pos_embedding = LearnedPositionEmbedding(self.num_tokens, dim)
        elif pos_embed == "sincos":
            self.pos_embedding = SinusoidalPositionEmbedding(self.num_tokens, dim)
        else:
            self.pos_embedding = nn.Identity()

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    drop_path=drop_path,
                    pre_norm=pre_norm,
                )
                for _ in range(depth)
            ]
        )

        self.norm = LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)         # (B, N, D)
        x = self.cls_token(x)           # (B, N+1, D); Identity in mean mode
        x = self.pos_embedding(x)       # (B, T, D)
        x = self.emb_dropout(x)
        x = self.blocks(x)              # (B, T, D)
        x = self.norm(x)
        x = pool_tokens(x, self.pool)   # (B, D)
        return self.head(x)             # (B, K)
