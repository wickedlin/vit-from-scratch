"""Core Swin Transformer components (S4).

Convention: features are channels-last (B, H, W, C) throughout;
tokens are flattened inside each window before attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vit_from_scratch.transformer import DropPath, LayerNorm, MLP


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split (B,H,W,C) into non-overlapping windows.

    Args:
        x: (B, H, W, C); H and W must be divisible by window_size.
        window_size: Window side length in tokens; must be a positive integer.

    Returns:
        (B * nH * nW, window_size * window_size, C), windows in row-major order:
        batch first, then window row i, then window column j.

    Raises:
        ValueError: x is not 4-D, window_size is not positive, or H/W are not
            divisible by it.
    """
    if x.dim() != 4:
        raise ValueError(f"expected 4-D (B,H,W,C), got shape {tuple(x.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    B, H, W, C = x.shape
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(
            f"H and W must be divisible by window_size={window_size}, "
            f"got H={H}, W={W}"
        )

    nH, nW = H // window_size, W // window_size
    # [B, nH, ws, nW, ws, C] -> [B, nH, nW, ws, ws, C] -> [B*nH*nW, ws*ws, C]
    x = x.reshape(B, nH, window_size, nW, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B * nH * nW, window_size * window_size, C)


def window_reverse(
        windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """Inverse of window_partition.

    Args:
        windows: (B * nH * nW, window_size * window_size, C).
        window_size: Window side length.
        H, W: Height/width of the reconstructed feature map; both must be
            divisible by window_size.

    Returns:
        (B, H, W, C).

    Raises:
        ValueError: windows is not 3-D; window_size is not positive; H/W are
            not divisible; windows.shape[1] is not window_size^2; the number of
            windows is not divisible by nH*nW.
    """
    if windows.dim() != 3:
        raise ValueError(f"expected 3-D windows, got shape {tuple(windows.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(
            f"H and W must be divisible by window_size={window_size}, "
            f"got H={H}, W={W}"
        )
    if windows.shape[1] != window_size * window_size:
        raise ValueError(
            f"windows.shape[1] must equal window_size**2="
            f"{window_size * window_size}, got {windows.shape[1]}"
        )

    nH, nW = H // window_size, W // window_size
    n_windows = windows.shape[0]
    if n_windows % (nH * nW) != 0:
        raise ValueError(
            f"number of windows {n_windows} is not divisible by "
            f"nH*nW={nH * nW}"
        )

    B = n_windows // (nH * nW)
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, C)


def shift_window(x: torch.Tensor, shift_size: int) -> torch.Tensor:
    """Cyclically shift (B,H,W,C) by (-shift_size, -shift_size).

    roll keeps the shape unchanged; rows/columns that fall outside wrap to the
    other side, so every partition is still a complete ws x ws window.
    """
    if x.dim() != 4:
        raise ValueError(f"expected 4-D (B,H,W,C), got shape {tuple(x.shape)}")
    if shift_size < 0:
        raise ValueError(f"shift_size must be non-negative, got {shift_size}")
    if shift_size == 0:
        return x
    return torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))


def unshift_window(x: torch.Tensor, shift_size: int) -> torch.Tensor:
    """Inverse of shift_window: cyclically shift (B,H,W,C) by (+s, +s)."""
    if x.dim() != 4:
        raise ValueError(f"expected 4-D (B,H,W,C), got shape {tuple(x.shape)}")
    if shift_size < 0:
        raise ValueError(f"shift_size must be non-negative, got {shift_size}")
    if shift_size == 0:
        return x
    return torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))


def compute_shifted_window_mask(
    H: int,
    W: int,
    window_size: int,
    shift_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the bool mask for shifted-window attention; True means "may attend".

    Args:
        H, W: Feature-map height/width; must be divisible by window_size.
        window_size: Window side length M.
        shift_size: Cyclic shift s; 0 <= s < M.
        device: Device for the mask; default None (CPU). During training, pass
            the input tensor's device.

    Returns:
        A bool tensor of shape (nH * nW, window_size**2, window_size**2).
        Window order matches window_partition; for window k, [i, j] says whether
        token i in that window may attend to token j.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if shift_size < 0 or shift_size >= window_size:
        raise ValueError(
            f"shift_size must satisfy 0 <= shift_size < window_size="
            f"{window_size}, got {shift_size}"
        )
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(
            f"H and W must be divisible by window_size={window_size}, "
            f"got H={H}, W={W}"
        )

    nH, nW = H // window_size, W // window_size
    ws2 = window_size * window_size
    if shift_size == 0:
        return torch.ones(nH * nW, ws2, ws2, dtype=torch.bool, device=device)
    if H <= window_size or W <= window_size:
        raise ValueError(
            "shift_size > 0 requires H > window_size and W > window_size; "
            "for a single window use shift_size=0"
        )

    img_mask = torch.zeros((1, H, W, 1), dtype=torch.int64, device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    return mask_windows.unsqueeze(1) == mask_windows.unsqueeze(2)


class PatchMerging(nn.Module):
    """Merge 2x2 neighborhoods: halve the spatial size and double the channels.

    Input:  (B, H, W, C), H/W must be even, C == dim
    Output: (B, H/2, W/2, 2*dim)
    Params: LayerNorm(4*dim) + Linear(4*dim -> 2*dim, bias=False)
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        # Note: nn.LayerNorm is used here because tests read .weight / .bias /
        # .eps; the S1 LayerNorm uses gamma/beta and has a different interface.
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected 4-D (B, H, W, C), got shape {tuple(x.shape)}")
        B, H, W, C = x.shape
        if C != self.dim:
            raise ValueError(f"expected {self.dim}, got C={C}")
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"H and W must be even, got H={H}, W={W}")

        x0 = x[:, 0::2, 0::2, :]    # top-left
        x1 = x[:, 1::2, 0::2, :]    # bottom-left
        x2 = x[:, 0::2, 1::2, :]    # top-right
        x3 = x[:, 1::2, 1::2, :]    # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)     # (B, H/2, W/2, 4C)
        x = self.norm(x)
        return self.reduction(x)


class RelativePositionBias(nn.Module):
    """Learnable relative-position bias for every token pair inside a window.

    Input:  none (window size is fixed at construction)
    Output: (num_heads, M*M, M*M), added directly to attention logits
    Params: bias_table, shape ((2M-1)^2, num_heads)
    Buffer: relative_position_index, shape (M*M, M*M), dtype=long, no gradient
    """
    def __init__(self, window_size: int, num_heads: int) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        self.window_size = window_size
        self.num_heads = num_heads

        # One (2M-1)^2 lookup table per head; learned, initialized with small normal values.
        self.bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # Precompute the (M^2, M^2) packed index; it is a geometric constant, not a parameter.
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )   # (2, M, M)
        coords_flatten = torch.flatten(coords, 1)   # (2, M^2)

        # relative_coords[:, i, j] = coords_i - coords_j
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )   # (2, M^2, M^2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()     # (M^2, M^2, 2)

        # Shift into [0, 2M-2], then multiply the row coordinate by the column span.
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)   # (M^2, M^2)

        # persistent=False: not saved in state_dict, but moves with the module.
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

    def forward(self) -> torch.Tensor:
        # (M^2 * M^2, H) -> (M^2, M^2, H) -> (H, M^2, M^2)
        bias = self.bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(
            self.window_size ** 2,
            self.window_size ** 2,
            self.num_heads,
        )
        return bias.permute(2, 0, 1).contiguous()


class WindowAttention(nn.Module):
    """Multi-head self-attention inside a window (shared by W-MSA / SW-MSA).

    Input:
        x:    (B_, M*M, C), where B_ = B * nW
        mask: None or a bool (nW, M*M, M*M), True = allowed
    Output:
        (B_, M*M, C)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_relative_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_position_bias = (
            RelativePositionBias(window_size, num_heads)
            if use_relative_bias
            else None
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3-D (B_, N, C), got {tuple(x.shape)}")
        B_, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"expected C={self.dim}, got {C}")
        if N != self.window_size ** 2:
            raise ValueError(
                f"expected N={self.window_size ** 2}, got {N}"
            )

        # (B_, N, 3C) -> (B_, N, 3, h, d) -> (3, B_, h, N, d)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # (B_, h, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_position_bias is not None:
            attn = attn + self.relative_position_bias()     # (h, N, N)

        if mask is not None:
            if mask.dim() != 3 or mask.shape[-2:] != (N, N):
                raise ValueError(
                    f"mask must be (nW, {N}, {N}), got {tuple(mask.shape)}"
                )
            n_windows = mask.shape[0]
            if B_ % n_windows != 0:
                raise ValueError(
                    f"B_={B_} is not divisible by nW={n_windows}"
                )
            batch = B_ // n_windows
            # (nW, 1, N, N) -> (B*nW, 1, N, N), matching window_partition's batch-first order.
            mask = mask.unsqueeze(1).repeat(batch, 1, 1, 1)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v      # (B_, h, N, d)
        out = out.transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


class SwinTransformerBlock(nn.Module):
    """Pre-LN Swin block: W-MSA / SW-MSA + MLP.

    x: (B, H, W, C) -> (B, H, W, C)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_relative_bias: bool = True,
    ) -> None:
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError(
                f"shift_size must satisfy 0 <= shift_size < window_size="
                f"{window_size}, got {shift_size}"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_relative_bias=use_relative_bias,
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim=int(dim * mlp_ratio), dropout=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected 4-D (B, H, W, C), got {tuple(x.shape)}")
        B, H, W, C = x.shape
        if C != self.dim:
            raise ValueError(f"expected C={self.dim}, got {C}")
        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(
                f"H and W must be divisible by window_size={self.window_size}, "
                f"got H={H}, W={W}"
            )

        shortcut = x
        h = self.norm1(x)

        if self.shift_size > 0:
            shifted = shift_window(h, self.shift_size)
            mask = compute_shifted_window_mask(
                H, W, self.window_size, self.shift_size, device=x.device
            )
        else:
            shifted = h
            mask = None

        windows = window_partition(shifted, self.window_size)
        windows = self.attn(windows, mask=mask)
        shifted = window_reverse(windows, self.window_size, H, W)
        h = unshift_window(shifted, self.shift_size) if self.shift_size > 0 else shifted

        x = shortcut + self.drop_path(h)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed2D(nn.Module):
    """Convolutional patch embedding returning a channels-last feature map.

    Input:  (B, C_in, H, W)
    Output: (B, H//P, W//P, dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 64,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.in_channels = in_channels
        self.dim = dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected 4-D input, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size={self.patch_size}, "
                f"got H={x.shape[-2]}, W={x.shape[-1]}"
            )
        x = self.proj(x)    # (B, dim, H/P, W/P)
        return x.permute(0, 2, 3, 1).contiguous()   # (B, H/P, W/P, dim)


class SwinStage(nn.Module):
    """A group of blocks at the same resolution; even blocks use W-MSA, odd blocks use SW-MSA."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_relative_bias: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    use_relative_bias=use_relative_bias,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SwinTransformer(nn.Module):
    """CIFAR-sized Swin: PatchEmbed -> SwinStages (PatchMerging between stages) -> head.

      Input:  (B, 3, image_size, image_size)
      Output: (B, num_classes)
      """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 64,
        depths: tuple[int, ...] = (2, 2, 6),
        heads: tuple[int, ...] = (2, 4, 8),
        window_sizes: tuple[int, ...] = (4, 2, 2),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        use_relative_bias: bool = True,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (len(depths) == len(heads) == len(window_sizes)):
            raise ValueError(
                "depths, heads and window_sizes must have the same length, got "
                f"{len(depths)}, {len(heads)}, {len(window_sizes)}"
            )
        if len(depths) == 0:
            raise ValueError("at least one stage is required")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        num_stages = len(depths)
        base_res = image_size // patch_size
        if base_res % (2 ** (num_stages - 1)) != 0:
            raise ValueError(
                f"stage-1 resolution {base_res} must be divisible by "
                f"2^(stages-1)={2 ** (num_stages - 1)} for PatchMerging"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = num_stages

        self.patch_embed = PatchEmbed2D(in_channels=in_channels, dim=dim, patch_size=patch_size)
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        stage_dim = dim
        for i in range(num_stages):
            res = base_res // (2 ** i)
            ws = window_sizes[i]
            if res % ws != 0:
                raise ValueError(
                    f"stage {i}: resolution {res} must be divisible by "
                    f"window_size={ws}"
                )
            # A single window cannot be shifted (by the block contract).
            shift = 0 if res <= ws else ws // 2
            self.stages.append(
                SwinStage(
                    stage_dim,
                    depth=depths[i],
                    num_heads=heads[i],
                    window_size=ws,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=dropout,
                    attn_drop=attn_dropout,
                    drop_path=drop_path,
                    use_relative_bias=use_relative_bias,
                )
            )
            if i < num_stages - 1:
                self.merges.append(PatchMerging(stage_dim))
                stage_dim *= 2

        self.norm = LayerNorm(stage_dim)
        self.head = nn.Linear(stage_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected 4-D input, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected input size {self.image_size}x{self.image_size}, "
                f"got {x.shape[-2]}x{x.shape[-1]}"
            )

        x = self.patch_embed(x)     # (B, H/P, W/P, dim)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.merges):
                x = self.merges[i](x)

        x = self.norm(x)
        x = x.mean(dim=(1, 2))      # (B, dim)
        return self.head(x)
