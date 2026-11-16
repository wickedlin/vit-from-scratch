"""MAE masking, model, and reconstruction loss (S5).

During pretraining the encoder receives only visible patches; the decoder is
assembled in 5.3.
Convention: x is a patch-token sequence (B, N, D) with N = (H/P)(W/P).
"""

from __future__ import annotations

import torch
from torch import nn

from vit_from_scratch.transformer import LayerNorm, TransformerBlock
from vit_from_scratch.vit import PatchEmbed


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly mask a fraction `mask_ratio` of patches.

    Args:
        x: (B, N, D) patch tokens; usually position embeddings are already added.
        mask_ratio: Fraction to mask; must be in (0, 1).
        generator: Optional RNG for reproducibility in tests; its device must
            match x's device.

    Returns:
        x_keep: (B, K, D) visible tokens only, in shuffled order.
        mask: (B, N) float32 in original patch order, 1=masked, 0=visible.
        ids_restore: (B, N) int64 indices that undo the shuffle.

    Raises:
        ValueError: x is not 3-D, mask_ratio is outside (0, 1), or K < 1.
    """
    if x.ndim != 3:
        raise ValueError(f"expected 3-D (B,N,D), got shape {tuple(x.shape)}")
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

    b, n, d = x.shape
    len_keep = int(n * (1.0 - mask_ratio))
    if len_keep < 1:
        raise ValueError(f"no visible token for N={n}, mask_ratio={mask_ratio}")

    # One random permutation per row; keep the first len_keep positions.
    noise = torch.rand(b, n, device=x.device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    # gather indices must match the output shape: all D channels of a patch use the same index.
    index = ids_keep.unsqueeze(-1).expand(-1, -1, d)
    x_keep = torch.gather(x, dim=1, index=index)

    # 1=masked, 0=visible; scatter writes back into original positions.
    mask = torch.ones(b, n, device=x.device, dtype=torch.float32)
    mask.scatter_(1, ids_keep, 0.0)
    return x_keep, mask, ids_restore


def unshuffle_tokens(
    x_shuffled: torch.Tensor,
    ids_restore: torch.Tensor,
) -> torch.Tensor:
    """Restore a shuffled (B,N,D) token sequence to original patch order.

    Args:
        x_shuffled: (B, N, D); the first K slots are encoded visible tokens,
            the remaining N-K slots are mask tokens.
        ids_restore: (B, N) int64 inverse permutation returned by random_masking.

    Returns:
        (B, N, D) tokens where out[:, p] corresponds to original patch p.

    Raises:
        ValueError: x_shuffled is not 3-D, or ids_restore has the wrong shape.
    """
    if x_shuffled.ndim != 3:
        raise ValueError(
            f"expected 3-D (B,N,D), got shape {tuple(x_shuffled.shape)}"
        )
    b, n, d = x_shuffled.shape
    if ids_restore.shape != (b, n):
        raise ValueError(
            f"ids_restore must have shape {(b, n)}, got "
            f"{tuple(ids_restore.shape)}"
        )
    index = ids_restore.unsqueeze(-1).expand(-1, -1, d)
    return torch.gather(x_shuffled, dim=1, index=index)


def mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    norm_pix_loss: bool = False,
) -> torch.Tensor:
    """Per-patch MSE over masked patches only.

    Args:
        pred: (B, N, P*P*C) predicted patches.
        target: Ground-truth patches with the same shape as pred.
        mask: (B, N), 1=masked, 0=visible; same device as pred.
        norm_pix_loss: If True, first normalize each patch with
            (x - mean) / sqrt(var + eps) before MSE (the official MAE
            norm_pix_loss). Default False.

    Returns:
        A 0-D scalar tensor: mean MSE over masked patches.

    Raises:
        ValueError: shapes or devices differ, or there is no masked patch.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if mask.shape != pred.shape[:2]:
        raise ValueError(
            f"mask must have shape {tuple(pred.shape[:2])}, got "
            f"{tuple(mask.shape)}"
        )
    if mask.device != pred.device:
        raise ValueError("mask and pred must be on the same device")

    mask = mask.to(dtype=torch.float32)
    if mask.sum() <= 0:
        raise ValueError("mask must contain at least one masked patch")

    if norm_pix_loss:
        # Per-patch mean removal and std division; D_p = P*P*C.
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5

    # (B, N): average the P*P*C elements inside each patch first.
    per_patch = ((pred - target) ** 2).mean(dim=-1)
    # Then average over masked patches only.
    return (per_patch * mask).sum() / mask.sum()


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, P*P*C), with patch elements ordered (C, P, P).

    Args:
        images: (B, C, H, W); H and W must be divisible by patch_size.
        patch_size: Patch side length P.

    Returns:
        (B, (H/P)*(W/P), C*P*P), matching PatchEmbed(use_conv=False)._patchify.

    Raises:
        ValueError: images is not 4-D, or H/W are not divisible by P.
    """
    if images.ndim != 4:
        raise ValueError(f"expected 4-D (B,C,H,W), got shape {tuple(images.shape)}")
    b, c, h, w = images.shape
    p = patch_size
    if h % p != 0 or w % p != 0:
        raise ValueError(f"H={h}, W={w} must be divisible by patch_size={p}")

    gh, gw = h // p, w // p
    x = images.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, gh * gw, c * p * p)


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    in_channels: int,
    grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Inverse of patchify: (B, N, P*P*C) -> (B, C, H, W).

    Args:
        patches: (B, N, P*P*C).
        patch_size: Patch side length P.
        in_channels: Number of image channels C.
        grid_size: Optional (gh, gw); by default N must be a perfect square and
            a square grid is inferred.

    Returns:
        (B, C, gh*P, gw*P).

    Raises:
        ValueError: shapes do not match, or N is not a valid grid.
    """
    if patches.ndim != 3:
        raise ValueError(f"expected 3-D (B,N,P*P*C), got shape {tuple(patches.shape)}")
    p = patch_size
    b, n, d = patches.shape
    if d != in_channels * p * p:
        raise ValueError(
            f"last dim must be {in_channels * p * p}, got {d}"
        )

    if grid_size is None:
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(
                f"N={n} is not a perfect square; pass grid_size=(gh, gw)"
            )
        gh = gw = side
    else:
        gh, gw = grid_size
        if gh * gw != n:
            raise ValueError(f"grid_size={grid_size} does not match N={n}")

    x = patches.reshape(b, gh, gw, in_channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, in_channels, gh * p, gw * p)


class MAEEncoder(nn.Module):
    """MAE encoder: patch embed + position embedding + Transformer blocks.

    Input (B, C, H, W); output: encodings of visible tokens (B, K, D) plus
    mask / ids_restore.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size=image_size, patch_size=patch_size, in_channels=in_channels, dim=dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    mlp_dim=int(dim * mlp_ratio),
                    dropout=dropout,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )
        self.norm = LayerNorm(dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """No masking; return encodings of all N tokens. Used during fine-tuning."""
        x = self.patch_embed(images) + self.pos_embed   # (B, N, D)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(
        self,
        images: torch.Tensor,
        mask_ratio: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask first, then send only visible tokens through the blocks."""
        x = self.patch_embed(images) + self.pos_embed   # (B, N, D)
        x_keep, mask, ids_restore = random_masking(
            x, mask_ratio, generator=generator
        )
        for block in self.blocks:
            x_keep = block(x_keep)
        return self.norm(x_keep), mask, ids_restore


class MAEDecoder(nn.Module):
    """Lightweight decoder: visible encodings + mask tokens -> pixel predictions for all patches."""

    def __init__(
        self,
        num_patches: int,
        patch_size: int = 4,
        in_channels: int = 3,
        encoder_dim: int = 192,
        decoder_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = in_channels * patch_size ** 2
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_dim,
                    heads=heads,
                    mlp_dim=int(decoder_dim * mlp_ratio),
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, self.patch_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def assemble_tokens(
            self,
            x_keep: torch.Tensor,
            ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate visible encodings with mask tokens and restore original order (before pos emb).

        Args:
            x_keep: (B, K, encoder_dim) encodings of visible tokens.
            ids_restore: (B, N) int64.

        Returns:
            (B, N, decoder_dim), where out[:, p] corresponds to original patch p.
        """
        if x_keep.ndim != 3:
            raise ValueError(f"expected 3-D x_keep, got {tuple(x_keep.shape)}")
        b, k, _ = x_keep.shape
        n = self.num_patches
        if k > n:
            raise ValueError(f"K={k} cannot exceed N={n}")
        if ids_restore.shape != (b, n):
            raise ValueError(
                f"ids_restore must have shape {(b, n)}, got "
                f"{tuple(ids_restore.shape)}"
            )

        x = self.enc_to_dec(x_keep)  # (B, K, D_dec)
        mask_tokens = self.mask_token.expand(b, n - k, -1)
        x = torch.cat([x, mask_tokens], dim=1)  # (B, N, D_dec), shuffled order
        return unshuffle_tokens(x, ids_restore)

    def forward(
            self,
            x_keep: torch.Tensor,
            ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        x = self.assemble_tokens(x_keep, ids_restore)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.pred(x)  # (B, N, P*P*C)


class MaskedAutoencoder(nn.Module):
    """MAE pretraining model: forward returns (loss, pred, mask)."""

    def __init__(
            self,
            image_size: int = 32,
            patch_size: int = 4,
            in_channels: int = 3,
            encoder_dim: int = 192,
            encoder_depth: int = 12,
            encoder_heads: int = 3,
            decoder_dim: int = 128,
            decoder_depth: int = 4,
            decoder_heads: int = 4,
            mlp_ratio: float = 4.0,
            mask_ratio: float = 0.75,
            norm_pix_loss: bool = False,
            dropout: float = 0.0,
            drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.encoder = MAEEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )
        self.num_patches = self.encoder.num_patches
        self.decoder = MAEDecoder(
            num_patches=self.num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
            self,
            images: torch.Tensor,
            generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, pred, mask).

        Args:
            images: (B, C, H, W), H=W=image_size.
            generator: Optional; passed to random_masking for reproducibility.

        Returns:
            loss: 0-D scalar.
            pred: (B, N, P*P*C) predictions in original patch order.
            mask: (B, N) in original patch order, 1=masked.
        """
        if images.ndim != 4:
            raise ValueError(f"expected 4-D (B,C,H,W), got {tuple(images.shape)}")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {images.shape[1]}"
            )
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size}, got "
                f"{images.shape[-2]}x{images.shape[-1]}"
            )

        target = patchify(images, self.patch_size)  # (B, N, P*P*C)
        x_keep, mask, ids_restore = self.encoder(
            images, self.mask_ratio, generator=generator
        )
        pred = self.decoder(x_keep, ids_restore)
        loss = mae_loss(pred, target, mask, norm_pix_loss=self.norm_pix_loss)
        return loss, pred, mask

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """For fine-tuning: no masking; return encoder features of all tokens (B, N, D_enc)."""
        return self.encoder.forward_features(images)


def load_mae_encoder_weights(vit: nn.Module, mae: MaskedAutoencoder) -> None:
    """Copy MAE encoder weights into an isomorphic ViT classifier.

    Copied: patch_embed, patch rows of pos_embed, all TransformerBlocks, final LN.
    Not copied: cls token, cls position row, classification head (keep new initialization).

    Args:
        vit: vit_from_scratch.vit.ViT; requires use_conv=True and pos_embed="learned".
        mae: A MaskedAutoencoder with pretrained weights loaded.

    Raises:
        ValueError: patch count, depth, width, or position-embedding layout differs.
    """
    if not isinstance(getattr(vit, "patch_embed", None), nn.Module):
        raise ValueError("vit must have a patch_embed module")
    if not isinstance(vit.patch_embed.proj, nn.Conv2d):
        raise ValueError("vit must use use_conv=True for MAE weight transfer")
    if not hasattr(vit, "pos_embedding") or not isinstance(
            getattr(vit.pos_embedding, "pos_embed", None), nn.Parameter
    ):
        raise ValueError("vit must use learned positional embeddings")

    if vit.patch_embed.num_patches != mae.num_patches:
        raise ValueError(
            f"patch count mismatch: vit={vit.patch_embed.num_patches}, "
            f"mae={mae.num_patches}"
        )
    if vit.patch_embed.proj.weight.shape != mae.encoder.patch_embed.proj.weight.shape:
        raise ValueError(
            f"patch embed shape mismatch: "
            f"{tuple(vit.patch_embed.proj.weight.shape)} vs "
            f"{tuple(mae.encoder.patch_embed.proj.weight.shape)}"
        )
    if len(vit.blocks) != len(mae.encoder.blocks):
        raise ValueError(
            f"depth mismatch: vit={len(vit.blocks)}, "
            f"mae={len(mae.encoder.blocks)}"
        )

    use_cls = getattr(vit, "use_cls", False)
    expected_tokens = mae.num_patches + (1 if use_cls else 0)
    if vit.pos_embedding.pos_embed.shape[:2] != (1, expected_tokens):
        raise ValueError(
            f"pos_embed shape mismatch: expected (1,{expected_tokens}), got "
            f"{tuple(vit.pos_embedding.pos_embed.shape)}"
        )

    with torch.no_grad():
        vit.patch_embed.load_state_dict(mae.encoder.patch_embed.state_dict())
        if use_cls:
            vit.pos_embedding.pos_embed[:, 1:].copy_(mae.encoder.pos_embed)
        else:
            vit.pos_embedding.pos_embed.copy_(mae.encoder.pos_embed)
        for vit_block, mae_block in zip(vit.blocks, mae.encoder.blocks):
            vit_block.load_state_dict(mae_block.state_dict())
        vit.norm.load_state_dict(mae.encoder.norm.state_dict())
