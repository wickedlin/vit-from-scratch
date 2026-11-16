"""Numerical alignment with vit-pytorch 1.26.12.

Run from the project root:
    python tests/test_vit_reference.py

The learner writes the core ViT by hand; this comparison file is generated
scaffolding. It builds the same architecture from the installed reference
library, copies our weights into it, and checks that both models produce the
same logits.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.vit import ViT  # noqa: E402
from vit_pytorch.vit import ViT as ReferenceViT  # noqa: E402


DIM = 192
HEADS = 3
DIM_HEAD = 64
MLP_DIM = 768
DEPTH = 2
NUM_CLASSES = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4
CHANNELS = 3


def build_reference(pool: str = "cls") -> ReferenceViT:
    """Create the reference model and strip its two patch-embedding LNs.

    vit-pytorch 1.26.12 inserts LayerNorm(patch_dim) before the patch Linear
    and LayerNorm(dim) after it. Neither exists in the ViT paper or in our
    PatchEmbed. Replacing them with Identity makes both models the same
    architecture, so the remaining parameters can be aligned one-to-one.
    """
    ref = ReferenceViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        mlp_dim=MLP_DIM,
        pool=pool,
        channels=CHANNELS,
        dropout=0.0,
        emb_dropout=0.0,
    ).double()

    if not isinstance(ref.to_patch_embedding[1], nn.LayerNorm):
        raise RuntimeError("unexpected vit-pytorch patch embedding layout")
    if not isinstance(ref.to_patch_embedding[2], nn.Linear):
        raise RuntimeError("unexpected vit-pytorch patch embedding layout")
    if not isinstance(ref.to_patch_embedding[3], nn.LayerNorm):
        raise RuntimeError("unexpected vit-pytorch patch embedding layout")

    ref.to_patch_embedding[1] = nn.Identity()
    ref.to_patch_embedding[3] = nn.Identity()
    return ref


def copy_weights(ours: ViT, ref: ReferenceViT) -> None:
    """Copy our parameters into the reference model.

    The patch projection is the only shape/order mismatch:
      ours: Conv2d weight (D, C, P, P), flattened in (C, P, P) order
      ref : Linear weight (D, P*P*C), flattened in (P, P, C) order
    so permute (D, C, P, P) -> (D, P, P, C) before flattening.
    """
    with torch.no_grad():
        w = ours.patch_embed.proj.weight                 # (D, C, P, P)
        ref.to_patch_embedding[2].weight.copy_(
            w.permute(0, 2, 3, 1).reshape(DIM, -1)       # (D, P*P*C)
        )
        ref.to_patch_embedding[2].bias.copy_(ours.patch_embed.proj.bias)

        if ref.cls_token.numel() > 0:
            ref.cls_token.copy_(
                ours.cls_token.token.reshape(ref.cls_token.shape)
            )
        ref.pos_embedding.copy_(
            ours.pos_embedding.pos_embed.reshape(ref.pos_embedding.shape)
        )

        for block, (attn, ff) in zip(ours.blocks, ref.transformer.layers):
            # Our Pre-LN norm1 <-> reference Attention.norm.
            attn.norm.weight.copy_(block.norm1.gamma)
            attn.norm.bias.copy_(block.norm1.beta)
            attn.to_qkv.weight.copy_(block.attn.to_qkv.weight)
            attn.to_out[0].weight.copy_(block.attn.to_out[0].weight)
            attn.to_out[0].bias.copy_(block.attn.to_out[0].bias)

            # Our Pre-LN norm2 <-> reference FeedForward.net[0].
            ff.net[0].weight.copy_(block.norm2.gamma)
            ff.net[0].bias.copy_(block.norm2.beta)
            ff.net[1].weight.copy_(block.mlp.net[0].weight)
            ff.net[1].bias.copy_(block.mlp.net[0].bias)
            ff.net[4].weight.copy_(block.mlp.net[3].weight)
            ff.net[4].bias.copy_(block.mlp.net[3].bias)

        # Reference Transformer.norm is the final LN after all blocks.
        ref.transformer.norm.weight.copy_(ours.norm.gamma)
        ref.transformer.norm.bias.copy_(ours.norm.beta)
        ref.mlp_head.weight.copy_(ours.head.weight)
        ref.mlp_head.bias.copy_(ours.head.bias)


class TestReferenceAlignment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _pair(self, pool: str):
        ours = ViT(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=CHANNELS,
            num_classes=NUM_CLASSES,
            dim=DIM,
            depth=DEPTH,
            heads=HEADS,
            mlp_dim=MLP_DIM,
            dropout=0.0,
            emb_dropout=0.0,
            drop_path=0.0,
            pool=pool,
            pos_embed="learned",
            pre_norm=True,
        ).double().eval()
        ref = build_reference(pool).eval()
        copy_weights(ours, ref)
        return ours, ref

    def test_cls_pool_matches_vit_pytorch(self):
        ours, ref = self._pair("cls")
        x = torch.randn(2, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64)
        with torch.no_grad():
            out_ours = ours(x)
            out_ref = ref(x)

        self.assertTrue(torch.allclose(out_ours, out_ref, atol=1e-5))
        self.assertLess((out_ours - out_ref).abs().max().item(), 1e-12)

    def test_mean_pool_matches_vit_pytorch(self):
        ours, ref = self._pair("mean")
        x = torch.randn(2, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64)
        with torch.no_grad():
            out_ours = ours(x)
            out_ref = ref(x)

        self.assertTrue(torch.allclose(out_ours, out_ref, atol=1e-5))
        self.assertLess((out_ours - out_ref).abs().max().item(), 1e-12)

    def test_parameter_counts_match_after_stripping_patch_norms(self):
        ours, ref = self._pair("cls")
        self.assertEqual(
            sum(p.numel() for p in ours.parameters()),
            sum(p.numel() for p in ref.parameters()),
        )

    def test_patch_weight_permutation_is_required(self):
        # Negative control: dropping the permute still runs, but silently
        # assigns patch positions to the wrong linear weights.
        ours, ref = self._pair("cls")
        with torch.no_grad():
            ref.to_patch_embedding[2].weight.copy_(
                ours.patch_embed.proj.weight.reshape(DIM, -1)
            )
        x = torch.randn(2, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64)
        with torch.no_grad():
            maxdiff = (ours(x) - ref(x)).abs().max().item()

        self.assertGreater(maxdiff, 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
