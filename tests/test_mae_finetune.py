"""Tests for S5-5.4b: MAE encoder -> ViT weight transfer.

Run from the project root:
    python tests/test_mae_finetune.py

The learner writes mae.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.mae import (  # noqa: E402
    MaskedAutoencoder,
    load_mae_encoder_weights,
)
from vit_from_scratch.vit import ViT  # noqa: E402


def make_mae(depth: int = 2) -> MaskedAutoencoder:
    torch.manual_seed(0)
    return MaskedAutoencoder(
        image_size=16,
        patch_size=4,
        in_channels=3,
        encoder_dim=32,
        encoder_depth=depth,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=2,
        mlp_ratio=2.0,
        mask_ratio=0.5,
    )


def make_vit(
    depth: int = 2,
    dim: int = 32,
    heads: int = 4,
    mlp_dim: int = 64,
    pool: str = "cls",
    pos_embed: str = "learned",
    use_conv: bool = True,
) -> ViT:
    torch.manual_seed(1)
    vit = ViT(
        image_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool=pool,
        pos_embed=pos_embed,
    )
    if not use_conv:
        # ViT's use_conv is fixed to True; replace patch_embed's proj directly here as a negative case.
        vit.patch_embed.proj = torch.nn.Linear(48, dim)
    return vit


class TestLoadMAEEncoderWeights(unittest.TestCase):
    def test_copies_encoder_and_leaves_new_parts_untouched(self):
        mae = make_mae()
        vit = make_vit()

        cls_before = vit.cls_token.token.detach().clone()
        cls_pos_before = vit.pos_embedding.pos_embed[:, :1].detach().clone()
        head_before = vit.head.weight.detach().clone()

        load_mae_encoder_weights(vit, mae)

        self.assertTrue(
            torch.equal(
                vit.patch_embed.proj.weight,
                mae.encoder.patch_embed.proj.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                vit.patch_embed.proj.bias,
                mae.encoder.patch_embed.proj.bias,
            )
        )
        self.assertTrue(
            torch.equal(
                vit.pos_embedding.pos_embed[:, 1:],
                mae.encoder.pos_embed,
            )
        )
        for vit_block, mae_block in zip(vit.blocks, mae.encoder.blocks):
            vit_state = vit_block.state_dict()
            mae_state = mae_block.state_dict()
            self.assertEqual(list(vit_state), list(mae_state))
            for key in vit_state:
                self.assertTrue(torch.equal(vit_state[key], mae_state[key]))
        self.assertTrue(torch.equal(vit.norm.gamma, mae.encoder.norm.gamma))
        self.assertTrue(torch.equal(vit.norm.beta, mae.encoder.norm.beta))

        # Newly initialized parts must stay unchanged
        self.assertTrue(torch.equal(vit.cls_token.token, cls_before))
        self.assertTrue(
            torch.equal(vit.pos_embedding.pos_embed[:, :1], cls_pos_before)
        )
        self.assertTrue(torch.equal(vit.head.weight, head_before))

    def test_mean_pool_copies_all_positions(self):
        mae = make_mae()
        vit = make_vit(pool="mean")
        load_mae_encoder_weights(vit, mae)
        self.assertTrue(
            torch.equal(vit.pos_embedding.pos_embed, mae.encoder.pos_embed)
        )

    def test_depth_mismatch_raises(self):
        mae = make_mae(depth=2)
        vit = make_vit(depth=3)
        with self.assertRaises(ValueError):
            load_mae_encoder_weights(vit, mae)

    def test_width_mismatch_raises(self):
        mae = make_mae()
        vit = make_vit(dim=64, heads=4, mlp_dim=128)
        with self.assertRaises(ValueError):
            load_mae_encoder_weights(vit, mae)

    def test_non_conv_patch_embed_raises(self):
        mae = make_mae()
        vit = make_vit(use_conv=False)
        with self.assertRaises(ValueError):
            load_mae_encoder_weights(vit, mae)

    def test_non_learned_pos_embed_raises(self):
        mae = make_mae()
        vit = make_vit(pos_embed="sincos")
        with self.assertRaises(ValueError):
            load_mae_encoder_weights(vit, mae)


if __name__ == "__main__":
    unittest.main(verbosity=2)
