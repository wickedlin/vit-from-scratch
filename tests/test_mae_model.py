"""Tests for S5-5.3: patchify / MAE encoder-decoder / full forward.

Run from the project root:
    python tests/test_mae_model.py

The learner writes mae.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.mae import (  # noqa: E402
    MAEDecoder,
    MAEEncoder,
    MaskedAutoencoder,
    patchify,
    unpatchify,
)
from vit_from_scratch.vit import PatchEmbed  # noqa: E402


def gather_tokens(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, ids.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class TestPatchify(unittest.TestCase):
    def test_roundtrip(self):
        x = torch.randn(2, 3, 32, 32)
        p = patchify(x, 4)
        self.assertEqual(p.shape, (2, 64, 48))
        self.assertTrue(torch.allclose(unpatchify(p, 4, 3), x, atol=1e-6))

    def test_order_matches_patch_embed(self):
        x = torch.randn(2, 3, 32, 32)
        p = patchify(x, 4)
        pe = PatchEmbed(32, 4, 3, dim=8, use_conv=False)
        self.assertTrue(torch.allclose(p, pe._patchify(x), atol=1e-6))

    def test_rectangular_grid(self):
        x = torch.randn(2, 3, 8, 12)
        p = patchify(x, 4)
        self.assertEqual(p.shape, (2, 6, 48))
        self.assertTrue(
            torch.allclose(
                unpatchify(p, 4, 3, grid_size=(2, 3)), x, atol=1e-6
            )
        )

    def test_invalid_shapes_raise(self):
        with self.assertRaises(ValueError):
            patchify(torch.randn(2, 3, 30, 32), 4)
        with self.assertRaises(ValueError):
            patchify(torch.randn(2, 3, 32), 4)
        with self.assertRaises(ValueError):
            unpatchify(torch.randn(2, 6, 48), 4, 3)  # N=6 is not a perfect square
        with self.assertRaises(ValueError):
            unpatchify(torch.randn(2, 8, 48), 4, 3, grid_size=(2, 3))


class TestMAEEncoder(unittest.TestCase):
    def test_output_shape_and_mask(self):
        torch.manual_seed(0)
        enc = MAEEncoder(
            image_size=16, patch_size=4, in_channels=3, dim=32, depth=1, heads=4
        )
        x_keep, mask, ids_restore = enc(torch.randn(2, 3, 16, 16), 0.75)
        self.assertEqual(x_keep.shape, (2, 4, 32))
        self.assertEqual(mask.shape, (2, 16))
        self.assertEqual(ids_restore.shape, (2, 16))
        self.assertEqual(int(mask.sum()), 2 * 12)
        self.assertTrue(torch.isfinite(x_keep).all())

    def test_position_embedding_is_added_before_masking(self):
        torch.manual_seed(0)
        enc = MAEEncoder(
            image_size=8, patch_size=4, in_channels=3, dim=8, depth=0, heads=2
        )
        with torch.no_grad():
            enc.patch_embed.proj.weight.zero_()
            enc.patch_embed.proj.bias.zero_()
        x_keep, _, ids_restore = enc(
            torch.zeros(1, 3, 8, 8),
            0.5,
            generator=torch.Generator().manual_seed(0),
        )
        ids_keep = torch.argsort(ids_restore, dim=1)[:, :2]
        expected = gather_tokens(enc.norm(enc.pos_embed), ids_keep)
        self.assertTrue(torch.allclose(x_keep, expected, atol=1e-6))


class TestMAEDecoder(unittest.TestCase):
    def _make(self):
        torch.manual_seed(0)
        return MAEDecoder(
            num_patches=16,
            patch_size=4,
            in_channels=3,
            encoder_dim=32,
            decoder_dim=16,
            depth=1,
            heads=2,
        )

    def test_output_shape(self):
        dec = self._make()
        x_keep = torch.randn(2, 4, 32)
        ids_restore = torch.argsort(torch.rand(2, 16), dim=1)
        pred = dec(x_keep, ids_restore)
        self.assertEqual(pred.shape, (2, 16, 48))
        self.assertTrue(torch.isfinite(pred).all())

    def test_assembles_visible_and_mask_tokens(self):
        dec = self._make()
        b, k, n = 2, 4, 16
        x_keep = torch.randn(b, k, 32)
        ids_restore = torch.argsort(torch.rand(b, n), dim=1)
        tokens = dec.assemble_tokens(x_keep, ids_restore)
        self.assertEqual(tokens.shape, (b, n, 16))

        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_keep = ids_shuffle[:, :k]
        ids_masked = ids_shuffle[:, k:]
        visible = gather_tokens(tokens, ids_keep)
        self.assertTrue(torch.allclose(visible, dec.enc_to_dec(x_keep)))
        masked = gather_tokens(tokens, ids_masked)
        expected = dec.mask_token.expand(b, n - k, -1)
        self.assertTrue(torch.allclose(masked, expected))


class TestMaskedAutoencoder(unittest.TestCase):
    def _tiny(self, **kwargs):
        torch.manual_seed(0)
        config = dict(
            image_size=16,
            patch_size=4,
            in_channels=3,
            encoder_dim=32,
            encoder_depth=1,
            encoder_heads=4,
            decoder_dim=16,
            decoder_depth=1,
            decoder_heads=2,
            mlp_ratio=2.0,
            mask_ratio=0.75,
        )
        config.update(kwargs)
        return MaskedAutoencoder(**config)

    def test_forward_shapes(self):
        model = self._tiny()
        loss, pred, mask = model(torch.randn(2, 3, 16, 16))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(pred.shape, (2, 16, 48))
        self.assertEqual(mask.shape, (2, 16))
        self.assertEqual(int(mask.sum()), 2 * 12)

    def test_parameter_count(self):
        torch.manual_seed(0)
        model = MaskedAutoencoder()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6_184_560)

    def test_backward_reaches_encoder_and_decoder(self):
        model = self._tiny()
        loss, _, _ = model(torch.randn(2, 3, 16, 16))
        loss.backward()

        self.assertIsNotNone(model.encoder.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.encoder.blocks[0].attn.to_qkv.weight.grad)
        self.assertIsNotNone(model.decoder.mask_token.grad)
        self.assertIsNotNone(model.decoder.pred.weight.grad)
        self.assertTrue(
            torch.isfinite(model.encoder.patch_embed.proj.weight.grad).all()
        )

    def test_encode_returns_full_grid(self):
        model = self._tiny()
        x = model.encode(torch.randn(2, 3, 16, 16))
        self.assertEqual(x.shape, (2, 16, 32))
        self.assertTrue(torch.isfinite(x).all())

    def test_reproducible_with_generator(self):
        model = self._tiny()
        images = torch.randn(2, 3, 16, 16)
        out1 = model(images, generator=torch.Generator().manual_seed(123))
        out2 = model(images, generator=torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(out1[0], out2[0]))
        self.assertTrue(torch.equal(out1[1], out2[1]))
        self.assertTrue(torch.equal(out1[2], out2[2]))

    def test_norm_pix_loss_variant(self):
        model = self._tiny(norm_pix_loss=True)
        loss, _, _ = model(torch.randn(2, 3, 16, 16))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_overfits_single_batch(self):
        torch.manual_seed(0)
        model = MaskedAutoencoder(
            image_size=8,
            patch_size=4,
            in_channels=3,
            encoder_dim=16,
            encoder_depth=1,
            encoder_heads=2,
            decoder_dim=16,
            decoder_depth=1,
            decoder_heads=2,
            mlp_ratio=2.0,
            mask_ratio=0.5,
        )
        images = torch.randn(2, 3, 8, 8)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

        first = None
        last = None
        for step in range(80):
            # Fixed seed -> same mask every step, so we can check memorization of this batch
            loss, _, _ = model(
                images, generator=torch.Generator().manual_seed(0)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())

        self.assertLess(last, 0.6 * first)

    def test_invalid_configs_raise(self):
        with self.assertRaises(ValueError):
            MaskedAutoencoder(image_size=30, patch_size=4)
        with self.assertRaises(ValueError):
            MaskedAutoencoder(mask_ratio=0.0)
        with self.assertRaises(ValueError):
            MaskedAutoencoder(mask_ratio=1.0)
        with self.assertRaises(ValueError):
            MaskedAutoencoder(encoder_dim=32, encoder_heads=5)

    def test_wrong_input_raises(self):
        model = self._tiny()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 8, 16))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 16, 16))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
