"""Tests for S4-4.5: PatchEmbed2D / SwinStage / SwinTransformer.

Run from the project root:
    python tests/test_swin_model.py

The learner writes swin.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.swin import (  # noqa: E402
    PatchEmbed2D,
    SwinStage,
    SwinTransformer,
)


class TestPatchEmbed2D(unittest.TestCase):
    def test_output_shape(self):
        m = PatchEmbed2D(in_channels=3, dim=64, patch_size=4)
        y = m(torch.randn(2, 3, 32, 32))
        self.assertEqual(y.shape, (2, 8, 8, 64))
        self.assertTrue(torch.isfinite(y).all())

    def test_non_divisible_spatial_raises(self):
        m = PatchEmbed2D(in_channels=3, dim=64, patch_size=4)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 3, 30, 32))
        with self.assertRaises(ValueError):
            m(torch.randn(2, 3, 32, 30))

    def test_channel_mismatch_raises(self):
        m = PatchEmbed2D(in_channels=3, dim=64, patch_size=4)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 1, 32, 32))


class TestSwinStage(unittest.TestCase):
    def _make(self, depth=2, ws=4, shift=2):
        torch.manual_seed(0)
        return SwinStage(64, depth=depth, num_heads=2, window_size=ws, shift_size=shift)

    def test_output_shape(self):
        stage = self._make()
        x = torch.randn(2, 8, 8, 64)
        y = stage(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_shift_schedule(self):
        stage = self._make(depth=2, ws=4, shift=2)
        self.assertEqual([b.shift_size for b in stage.blocks], [0, 2])

    def test_zero_shift_stage(self):
        stage = self._make(depth=3, ws=2, shift=0)
        self.assertEqual([b.shift_size for b in stage.blocks], [0, 0, 0])


class TestSwinTransformer(unittest.TestCase):
    def _make(self, **kwargs):
        torch.manual_seed(0)
        return SwinTransformer(**kwargs)

    def test_output_shape_and_finite(self):
        model = self._make()
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))
        self.assertTrue(torch.isfinite(y).all())

    def test_parameter_count(self):
        model = self._make()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5_407_366)

    def test_relative_bias_parameter_count(self):
        model = self._make()
        no_bias = self._make(use_relative_bias=False)
        diff = sum(p.numel() for p in model.parameters()) - sum(
            p.numel() for p in no_bias.parameters()
        )
        # stage1: 2*49*2, stage2: 2*9*4, stage3: 6*9*8
        self.assertEqual(diff, 700)

    def test_stage_configuration(self):
        model = self._make()
        self.assertEqual(len(model.stages), 3)
        self.assertEqual(len(model.merges), 2)
        self.assertEqual(
            [s.window_size for s in model.stages],
            [4, 2, 2],
        )
        self.assertEqual(
            [[b.shift_size for b in s.blocks] for s in model.stages],
            [[0, 2], [0, 1], [0, 0, 0, 0, 0, 0]],
        )
        self.assertEqual(
            [s.blocks[0].dim for s in model.stages],
            [64, 128, 256],
        )

    def test_backward(self):
        model = self._make()
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        model(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNotNone(model.stages[0].blocks[0].attn.qkv.weight.grad)
        self.assertIsNotNone(
            model.stages[0].blocks[1].attn.relative_position_bias.bias_table.grad
        )
        self.assertIsNotNone(model.stages[2].blocks[-1].mlp.net[0].weight.grad)

    def test_no_relative_bias_variant(self):
        model = self._make(use_relative_bias=False)
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (1, 10))
        self.assertTrue(torch.isfinite(y).all())

    def test_train_eval_dropout(self):
        model = self._make(dropout=0.1, attn_dropout=0.1, drop_path=0.1)
        x = torch.randn(2, 3, 32, 32)

        model.eval()
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        self.assertTrue(torch.allclose(y1, y2))

        model.train()
        torch.manual_seed(0)
        y3 = model(x)
        torch.manual_seed(1)
        y4 = model(x)
        self.assertFalse(torch.allclose(y3, y4))

    def test_invalid_configs_raise(self):
        with self.assertRaises(ValueError):
            SwinTransformer(image_size=30, patch_size=4)
        with self.assertRaises(ValueError):
            SwinTransformer(depths=(2, 2, 6), heads=(2, 4))
        with self.assertRaises(ValueError):
            SwinTransformer(window_sizes=(4, 4, 4))  # stage3 res=2 is not divisible by 4
        with self.assertRaises(ValueError):
            SwinTransformer(heads=(3, 4, 8))  # 64 is not divisible by 3
        with self.assertRaises(ValueError):
            SwinTransformer(drop_path=1.0)
        with self.assertRaises(ValueError):
            SwinTransformer(num_classes=0)

    def test_wrong_input_raises(self):
        model = self._make()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 16, 16))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
