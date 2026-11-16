"""Tests for S4-4.4: WindowAttention / SwinTransformerBlock.

Run from the project root:
    python tests/test_swin_block.py

The learner writes swin.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.swin import (  # noqa: E402
    SwinTransformerBlock,
    WindowAttention,
)


class TestWindowAttention(unittest.TestCase):
    def _make(self, dim=16, heads=2, ws=4, **kwargs):
        torch.manual_seed(0)
        return WindowAttention(dim, heads, ws, **kwargs)

    def test_output_shape_and_finite(self):
        attn = self._make()
        x = torch.randn(8, 16, 16)  # B_=8, N=M^2=16, C=16
        y = attn(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_gradients(self):
        attn = self._make()
        x = torch.randn(8, 16, 16, requires_grad=True)
        attn(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(attn.qkv.weight.grad)
        self.assertIsNotNone(attn.proj.weight.grad)
        self.assertIsNotNone(attn.relative_position_bias.bias_table.grad)

    def test_mask_blocks_pair(self):
        attn = self._make()
        attn.eval()
        x = torch.randn(1, 16, 16)

        mask = torch.ones(1, 16, 16, dtype=torch.bool)
        mask[0, 0, 5] = False  # query 0 cannot attend to key 5

        out = attn(x, mask=mask)
        x2 = x.clone()
        x2[0, 5] += 10.0  # change the masked token
        out2 = attn(x2, mask=mask)
        self.assertTrue(torch.allclose(out[0, 0], out2[0, 0], atol=1e-6))

        # Control: without masking, token 0's output must change
        mask_all = torch.ones(1, 16, 16, dtype=torch.bool)
        out_c = attn(x, mask=mask_all)
        out_c2 = attn(x2, mask=mask_all)
        self.assertFalse(torch.allclose(out_c[0, 0], out_c2[0, 0], atol=1e-6))

    def test_mask_repeats_across_batch(self):
        attn = self._make()
        attn.eval()
        x = torch.randn(4, 16, 16)  # B=2, nW=2

        mask = torch.ones(2, 16, 16, dtype=torch.bool)
        mask[1, 0, 5] = False  # second window: query 0 cannot attend to key 5

        out = attn(x, mask=mask)
        x2 = x.clone()
        x2[1, 5] += 10.0  # window 1 of batch 0
        x2[3, 5] += 10.0  # window 1 of batch 1
        out2 = attn(x2, mask=mask)
        self.assertTrue(torch.allclose(out[1, 0], out2[1, 0], atol=1e-6))
        self.assertTrue(torch.allclose(out[3, 0], out2[3, 0], atol=1e-6))

    def test_all_true_mask_equals_none(self):
        attn = self._make()
        x = torch.randn(4, 16, 16)
        mask = torch.ones(1, 16, 16, dtype=torch.bool)
        self.assertTrue(torch.allclose(attn(x), attn(x, mask=mask), atol=1e-6))

    def test_relative_bias_affects_output(self):
        attn = self._make()
        x = torch.randn(4, 16, 16)
        out = attn(x)
        with torch.no_grad():
            attn.relative_position_bias.bias_table.normal_(0, 0.5)
        out2 = attn(x)
        self.assertFalse(torch.allclose(out, out2, atol=1e-6))

    def test_no_relative_bias_variant(self):
        attn = self._make(use_relative_bias=False)
        self.assertIsNone(attn.relative_position_bias)
        x = torch.randn(4, 16, 16)
        self.assertEqual(attn(x).shape, x.shape)

    def test_invalid_heads_raises(self):
        with self.assertRaises(ValueError):
            WindowAttention(dim=15, num_heads=4, window_size=4)
        with self.assertRaises(ValueError):
            WindowAttention(dim=16, num_heads=0, window_size=4)

    def test_bad_token_count_raises(self):
        attn = self._make()
        with self.assertRaises(ValueError):
            attn(torch.randn(4, 15, 16))

    def test_bad_mask_shape_raises(self):
        attn = self._make()
        x = torch.randn(4, 16, 16)
        with self.assertRaises(ValueError):
            attn(x, mask=torch.ones(1, 15, 15, dtype=torch.bool))
        with self.assertRaises(ValueError):
            attn(x, mask=torch.ones(3, 16, 16, dtype=torch.bool))


class TestSwinTransformerBlock(unittest.TestCase):
    def _make(self, shift=0, **kwargs):
        torch.manual_seed(0)
        return SwinTransformerBlock(
            dim=16,
            num_heads=4,
            window_size=4,
            shift_size=shift,
            **kwargs,
        )

    def test_shape_and_finite_noshift(self):
        block = self._make(0)
        x = torch.randn(2, 8, 8, 16)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_shape_and_finite_shift(self):
        block = self._make(2)
        x = torch.randn(2, 8, 8, 16)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_backward(self):
        block = self._make(2)
        x = torch.randn(2, 8, 8, 16, requires_grad=True)
        block(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, p in block.named_parameters():
            self.assertIsNotNone(p.grad, msg=name)
            self.assertTrue(torch.isfinite(p.grad).all(), msg=name)

    def test_residual_identity_when_branches_zeroed(self):
        block = self._make(2)
        block.eval()
        with torch.no_grad():
            block.attn.proj.weight.zero_()
            block.attn.proj.bias.zero_()
            block.mlp.net[-2].weight.zero_()
            block.mlp.net[-2].bias.zero_()

        x = torch.randn(2, 8, 8, 16)
        self.assertTrue(torch.allclose(block(x), x, atol=1e-6))

    def test_shift_changes_output(self):
        block0 = self._make(0)
        block1 = copy.deepcopy(block0)
        block1.shift_size = 2
        block0.eval()
        block1.eval()

        x = torch.randn(2, 8, 8, 16)
        self.assertFalse(torch.allclose(block0(x), block1(x), atol=1e-6))

    def test_invalid_shift_raises(self):
        with self.assertRaises(ValueError):
            SwinTransformerBlock(16, 4, 4, shift_size=4)
        with self.assertRaises(ValueError):
            SwinTransformerBlock(16, 4, 4, shift_size=-1)

    def test_single_window_shift_raises(self):
        block = self._make(2)
        with self.assertRaises(ValueError):
            block(torch.randn(1, 4, 4, 16))

    def test_bad_grid_raises(self):
        block = self._make(0)
        with self.assertRaises(ValueError):
            block(torch.randn(1, 6, 8, 16))
        with self.assertRaises(ValueError):
            block(torch.randn(1, 8, 6, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
