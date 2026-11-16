"""Tests for src/vit_from_scratch/transformer.py (LayerNorm).

Run from the project root:
    python tests/test_transformer.py

The learner writes transformer.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.transformer import (  # noqa: E402
    DropPath,
    LayerNorm,
    MLP,
    TransformerBlock,
)


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, 16, dtype=torch.float64)

    def test_matches_nn_layer_norm(self):
        ours = LayerNorm(16).double()
        ref = nn.LayerNorm(16).double()
        self.assertTrue(torch.allclose(ours(self.x), ref(self.x), atol=1e-6))

    def test_output_statistics(self):
        # With gamma=1, beta=0, every token is zero-mean and unit-variance.
        y = LayerNorm(16).double()(self.x)
        self.assertTrue(
            torch.allclose(y.mean(dim=-1), torch.zeros(2, 5, dtype=torch.float64), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                y.var(dim=-1, unbiased=False),
                torch.ones(2, 5, dtype=torch.float64),
                atol=1e-4,
            )
        )

    def test_shift_scale_invariance(self):
        # With eps=0 and identity affine, LN(2x + 3) == LN(x).
        ln = LayerNorm(16, eps=0.0).double()
        self.assertTrue(torch.allclose(ln(self.x * 2 + 3), ln(self.x), atol=1e-12))

    def test_tokens_are_independent(self):
        ln = LayerNorm(16).double()
        x2 = self.x.clone()
        x2[:, 0, :] += torch.randn(2, 16, dtype=torch.float64)
        y1, y2 = ln(self.x), ln(x2)

        self.assertTrue(torch.allclose(y1[:, 1:], y2[:, 1:], atol=1e-12))
        self.assertFalse(torch.allclose(y1[:, 0], y2[:, 0]))

    def test_gradients_match_nn_layer_norm(self):
        x1 = self.x.clone().requires_grad_(True)
        x2 = self.x.clone().requires_grad_(True)
        ours = LayerNorm(16).double()
        ref = nn.LayerNorm(16).double()
        upstream = torch.randn_like(self.x)

        (ours(x1) * upstream).sum().backward()
        (ref(x2) * upstream).sum().backward()

        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=1e-6))
        self.assertTrue(torch.allclose(ours.gamma.grad, ref.weight.grad, atol=1e-6))
        self.assertTrue(torch.allclose(ours.beta.grad, ref.bias.grad, atol=1e-6))

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in LayerNorm(16).parameters()), 2 * 16)

    def test_accepts_2d_input(self):
        x = torch.randn(3, 16, dtype=torch.float64)
        self.assertEqual(LayerNorm(16).double()(x).shape, (3, 16))


class TestMLP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, 16, dtype=torch.float64)

    def test_output_shape(self):
        mlp = MLP(16, mlp_dim=64)
        self.assertEqual(mlp(torch.randn(2, 5, 16)).shape, (2, 5, 16))

    def test_parameter_count(self):
        mlp = MLP(16, mlp_dim=64)
        expected = 2 * 16 * 64 + 64 + 16
        self.assertEqual(sum(p.numel() for p in mlp.parameters()), expected)

    def test_matches_manual(self):
        mlp = MLP(16, mlp_dim=64).double().eval()
        hidden = F.gelu(
            F.linear(self.x, mlp.net[0].weight, mlp.net[0].bias)
        )
        expected = F.linear(hidden, mlp.net[3].weight, mlp.net[3].bias)
        self.assertTrue(torch.allclose(mlp(self.x), expected, atol=1e-12))

    def test_gradients_flow(self):
        mlp = MLP(16, mlp_dim=64)
        x = torch.randn(2, 5, 16, requires_grad=True)
        mlp(x).sum().backward()
        self.assertIsNotNone(x.grad)
        for name, p in mlp.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)


class TestDropPath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(64, 4, 8)

    def test_eval_is_identity(self):
        drop = DropPath(0.5).eval()
        self.assertTrue(torch.allclose(drop(self.x), self.x))

    def test_zero_prob_is_identity(self):
        drop = DropPath(0.0)
        self.assertTrue(torch.allclose(drop(self.x), self.x))

    def test_drops_whole_sample_and_rescales(self):
        drop = DropPath(0.5)
        out = drop(self.x)

        dropped = out.flatten(1).abs().sum(dim=1) == 0
        kept = ~dropped

        # Both outcomes must occur in a batch of 64.
        self.assertTrue(dropped.any())
        self.assertTrue(kept.any())
        # A kept sample is scaled by 1 / keep_prob = 2 on every token.
        self.assertTrue(torch.allclose(out[kept], self.x[kept] * 2))

    def test_invalid_prob_raises(self):
        with self.assertRaises(ValueError):
            DropPath(1.0)
        with self.assertRaises(ValueError):
            DropPath(-0.1)


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, 16, dtype=torch.float64)

    def _block(self, drop_path=0.0):
        return TransformerBlock(
            dim=16,
            heads=4,
            dim_head=4,
            mlp_dim=64,
            dropout=0.0,
            drop_path=drop_path,
        ).double()

    def test_output_shape(self):
        block = self._block().eval()
        self.assertEqual(block(self.x).shape, (2, 5, 16))

    def test_matches_manual_pre_ln(self):
        # drop_path > 0 in eval mode must be identity.
        block = self._block(drop_path=0.5).eval()
        expected = self.x + block.attn(block.norm1(self.x))
        expected = expected + block.mlp(block.norm2(expected))
        self.assertTrue(torch.allclose(block(self.x), expected, atol=1e-12))

    def test_parameter_count(self):
        block = self._block()
        # 2 LayerNorms: 2 * 2*16; attention: 16*48 + 16*16 + 16; MLP: 2*16*64 + 64 + 16.
        expected = 2 * (2 * 16) + (16 * 48 + 16 * 16 + 16) + (2 * 16 * 64 + 64 + 16)
        self.assertEqual(sum(p.numel() for p in block.parameters()), expected)

    def test_gradients_flow(self):
        block = self._block()
        x = torch.randn(2, 5, 16, requires_grad=True)
        block(x).sum().backward()
        self.assertIsNotNone(x.grad)
        for name, p in block.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_mask_is_forwarded_to_attention(self):
        block = self._block().eval()
        mask = torch.eye(5, dtype=torch.bool)
        expected = self.x + block.attn(block.norm1(self.x), mask=mask)
        expected = expected + block.mlp(block.norm2(expected))
        self.assertTrue(torch.allclose(block(self.x, mask=mask), expected, atol=1e-12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
