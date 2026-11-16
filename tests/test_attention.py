"""Tests for src/vit_from_scratch/attention.py.

Run from the project root:
    python tests/test_attention.py

The learner writes attention.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.attention import (  # noqa: E402
    MultiHeadSelfAttention,
    scaled_dot_product_attention,
    scaled_dot_product_attention_naive,
)


class TestSDPA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.q = torch.randn(2, 3, 5, 8, dtype=torch.float64)
        self.k = torch.randn(2, 3, 7, 8, dtype=torch.float64)
        self.v = torch.randn(2, 3, 7, 6, dtype=torch.float64)

    def test_matches_torch_sdpa(self):
        out = scaled_dot_product_attention(self.q, self.k, self.v)
        ref = F.scaled_dot_product_attention(self.q, self.k, self.v)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_naive_matches_vectorized(self):
        q = torch.randn(1, 2, 3, 4, dtype=torch.float64)
        k = torch.randn(1, 2, 5, 4, dtype=torch.float64)
        v = torch.randn(1, 2, 5, 4, dtype=torch.float64)
        out = scaled_dot_product_attention(q, k, v)
        ref = scaled_dot_product_attention_naive(q, k, v)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_bool_mask_zeros_weights(self):
        # v = identity -> each output row is the attention weight vector.
        q = torch.randn(1, 1, 4, 8, dtype=torch.float64)
        k = torch.randn(1, 1, 6, 8, dtype=torch.float64)
        v = torch.eye(6, dtype=torch.float64)[None, None]
        mask = torch.ones(4, 6, dtype=torch.bool)
        mask[:, -2:] = False

        out = scaled_dot_product_attention(q, k, v, mask=mask)

        self.assertAlmostEqual(out[..., -2:].abs().max().item(), 0.0)
        self.assertTrue(
            torch.allclose(
                out.sum(dim=-1),
                torch.ones(1, 1, 4, dtype=torch.float64),
            )
        )

        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_causal_matches_torch(self):
        q = torch.randn(1, 2, 5, 8, dtype=torch.float64)
        k = torch.randn(1, 2, 5, 8, dtype=torch.float64)
        v = torch.randn(1, 2, 5, 8, dtype=torch.float64)

        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_gradients_match_torch(self):
        q1 = self.q.clone().requires_grad_(True)
        k1 = self.k.clone().requires_grad_(True)
        v1 = self.v.clone().requires_grad_(True)
        q2 = self.q.clone().requires_grad_(True)
        k2 = self.k.clone().requires_grad_(True)
        v2 = self.v.clone().requires_grad_(True)

        scaled_dot_product_attention(q1, k1, v1).sum().backward()
        F.scaled_dot_product_attention(q2, k2, v2).sum().backward()

        for a, b in ((q1.grad, q2.grad), (k1.grad, k2.grad), (v1.grad, v2.grad)):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))


class TestMHSA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shape(self):
        mha = MultiHeadSelfAttention(dim=16, heads=4, dim_head=4)
        x = torch.randn(2, 7, 16)
        self.assertEqual(mha(x).shape, (2, 7, 16))

    def test_heads_one_matches_single_head(self):
        # heads=1, dim_head=dim must reduce to plain single-head attention.
        mha = MultiHeadSelfAttention(dim=16, heads=1, dim_head=16).double()
        x = torch.randn(2, 5, 16, dtype=torch.float64)

        w = mha.to_qkv.weight  # (3*dim, dim), row blocks [W_q; W_k; W_v]
        q, k, v = x @ w[:16].T, x @ w[16:32].T, x @ w[32:].T
        manual = scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        ).squeeze(1)
        manual = mha.to_out[0](manual)

        self.assertTrue(torch.allclose(mha(x), manual, atol=1e-12))

    def test_single_key_mask(self):
        # query i may only see key i -> each head outputs the value of token i.
        mha = MultiHeadSelfAttention(dim=16, heads=4, dim_head=4).double()
        x = torch.randn(2, 5, 16, dtype=torch.float64)
        mask = torch.eye(5, dtype=torch.bool)
        v = mha.to_qkv(x).chunk(3, dim=-1)[2].reshape(2, 5, 16)
        expected = mha.to_out[0](v)

        self.assertTrue(torch.allclose(mha(x, mask=mask), expected, atol=1e-12))

    def test_ones_mask_matches_no_mask(self):
        mha = MultiHeadSelfAttention(dim=16, heads=4, dim_head=4).double()
        x = torch.randn(2, 5, 16, dtype=torch.float64)
        # A mask shaped (B, 1, N, M) broadcasts across heads.
        mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)

        self.assertTrue(torch.allclose(mha(x, mask=mask), mha(x), atol=1e-12))

    def test_matches_lucidrains(self):
        # vit-pytorch 1.26.12 has a pre-LN inside Attention; strip it so this
        # test compares the attention math only.
        from vit_pytorch.vit import Attention as ReferenceAttention

        mha = MultiHeadSelfAttention(dim=16, heads=2, dim_head=8).double()
        ref = ReferenceAttention(dim=16, heads=2, dim_head=8, dropout=0.0).double()
        ref.norm = torch.nn.Identity()

        mha.load_state_dict(
            {
                "to_qkv.weight": ref.to_qkv.weight,
                "to_out.0.weight": ref.to_out[0].weight,
                "to_out.0.bias": ref.to_out[0].bias,
            }
        )
        x = torch.randn(3, 5, 16, dtype=torch.float64)

        self.assertTrue(torch.allclose(mha(x), ref(x), atol=1e-6))

    def test_gradients_flow(self):
        mha = MultiHeadSelfAttention(dim=16, heads=4, dim_head=4)
        x = torch.randn(2, 5, 16, requires_grad=True)
        mha(x).sum().backward()

        self.assertIsNotNone(x.grad)
        for name, p in mha.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_parameter_count(self):
        mha = MultiHeadSelfAttention(dim=16, heads=4, dim_head=4)
        # to_qkv: 16 x 48 (no bias); to_out: 16 x 16 weight + 16 bias.
        expected = 16 * 48 + 16 * 16 + 16
        self.assertEqual(sum(p.numel() for p in mha.parameters()), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
