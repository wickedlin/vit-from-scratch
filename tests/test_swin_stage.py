"""Tests for S4-4.3: PatchMerging / RelativePositionBias.

Run from the project root:
    python tests/test_swin_stage.py

The learner writes swin.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.swin import PatchMerging, RelativePositionBias  # noqa: E402


class TestPatchMerging(unittest.TestCase):
    def test_output_shape(self):
        m = PatchMerging(dim=32)
        y = m(torch.randn(2, 8, 6, 32))
        self.assertEqual(y.shape, (2, 4, 3, 64))

    def test_matches_manual_concat_order(self):
        torch.manual_seed(0)
        dim = 8
        m = PatchMerging(dim)
        x = torch.randn(2, 4, 4, dim)
        with torch.no_grad():
            x0 = x[:, 0::2, 0::2, :]  # top-left
            x1 = x[:, 1::2, 0::2, :]  # bottom-left
            x2 = x[:, 0::2, 1::2, :]  # top-right
            x3 = x[:, 1::2, 1::2, :]  # bottom-right
            h = torch.cat([x0, x1, x2, x3], dim=-1)
            h = F.layer_norm(
                h, (4 * dim,), m.norm.weight, m.norm.bias, m.norm.eps
            )
            expected = F.linear(h, m.reduction.weight)
        self.assertTrue(torch.allclose(m(x), expected, atol=1e-6))

    def test_parameter_count_and_no_bias(self):
        dim = 8
        m = PatchMerging(dim)
        self.assertIsNone(m.reduction.bias)
        self.assertEqual(
            sum(p.numel() for p in m.parameters()), 8 * dim * dim + 8 * dim
        )

    def test_gradient_flows(self):
        # Note: cannot assert x.grad is all ones -- the path has LayerNorm, so the gradient is not identity
        m = PatchMerging(4)
        x = torch.randn(1, 4, 4, 4, requires_grad=True)
        m(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(m.norm.weight.grad)
        self.assertIsNotNone(m.reduction.weight.grad)

    def test_odd_grid_raises(self):
        m = PatchMerging(4)
        with self.assertRaises(ValueError):
            m(torch.randn(1, 7, 8, 4))
        with self.assertRaises(ValueError):
            m(torch.randn(1, 8, 5, 4))

    def test_non_4d_raises(self):
        with self.assertRaises(ValueError):
            PatchMerging(4)(torch.randn(8, 8, 4))

    def test_stage_chain_shapes(self):
        x = torch.randn(2, 16, 16, 64)
        x = PatchMerging(64)(x)
        self.assertEqual(x.shape, (2, 8, 8, 128))
        x = PatchMerging(128)(x)
        self.assertEqual(x.shape, (2, 4, 4, 256))


class TestRelativePositionBias(unittest.TestCase):
    def test_index_shape_range_dtype(self):
        m = RelativePositionBias(window_size=4, num_heads=2)
        idx = m.relative_position_index
        self.assertEqual(idx.shape, (16, 16))
        self.assertEqual(idx.dtype, torch.long)
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), (2 * 4 - 1) ** 2)
        self.assertFalse(idx.requires_grad)

    def test_index_matches_formula(self):
        M = 4
        idx = RelativePositionBias(M, 2).relative_position_index
        for i, j in [(0, 0), (0, 15), (15, 0), (5, 10), (3, 12), (9, 9)]:
            r1, c1 = divmod(i, M)
            r2, c2 = divmod(j, M)
            want = ((r1 - r2) + M - 1) * (2 * M - 1) + ((c1 - c2) + M - 1)
            self.assertEqual(int(idx[i, j]), want, msg=f"i={i}, j={j}")

    def test_mirror_index_identity(self):
        M = 5
        idx = RelativePositionBias(M, 3).relative_position_index
        self.assertTrue(torch.equal(idx.T, (2 * M - 1) ** 2 - 1 - idx))

    def test_diagonal_is_center_index(self):
        for M in (2, 4, 7):
            idx = RelativePositionBias(M, 1).relative_position_index
            center = (M - 1) * (2 * M - 1) + (M - 1)
            self.assertTrue(torch.all(torch.diagonal(idx) == center))

    def test_forward_shape(self):
        self.assertEqual(RelativePositionBias(4, 3)().shape, (3, 16, 16))

    def test_gather_values(self):
        M, H = 4, 2
        m = RelativePositionBias(M, H)
        table = torch.arange((2 * M - 1) ** 2).float()
        with torch.no_grad():
            m.bias_table.copy_(table.unsqueeze(1).expand(-1, H))
        out = m()
        idx = m.relative_position_index.float()
        for h in range(H):
            self.assertTrue(torch.equal(out[h], idx))

    def test_gradient_reaches_table(self):
        M, H = 4, 2
        m = RelativePositionBias(M, H)
        m().sum().backward()
        self.assertIsNotNone(m.bias_table.grad)
        self.assertEqual(int(m.bias_table.grad.sum().item()), M * M * M * M * H)


if __name__ == "__main__":
    unittest.main(verbosity=2)
