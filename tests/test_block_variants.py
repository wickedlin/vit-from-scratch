"""S1 phase acceptance: TransformerBlock with a pre_norm switch.

Target API:
    TransformerBlock(..., pre_norm: bool = True)

- pre_norm=True  -> x = x + DropPath(Attn(LN1(x))); x = x + DropPath(MLP(LN2(x)))
- pre_norm=False -> x = LN1(x + DropPath(Attn(x))); x = LN2(x + DropPath(MLP(x)))

Run after adding the parameter:
    python tests/test_block_variants.py

Expected: Ran 5 tests ... OK
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.transformer import TransformerBlock  # noqa: E402


class _Half(nn.Module):
    """Deterministic stand-in for DropPath: scales the branch by 0.5."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x


class TestPreNormSwitch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, 16, dtype=torch.float64)

    def _block(self, pre_norm: bool) -> TransformerBlock:
        torch.manual_seed(1)
        block = TransformerBlock(
            dim=16,
            heads=4,
            dim_head=4,
            mlp_dim=32,
            dropout=0.0,
            drop_path=0.0,
            pre_norm=pre_norm,
        )
        return block.double()

    def test_pre_norm_matches_manual(self):
        block = self._block(pre_norm=True)
        expected = self.x + block.attn(block.norm1(self.x))
        expected = expected + block.mlp(block.norm2(expected))
        self.assertTrue(torch.allclose(block(self.x), expected, atol=1e-12))

    def test_post_norm_matches_manual(self):
        block = self._block(pre_norm=False)
        expected = block.norm1(self.x + block.attn(self.x))
        expected = block.norm2(expected + block.mlp(expected))
        self.assertTrue(torch.allclose(block(self.x), expected, atol=1e-12))

    def test_both_modes_have_finite_gradients(self):
        for pre_norm in (True, False):
            block = self._block(pre_norm=pre_norm)
            block(self.x).sum().backward()
            for name, param in block.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_pre_norm_drop_path_wraps_branch_only(self):
        block = self._block(pre_norm=True)
        block.drop_path = _Half()
        expected = self.x + 0.5 * block.attn(block.norm1(self.x))
        expected = expected + 0.5 * block.mlp(block.norm2(expected))
        self.assertTrue(torch.allclose(block(self.x), expected, atol=1e-12))

    def test_post_norm_drop_path_wraps_branch_only(self):
        block = self._block(pre_norm=False)
        block.drop_path = _Half()
        expected = block.norm1(self.x + 0.5 * block.attn(self.x))
        expected = block.norm2(expected + 0.5 * block.mlp(expected))
        self.assertTrue(torch.allclose(block(self.x), expected, atol=1e-12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
