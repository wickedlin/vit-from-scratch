"""Tests for S5-5.2: MAE random masking / unshuffle / masked loss.

Run from the project root:
    python tests/test_mae.py

The learner writes mae.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.mae import mae_loss, random_masking, unshuffle_tokens  # noqa: E402


def ids_keep_from(ids_restore: torch.Tensor, len_keep: int) -> torch.Tensor:
    """ids_shuffle = argsort(ids_restore); visible ids are its first len_keep entries."""
    return torch.argsort(ids_restore, dim=1)[:, :len_keep]


def gather_tokens(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, ids.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class TestRandomMasking(unittest.TestCase):
    def test_shapes_and_counts(self):
        cases = [
            (4, 64, 8, 0.75, 16),  # CIFAR-10 32x32 / patch 4
            (2, 144, 8, 0.75, 36),  # STL-10 96x96 / patch 8
            (3, 16, 5, 0.5, 8),
        ]
        for b, n, d, ratio, k in cases:
            with self.subTest(b=b, n=n, ratio=ratio):
                x = torch.randn(b, n, d)
                x_keep, mask, ids_restore = random_masking(x, ratio)
                self.assertEqual(x_keep.shape, (b, k, d))
                self.assertEqual(mask.shape, (b, n))
                self.assertEqual(ids_restore.shape, (b, n))
                self.assertEqual(mask.dtype, torch.float32)
                self.assertEqual(ids_restore.dtype, torch.int64)
                self.assertTrue(
                    torch.equal(mask.sum(1), torch.full((b,), float(n - k)))
                )
                self.assertTrue(set(mask.unique().tolist()) <= {0.0, 1.0})
                self.assertTrue(
                    torch.equal(
                        ids_restore.sort(dim=1).values,
                        torch.arange(n).expand(b, n),
                    )
                )

    def test_gather_uses_same_patch_index_for_all_channels(self):
        b, n, d, k = 2, 16, 6, 4
        x = torch.arange(b * n * d, dtype=torch.float32).reshape(b, n, d)
        x_keep, _, ids_restore = random_masking(
            x, 1 - k / n, generator=torch.Generator().manual_seed(0)
        )
        ids_keep = ids_keep_from(ids_restore, k)
        self.assertTrue(torch.equal(x_keep, gather_tokens(x, ids_keep)))
        # Each row maps to the same original patch; all channels use the same index
        for i in range(b):
            for j in range(k):
                self.assertTrue(torch.equal(x_keep[i, j], x[i, ids_keep[i, j]]))

    def test_mask_marks_exactly_the_dropped_patches(self):
        b, n, k = 4, 32, 8
        x = torch.randn(b, n, 3)
        _, mask, ids_restore = random_masking(
            x, 1 - k / n, generator=torch.Generator().manual_seed(1)
        )
        ids_keep = ids_keep_from(ids_restore, k)
        self.assertTrue(
            torch.equal(torch.gather(mask, 1, ids_keep), torch.zeros(b, k))
        )
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_masked = ids_shuffle[:, k:]
        self.assertTrue(
            torch.equal(torch.gather(mask, 1, ids_masked), torch.ones(b, n - k))
        )

    def test_ids_restore_is_inverse_permutation(self):
        b, n, k = 5, 24, 6
        x = torch.randn(b, n, 4)
        _, _, ids_restore = random_masking(
            x, 1 - k / n, generator=torch.Generator().manual_seed(2)
        )
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_keep = ids_shuffle[:, :k]
        self.assertTrue(
            torch.equal(
                torch.gather(ids_restore, 1, ids_keep),
                torch.arange(k).expand(b, k),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.gather(ids_restore, 1, ids_shuffle),
                torch.arange(n).expand(b, n),
            )
        )

    def test_reproducible_with_generator(self):
        x = torch.randn(3, 20, 5)
        out1 = random_masking(x, 0.5, generator=torch.Generator().manual_seed(42))
        out2 = random_masking(x, 0.5, generator=torch.Generator().manual_seed(42))
        out3 = random_masking(x, 0.5, generator=torch.Generator().manual_seed(43))
        self.assertTrue(torch.equal(out1[2], out2[2]))
        self.assertTrue(torch.equal(out1[1], out2[1]))
        self.assertFalse(torch.equal(out1[2], out3[2]))

    def test_invalid_inputs_raise(self):
        x = torch.randn(2, 8, 4)
        for ratio in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    random_masking(x, ratio)
        with self.assertRaises(ValueError):
            random_masking(torch.randn(2, 8), 0.5)


class TestUnshuffle(unittest.TestCase):
    def test_roundtrip_restores_full_permutation(self):
        b, n, d, k = 3, 32, 7, 8
        x = torch.randn(b, n, d)
        _, _, ids_restore = random_masking(
            x, 1 - k / n, generator=torch.Generator().manual_seed(3)
        )
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        shuffled = gather_tokens(x, ids_shuffle)
        self.assertTrue(torch.equal(unshuffle_tokens(shuffled, ids_restore), x))

    def test_visible_slots_roundtrip_and_masked_slots_are_filler(self):
        b, n, d, k = 2, 32, 6, 8
        x = torch.randn(b, n, d)
        x_keep, mask, ids_restore = random_masking(
            x, 1 - k / n, generator=torch.Generator().manual_seed(4)
        )
        filler = torch.full((b, n - k, d), -7.0)
        x_shuffled = torch.cat([x_keep, filler], dim=1)
        restored = unshuffle_tokens(x_shuffled, ids_restore)

        ids_keep = ids_keep_from(ids_restore, k)
        ids_masked = torch.argsort(ids_restore, dim=1)[:, k:]
        self.assertTrue(torch.equal(gather_tokens(restored, ids_keep), x_keep))
        self.assertTrue(
            torch.equal(
                gather_tokens(restored, ids_masked),
                torch.full((b, n - k, d), -7.0),
            )
        )
        self.assertTrue(
            torch.equal(
                gather_tokens(restored, ids_keep),
                gather_tokens(x, ids_keep),
            )
        )
        self.assertEqual(int(mask.sum()), b * (n - k))

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            unshuffle_tokens(
                torch.randn(2, 8, 4), torch.zeros(2, 7, dtype=torch.long)
            )
        with self.assertRaises(ValueError):
            unshuffle_tokens(torch.randn(2, 8), torch.zeros(2, 8, dtype=torch.long))


class TestMaskedLoss(unittest.TestCase):
    def test_visible_errors_do_not_contribute(self):
        b, n, dp = 2, 4, 3
        target = torch.zeros(b, n, dp)
        pred = torch.zeros(b, n, dp)
        mask = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0],
             [0.0, 1.0, 1.0, 0.0]]
        )
        visible = (mask == 0).unsqueeze(-1)
        pred = torch.where(visible, torch.full_like(pred, 100.0), pred)
        self.assertEqual(float(mae_loss(pred, target, mask)), 0.0)

    def test_masked_errors_only_and_matches_manual(self):
        torch.manual_seed(0)
        b, n, dp = 3, 16, 5
        pred = torch.randn(b, n, dp)
        target = torch.randn(b, n, dp)
        mask = (torch.rand(b, n) > 0.4).float()
        mask[0, 0] = 1.0
        got = mae_loss(pred, target, mask)
        want_per_patch = ((pred - target) ** 2).mean(dim=-1)
        want = (want_per_patch * mask).sum() / mask.sum()
        self.assertTrue(torch.allclose(got, want))
        # Equivalent to "select masked elements first, then average"
        flat = ((pred - target) ** 2)[mask.bool()].mean()
        self.assertTrue(torch.allclose(got, flat))
        self.assertFalse(torch.allclose(got, ((pred - target) ** 2).mean()))

    def test_gradient_is_zero_on_visible_patches(self):
        b, n, dp = 2, 8, 4
        pred = torch.randn(b, n, dp, requires_grad=True)
        target = torch.randn(b, n, dp)
        mask = torch.zeros(b, n)
        mask[:, n // 2:] = 1.0
        loss = mae_loss(pred, target, mask)
        loss.backward()
        self.assertTrue(torch.all(pred.grad[mask == 0] == 0))
        self.assertTrue(torch.all(pred.grad[mask == 1] != 0))

    def test_no_masked_patch_raises(self):
        with self.assertRaises(ValueError):
            mae_loss(torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), torch.zeros(2, 4))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mae_loss(torch.zeros(2, 4, 3), torch.zeros(2, 4, 2), torch.ones(2, 4))
        with self.assertRaises(ValueError):
            mae_loss(torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), torch.ones(2, 3))


class TestNormPixLoss(unittest.TestCase):
    def test_per_patch_normalization_is_affine_invariant(self):
        torch.manual_seed(0)
        b, n, dp = 2, 8, 6
        target = torch.randn(b, n, dp)
        scale = torch.rand(b, n, 1) + 0.5
        shift = torch.randn(b, n, 1) * 10.0
        target2 = target * scale + shift
        mask = torch.ones(b, n)

        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred = (target - mean) / (var + 1e-6) ** 0.5

        loss1 = mae_loss(pred, target, mask, norm_pix_loss=True)
        loss2 = mae_loss(pred, target2, mask, norm_pix_loss=True)
        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-4))
        self.assertLess(float(loss1), 1e-5)

        raw1 = mae_loss(pred, target, mask)
        raw2 = mae_loss(pred, target2, mask)
        self.assertFalse(torch.allclose(raw1, raw2, atol=1e-2))

    def test_constant_patch_is_finite(self):
        target = torch.full((2, 4, 6), 3.0)
        pred = torch.randn(2, 4, 6)
        mask = torch.ones(2, 4)
        loss = mae_loss(pred, target, mask, norm_pix_loss=True)
        self.assertTrue(torch.isfinite(loss))
        # A constant patch normalizes to all zeros, so loss = mean(pred^2)
        self.assertTrue(torch.allclose(loss, (pred ** 2).mean()))

    def test_norm_loss_still_ignores_visible_errors(self):
        b, n, dp = 2, 4, 3
        target = torch.randn(b, n, dp)
        mask = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0],
             [0.0, 1.0, 1.0, 0.0]]
        )
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean) / (var + 1e-6) ** 0.5
        visible = (mask == 0).unsqueeze(-1)
        pred = torch.where(visible, torch.full_like(target, 100.0), target_norm)
        self.assertEqual(
            float(mae_loss(pred, target, mask, norm_pix_loss=True)), 0.0
        )


if __name__ == "__main__":
    unittest.main()
