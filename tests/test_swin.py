"""Tests for src/vit_from_scratch/swin.py (window partition / reverse / shift / mask).

Run from the project root:
    python tests/test_swin.py

The learner writes swin.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.swin import (  # noqa: E402
    compute_shifted_window_mask,
    shift_window,
    unshift_window,
    window_partition,
    window_reverse,
)


def brute_force_shifted_mask(H: int, W: int, ws: int, s: int) -> torch.Tensor:
    """Brute-force mask from "which corner of the original image each real pixel
    comes from", as a reference for the official 3x3-region method.

    In window k (k = i*nW + j), token (lr, lc) comes from source coordinate
    (i*ws + s + lr, j*ws + s + lc). Since s < ws and H, W are divisible by ws,
    each coordinate goes out of bounds at most once, so two bits (row OOB,
    col OOB) uniquely identify which piece it belongs to.
    """
    nH, nW = H // ws, W // ws
    masks = []
    for i in range(nH):
        for j in range(nW):
            labels = []
            for lr in range(ws):
                for lc in range(ws):
                    r = i * ws + s + lr
                    c = j * ws + s + lc
                    row_piece = 1 if r >= H else 0
                    col_piece = 1 if c >= W else 0
                    labels.append(row_piece * 2 + col_piece)
            labels = torch.tensor(labels)
            masks.append(labels[:, None] == labels[None, :])
    return torch.stack(masks)


class TestWindowPartition(unittest.TestCase):
    def test_partition_shape(self):
        x = torch.randn(2, 8, 8, 3)
        windows = window_partition(x, window_size=4)
        self.assertEqual(windows.shape, (8, 16, 3))

    def test_roundtrip(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 8, 3)
        y = window_reverse(window_partition(x, 4), 4, 8, 8)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x))

    def test_window_order_and_values(self):
        x = torch.arange(2 * 8 * 8).reshape(2, 8, 8, 1).float()
        windows = window_partition(x, 4)
        # Row-major: b -> window row i -> window col j
        self.assertTrue(
            torch.equal(windows[0, :, 0], x[0, 0:4, 0:4, 0].reshape(-1))
        )
        self.assertTrue(
            torch.equal(windows[1, :, 0], x[0, 0:4, 4:8, 0].reshape(-1))
        )
        self.assertTrue(
            torch.equal(windows[2, :, 0], x[0, 4:8, 0:4, 0].reshape(-1))
        )
        self.assertTrue(
            torch.equal(windows[3, :, 0], x[0, 4:8, 4:8, 0].reshape(-1))
        )
        self.assertTrue(
            torch.equal(windows[4, :, 0], x[1, 0:4, 0:4, 0].reshape(-1))
        )

    def test_non_divisible_raises(self):
        x = torch.randn(1, 7, 8, 3)
        with self.assertRaises(ValueError):
            window_partition(x, 4)

    def test_invalid_window_size_raises(self):
        x = torch.randn(1, 8, 8, 3)
        with self.assertRaises(ValueError):
            window_partition(x, 0)


class TestWindowReverse(unittest.TestCase):
    def test_reverse_shape(self):
        windows = torch.randn(8, 16, 5)
        x = window_reverse(windows, 4, 8, 8)
        self.assertEqual(x.shape, (2, 8, 8, 5))

    def test_reverse_non_divisible_raises(self):
        windows = torch.randn(4, 16, 3)
        with self.assertRaises(ValueError):
            window_reverse(windows, 4, 7, 8)

    def test_reverse_bad_window_elems_raises(self):
        windows = torch.randn(4, 15, 3)
        with self.assertRaises(ValueError):
            window_reverse(windows, 4, 8, 8)

    def test_reverse_bad_window_count_raises(self):
        windows = torch.randn(5, 16, 3)  # 4 windows are needed to form a B=1 8x8 image
        with self.assertRaises(ValueError):
            window_reverse(windows, 4, 8, 8)

    def test_gradient_flows(self):
        x = torch.randn(1, 8, 8, 3, requires_grad=True)
        y = window_reverse(window_partition(x, 4), 4, 8, 8)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))


class TestShiftWindow(unittest.TestCase):
    def test_shift_direction(self):
        x = torch.arange(8 * 8).reshape(1, 8, 8, 1).float()
        y = shift_window(x, 2)
        # roll(-2, -2): original (2, 2) moves to (0, 0)
        self.assertTrue(torch.equal(y[0, 0, 0], x[0, 2, 2]))
        self.assertTrue(torch.equal(y[0, 0, 7], x[0, 2, 1]))

    def test_shift_zero_is_identity(self):
        x = torch.randn(2, 8, 8, 3)
        self.assertTrue(torch.equal(shift_window(x, 0), x))

    def test_shift_unshift_roundtrip(self):
        x = torch.randn(2, 8, 8, 3)
        self.assertTrue(torch.allclose(unshift_window(shift_window(x, 3), 3), x))

    def test_shift_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            shift_window(torch.randn(8, 8, 3), 2)
        with self.assertRaises(ValueError):
            shift_window(torch.randn(1, 8, 8, 3), -1)

    def test_gradient_flows_through_shift(self):
        x = torch.randn(1, 8, 8, 3, requires_grad=True)
        unshift_window(shift_window(x, 2), 2).sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))


class TestUnshiftWindow(unittest.TestCase):
    def test_unshift_direction(self):
        x = torch.arange(8 * 8).reshape(1, 8, 8, 1).float()
        y = unshift_window(x, 2)
        # roll(+2, +2): original (0, 0) moves to (2, 2)
        self.assertTrue(torch.equal(y[0, 2, 2], x[0, 0, 0]))
        self.assertTrue(torch.equal(y[0, 2, 1], x[0, 0, 7]))

    def test_unshift_zero_is_identity(self):
        x = torch.randn(1, 8, 8, 3)
        self.assertTrue(torch.equal(unshift_window(x, 0), x))

    def test_unshift_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            unshift_window(torch.randn(8, 8, 3), 2)
        with self.assertRaises(ValueError):
            unshift_window(torch.randn(1, 8, 8, 3), -2)


class TestShiftedWindowMask(unittest.TestCase):
    def test_mask_shape_dtype(self):
        mask = compute_shifted_window_mask(8, 8, 4, 2)
        self.assertEqual(mask.shape, (4, 16, 16))
        self.assertEqual(mask.dtype, torch.bool)

    def test_diagonal_and_symmetry(self):
        mask = compute_shifted_window_mask(8, 8, 4, 2)
        diag = torch.diagonal(mask, dim1=-2, dim2=-1)
        self.assertTrue(diag.all())
        self.assertTrue(torch.equal(mask, mask.transpose(-1, -2)))

    def test_per_window_allowed_fraction(self):
        # 8x8 / ws=4 / s=2: window 0 no wraparound; window 1 horizontal; window 2 vertical; window 3 both
        mask = compute_shifted_window_mask(8, 8, 4, 2)
        frac = mask.float().mean(dim=(-2, -1))
        expected = torch.tensor([1.0, 0.5, 0.5, 0.25])
        self.assertTrue(torch.allclose(frac, expected))

    def test_zero_shift_all_true(self):
        mask = compute_shifted_window_mask(8, 8, 4, 0)
        self.assertEqual(mask.shape, (4, 16, 16))
        self.assertTrue(mask.all())

    def test_matches_brute_force(self):
        cases = [
            (8, 8, 4, 2),
            (12, 16, 4, 2),
            (16, 12, 4, 3),
            (16, 16, 8, 4),
            (12, 12, 4, 1),
        ]
        for H, W, ws, s in cases:
            got = compute_shifted_window_mask(H, W, ws, s)
            want = brute_force_shifted_mask(H, W, ws, s)
            self.assertTrue(
                torch.equal(got, want), msg=f"H={H}, W={W}, ws={ws}, s={s}"
            )

    def test_bad_args_raise(self):
        with self.assertRaises(ValueError):
            compute_shifted_window_mask(8, 8, 4, 4)  # shift must be < ws
        with self.assertRaises(ValueError):
            compute_shifted_window_mask(8, 8, 4, -1)
        with self.assertRaises(ValueError):
            compute_shifted_window_mask(10, 8, 4, 2)  # not divisible
        with self.assertRaises(ValueError):
            compute_shifted_window_mask(4, 4, 4, 2)  # a single window should not shift
        with self.assertRaises(ValueError):
            compute_shifted_window_mask(4, 8, 4, 2)  # H <= ws


class TestShiftedAttentionPipeline(unittest.TestCase):
    def test_shift_partition_reverse_unshift_roundtrip(self):
        x = torch.randn(2, 8, 8, 3)
        ws, s = 4, 2
        windows = window_partition(shift_window(x, s), ws)
        self.assertEqual(windows.shape, (8, 16, 3))
        y = unshift_window(window_reverse(windows, ws, 8, 8), s)
        self.assertTrue(torch.allclose(y, x))

    def test_mask_drops_only_wrapped_pairs(self):
        mask = compute_shifted_window_mask(8, 8, 4, 2)
        # Window 3 (wraparound in both directions): token 0 = (6,6) and token 15 = (1,1) span four pieces, must be masked
        self.assertFalse(mask[3, 0, 15])
        # Within the same original 2x2 piece ((6,6) and (6,7)), must keep
        self.assertTrue(mask[3, 0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
