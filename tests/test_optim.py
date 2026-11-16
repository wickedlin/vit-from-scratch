"""Tests for src/vit_from_scratch/optim.py (S3-3.2).

Run from the project root:
    python tests/test_optim.py -v

The learner writes optim.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.optim import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    clip_gradients,
    label_smoothing_loss,
    split_params_for_weight_decay,
    warmup_cosine_lr,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 16)              # weight (16, 8) + bias (16,)
        self.norm = nn.LayerNorm(16)            # weight (16,) + bias (16,)
        self.proj = nn.Linear(16, 4, bias=False)  # weight (4, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(self.fc(x)))


class TestWeightDecayGroups(unittest.TestCase):
    def test_split_rule(self):
        model = TinyModel()
        decay, no_decay = split_params_for_weight_decay(model)
        decay_ids = {id(p) for p in decay}
        no_decay_ids = {id(p) for p in no_decay}

        self.assertEqual(decay_ids, {id(model.fc.weight), id(model.proj.weight)})
        self.assertEqual(
            no_decay_ids,
            {id(model.fc.bias), id(model.norm.weight), id(model.norm.bias)},
        )
        self.assertTrue(decay_ids.isdisjoint(no_decay_ids))
        self.assertEqual(
            len(decay) + len(no_decay),
            sum(1 for p in model.parameters() if p.requires_grad),
        )

    def test_optimizer_groups(self):
        model = TinyModel()
        opt = build_optimizer(model, lr=1e-3, weight_decay=0.05)
        self.assertEqual(len(opt.param_groups), 2)

        decay_group = opt.param_groups[0]
        no_decay_group = opt.param_groups[1]
        self.assertAlmostEqual(decay_group["weight_decay"], 0.05)
        self.assertAlmostEqual(no_decay_group["weight_decay"], 0.0)
        self.assertAlmostEqual(decay_group["lr"], 1e-3)
        self.assertAlmostEqual(no_decay_group["lr"], 1e-3)

        counted = [p for g in opt.param_groups for p in g["params"]]
        self.assertEqual(
            len(counted),
            sum(1 for p in model.parameters() if p.requires_grad),
        )
        self.assertEqual(len({id(p) for p in counted}), len(counted))

    def test_no_trainable_params_raises(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        for p in model.parameters():
            p.requires_grad = False
        with self.assertRaises(ValueError):
            build_optimizer(model)


class TestAdamWDecay(unittest.TestCase):
    def test_decoupled_weight_decay(self):
        """With zero gradients, AdamW still shrinks params by lr * wd (L2 does not)."""
        p = nn.Parameter(torch.tensor([1.0, -2.0]))
        p.grad = torch.zeros_like(p)
        opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.1)
        opt.step()
        torch.testing.assert_close(
            p.detach(), torch.tensor([0.99, -1.98]), atol=1e-6, rtol=0
        )


class TestSchedule(unittest.TestCase):
    def test_curve_values(self):
        base, warmup, total, min_lr = 1.0, 4, 10, 0.0
        values = [warmup_cosine_lr(t, warmup, total, base, min_lr) for t in range(11)]

        for t, expected in enumerate([0.25, 0.5, 0.75, 1.0]):
            self.assertAlmostEqual(values[t], expected, places=6)
        self.assertAlmostEqual(values[4], 1.0, places=6)
        # The last training step is step=9; step=10 only verifies the endpoint.
        self.assertAlmostEqual(
            values[9], 0.5 * (1 + math.cos(math.pi * 5 / 6)), places=6
        )
        self.assertAlmostEqual(values[10], 0.0, places=6)

    def test_monotonic(self):
        values = [
            warmup_cosine_lr(t, 4, 20, 1.0, 0.0) for t in range(21)
        ]
        for t in range(4):
            self.assertLessEqual(values[t], values[t + 1])
        for t in range(4, 20):
            self.assertGreaterEqual(values[t], values[t + 1])

    def test_scheduler_applied_to_optimizer(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=0.1)
        sched = build_scheduler(opt, warmup_steps=4, total_steps=10, base_lr=0.1)

        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.025, places=6)
        # Real training order: optimizer.step() first, then scheduler.step().
        opt.step()
        sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.05, places=6)
        for _ in range(3):
            opt.step()
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.1, places=6)

    def test_invalid_schedule_args(self):
        with self.assertRaises(ValueError):
            warmup_cosine_lr(0, 10, 10, 1.0, 0.0)
        with self.assertRaises(ValueError):
            warmup_cosine_lr(0, 2, 10, 0.0, 0.0)
        with self.assertRaises(ValueError):
            warmup_cosine_lr(0, 2, 10, 0.1, 0.2)
        with self.assertRaises(ValueError):
            build_scheduler(torch.optim.AdamW([nn.Parameter(torch.zeros(1))]), 2, 10, 0.0)


class TestLabelSmoothing(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(4, 10)
        self.targets = torch.tensor([0, 3, 9, 1])

    def test_matches_pytorch(self):
        ours = label_smoothing_loss(self.logits, self.targets, smoothing=0.1)
        ref = nn.CrossEntropyLoss(label_smoothing=0.1)(self.logits, self.targets)
        torch.testing.assert_close(ours, ref, atol=1e-6, rtol=0)

    def test_zero_smoothing_equals_ce(self):
        ours = label_smoothing_loss(self.logits, self.targets, smoothing=0.0)
        ref = F.cross_entropy(self.logits, self.targets)
        torch.testing.assert_close(ours, ref, atol=1e-6, rtol=0)

    def test_smoothing_penalizes_overconfidence(self):
        logits = torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([0])
        ce = F.cross_entropy(logits, target)
        smoothed = label_smoothing_loss(logits, target, smoothing=0.1)
        self.assertGreater(float(smoothed), float(ce))

    def test_invalid_smoothing(self):
        with self.assertRaises(ValueError):
            label_smoothing_loss(self.logits, self.targets, smoothing=1.0)
        with self.assertRaises(ValueError):
            label_smoothing_loss(self.logits, self.targets, smoothing=-0.1)


class TestGradientClipping(unittest.TestCase):
    def test_clips_global_norm(self):
        p1 = nn.Parameter(torch.zeros(2))
        p1.grad = torch.tensor([3.0, 4.0])
        p2 = nn.Parameter(torch.zeros(1))
        p2.grad = torch.tensor([0.0])

        norm = clip_gradients([p1, p2], max_norm=1.0)
        self.assertEqual(norm.dim(), 0)  # same as clip_grad_norm_: a 0-d scalar
        self.assertAlmostEqual(float(norm), 5.0, places=6)
        torch.testing.assert_close(p1.grad, torch.tensor([0.6, 0.8]), atol=1e-6, rtol=0)
        torch.testing.assert_close(p2.grad, torch.tensor([0.0]), atol=0, rtol=0)

    def test_below_max_unchanged(self):
        p = nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([0.3, 0.4])
        norm = clip_gradients([p], max_norm=1.0)
        self.assertAlmostEqual(float(norm), 0.5, places=6)
        torch.testing.assert_close(p.grad, torch.tensor([0.3, 0.4]), atol=0, rtol=0)

    def test_no_gradients(self):
        p = nn.Parameter(torch.zeros(2))
        norm = clip_gradients([p], max_norm=1.0)
        self.assertEqual(norm.dim(), 0)  # 0-d, matching the shape when gradients exist
        self.assertEqual(float(norm), 0.0)

    def test_invalid_max_norm(self):
        with self.assertRaises(ValueError):
            clip_gradients([nn.Parameter(torch.zeros(1))], max_norm=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
