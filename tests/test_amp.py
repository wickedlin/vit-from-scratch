"""Tests for src/vit_from_scratch/amp.py (S3-3.3).

Run from the project root:
    python tests/test_amp.py -v

The learner writes amp.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.amp import (  # noqa: E402
    autocast_context,
    backward_microbatch,
    make_grad_scaler,
    optimizer_step,
    resolve_amp_dtype,
)
from vit_from_scratch.optim import build_optimizer, build_scheduler  # noqa: E402

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 16)
        self.out = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.fc(x)))


class TestAmpDtype(unittest.TestCase):
    def test_off(self):
        self.assertIsNone(resolve_amp_dtype(CPU, "off"))

    def test_explicit_bf16(self):
        self.assertEqual(resolve_amp_dtype(CPU, "bf16"), torch.bfloat16)

    def test_fp16_requires_cuda(self):
        with self.assertRaises(ValueError):
            resolve_amp_dtype(CPU, "fp16")

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            resolve_amp_dtype(CPU, "tf32")

    def test_auto_cpu_disabled(self):
        self.assertIsNone(resolve_amp_dtype(CPU, "auto"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_auto_cuda(self):
        expected = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.assertEqual(resolve_amp_dtype(CUDA, "auto"), expected)


class TestGradScaler(unittest.TestCase):
    def test_disabled_on_cpu(self):
        scaler = make_grad_scaler(CPU, None)
        self.assertFalse(scaler.is_enabled())

    def test_disabled_for_bf16(self):
        scaler = make_grad_scaler(CPU, torch.bfloat16)
        self.assertFalse(scaler.is_enabled())

    def test_fp16_on_cpu_raises(self):
        with self.assertRaises(ValueError):
            make_grad_scaler(CPU, torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_enabled_for_fp16_cuda(self):
        scaler = make_grad_scaler(CUDA, torch.float16)
        self.assertTrue(scaler.is_enabled())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_disabled_for_bf16_cuda(self):
        scaler = make_grad_scaler(CUDA, torch.bfloat16)
        self.assertFalse(scaler.is_enabled())


class TestAutocast(unittest.TestCase):
    def test_null_context_keeps_fp32(self):
        model = nn.Linear(4, 4)
        x = torch.randn(2, 4)
        with autocast_context(CPU, None):
            y = model(x)
        self.assertEqual(y.dtype, torch.float32)

    def test_cpu_bf16_changes_dtype(self):
        model = nn.Linear(4, 4)
        x = torch.randn(2, 4)
        with autocast_context(CPU, torch.bfloat16):
            y = model(x)
        self.assertEqual(y.dtype, torch.bfloat16)


class TestBackwardAndStep(unittest.TestCase):
    def _setup(self):
        torch.manual_seed(0)
        model = TinyModel()
        optimizer = build_optimizer(model, lr=0.1, weight_decay=0.0)
        scheduler = build_scheduler(optimizer, warmup_steps=2, total_steps=10, base_lr=0.1)
        scaler = make_grad_scaler(CPU, None)
        return model, optimizer, scheduler, scaler

    def test_backward_populates_grads(self):
        model, _, _, scaler = self._setup()
        x = torch.randn(4, 8)
        y = torch.randint(0, 4, (4,))
        loss = backward_microbatch(
            model, x, y, nn.CrossEntropyLoss(), scaler, None, loss_weight=0.5
        )
        self.assertTrue(torch.isfinite(loss))
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_optimizer_step_updates_params_and_lr(self):
        model, optimizer, scheduler, scaler = self._setup()
        x = torch.randn(4, 8)
        y = torch.randint(0, 4, (4,))
        before = [p.detach().clone() for p in model.parameters()]
        lr_before = optimizer.param_groups[0]["lr"]

        backward_microbatch(model, x, y, nn.CrossEntropyLoss(), scaler, None)
        norm = optimizer_step(model, optimizer, scheduler, scaler, max_norm=1.0)

        self.assertEqual(norm.dim(), 0)
        self.assertTrue(torch.isfinite(norm))
        self.assertNotAlmostEqual(lr_before, optimizer.param_groups[0]["lr"])
        for old, p in zip(before, model.parameters()):
            self.assertFalse(torch.equal(old, p.detach()))
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_microbatch_gradients_match_full_batch(self):
        torch.manual_seed(1)
        model = TinyModel()
        x = torch.randn(8, 8)
        y = torch.randint(0, 4, (8,))
        criterion = nn.CrossEntropyLoss()
        scaler = make_grad_scaler(CPU, None)

        model.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        full = {name: p.grad.clone() for name, p in model.named_parameters()}

        model.zero_grad(set_to_none=True)
        for xb, yb in ((x[:4], y[:4]), (x[4:], y[4:])):
            backward_microbatch(
                model, xb, yb, criterion, scaler, None, loss_weight=0.5
            )

        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, full[name], atol=1e-6, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFp16Cuda(unittest.TestCase):
    def test_fp16_scaler_step_is_finite(self):
        torch.manual_seed(0)
        model = TinyModel().to(CUDA)
        optimizer = build_optimizer(model, lr=0.1, weight_decay=0.0)
        scheduler = build_scheduler(optimizer, warmup_steps=2, total_steps=10, base_lr=0.1)
        scaler = make_grad_scaler(CUDA, torch.float16)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 8, device=CUDA)
        y = torch.randint(0, 4, (4,), device=CUDA)
        loss = backward_microbatch(model, x, y, criterion, scaler, torch.float16)
        self.assertTrue(torch.isfinite(loss))
        norm = optimizer_step(model, optimizer, scheduler, scaler, max_norm=1.0)
        self.assertTrue(torch.isfinite(norm))
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
