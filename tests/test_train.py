"""Tests for src/vit_from_scratch/train.py (S3-3.4).

Run from the project root:
    python tests/test_train.py -v

The learner writes train.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.amp import make_grad_scaler  # noqa: E402
from vit_from_scratch.optim import build_optimizer, build_scheduler  # noqa: E402
from vit_from_scratch.train import (  # noqa: E402
    TrainConfig,
    evaluate,
    fit,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from vit_from_scratch.utils import set_seed  # noqa: E402

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_data(n: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 8, 8, generator=generator)
    y = torch.randint(0, 10, (n,), generator=generator)
    return x, y


def make_loader(
    n: int = 16, batch: int = 4, seed: int = 0, shuffle: bool = False
) -> DataLoader:
    x, y = make_data(n, seed)
    generator = torch.Generator().manual_seed(seed + 100)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch,
        shuffle=shuffle,
        generator=generator,
    )


def make_optimizer_scheduler_scaler(model, config, device=CPU, amp_dtype=None):
    optimizer = build_optimizer(
        model, lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, warmup_steps=0, total_steps=100, base_lr=config.lr
    )
    scaler = make_grad_scaler(device, amp_dtype)
    return optimizer, scheduler, scaler


class TestTrainConfig(unittest.TestCase):
    def test_defaults_match_s3_baseline(self):
        config = TrainConfig()
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.accum_steps, 4)
        self.assertEqual(config.lr, 1e-3)
        self.assertEqual(config.weight_decay, 0.05)
        self.assertEqual(config.warmup_epochs, 5)
        self.assertEqual(config.amp, "auto")

    def test_invalid_values_raise(self):
        for kwargs in (
            {"epochs": 0},
            {"batch_size": 0},
            {"accum_steps": 0},
            {"lr": 0.0},
            {"warmup_epochs": -1},
            {"max_grad_norm": 0.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TrainConfig(**kwargs)


class TestTrainOneEpoch(unittest.TestCase):
    def test_metrics_updates_and_clears_grads(self):
        set_seed(0)
        model = TinyModel()
        loader = make_loader(n=16, batch=4)  # 4 micro-batches
        config = TrainConfig(epochs=2, accum_steps=2, log_every=0)
        optimizer, scheduler, scaler = make_optimizer_scheduler_scaler(model, config)
        before = [p.detach().clone() for p in model.parameters()]

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            None,
            nn.CrossEntropyLoss(),
            CPU,
            config,
            epoch=0,
        )

        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertGreaterEqual(metrics["acc"], 0.0)
        self.assertLessEqual(metrics["acc"], 1.0)
        for param in model.parameters():
            self.assertIsNone(param.grad)
        changed = sum(
            not torch.equal(old, new)
            for old, new in zip(before, model.parameters())
        )
        self.assertGreater(changed, 0)

    def test_last_partial_group_is_flushed(self):
        set_seed(1)
        model = TinyModel()
        loader = make_loader(n=20, batch=4)  # 5 micro-batches
        config = TrainConfig(epochs=1, accum_steps=2, log_every=0)
        optimizer, scheduler, scaler = make_optimizer_scheduler_scaler(model, config)

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            None,
            nn.CrossEntropyLoss(),
            CPU,
            config,
            epoch=0,
        )

        self.assertEqual(metrics["optimizer_steps"], 3)  # 2 + 2 + 1

    def test_empty_loader_returns_nan(self):
        model = TinyModel()
        loader = DataLoader(
            TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long)),
            batch_size=4,
        )
        config = TrainConfig(epochs=1, accum_steps=2, log_every=0)
        optimizer, scheduler, scaler = make_optimizer_scheduler_scaler(model, config)

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            None,
            nn.CrossEntropyLoss(),
            CPU,
            config,
            epoch=0,
        )

        self.assertEqual(metrics["optimizer_steps"], 0)
        self.assertTrue(math.isnan(metrics["loss"]))
        self.assertTrue(math.isnan(metrics["acc"]))
        self.assertTrue(math.isnan(metrics["lr"]))

    def test_reported_lr_is_last_used_not_next_step(self):
        """Report the LR used by the last update, not the scheduler's next-step LR at epoch end."""
        model = TinyModel()
        loader = make_loader(n=8, batch=2)  # 4 micro-batches → 2 optimizer steps
        config = TrainConfig(epochs=1, accum_steps=2, log_every=0, lr=1e-3)
        optimizer = build_optimizer(
            model, lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = build_scheduler(
            optimizer, warmup_steps=0, total_steps=2, base_lr=config.lr
        )
        scaler = make_grad_scaler(CPU, None)

        # Initial LR=1e-3 (step 0); after step 0 the scheduler moves to 5e-4 (step 1);
        # after step 1 the scheduler moves to 0 (step 2 does not exist).
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            None,
            nn.CrossEntropyLoss(),
            CPU,
            config,
            epoch=0,
        )

        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertAlmostEqual(metrics["lr"], 5e-4, places=12)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)


class TestEvaluate(unittest.TestCase):
    def test_eval_does_not_update_or_enable_train_mode(self):
        set_seed(2)
        model = TinyModel()
        loader = make_loader(n=8, batch=4)
        before = [p.detach().clone() for p in model.parameters()]

        metrics = evaluate(model, loader, nn.CrossEntropyLoss(), CPU)

        self.assertFalse(model.training)
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new))
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertGreaterEqual(metrics["acc"], 0.0)
        self.assertLessEqual(metrics["acc"], 1.0)


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_restores_model_and_optimizer(self):
        set_seed(3)
        model = TinyModel()
        loader = make_loader(n=8, batch=4)
        config = TrainConfig(epochs=2, accum_steps=2, log_every=0)
        optimizer, scheduler, scaler = make_optimizer_scheduler_scaler(model, config)
        train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            None,
            nn.CrossEntropyLoss(),
            CPU,
            config,
            epoch=0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=3,
                best_acc=0.75,
                config=config,
            )

            model2 = TinyModel()
            optimizer2, scheduler2, scaler2 = make_optimizer_scheduler_scaler(
                model2, config
            )
            checkpoint = load_checkpoint(
                path, model2, optimizer2, scheduler2, scaler2
            )

            self.assertEqual(checkpoint["epoch"], 3)
            self.assertAlmostEqual(checkpoint["best_acc"], 0.75)
            self.assertEqual(checkpoint["config"]["epochs"], 2)
            for (name1, value1), (name2, value2) in zip(
                model.state_dict().items(), model2.state_dict().items()
            ):
                self.assertEqual(name1, name2)
                self.assertTrue(torch.equal(value1, value2))
            self.assertEqual(
                set(optimizer.state_dict()["state"]),
                set(optimizer2.state_dict()["state"]),
            )
            self.assertEqual(
                scheduler.state_dict()["last_epoch"],
                scheduler2.state_dict()["last_epoch"],
            )


class TestFit(unittest.TestCase):
    def _run(self, seed: int, output_dir: Path):
        set_seed(seed)
        model = TinyModel()
        train_loader = make_loader(n=16, batch=4, seed=seed, shuffle=True)
        val_loader = make_loader(n=8, batch=4, seed=seed + 1, shuffle=False)
        config = TrainConfig(
            epochs=2,
            accum_steps=2,
            seed=seed,
            log_every=0,
            output_dir=str(output_dir),
        )
        history = fit(model, train_loader, val_loader, config, CPU)
        return history, config

    def test_same_seed_gives_identical_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_a, config = self._run(7, Path(tmp) / "a")
            history_b, _ = self._run(7, Path(tmp) / "b")

            self.assertEqual(history_a, history_b)
            self.assertEqual(len(history_a["train_loss"]), config.epochs)
            for name in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
                self.assertEqual(len(history_a[name]), config.epochs)

    def test_logs_and_checkpoints_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _, config = self._run(11, run_dir)

            config_json = json.loads(
                (run_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config_json["epochs"], config.epochs)
            self.assertEqual(config_json["seed"], config.seed)
            self.assertEqual(config_json["output_dir"], str(run_dir))

            lines = (
                (run_dir / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .strip()
                .splitlines()
            )
            self.assertEqual(len(lines), config.epochs)
            record = json.loads(lines[0])
            self.assertEqual(
                set(record),
                {"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"},
            )
            self.assertTrue((run_dir / "last.pt").exists())
            self.assertTrue((run_dir / "best.pt").exists())


class TestGradientAccumulation(unittest.TestCase):
    def test_accumulation_matches_full_batch(self):
        set_seed(42)
        base = TinyModel()
        model_accum = copy.deepcopy(base)
        model_full = copy.deepcopy(base)

        x, y = make_data(n=8, seed=3)
        loader_accum = DataLoader(
            TensorDataset(x, y), batch_size=4, shuffle=False
        )
        loader_full = DataLoader(
            TensorDataset(x, y), batch_size=8, shuffle=False
        )

        config_accum = TrainConfig(epochs=1, accum_steps=2, log_every=0)
        config_full = TrainConfig(epochs=1, accum_steps=1, log_every=0)
        opt_a, sched_a, scaler_a = make_optimizer_scheduler_scaler(
            model_accum, config_accum
        )
        opt_f, sched_f, scaler_f = make_optimizer_scheduler_scaler(
            model_full, config_full
        )

        criterion = nn.CrossEntropyLoss()
        train_one_epoch(
            model_accum,
            loader_accum,
            opt_a,
            sched_a,
            scaler_a,
            None,
            criterion,
            CPU,
            config_accum,
        )
        train_one_epoch(
            model_full,
            loader_full,
            opt_f,
            sched_f,
            scaler_f,
            None,
            criterion,
            CPU,
            config_full,
        )

        for param_accum, param_full in zip(
            model_accum.parameters(), model_full.parameters()
        ):
            torch.testing.assert_close(
                param_accum, param_full, atol=1e-6, rtol=0
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestAmpTraining(unittest.TestCase):
    def test_fp16_epoch_and_eval_are_finite(self):
        set_seed(0)
        model = TinyModel().to(CUDA)
        loader = make_loader(n=8, batch=4)
        config = TrainConfig(epochs=1, accum_steps=2, amp="fp16", log_every=0)
        optimizer, scheduler, scaler = make_optimizer_scheduler_scaler(
            model, config, device=CUDA, amp_dtype=torch.float16
        )

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            torch.float16,
            nn.CrossEntropyLoss(),
            CUDA,
            config,
            epoch=0,
        )
        self.assertTrue(math.isfinite(metrics["loss"]))
        for param in model.parameters():
            self.assertTrue(torch.isfinite(param).all())

        val_metrics = evaluate(
            model, loader, nn.CrossEntropyLoss(), CUDA, torch.float16
        )
        self.assertTrue(math.isfinite(val_metrics["loss"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
