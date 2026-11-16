"""CIFAR-10 training loop: train/eval, gradient accumulation, AMP, checkpoints, logging (S3-3.4).

Minimal run:
    python -m vit_from_scratch.train --epochs 2 --output-dir experiments/repro_a

Full-argument example (a regularization run; adjust --dropout / --drop-path / --lr):
    python -m vit_from_scratch.train --epochs 100 --batch-size 32 --accum-steps 4 --lr 1e-3 --weight-decay 0.05 --warmup-epochs 5 --label-smoothing 0.1 --max-grad-norm 1.0 --amp off --seed 0 --num-workers 0 --log-every 200 --output-dir experiments/reg_dp01_do01_lr1e-3 --image-size 32 --patch-size 4 --dim 192 --depth 12 --heads 3 --mlp-dim 768 --dropout 0.1 --emb-dropout 0.0 --drop-path 0.1

Regularization notes:
- To enable drop_path only: set --dropout 0.0; start with drop_path 0.05, then try 0.1.
- Do not start with dropout=0.1 + drop_path=0.1; both at 0.05 is already enough.
- Fixed settings: in_channels=3, num_classes=10, pool=cls, pos_embed=learned, pre_norm=True.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .amp import (
    autocast_context,
    make_grad_scaler,
    optimizer_step,
    resolve_amp_dtype,
)
from .optim import build_optimizer, build_scheduler, label_smoothing_loss
from .utils import get_device, set_seed

Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainConfig:
    """All hyperparameters and paths for one training run."""

    epochs: int = 100
    batch_size: int = 32
    accum_steps: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    amp: str = "auto"
    seed: int = 0
    num_workers: int = 0
    log_every: int = 50
    output_dir: str = "experiments/run"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.accum_steps <= 0:
            raise ValueError(f"accum_steps must be positive, got {self.accum_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
    criterion: Criterion,
    device: torch.device,
    config: TrainConfig,
    epoch: int = 0,
) -> dict[str, float]:
    """Train for one epoch and return sample-weighted loss/accuracy, the number of
    optimizer steps, and the LR actually used by the last parameter update."""
    model.train()
    num_micro = len(loader)
    if num_micro == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "optimizer_steps": 0,
            "lr": float("nan"),
        }

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0
    optimizer_steps = 0
    last_lr = float("nan")

    for micro_idx, (x, y) in enumerate(loader):
        # Read the current LR before optimizer_step; the scheduler.step() inside
        # it advances the LR to the "next" update, so reading afterwards is one step late.
        step_lr = optimizer.param_groups[0]["lr"]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Accumulation group of this micro-batch: split the group loss evenly.
        group_start = (micro_idx // config.accum_steps) * config.accum_steps
        group_size = min(config.accum_steps, num_micro - group_start)
        loss_weight = 1.0 / group_size

        with autocast_context(device, amp_dtype=amp_dtype):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss * loss_weight).backward()

        # Accumulate metrics on the device and synchronize once per epoch; this
        # avoids a .item() sync on every step.
        with torch.no_grad():
            total_loss += loss.detach().float() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum()
            total_samples += x.size(0)

        is_group_end = (micro_idx + 1) % config.accum_steps == 0
        is_last_micro = (micro_idx + 1) == num_micro
        if is_group_end or is_last_micro:
            last_lr = step_lr
            optimizer_step(
                model, optimizer, scheduler, scaler, max_norm=config.max_grad_norm
            )
            optimizer_steps += 1

        if config.log_every > 0 and (micro_idx + 1) % config.log_every == 0:
            print(
                f"epoch {epoch} micro {micro_idx + 1}/{num_micro} "
                f"loss {loss.item():.4f} lr {step_lr:.3e}",
                flush=True,
            )

    return {
        "loss": (total_loss / total_samples).item(),
        "acc": (total_correct.float() / total_samples).item(),
        "optimizer_steps": optimizer_steps,
        "lr": last_lr,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Criterion,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Compute validation/test loss and accuracy in eval mode without updating parameters."""
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.float() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum()
        total_samples += x.size(0)

    if total_samples == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {
        "loss": (total_loss / total_samples).item(),
        "acc": (total_correct.float() / total_samples).item(),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_acc: float,
    config: TrainConfig,
) -> None:
    """Save a portable state_dict (tensors and config only; never pickle the whole model)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint and fill its state_dicts into existing model/optimizer/scheduler/scaler objects."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    criterion: Criterion | None = None,
) -> dict[str, list[float]]:
    """Full training flow: seed, optimizer/scheduler, per-epoch train+val, logging, last/best checkpoints."""
    set_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    if criterion is None:
        def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return label_smoothing_loss(logits, targets, config.label_smoothing)

    optimizer = build_optimizer(model, lr=config.lr, weight_decay=config.weight_decay)
    micro_per_epoch = len(train_loader)
    opt_per_epoch = math.ceil(micro_per_epoch / config.accum_steps)
    total_steps = max(1, config.epochs * opt_per_epoch)
    warmup_steps = min(config.warmup_epochs * opt_per_epoch, total_steps - 1)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=config.lr,
    )
    amp_dtype = resolve_amp_dtype(device, config.amp)
    scaler = make_grad_scaler(device, amp_dtype)

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_acc = -1.0

    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            amp_dtype,
            criterion,
            device,
            config,
            epoch,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, amp_dtype)
        # Use the LR actually used by the last optimizer step in the epoch.
        # Reading the optimizer here would already see the next step; the last
        # epoch would report a nonexistent "next" LR of 0.
        lr = train_metrics["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(lr)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": lr,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_acc,
                config,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_acc,
            config,
        )

        print(
            f"epoch {epoch + 1}/{config.epochs} "
            f"train_loss {train_metrics['loss']:.4f} "
            f"train_acc {train_metrics['acc']:.4f} "
            f"val_loss {val_metrics['loss']:.4f} "
            f"val_acc {val_metrics['acc']:.4f} lr {lr:.3e}",
            flush=True,
        )

    return history


MODEL_ARG_NAMES = (
    "image_size",
    "patch_size",
    "dim",
    "depth",
    "heads",
    "mlp_dim",
    "dropout",
    "emb_dropout",
    "drop_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT on CIFAR-10 (S3-3.4)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--amp", choices=["off", "auto", "bf16", "fp16"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="experiments/run")

    # Model architecture: defaults = S3 baseline; for ablations change only
    # these and keep the training recipe fixed.
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--emb-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    from .data import build_dataloaders
    from .vit import ViT

    args = parse_args()
    model_config = {name: getattr(args, name) for name in MODEL_ARG_NAMES}
    train_config = {
        key: value for key, value in vars(args).items()
        if key not in MODEL_ARG_NAMES
    }
    config = TrainConfig(**train_config)
    device = get_device()
    set_seed(config.seed)

    # Data split is fixed at seed=42; shuffle/worker seeds follow the training seed.
    loaders = build_dataloaders(
        data_root="data",
        seed=42,
        shuffle_seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Save the model architecture separately so eval_checkpoint.py can read it
    # directly instead of repeating ablation parameters on the command line.
    model_record = {
        **model_config,
        "in_channels": 3,
        "num_classes": 10,
        "pool": "cls",
        "pos_embed": "learned",
        "pre_norm": True,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_config.json").write_text(
        json.dumps(model_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model = ViT(**model_record).to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")
    fit(model, loaders["train"], loaders["val"], config, device)


if __name__ == "__main__":
    main()
