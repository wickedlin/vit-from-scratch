"""MAE pretraining script (S5-5.4a).

Reuses the S3 optimizer / scheduler / AMP utilities and the CIFAR-10 data
pipeline; trains only the MaskedAutoencoder reconstruction objective and does
not use labels.

Smoke example (CPU, a few tens of seconds):
    python scripts/pretrain_mae.py --epochs 1 --max-steps 20 --val-batches 5 \
        --batch-size 16 --accum-steps 2 --device cpu \
        --output-dir experiments/mae_pretrain_smoke

Full pilot example:
    python scripts/pretrain_mae.py --epochs 100 --batch-size 64 --accum-steps 2 \
        --lr 1e-3 --weight-decay 0.05 --warmup-epochs 5 --mask-ratio 0.75 \
        --norm-pix-loss --amp auto --output-dir experiments/mae_pretrain_100ep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.amp import (  # noqa: E402
    autocast_context,
    make_grad_scaler,
    optimizer_step,
    resolve_amp_dtype,
)
from vit_from_scratch.data import build_dataloaders  # noqa: E402
from vit_from_scratch.mae import MaskedAutoencoder  # noqa: E402
from vit_from_scratch.optim import build_optimizer, build_scheduler  # noqa: E402
from vit_from_scratch.utils import get_device, set_seed  # noqa: E402


MODEL_ARG_NAMES = (
    "image_size",
    "patch_size",
    "in_channels",
    "encoder_dim",
    "encoder_depth",
    "encoder_heads",
    "decoder_dim",
    "decoder_depth",
    "decoder_heads",
    "mlp_ratio",
    "mask_ratio",
    "norm_pix_loss",
    "dropout",
    "drop_path",
)


@dataclass
class MAEPretrainConfig:
    """All training hyperparameters and paths for one MAE pretraining run."""

    epochs: int = 100
    batch_size: int = 64
    accum_steps: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    amp: str = "auto"
    seed: int = 0
    num_workers: int = 0
    log_every: int = 100
    output_dir: str = "experiments/mae_pretrain"
    aug: str = "basic"
    max_steps: int = 0
    val_batches: int = 0

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
        if self.max_steps < 0 or self.val_batches < 0:
            raise ValueError("max_steps and val_batches must be non-negative")


def save_mae_checkpoint(
    path: str | Path,
    model: MaskedAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    config: MAEPretrainConfig,
    model_config: dict,
) -> None:
    """Save a portable state_dict and the config needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": asdict(config),
            "model_config": model_config,
        },
        path,
    )


def train_one_epoch(
    model: MaskedAutoencoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    config: MAEPretrainConfig,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch; only images are used, labels are ignored."""
    model.train()
    num_micro = len(loader)
    if config.max_steps > 0:
        num_micro = min(num_micro, config.max_steps)
    if num_micro == 0:
        return {
            "loss": float("nan"),
            "optimizer_steps": 0,
            "lr": float("nan"),
        }

    total_loss = torch.zeros((), device=device)
    total_samples = 0
    optimizer_steps = 0
    last_lr = float("nan")

    for micro_idx, (images, _) in enumerate(loader):
        if micro_idx >= num_micro:
            break

        # Same convention as train.py: read the LR actually used for this step, then optimizer_step.
        step_lr = optimizer.param_groups[0]["lr"]
        images = images.to(device, non_blocking=True)

        group_start = (micro_idx // config.accum_steps) * config.accum_steps
        group_size = min(config.accum_steps, num_micro - group_start)
        loss_weight = 1.0 / group_size

        with autocast_context(device, amp_dtype=amp_dtype):
            loss, _, _ = model(images)
        scaler.scale(loss * loss_weight).backward()

        with torch.no_grad():
            total_loss += loss.detach().float() * images.size(0)
            total_samples += images.size(0)

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
        "optimizer_steps": optimizer_steps,
        "lr": last_lr,
    }


@torch.no_grad()
def evaluate_mae(
    model: MaskedAutoencoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    max_batches: int = 0,
) -> dict[str, float]:
    """Compute reconstruction loss on the validation set (masks are random; monitoring only)."""
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_samples = 0

    for batch_idx, (images, _) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            loss, _, _ = model(images)
        total_loss += loss.float() * images.size(0)
        total_samples += images.size(0)

    if total_samples == 0:
        return {"loss": float("nan"), "samples": 0}
    return {
        "loss": (total_loss / total_samples).item(),
        "samples": total_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MAE on CIFAR-10 (S5-5.4a)")

    # Training recipe
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--amp", choices=["off", "auto", "bf16", "fp16"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="experiments/mae_pretrain")
    parser.add_argument("--aug", choices=["none", "basic", "randaugment"], default="basic")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = no limit")
    parser.add_argument("--val-batches", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )

    # Model architecture: defaults = the 5.3 CIFAR config
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--encoder-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=12)
    parser.add_argument("--encoder-heads", type=int, default=3)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--norm-pix-loss", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_kwargs = {name: getattr(args, name) for name in MODEL_ARG_NAMES}
    train_kwargs = {
        key: value
        for key, value in vars(args).items()
        if key not in MODEL_ARG_NAMES and key != "device"
    }
    config = MAEPretrainConfig(**train_kwargs)

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = {"arch": "mae", **model_kwargs}
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    loaders = build_dataloaders(
        data_root="data",
        seed=42,
        shuffle_seed=config.seed,
        aug=config.aug,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MaskedAutoencoder(**model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} arch=mae params={num_params:,} "
        f"N={model.num_patches} mask_ratio={model.mask_ratio} "
        f"norm_pix_loss={model.norm_pix_loss}",
        flush=True,
    )

    micro_per_epoch = len(loaders["train"])
    if config.max_steps > 0:
        micro_per_epoch = min(micro_per_epoch, config.max_steps)
    opt_per_epoch = math.ceil(micro_per_epoch / config.accum_steps)
    total_steps = max(1, config.epochs * opt_per_epoch)
    warmup_steps = min(config.warmup_epochs * opt_per_epoch, total_steps - 1)

    optimizer = build_optimizer(
        model, lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=config.lr,
    )
    amp_dtype = resolve_amp_dtype(device, config.amp)
    scaler = make_grad_scaler(device, amp_dtype)

    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scheduler,
            scaler,
            amp_dtype,
            device,
            config,
            epoch,
        )
        val_metrics = evaluate_mae(
            model,
            loaders["val"],
            device,
            amp_dtype=amp_dtype,
            max_batches=config.val_batches,
        )
        lr = train_metrics["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_samples": val_metrics["samples"],
            "optimizer_steps": train_metrics["optimizer_steps"],
            "lr": lr,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_mae_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                config,
                model_config,
            )
        save_mae_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            config,
            model_config,
        )

        print(
            f"epoch {epoch + 1}/{config.epochs} "
            f"train_loss {train_metrics['loss']:.4f} "
            f"val_loss {val_metrics['loss']:.4f} "
            f"steps {train_metrics['optimizer_steps']} lr {lr:.3e}",
            flush=True,
        )

    print(f"done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
