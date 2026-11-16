"""Learning-rate sweep (S3-3.5): run several LRs on a training subset and compare validation curves.

Goal:
- Observe three curve regimes: underfitting (LR too small), just right, and
  divergence (LR too large).
- Pick a reasonable LR for the full 3.6 baseline. Rankings on the subset are a
  guide only; the final choice is based on the full 100-epoch run.

Run:
    python scripts/lr_scan.py
    python scripts/lr_scan.py --epochs 6 --fraction 0.25
    python scripts/lr_scan.py --lrs 1e-4,3e-4,1e-3,3e-3

Outputs:
    experiments/lr_scan/lr_3e-04/metrics.jsonl
    experiments/lr_scan/lr_1e-03/metrics.jsonl
    ...
    experiments/lr_scan/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vit_from_scratch.amp import make_grad_scaler, resolve_amp_dtype
from vit_from_scratch.data import build_datasets
from vit_from_scratch.optim import (
    build_optimizer,
    build_scheduler,
    label_smoothing_loss,
)
from vit_from_scratch.train import TrainConfig, evaluate, train_one_epoch
from vit_from_scratch.utils import get_device, set_seed
from vit_from_scratch.vit import ViT


def select_fraction_indices(
    targets: np.ndarray,
    indices: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Stratified-sample a `fraction` of absolute indices from the training split, per class."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    targets = np.asarray(targets)
    indices = np.asarray(indices)
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(targets[indices]):
        cls_indices = indices[targets[indices] == cls]
        n = max(1, int(round(len(cls_indices) * fraction)))
        chosen.append(rng.choice(cls_indices, size=n, replace=False))
    return np.sort(np.concatenate(chosen))


def build_model(device: torch.device) -> ViT:
    """Build the same ViT-tiny as the S3 baseline."""
    return ViT(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0.0,
        emb_dropout=0.0,
        drop_path=0.0,
        pool="cls",
        pos_embed="learned",
        pre_norm=True,
    ).to(device)


def run_one_lr(
    lr: float,
    args: argparse.Namespace,
    device: torch.device,
    scan_set: Subset,
    val_loader: DataLoader,
    output_root: Path,
) -> dict:
    """Train one full short run for a single LR and return a summary dict."""
    set_seed(args.seed)
    model = build_model(device)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        max_grad_norm=1.0,
        amp=args.amp,
        seed=args.seed,
        num_workers=args.num_workers,
        log_every=0,
        output_dir=str(output_root / f"lr_{lr:.0e}"),
    )

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return label_smoothing_loss(logits, targets, smoothing=args.label_smoothing)

    optimizer = build_optimizer(
        model, lr=lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        scan_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )

    micro_per_epoch = len(train_loader)
    opt_per_epoch = math.ceil(micro_per_epoch / args.accum_steps)
    total_steps = max(1, args.epochs * opt_per_epoch)
    warmup_steps = min(args.warmup_epochs * opt_per_epoch, total_steps - 1)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=lr,
    )
    amp_dtype = resolve_amp_dtype(device, args.amp)
    scaler = make_grad_scaler(device, amp_dtype)

    run_dir = output_root / f"lr_{lr:.0e}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "lr": lr,
                "fraction": args.fraction,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "accum_steps": args.accum_steps,
                "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "aug": args.aug,
                "amp": args.amp,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    history: list[dict] = []
    diverged = False
    for epoch in range(args.epochs):
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
        lr_now = train_metrics["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": lr_now,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"lr {lr:.0e} epoch {epoch + 1:2d}/{args.epochs} "
            f"train_loss {train_metrics['loss']:.4f} "
            f"train_acc {train_metrics['acc']:.4f} "
            f"val_loss {val_metrics['loss']:.4f} "
            f"val_acc {val_metrics['acc']:.4f} "
            f"lr_now {lr_now:.3e}",
            flush=True,
        )

        if not (
            math.isfinite(train_metrics["loss"])
            and math.isfinite(val_metrics["loss"])
        ):
            diverged = True
            print(f"lr {lr:.0e} diverged at epoch {epoch}; stop this run", flush=True)
            break

    last = history[-1]
    return {
        "lr": lr,
        "epochs_run": len(history),
        "final_train_loss": last["train_loss"],
        "final_train_acc": last["train_acc"],
        "final_val_loss": last["val_loss"],
        "final_val_acc": last["val_acc"],
        "best_val_acc": max(h["val_acc"] for h in history),
        "min_val_loss": min(h["val_loss"] for h in history),
        "diverged": diverged,
        "metrics": str(metrics_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep learning rates on a CIFAR-10 subset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lrs", type=str, default="3e-4,1e-3,3e-3,1e-2")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--aug", choices=["none", "basic", "randaugment"], default="basic"
    )
    parser.add_argument("--amp", choices=["off", "auto", "bf16", "fp16"], default="off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="experiments/lr_scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    if any(lr <= 0 for lr in lrs):
        raise SystemExit(f"all lrs must be positive, got {lrs}")

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _ = build_datasets(
        data_root="data", val_ratio=0.1, seed=42, aug=args.aug
    )
    targets = np.asarray(train_set.dataset.targets)
    indices = np.asarray(train_set.indices)
    scan_indices = select_fraction_indices(
        targets, indices, fraction=args.fraction, seed=args.seed
    )
    scan_set = Subset(train_set.dataset, scan_indices)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(
        f"device={device} scan_samples={len(scan_set)} "
        f"val_samples={len(val_set)} lrs={lrs}",
        flush=True,
    )

    summaries = []
    for lr in lrs:
        summaries.append(
            run_one_lr(lr, args, device, scan_set, val_loader, output_root)
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nsummary")
    print(
        f"{'lr':>8} {'epochs':>6} {'train_acc':>10} {'val_acc':>8} "
        f"{'best_val':>9} {'min_val_loss':>13} {'diverged':>9}"
    )
    for s in summaries:
        print(
            f"{s['lr']:>8.0e} {s['epochs_run']:>6d} "
            f"{s['final_train_acc']:>10.4f} {s['final_val_acc']:>8.4f} "
            f"{s['best_val_acc']:>9.4f} {s['min_val_loss']:>13.4f} "
            f"{str(s['diverged']):>9}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
