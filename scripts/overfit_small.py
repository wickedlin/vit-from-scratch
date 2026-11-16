"""1% data overfitting diagnostic (S3-3.5).

Goal: stratified-sample 1% of the 45,000 training images (about 450 images,
45 per class), disable augmentation and regularization, and repeatedly train on
the training set itself until train acc approaches 100%.

If the model cannot even memorize 450 images, the problem is in the **pipeline
or training loop** (misaligned labels, wrong normalization, forgotten
zero_grad, learning rate decaying to 0 too early, etc.), not model capacity.
This test only proves that learning is possible; it does not prove generalization.

Run:
    python scripts/overfit_small.py
    python scripts/overfit_small.py --epochs 200 --target-train-acc 0.999
    python scripts/overfit_small.py --lr 3e-3 --output-dir experiments/overfit_lr3e-3
"""

from __future__ import annotations

import argparse
import json
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
    """Stratified-sample `fraction` of the training-split indices by class.

    Args:
        targets: labels of the full training set (length 50,000).
        indices: absolute indices of the training split (length 45,000).
        fraction: sampling fraction in (0, 1].
        seed: numpy Generator seed; affects only this sampling.

    Returns:
        Ascending np.int64 array of absolute indices, about
        len(indices) * fraction long, sampling round(class_train_count * fraction)
        images per class.
    """
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit 1% of CIFAR-10 train split")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--target-train-acc", type=float, default=0.99)
    parser.add_argument("--amp", choices=["off", "auto", "bf16", "fp16"], default="off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="experiments/overfit_1pct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # 1) Data: aug="none", so the training and evaluation views match exactly
    # (no random augmentation).
    train_set, _, _ = build_datasets(
        data_root="data", val_ratio=0.1, seed=42, aug="none"
    )
    targets = np.asarray(train_set.dataset.targets)
    indices = np.asarray(train_set.indices)
    debug_indices = select_fraction_indices(
        targets, indices, fraction=args.fraction, seed=args.seed
    )
    debug_set = Subset(train_set.dataset, debug_indices)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        debug_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator=generator,
    )
    eval_loader = DataLoader(
        debug_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # 2) Model and training config: disable augmentation, weight decay, label smoothing, dropout.
    model = ViT(
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

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=1,
        lr=args.lr,
        weight_decay=0.0,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=0.0,
        max_grad_norm=1.0,
        amp=args.amp,
        seed=args.seed,
        num_workers=args.num_workers,
        log_every=0,
        output_dir=args.output_dir,
    )

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return label_smoothing_loss(logits, targets, smoothing=0.0)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=0.0)
    micro_per_epoch = len(train_loader)
    total_steps = args.epochs * micro_per_epoch
    warmup_steps = min(args.warmup_epochs * micro_per_epoch, total_steps - 1)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=args.lr,
    )
    amp_dtype = resolve_amp_dtype(device, args.amp)
    scaler = make_grad_scaler(device, amp_dtype)

    # 3) Logging: write only a small JSONL curve, no checkpoint (not needed for a diagnostic).
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    print(
        f"device={device} samples={len(debug_set)} "
        f"micro/epoch={micro_per_epoch} "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    reached_at = None
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
        eval_metrics = evaluate(model, eval_loader, criterion, device, amp_dtype)
        lr = train_metrics["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "eval_loss": eval_metrics["loss"],
            "eval_acc": eval_metrics["acc"],
            "lr": lr,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"epoch {epoch + 1:3d}/{args.epochs} "
            f"train_loss {train_metrics['loss']:.4f} "
            f"train_acc {train_metrics['acc']:.4f} "
            f"eval_acc {eval_metrics['acc']:.4f} "
            f"lr {lr:.3e}",
            flush=True,
        )

        if train_metrics["acc"] >= args.target_train_acc:
            reached_at = epoch + 1
            print(
                f"target train acc {args.target_train_acc:.3f} reached "
                f"at epoch {reached_at}",
                flush=True,
            )
            break

    if reached_at is None:
        print(
            f"target not reached after {args.epochs} epochs; "
            f"check labels / normalization / zero_grad / lr schedule",
            flush=True,
        )
    print(f"metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
