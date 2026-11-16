"""Evaluate a saved checkpoint on val/test (S3-3.6).

`train.py` only runs train/val; this script loads `best.pt` / `last.pt` and
computes loss and accuracy on the held-out test split (10,000 images that were
not used for training or validation).

Run:
    python scripts/eval_checkpoint.py --checkpoint experiments/run/best.pt
    python scripts/eval_checkpoint.py --checkpoint experiments/run/last.pt --split val
    python scripts/eval_checkpoint.py \
        --checkpoint experiments/baseline_vit_tiny_p4/best.pt \
        --output-json experiments/baseline_vit_tiny_p4/test_metrics.json

Notes:
- The checkpoint stores only the training config, not the model architecture;
  architecture parameters come from this script's CLI, whose defaults match the
  S3 baseline ViT-tiny (p4/d192/L12/h3).
- Newer training runs write `model_config.json` to output_dir; this script reads
  that file from the checkpoint's directory by default, so ablation evaluations
  do not need model parameters copied by hand. You can also pass
  `--model-config` explicitly; that file takes precedence over the CLI model
  arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vit_from_scratch.amp import resolve_amp_dtype
from vit_from_scratch.data import build_dataloaders
from vit_from_scratch.optim import label_smoothing_loss
from vit_from_scratch.train import evaluate, load_checkpoint
from vit_from_scratch.utils import get_device
from vit_from_scratch.vit import ViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a ViT checkpoint on CIFAR-10 val/test"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="experiments/run/best.pt",
        help="checkpoint path (best.pt / last.pt)",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp", choices=["off", "auto", "bf16", "fp16"], default="off",
        help="evaluation precision; defaults to off for reproducible results",
    )

    # Model architecture arguments: defaults = S3 baseline config.
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--emb-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--pool", choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--pos-embed", choices=["learned", "sincos", "none"], default="learned"
    )
    parser.add_argument("--pre-norm", action="store_true", default=True)
    parser.add_argument("--post-norm", dest="pre_norm", action="store_false")
    parser.add_argument(
        "--model-config", type=str, default="",
        help=(
            "optional: model architecture JSON; defaults to "
            "model_config.json next to the checkpoint"
        ),
    )

    parser.add_argument(
        "--output-json", type=str, default="",
        help="optional: write evaluation results to JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model_kwargs = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": args.mlp_dim,
        "dropout": args.dropout,
        "emb_dropout": args.emb_dropout,
        "drop_path": args.drop_path,
        "pool": args.pool,
        "pos_embed": args.pos_embed,
        "pre_norm": args.pre_norm,
    }
    model_config_path = (
        Path(args.model_config)
        if args.model_config
        else Path(args.checkpoint).parent / "model_config.json"
    )
    if model_config_path.exists():
        model_kwargs.update(
            json.loads(model_config_path.read_text(encoding="utf-8"))
        )
        print(f"model_config: {model_config_path}", flush=True)
    elif args.model_config:
        raise SystemExit(f"model config not found: {model_config_path}")

    model = ViT(**model_kwargs).to(device)

    checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
    config = checkpoint.get("config", {})
    smoothing = float(config.get("label_smoothing", 0.1))

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return label_smoothing_loss(logits, targets, smoothing=smoothing)

    loaders = build_dataloaders(
        data_root=args.data_root,
        seed=42,
        shuffle_seed=int(config.get("seed", 0)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader = loaders[args.split]
    amp_dtype = resolve_amp_dtype(device, args.amp)

    metrics = evaluate(model, loader, criterion, device, amp_dtype)

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "split": args.split,
        "samples": len(loader.dataset),
        "loss": metrics["loss"],
        "acc": metrics["acc"],
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_acc": float(checkpoint.get("best_acc", float("nan"))),
        "label_smoothing": smoothing,
        "amp": args.amp,
    }

    print(
        f"checkpoint={result['checkpoint']} split={result['split']} "
        f"samples={result['samples']} loss={result['loss']:.4f} "
        f"acc={result['acc']:.4f} epoch={result['epoch']} "
        f"best_acc={result['best_acc']:.4f}",
        flush=True,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"metrics: {output_path}", flush=True)


if __name__ == "__main__":
    main()
