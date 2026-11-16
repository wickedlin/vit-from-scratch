"""Train the hand-written Swin Transformer on CIFAR-10 (S4-4.5).

Reuses the S3 training loop:
  - `TrainConfig` / `fit` from `vit_from_scratch.train`
  - `build_dataloaders` from `vit_from_scratch.data`
  - `get_device` / `set_seed` from `vit_from_scratch.utils`

Default model = the 4.5 config:
  patch 4, 3 stages (8x8 -> 4x4 -> 2x2),
  dim 64/128/256, depths 2/2/6, heads 2/4/8, windows 4/2/2.

Example runs:
    python scripts/train_swin.py --epochs 1 --output-dir experiments/swin_smoke
    python scripts/train_swin.py --epochs 30 --output-dir experiments/swin_30ep
    python scripts/train_swin.py --epochs 30 --drop-path 0.1 \
        --output-dir experiments/swin_30ep_dp01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.data import build_dataloaders  # noqa: E402
from vit_from_scratch.swin import SwinTransformer  # noqa: E402
from vit_from_scratch.train import TrainConfig, fit  # noqa: E402
from vit_from_scratch.utils import get_device, set_seed  # noqa: E402


MODEL_ARG_NAMES = (
    "image_size",
    "patch_size",
    "in_channels",
    "num_classes",
    "dim",
    "depths",
    "heads",
    "window_sizes",
    "mlp_ratio",
    "dropout",
    "attn_dropout",
    "drop_path",
    "use_relative_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hand-written Swin Transformer on CIFAR-10"
    )

    # Training recipe: defaults match the S3 baseline for comparison with ViT control_30.
    parser.add_argument("--epochs", type=int, default=30)
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
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="experiments/swin_run")

    # Model architecture: defaults = the 4.5 config.
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 2, 6])
    parser.add_argument("--heads", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[4, 2, 2])
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument(
        "--no-relative-bias",
        dest="use_relative_bias",
        action="store_false",
        help="ablation switch: disable the relative position bias",
    )
    parser.set_defaults(use_relative_bias=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_kwargs = {name: getattr(args, name) for name in MODEL_ARG_NAMES}
    train_kwargs = {
        key: value
        for key, value in vars(args).items()
        if key not in MODEL_ARG_NAMES
    }
    config = TrainConfig(**train_kwargs)
    device = get_device()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_record = {"arch": "swin", **model_kwargs}
    (output_dir / "model_config.json").write_text(
        json.dumps(model_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Data split uses a fixed seed=42; shuffle/worker seeds follow the training seed.
    loaders = build_dataloaders(
        data_root="data",
        seed=42,
        shuffle_seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SwinTransformer(**model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} arch=swin params={num_params:,} "
        f"stages={len(model.stages)} merges={len(model.merges)}",
        flush=True,
    )
    print(f"model_config: {output_dir / 'model_config.json'}", flush=True)

    fit(model, loaders["train"], loaders["val"], config, device)


if __name__ == "__main__":
    main()
