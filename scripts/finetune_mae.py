"""MAE fine-tuning script (S5-5.4b).

Loads the encoder from an MAE pretraining checkpoint, maps it into the S3 ViT
classifier, then reuses `TrainConfig` / `fit` to fine-tune on CIFAR-10 labels.

Smoke example:
    python scripts/finetune_mae.py \
        --mae-checkpoint experiments/mae_pretrain_smoke/best.pt \
        --epochs 1 --train-limit 128 --val-limit 128 \
        --batch-size 32 --accum-steps 4 --device cpu \
        --output-dir experiments/mae_finetune_smoke

Full fine-tuning example (same recipe as the S3 baseline):
    python scripts/finetune_mae.py \
        --mae-checkpoint experiments/mae_pretrain_100ep/best.pt \
        --epochs 100 --batch-size 32 --accum-steps 4 --lr 1e-3 \
        --weight-decay 0.05 --warmup-epochs 5 --label-smoothing 0.1 \
        --output-dir experiments/mae_finetune_100ep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.data import build_dataloaders  # noqa: E402
from vit_from_scratch.mae import (  # noqa: E402
    MaskedAutoencoder,
    load_mae_encoder_weights,
)
from vit_from_scratch.train import TrainConfig, fit  # noqa: E402
from vit_from_scratch.utils import get_device, set_seed  # noqa: E402
from vit_from_scratch.vit import ViT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a ViT initialized from MAE (S5-5.4b)"
    )
    parser.add_argument(
        "--mae-checkpoint",
        type=str,
        required=True,
        help="MAE pretraining checkpoint (best.pt / last.pt)",
    )
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
    parser.add_argument("--output-dir", type=str, default="experiments/mae_finetune")
    parser.add_argument(
        "--aug", choices=["none", "basic", "randaugment"], default="basic"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )

    # Smoke testing: use only the first N samples; 0 = use the full split.
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)

    # Fine-tuning classification head and regularization; the encoder
    # architecture comes from the MAE checkpoint.
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--pool", choices=["cls", "mean"], default="cls")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--emb-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)
    return parser.parse_args()


def make_subset_loader(
    loader: DataLoader,
    limit: int,
    batch_size: int,
    seed: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    """Build a loader from the first `limit` samples of an existing loader's dataset.

    Intended for smoke tests.
    """
    dataset = loader.dataset
    limit = min(limit, len(dataset))
    subset = Subset(dataset, range(limit))
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=loader.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=loader.worker_init_fn,
    )


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    # 1. Read the MAE checkpoint, rebuild the model, and load the pretrained weights.
    checkpoint = torch.load(
        Path(args.mae_checkpoint), map_location="cpu", weights_only=True
    )
    if "model_config" not in checkpoint:
        raise SystemExit("checkpoint has no model_config; was it produced by pretrain_mae.py?")
    mae_config = dict(checkpoint["model_config"])
    mae_config.pop("arch", None)
    mae = MaskedAutoencoder(**mae_config)
    mae.load_state_dict(checkpoint["model"])

    image_size = mae_config["image_size"]
    patch_size = mae_config["patch_size"]
    in_channels = mae_config["in_channels"]
    if (image_size, in_channels) != (32, 3):
        raise SystemExit(
            "finetune_mae.py currently assumes CIFAR-10 (32x32x3); "
            f"got {image_size}x{image_size}x{in_channels}"
        )

    dim = mae_config["encoder_dim"]
    depth = mae_config["encoder_depth"]
    heads = mae_config["encoder_heads"]
    mlp_dim = int(dim * mae_config["mlp_ratio"])

    model_config = {
        "image_size": image_size,
        "patch_size": patch_size,
        "in_channels": in_channels,
        "num_classes": args.num_classes,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "mlp_dim": mlp_dim,
        "dropout": args.dropout,
        "emb_dropout": args.emb_dropout,
        "drop_path": args.drop_path,
        "pool": args.pool,
        "pos_embed": "learned",
        "pre_norm": True,
    }

    # 2. Build an isomorphic ViT and copy the encoder weights; discard the decoder.
    set_seed(args.seed)
    vit = ViT(**model_config)
    load_mae_encoder_weights(vit, mae)
    del mae
    vit = vit.to(device)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        amp=args.amp,
        seed=args.seed,
        num_workers=args.num_workers,
        log_every=args.log_every,
        output_dir=args.output_dir,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "mae_source.json").write_text(
        json.dumps(
            {
                "mae_checkpoint": str(Path(args.mae_checkpoint)),
                "mae_epoch": int(checkpoint.get("epoch", -1)),
                "mae_best_val_loss": float(
                    checkpoint.get("best_val_loss", float("nan"))
                ),
                "mae_model_config": mae_config,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 3. Data and training: reuse the S3 TrainConfig / fit.
    loaders = build_dataloaders(
        data_root="data",
        seed=42,
        shuffle_seed=config.seed,
        aug=args.aug,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    if args.train_limit > 0:
        train_loader = make_subset_loader(
            train_loader,
            args.train_limit,
            config.batch_size,
            config.seed,
            pin_memory=device.type == "cuda",
            shuffle=True,
        )
    if args.val_limit > 0:
        val_loader = make_subset_loader(
            val_loader,
            args.val_limit,
            config.batch_size,
            config.seed,
            pin_memory=device.type == "cuda",
            shuffle=False,
        )

    num_params = sum(p.numel() for p in vit.parameters())
    print(
        f"device={device} arch=vit params={num_params:,} "
        f"init=mae({args.mae_checkpoint}) pool={args.pool}",
        flush=True,
    )
    print(f"model_config: {output_dir / 'model_config.json'}", flush=True)

    fit(vit, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
