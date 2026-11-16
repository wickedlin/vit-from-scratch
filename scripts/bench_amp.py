"""Benchmark fp32 / bf16 / fp16 training steps on CUDA (S3-3.3).

Example:
    python scripts/bench_amp.py --batch-size 32 --iters 10
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vit_from_scratch.amp import (  # noqa: E402
    backward_microbatch,
    make_grad_scaler,
    optimizer_step,
    resolve_amp_dtype,
)
from vit_from_scratch.optim import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    label_smoothing_loss,
)
from vit_from_scratch.vit import ViT  # noqa: E402

MODES = {"fp32": "off", "bf16": "bf16", "fp16": "fp16"}


def build_model(args: argparse.Namespace, device: torch.device) -> ViT:
    return ViT(
        image_size=32,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=0.0,
        emb_dropout=0.0,
        drop_path=0.0,
        pool="cls",
        pos_embed="learned",
        pre_norm=True,
    ).to(device)


def run_mode(mode: str, args: argparse.Namespace, device: torch.device) -> tuple[float, float]:
    torch.cuda.empty_cache()
    gc.collect()

    model = build_model(args, device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.05)
    scheduler = build_scheduler(optimizer, warmup_steps=5, total_steps=100, base_lr=1e-3)
    amp_dtype = resolve_amp_dtype(device, MODES[mode])
    scaler = make_grad_scaler(device, amp_dtype)

    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)
    criterion = lambda logits, targets: label_smoothing_loss(logits, targets, 0.1)

    def one_step() -> None:
        backward_microbatch(model, x, y, criterion, scaler, amp_dtype)
        optimizer_step(model, optimizer, scheduler, scaler, max_norm=1.0)

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.iters):
        one_step()
    torch.cuda.synchronize()
    ms_per_step = (time.perf_counter() - start) * 1000.0 / args.iters
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    del model, optimizer, scheduler, x, y, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return ms_per_step, peak_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ViT AMP modes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp-dim", type=int, default=768)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda")
    print(
        f"config: patch={args.patch_size} dim={args.dim} depth={args.depth} "
        f"heads={args.heads} mlp={args.mlp_dim} batch={args.batch_size}"
    )
    print(f"{'mode':>6}  {'ms/step':>10}  {'peak MB':>10}")
    for mode in MODES:
        ms, peak = run_mode(mode, args, device)
        print(f"{mode:>6}  {ms:>10.2f}  {peak:>10.0f}")


if __name__ == "__main__":
    main()
