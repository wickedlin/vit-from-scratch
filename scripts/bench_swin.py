"""Estimate Swin training speed on the current device (S4-4.5).

Measures only the model forward + backward + optimizer step, without data
loading; used to decide the epoch budget for full runs.

Run:
    python scripts/bench_swin.py --batch-size 32 --steps 12 --warmup 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.swin import SwinTransformer  # noqa: E402
from vit_from_scratch.utils import get_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Swin training step")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=45_000)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    device = get_device()
    set_seed(0)

    model = SwinTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)

    model.train()
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    sync(device)
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed / args.steps * 1000.0
    steps_per_epoch = (args.train_size + args.batch_size - 1) // args.batch_size
    epoch_min = ms_per_step * steps_per_epoch / 1000.0 / 60.0
    params = sum(p.numel() for p in model.parameters())

    print(f"device={device} params={params:,} batch_size={args.batch_size}")
    print(f"{ms_per_step:.2f} ms/step ({args.steps} steps, no data loading)")
    print(f"steps/epoch={steps_per_epoch} -> est. {epoch_min:.1f} min/epoch")
    print(f"est. 30 epochs = {epoch_min * 30 / 60:.1f} h; 100 epochs = {epoch_min * 100 / 60:.1f} h")


if __name__ == "__main__":
    main()
