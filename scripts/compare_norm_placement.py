"""Compare Pre-LN vs Post-LN placement.

Run from the project root (after adding LayerNorm to transformer.py):
    python scripts/compare_norm_placement.py

Part A: at initialization, measure each block's parameter gradient norm.
Part B: train both placements for 60 Adam steps at a high learning rate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.transformer import LayerNorm  # noqa: E402
from vit_from_scratch.utils import get_device, set_seed  # noqa: E402


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim: int, pre_ln: bool) -> None:
        super().__init__()
        self.pre_ln = pre_ln
        self.norm = LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_ln:
            return x + self.mlp(self.norm(x))
        return self.norm(x + self.mlp(x))


class TinyStack(nn.Module):
    def __init__(self, dim: int, depth: int, pre_ln: bool, num_classes: int = 10) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, pre_ln) for _ in range(depth)])
        self.final_norm = LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


def block_grad_norms(model: TinyStack) -> list[float]:
    return [
        math.sqrt(sum(p.grad.pow(2).sum().item() for p in block.parameters()))
        for block in model.blocks
    ]


def gradient_norms(pre_ln: bool, depth: int, dim: int, device: torch.device) -> list[float]:
    set_seed(0)
    model = TinyStack(dim, depth, pre_ln).to(device)
    x = torch.randn(16, 32, dim, device=device)
    target = torch.randn(16, 32, 10, device=device)

    loss = nn.functional.mse_loss(model(x), target)
    loss.backward()
    return block_grad_norms(model)


def train_steps(
    pre_ln: bool,
    depth: int,
    dim: int,
    lr: float,
    steps: int,
    device: torch.device,
) -> tuple[float, float]:
    set_seed(0)
    model = TinyStack(dim, depth, pre_ln).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.randn(16, 32, dim, device=device)
    target = torch.randn(16, 32, 10, device=device)

    first = last = float("nan")
    for step in range(steps):
        loss = nn.functional.mse_loss(model(x), target)
        if step == 0:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
        if not math.isfinite(last):
            break
    return first, last


def main() -> None:
    device = get_device()
    dim = 64

    print(f"device: {device}")
    print("\nPart A: per-block parameter grad norm at init (no training)")
    header = f"{'depth':>6} {'placement':>10} {'first':>10} {'last':>10} {'last/first':>11} {'max/min':>9}"
    print(header)
    for depth in (24, 48, 96):
        for pre_ln in (True, False):
            norms = gradient_norms(pre_ln, depth, dim, device)
            print(
                f"{depth:>6} {'Pre-LN' if pre_ln else 'Post-LN':>10}"
                f" {norms[0]:>10.3e} {norms[-1]:>10.3e}"
                f" {norms[-1] / norms[0]:>11.2f} {max(norms) / min(norms):>9.2f}"
            )

    print("\nPart B: 60 Adam steps at lr=5e-3 (depth=32)")
    print(f"{'placement':>10} {'loss[0]':>10} {'loss[59]':>10}")
    for pre_ln in (True, False):
        first, last = train_steps(pre_ln, depth=32, dim=dim, lr=5e-3, steps=60, device=device)
        print(f"{'Pre-LN' if pre_ln else 'Post-LN':>10} {first:>10.3f} {last:>10.4f}")

    print(
        "\nReading: Post-LN's gradient norm drifts across depth and it fails to fit\n"
        "at a learning rate that is fine for Pre-LN. The identity residual path in\n"
        "Pre-LN keeps gradients uniform; Post-LN routes every gradient through LN."
    )


if __name__ == "__main__":
    main()
