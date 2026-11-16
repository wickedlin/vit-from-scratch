"""Arrange local CIFAR-10 images into a contact sheet for quick browsing (S3-3.6).

Run:
    python scripts/cifar10_samples.py
    python scripts/cifar10_samples.py --split train --per-class 10
    python scripts/cifar10_samples.py --output reports/cifar10_samples.png

Output defaults to `reports/cifar10_samples.png`: 10 rows (classes) × N columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save PNGs in headless environments too
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a CIFAR-10 contact sheet")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument(
        "--output", type=str, default="reports/cifar10_samples.png"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CIFAR10(
        root=args.data_root,
        train=args.split == "train",
        download=True,
        transform=None,
    )

    # CIFAR-10 targets is a list[int]; collect the first N images per class.
    by_class: list[list[int]] = [[] for _ in range(10)]
    for idx, label in enumerate(dataset.targets):
        if len(by_class[label]) < args.per_class:
            by_class[label].append(idx)

    fig, axes = plt.subplots(
        10,
        args.per_class,
        figsize=(1.5 * args.per_class, 15),
        squeeze=False,
    )
    for cls in range(10):
        for col in range(args.per_class):
            ax = axes[cls, col]
            if col < len(by_class[cls]):
                img = np.asarray(dataset[by_class[cls][col]][0])
                ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[cls, 0].set_ylabel(
            dataset.classes[cls],
            rotation=0,
            ha="right",
            va="center",
            fontsize=10,
            labelpad=24,
        )

    fig.suptitle(
        f"CIFAR-10 {args.split} split — {args.per_class} samples per class",
        fontsize=14,
    )
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path} ({len(dataset)} images in split)")


if __name__ == "__main__":
    main()
