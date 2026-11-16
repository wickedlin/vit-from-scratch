"""Plot training curves from metrics.jsonl (S3-3.6).

Three independent panels (shared epoch axis, no dual y-axes):
  1. train / val loss
  2. train / val accuracy
  3. learning rate (units ×1e-3)

Run:
    python scripts/plot_metrics.py \
        --metrics experiments/pilot_lr1e-3_w5/metrics.jsonl
    python scripts/plot_metrics.py \
        --metrics experiments/baseline_vit_tiny_p4/metrics.jsonl \
        --output reports/baseline_vit_tiny_p4_curves.png
    python scripts/plot_metrics.py --metrics ... --theme dark

Output defaults to `reports/<run>_curves.png`. Colors use CVD-validated blue /
orange categorical slots; LR is a single series drawn in neutral ink to avoid
confusion with the train/val identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, PercentFormatter


THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "train": "#2a78d6",
        "val": "#eb6834",
        "lr": "#52514e",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "train": "#3987e5",
        "val": "#d95926",
        "lr": "#c3c2b7",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics from JSONL")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--title", type=str, default="CIFAR-10 ViT-tiny — training curves")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no metrics in {path}")
    return records


def style_axes(ax, colors: dict) -> None:
    ax.set_facecolor(colors["surface"])
    ax.grid(True, axis="y", color=colors["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=colors["muted"], labelsize=9, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))


def line(ax, x, y, color: str, label: str):
    return ax.plot(
        x,
        y,
        color=color,
        linewidth=1.0,          # ~2 px at 150 dpi
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
        zorder=3,
    )[0]


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    records = load_metrics(metrics_path)
    colors = THEMES[args.theme]

    epochs = [int(r["epoch"]) + 1 for r in records]  # JSONL epochs are 0-based
    train_loss = [float(r["train_loss"]) for r in records]
    val_loss = [float(r["val_loss"]) for r in records]
    train_acc = [float(r["train_acc"]) for r in records]
    val_acc = [float(r["val_acc"]) for r in records]
    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    best_epoch = epochs[best_idx]
    best_val = val_acc[best_idx]
    has_lr = all("lr" in r for r in records)
    lr_milli = [float(r["lr"]) * 1e3 for r in records] if has_lr else []

    run_name = metrics_path.parent.name
    output_path = Path(args.output) if args.output else Path("reports") / f"{run_name}_curves.png"

    config = {}
    config_path = metrics_path.parent / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    recipe = " · ".join(
        part
        for part in [
            f"lr {config.get('lr')}" if "lr" in config else "",
            f"warmup {config.get('warmup_epochs')}" if "warmup_epochs" in config else "",
            f"amp {config.get('amp')}" if "amp" in config else "",
            (
                f"bs {config.get('batch_size')} × accum {config.get('accum_steps')}"
                if "batch_size" in config and "accum_steps" in config
                else ""
            ),
        ]
        if part
    )
    subtitle = (
        f"run: {run_name}"
        + (f" · {recipe}" if recipe else "")
        + f" · best val {best_val:.4f} @ e{best_epoch}"
    )

    n_panels = 3 if has_lr else 2
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(9, 2.9 * n_panels + 0.8),
        sharex=True,
    )
    fig.patch.set_facecolor(colors["surface"])

    # 1) Loss
    ax = axes[0]
    style_axes(ax, colors)
    line(ax, epochs, train_loss, colors["train"], "train")
    line(ax, epochs, val_loss, colors["val"], "val")
    ax.set_ylabel("Loss", color=colors["secondary"], fontsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        labelcolor=colors["secondary"],
    )

    # 2) Accuracy + mark best val (selective direct annotation)
    ax = axes[1]
    style_axes(ax, colors)
    line(ax, epochs, train_acc, colors["train"], "train")
    line(ax, epochs, val_acc, colors["val"], "val")
    ax.set_ylabel("Accuracy", color=colors["secondary"], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
        labelcolor=colors["secondary"],
    )

    ax.plot(
        epochs[best_idx],
        val_acc[best_idx],
        marker="o",
        markersize=4.5,
        color=colors["val"],
        markeredgecolor=colors["surface"],
        markeredgewidth=1.5,
        zorder=5,
    )

    # 3) Learning rate (single series, no legend needed)
    if has_lr:
        ax = axes[2]
        style_axes(ax, colors)
        line(ax, epochs, lr_milli, colors["lr"], "lr")
        ax.set_ylabel("LR (×10⁻³)", color=colors["secondary"], fontsize=10)
        ax.set_xlabel("Epoch", color=colors["secondary"], fontsize=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        peak_idx = max(range(len(lr_milli)), key=lambda i: lr_milli[i])
        ax.annotate(
            f"peak {lr_milli[peak_idx]:.2f}e-3",
            xy=(epochs[peak_idx], lr_milli[peak_idx]),
            xytext=(8, 6),
            textcoords="offset points",
            color=colors["secondary"],
            fontsize=8.5,
            annotation_clip=False,
        )
    else:
        axes[1].set_xlabel("Epoch", color=colors["secondary"], fontsize=10)

    fig.suptitle(args.title, color=colors["text"], fontsize=14, y=0.985)
    fig.text(
        0.5,
        0.948,
        subtitle,
        ha="center",
        color=colors["secondary"],
        fontsize=9,
    )
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.10, right=0.98, hspace=0.32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=colors["surface"])
    plt.close(fig)

    print(
        f"run={run_name} epochs={len(records)} "
        f"final train_loss={train_loss[-1]:.4f} train_acc={train_acc[-1]:.4f} "
        f"val_loss={val_loss[-1]:.4f} val_acc={val_acc[-1]:.4f} "
        f"best_val={val_acc[best_idx]:.4f} @ e{epochs[best_idx]}",
        flush=True,
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
