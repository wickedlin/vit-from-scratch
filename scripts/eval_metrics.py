"""Full classification evaluation of one or more checkpoints on val/test (S3-3.7).

Division of labor with `eval_checkpoint.py`:
- `eval_checkpoint.py` only computes loss/acc for a quick check of test targets;
- this script collects all logits in one forward pass, computes P/R/F1,
  confusion matrices, top-k, ROC-AUC/PR-AUC, and NLL/Brier/ECE, and runs paired
  McNemar + bootstrap comparisons across multiple models.

Run (example with three dropout runs):
    python scripts/eval_metrics.py \
        --checkpoint experiments/baseline_vit_tiny_p4/best.pt \
        --checkpoint experiments/reg_dp005_do005_lr1e-3/best.pt \
        --checkpoint experiments/reg_dp01_do01_lr1e-3/best.pt \
        --names baseline,drop005,drop01 \
        --split test --output-dir experiments/eval_3runs

Outputs:
    metrics_full.json       metrics for all models + pairwise comparisons
    summary.csv             key metrics, one row per model
    summary.txt             copy of the console summary
    predictions_<name>.npz  logits/probs/targets/pred for later re-analysis
    confusion_matrices.png  per-class_f1.png  roc_curves.png  reliability.png

Notes:
- Model architecture is read from `model_config.json` next to the checkpoint by
  default; older checkpoints (e.g. the S3 baseline) lack that file, in which
  case the script falls back to the S3 baseline defaults and prints a warning.
- All models share the same DataLoader (shuffle=False), so sample order is
  identical and the paired McNemar assumption holds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vit_from_scratch.amp import autocast_context, resolve_amp_dtype
from vit_from_scratch.data import build_dataloaders
from vit_from_scratch.metrics import (
    compute_metrics,
    holm_adjust,
    mcnemar_test,
    paired_bootstrap_ci,
    softmax,
    wilson_interval,
)
from vit_from_scratch.optim import label_smoothing_loss
from vit_from_scratch.train import load_checkpoint
from vit_from_scratch.swin import SwinTransformer
from vit_from_scratch.utils import get_device
from vit_from_scratch.vit import ViT


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEFAULT_MODEL_KWARGS = {
    "image_size": 32,
    "patch_size": 4,
    "in_channels": 3,
    "num_classes": 10,
    "dim": 192,
    "depth": 12,
    "heads": 3,
    "mlp_dim": 768,
    "dropout": 0.0,
    "emb_dropout": 0.0,
    "drop_path": 0.0,
    "pool": "cls",
    "pos_embed": "learned",
    "pre_norm": True,
}

# Four categorical series slots validated by the dataviz validator (fixed order, no cycling).
# In the light theme, aqua (2.74) and yellow (2.11) fall below 3:1, so the figures
# also provide a value-labelled legend and a summary table as relief.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "chance": "#898781",
        "seq_low": "#cde2fb",
        "seq_high": "#0d366b",
        "on_dark": "#ffffff",
        "on_light": "#0b0b0b",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "chance": "#898781",
        "seq_low": "#0d366b",
        "seq_high": "#cde2fb",
        "on_dark": "#0b0b0b",
        "on_light": "#f0efec",
    },
}


def series_color(theme: dict, index: int) -> str:
    """Pick a categorical color in fixed order; no cycling.

    Raise a clear error when the requested slot does not exist.
    """
    colors = theme["series"]
    if index >= len(colors):
        raise ValueError(
            f"palette has {len(colors)} series slots, need {index + 1}; "
            "extend THEMES['light'/'dark']['series'] with validated colors"
        )
    return colors[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full classification evaluation for one or more ViT checkpoints"
    )
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="checkpoint path; repeat to pass multiple",
    )
    parser.add_argument(
        "--names", type=str, default="",
        help="comma-separated display names; defaults to each checkpoint's parent directory name",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp", choices=["off", "auto", "bf16", "fp16"], default="off",
        help="evaluation precision; defaults to off for reproducible metrics",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument(
        "--min-bin-count", type=int, default=50,
        help=(
            "reliability plot only draws bins with at least this many samples; "
            "ECE still uses all bins"
        ),
    )
    parser.add_argument("--top-confusions", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="experiments/eval_metrics")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument(
        "--model-config", action="append", default=[],
        help="optional: NAME=PATH to explicitly set the architecture JSON for a checkpoint",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def resolve_names(raw: str, checkpoints: list[Path]) -> list[str]:
    if raw.strip():
        names = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        names = [path.parent.name for path in checkpoints]
    if len(names) != len(checkpoints):
        raise SystemExit(
            f"--names has {len(names)} entries but --checkpoint has {len(checkpoints)}"
        )
    # Append a suffix to duplicate names to avoid JSON key / legend collisions.
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def parse_model_configs(items: list[str]) -> dict[str, Path]:
    configs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--model-config expects NAME=PATH")
        name, raw_path = item.split("=", 1)
        configs[name.strip()] = Path(raw_path.strip())
    return configs


def resolve_model_kwargs(
    checkpoint_path: Path,
    name: str,
    explicit: dict[str, Path],
) -> tuple[dict, str, str]:
    """Return (model_kwargs, arch, config_source). arch is determined by model_config.json."""
    config_path = explicit.get(name, checkpoint_path.parent / "model_config.json")
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        arch = raw.pop("arch", "vit")
        if arch == "swin":
            return raw, arch, str(config_path)
        kwargs = dict(DEFAULT_MODEL_KWARGS)
        kwargs.update(raw)
        return kwargs, arch, str(config_path)
    print(
        f"[warn] {name}: no model_config.json; using S3 baseline defaults "
        f"(p4/d192/L12/h3). Pass --model-config {name}=PATH if this is wrong.",
        flush=True,
    )
    return dict(DEFAULT_MODEL_KWARGS), "vit", "defaults"


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collect logits, targets, and smoothed loss for the whole split in one forward pass."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            logits = model(x)
            loss = label_smoothing_loss(logits, y, smoothing=smoothing)
        all_logits.append(logits.float().cpu())
        all_targets.append(y.cpu())
        total_loss += float(loss.float().item()) * x.size(0)
        total_samples += x.size(0)
    if total_samples == 0:
        raise SystemExit("empty evaluation loader")
    logits_np = torch.cat(all_logits).numpy()
    targets_np = torch.cat(all_targets).numpy()
    return logits_np, targets_np, total_loss / total_samples


def evaluate_checkpoint(
    checkpoint_path: Path,
    name: str,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    args: argparse.Namespace,
    explicit_configs: dict[str, Path],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    kwargs, arch, config_source = resolve_model_kwargs(
        checkpoint_path, name, explicit_configs
    )
    if arch == "swin":
        model = SwinTransformer(**kwargs).to(device)
    elif arch == "vit":
        model = ViT(**kwargs).to(device)
    else:
        raise SystemExit(f"{name}: unknown arch {arch!r}")
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    config = checkpoint.get("config", {})
    smoothing = float(config.get("label_smoothing", 0.1))

    logits, targets, loss_smoothed = collect_logits(
        model, loader, device, amp_dtype, smoothing
    )
    metrics = compute_metrics(
        logits,
        targets,
        n_bins=args.n_bins,
        topk=(3,),
        class_names=CIFAR10_CLASSES,
        top_confusion_pairs=args.top_confusions,
    )
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    correct = pred == targets

    result = {
        "name": name,
        "arch": arch,
        "checkpoint": str(checkpoint_path),
        "model_config_source": config_source,
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_val_acc": float(checkpoint.get("best_acc", float("nan"))),
        "label_smoothing": smoothing,
        "loss_smoothed": float(loss_smoothed),
        "loss_hard": float(metrics["nll"]),
        "wilson_accuracy_ci": wilson_interval(correct),
        "metrics": metrics,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, correct, logits, probs, targets


def jsonable(value):
    """Convert numpy scalars and NaN/Inf to plain JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def build_comparisons(
    names: list[str],
    correct: dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> list[dict]:
    comparisons: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mcnemar = mcnemar_test(correct[a], correct[b])
            bootstrap = paired_bootstrap_ci(
                correct[a], correct[b], n_boot=n_boot, seed=seed
            )
            comparisons.append(
                {"a": a, "b": b, **mcnemar, "bootstrap": bootstrap}
            )
    adjusted = holm_adjust([row["p_value_exact"] for row in comparisons])
    for row, p_holm in zip(comparisons, adjusted):
        row["p_value_holm"] = float(p_holm)
    return comparisons


def style_axes(ax, theme: dict, grid_axis: str = "y") -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, axis=grid_axis, color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=8.5, length=0)
    ax.xaxis.label.set_color(theme["secondary"])
    ax.yaxis.label.set_color(theme["secondary"])
    ax.title.set_color(theme["text"])


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {path}", flush=True)


def plot_confusion_matrices(models: dict[str, dict], theme: dict, output_dir: Path) -> None:
    names = list(models)
    count = len(names)
    ncols = min(3, count)
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols + 0.8, 4.0 * nrows + 0.9),
        squeeze=False,
    )
    fig.patch.set_facecolor(theme["surface"])
    # With a two-row layout, the upper row's xlabel collides with the lower
    # row's title; increase row spacing explicitly.
    fig.subplots_adjust(hspace=0.5, wspace=0.28)
    cmap = LinearSegmentedColormap.from_list(
        "cifar_blue", [theme["seq_low"], theme["seq_high"]]
    )
    image = None
    for index, name in enumerate(names):
        ax = axes[index // ncols][index % ncols]
        matrix = np.asarray(
            models[name]["metrics"]["confusion_matrix"], dtype=np.float64
        )
        support = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            support,
            out=np.zeros_like(matrix),
            where=support > 0,
        )
        image = ax.imshow(normalized, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(CIFAR10_CLASSES)))
        ax.set_yticks(range(len(CIFAR10_CLASSES)))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=7)
        ax.tick_params(colors=theme["muted"], length=0)
        ax.set_xlabel("Predicted", color=theme["secondary"], fontsize=9)
        ax.set_ylabel("True", color=theme["secondary"], fontsize=9)
        ax.set_title(
            f"{name} · acc {models[name]['metrics']['accuracy']:.4f}",
            color=theme["text"],
            fontsize=10,
        )
        for row in range(normalized.shape[0]):
            for col in range(normalized.shape[1]):
                value = normalized[row, col]
                if value < 0.01:
                    continue
                color = (
                    theme["on_dark"] if value >= 0.55 else theme["on_light"]
                )
                ax.text(
                    col,
                    row,
                    f"{value * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=color,
                )
    for index in range(count, nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    if image is not None:
        colorbar = fig.colorbar(
            image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02
        )
        colorbar.set_label(
            "Row-normalized share", color=theme["secondary"], fontsize=9
        )
        colorbar.ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )
        colorbar.ax.tick_params(colors=theme["muted"], labelsize=8)
    fig.suptitle(
        "Confusion matrices — row-normalized (%), test",
        color=theme["text"],
        fontsize=13,
    )
    save_figure(fig, output_dir / "confusion_matrices.png")


def plot_per_class_f1(models: dict[str, dict], theme: dict, output_dir: Path) -> None:
    names = list(models)
    class_names = models[names[0]]["metrics"]["class_names"]
    x = np.arange(len(class_names))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor(theme["surface"])
    style_axes(ax, theme, grid_axis="y")
    for index, name in enumerate(names):
        metrics = models[name]["metrics"]
        f1 = [row["f1"] for row in metrics["per_class"]]
        offset = (index - (len(names) - 1) / 2) * width
        ax.bar(
            x + offset,
            f1,
            width=width,
            color=series_color(theme, index),
            edgecolor=theme["surface"],
            linewidth=1.0,
            label=f"{name} (macro F1 {metrics['f1_macro']:.4f})",
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("F1", color=theme["secondary"], fontsize=10)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8.5,
        labelcolor=theme["secondary"],
        ncol=len(names),
    )
    ax.set_title("Per-class F1 — test", color=theme["text"], fontsize=13)
    save_figure(fig, output_dir / "per_class_f1.png")


def plot_roc(
    models: dict[str, dict],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    theme: dict,
    output_dir: Path,
) -> None:
    names = list(models)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fig.patch.set_facecolor(theme["surface"])
    style_axes(ax, theme, grid_axis="both")
    grid = np.linspace(0.0, 1.0, 201)
    for index, name in enumerate(names):
        probs, targets = predictions[name]
        curves = []
        for class_id in range(probs.shape[1]):
            binary = (targets == class_id).astype(np.int64)
            fpr, tpr, _ = roc_curve(binary, probs[:, class_id])
            curves.append(np.interp(grid, fpr, tpr))
        macro_tpr = np.mean(curves, axis=0)
        auc = models[name]["metrics"]["roc_auc_ovr_macro"]
        ax.plot(
            grid,
            macro_tpr,
            color=series_color(theme, index),
            linewidth=1.4,
            solid_capstyle="round",
            label=f"{name} (macro AUC {auc:.4f})",
            zorder=3,
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color=theme["chance"],
        linewidth=1.0,
        label="chance",
        zorder=2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("False positive rate", color=theme["secondary"], fontsize=10)
    ax.set_ylabel("True positive rate", color=theme["secondary"], fontsize=10)
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8.5,
        labelcolor=theme["secondary"],
    )
    ax.set_title("Macro one-vs-rest ROC — test", color=theme["text"], fontsize=13)
    save_figure(fig, output_dir / "roc_curves.png")


def plot_reliability(
    models: dict[str, dict],
    theme: dict,
    output_dir: Path,
    min_bin_count: int,
) -> None:
    names = list(models)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fig.patch.set_facecolor(theme["surface"])
    style_axes(ax, theme, grid_axis="both")
    for index, name in enumerate(names):
        metrics = models[name]["metrics"]
        rows = [
            row
            for row in metrics["reliability"]
            if row["count"] >= min_bin_count
        ]
        xs = [row["confidence"] for row in rows]
        ys = [row["accuracy"] for row in rows]
        ax.plot(
            xs,
            ys,
            color=series_color(theme, index),
            linewidth=1.4,
            marker="o",
            markersize=4.5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
            label=f"{name} (ECE {metrics['ece']:.4f})",
            zorder=3,
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color=theme["chance"],
        linewidth=1.0,
        label="perfect calibration",
        zorder=2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Mean confidence in bin", color=theme["secondary"], fontsize=10)
    ax.set_ylabel("Empirical accuracy", color=theme["secondary"], fontsize=10)
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8.5,
        labelcolor=theme["secondary"],
    )
    ax.text(
        0.02,
        0.98,
        f"shown bins: n≥{min_bin_count}; ECE uses all bins",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=theme["muted"],
        fontsize=8,
    )
    ax.set_title(
        f"Reliability diagram — test ({models[names[0]]['metrics']['n']} samples)",
        color=theme["text"],
        fontsize=13,
    )
    save_figure(fig, output_dir / "reliability.png")


def format_p(value: float) -> str:
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def summary_lines(
    models: dict[str, dict],
    comparisons: list[dict],
    split: str,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"split={split} samples={next(iter(models.values()))['metrics']['n']}")
    lines.append("")
    header = (
        f"{'name':<20} {'acc':>7} {'95% CI':>15} {'top3':>7} {'macroF1':>8} "
        f"{'macroP':>8} {'macroR':>8} {'ROC-AUC':>8} {'AP':>7} {'NLL':>7} {'ECE':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, row in models.items():
        metrics = row["metrics"]
        ci = row["wilson_accuracy_ci"]
        top3 = metrics["top_k_accuracy"].get("3")
        top3_text = f"{top3:.4f}" if top3 is not None else "-"
        lines.append(
            f"{name:<20} {metrics['accuracy']:>7.4f} "
            f"[{ci['low']:.3f},{ci['high']:.3f}] "
            f"{top3_text:>7} {metrics['f1_macro']:>8.4f} "
            f"{metrics['precision_macro']:>8.4f} {metrics['recall_macro']:>8.4f} "
            f"{metrics['roc_auc_ovr_macro']:>8.4f} "
            f"{metrics['average_precision_macro']:>7.4f} "
            f"{metrics['nll']:>7.4f} {metrics['ece']:>7.4f}"
        )
    lines.append("")
    lines.append("Pairwise McNemar (accuracy; diff = A - B):")
    pair_header = (
        f"{'A vs B':<30} {'diff':>8} {'b':>6} {'c':>6} "
        f"{'p(exact)':>10} {'p(Holm)':>10} {'bootstrap 95% CI':>22}"
    )
    lines.append(pair_header)
    lines.append("-" * len(pair_header))
    for row in comparisons:
        ci = row["bootstrap"]
        lines.append(
            f"{row['a'] + ' vs ' + row['b']:<30} "
            f"{row['diff'] * 100:>+7.2f}pp "
            f"{row['n_a_correct_b_wrong']:>6} {row['n_a_wrong_b_correct']:>6} "
            f"{format_p(row['p_value_exact']):>10} "
            f"{format_p(row['p_value_holm']):>10} "
            f"[{ci['ci_low'] * 100:+.2f}, {ci['ci_high'] * 100:+.2f}]pp"
        )
    return lines


def write_summary_csv(models: dict[str, dict], output_path: Path) -> None:
    fields = [
        "name",
        "checkpoint",
        "epoch",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "top3",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "average_precision_macro",
        "nll",
        "brier",
        "ece",
        "loss_smoothed",
        "loss_hard",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in models.items():
            metrics = row["metrics"]
            writer.writerow(
                {
                    "name": name,
                    "checkpoint": row["checkpoint"],
                    "epoch": row["epoch"],
                    "accuracy": f"{metrics['accuracy']:.6f}",
                    "accuracy_ci_low": f"{row['wilson_accuracy_ci']['low']:.6f}",
                    "accuracy_ci_high": f"{row['wilson_accuracy_ci']['high']:.6f}",
                    "top3": f"{metrics['top_k_accuracy'].get('3', float('nan')):.6f}",
                    "precision_macro": f"{metrics['precision_macro']:.6f}",
                    "recall_macro": f"{metrics['recall_macro']:.6f}",
                    "f1_macro": f"{metrics['f1_macro']:.6f}",
                    "f1_weighted": f"{metrics['f1_weighted']:.6f}",
                    "roc_auc_ovr_macro": f"{metrics['roc_auc_ovr_macro']:.6f}",
                    "average_precision_macro": f"{metrics['average_precision_macro']:.6f}",
                    "nll": f"{metrics['nll']:.6f}",
                    "brier": f"{metrics['brier']:.6f}",
                    "ece": f"{metrics['ece']:.6f}",
                    "loss_smoothed": f"{row['loss_smoothed']:.6f}",
                    "loss_hard": f"{row['loss_hard']:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    checkpoints = [Path(path) for path in args.checkpoint]
    names = resolve_names(args.names, checkpoints)
    explicit_configs = parse_model_configs(args.model_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        data_root=args.data_root,
        seed=42,
        shuffle_seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader = loaders[args.split]
    amp_dtype = resolve_amp_dtype(device, args.amp)
    print(
        f"device={device.type} split={args.split} samples={len(loader.dataset)} "
        f"amp={args.amp}",
        flush=True,
    )

    models: dict[str, dict] = {}
    correct: dict[str, np.ndarray] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for checkpoint_path, name in zip(checkpoints, names):
        print(f"[eval] {name}: {checkpoint_path}", flush=True)
        result, model_correct, _logits, probs, targets = evaluate_checkpoint(
            checkpoint_path,
            name,
            loader,
            device,
            amp_dtype,
            args,
            explicit_configs,
        )
        models[name] = result
        correct[name] = model_correct
        predictions[name] = (probs, targets)
        np.savez_compressed(
            output_dir / f"predictions_{name}.npz",
            logits=_logits,
            probs=probs,
            targets=targets,
            pred=probs.argmax(axis=1),
            correct=model_correct,
        )
        print(
            f"  acc={result['metrics']['accuracy']:.4f} "
            f"macroF1={result['metrics']['f1_macro']:.4f} "
            f"rocAUC={result['metrics']['roc_auc_ovr_macro']:.4f} "
            f"ece={result['metrics']['ece']:.4f}",
            flush=True,
        )

    comparisons = build_comparisons(names, correct, args.bootstrap, args.seed)
    payload = {
        "split": args.split,
        "data_root": args.data_root,
        "samples": next(iter(models.values()))["metrics"]["n"],
        "class_names": CIFAR10_CLASSES,
        "n_bins": args.n_bins,
        "models": models,
        "comparisons": comparisons,
    }
    metrics_path = output_dir / "metrics_full.json"
    metrics_path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_csv(models, output_dir / "summary.csv")
    lines = summary_lines(models, comparisons, args.split)
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    if not args.no_figures:
        theme = THEMES[args.theme]
        plot_confusion_matrices(models, theme, output_dir)
        plot_per_class_f1(models, theme, output_dir)
        plot_roc(models, predictions, theme, output_dir)
        plot_reliability(models, theme, output_dir, args.min_bin_count)

    print(f"\nmetrics: {metrics_path}", flush=True)
    print(f"summary: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
