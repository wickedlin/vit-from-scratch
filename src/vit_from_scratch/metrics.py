"""Evaluation metrics for single-label multi-class classification (S3-3.7).

Core conventions
----------------
1. Every function takes arrays for the **entire evaluation set**: `logits` has
   shape `(N, K)` and `targets` has shape `(N,)`. Do not compute P/R/F1/ECE per
   batch and average them; these metrics are nonlinear, so aggregate predictions
   first and then compute the metric.
2. Probabilities are computed as `softmax(logits)` in float64 to avoid rounding
   from fp16/bf16 logits affecting probability metrics such as ECE/NLL.
3. This is an **evaluation utility**, not core ViT network code; the test
   scaffolding may be generated directly.

Main entry points
-----------------
`compute_metrics(logits, targets, ...)`: computes accuracy, top-k, per-class
P/R/F1, confusion matrix, ROC-AUC, PR-AUC, NLL, Brier, ECE, and top confusion
pairs in one call.

`mcnemar_test(correct_a, correct_b)`: paired accuracy test for two models on the
same test set. `paired_bootstrap_ci` gives a paired bootstrap confidence interval
for the accuracy difference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax returning float64 `(N, K)` with each row summing to 1."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, K), got shape {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _as_numpy(value) -> np.ndarray:
    """Accept numpy / torch / list and return numpy (torch tensors are detached to CPU)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict]]:
    """Equal-width ECE and reliability-diagram data.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|
    conf_b is the mean max(prob) in the bin; acc_b is the top-1 accuracy in the bin.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = _as_numpy(targets).astype(np.int64)
    if probs.ndim != 2 or len(probs) != len(targets):
        raise ValueError("probs and targets must have matching lengths")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # conf == 1.0 maps to n_bins; clip it back into the last bin.
    bin_ids = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)

    n = len(targets)
    ece = 0.0
    rows: list[dict] = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": b,
                    "lower": float(edges[b]),
                    "upper": float(edges[b + 1]),
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue
        mean_conf = float(confidence[mask].mean())
        mean_acc = float(correct[mask].mean())
        gap = abs(mean_acc - mean_conf)
        ece += (count / n) * gap
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "confidence": mean_conf,
                "accuracy": mean_acc,
                "gap": gap,
            }
        )
    return float(ece), rows


def top_confusions(
    matrix: np.ndarray,
    class_names: Sequence[str] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Find the most frequent "true class -> predicted class" errors in a confusion matrix (off-diagonal)."""
    matrix = np.asarray(matrix, dtype=np.int64)
    k = matrix.shape[0]
    names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(k)]
    support = matrix.sum(axis=1)
    pairs: list[dict] = []
    for true_id in range(k):
        for pred_id in range(k):
            if true_id == pred_id:
                continue
            count = int(matrix[true_id, pred_id])
            if count == 0:
                continue
            rate = float(count / support[true_id]) if support[true_id] else 0.0
            pairs.append(
                {
                    "true_id": true_id,
                    "true_name": names[true_id],
                    "pred_id": pred_id,
                    "pred_name": names[pred_id],
                    "count": count,
                    "rate": rate,
                }
            )
    pairs.sort(key=lambda row: (row["count"], row["rate"]), reverse=True)
    return pairs[:top_n]


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 15,
    topk: Sequence[int] = (3,),
    class_names: Sequence[str] | None = None,
    top_confusion_pairs: int = 5,
) -> dict:
    """Compute the full set of single-label multi-class metrics in one call.

    Args:
        logits: `(N, K)` raw scores (before softmax).
        targets: `(N,)` integer labels in `[0, K)`.
        n_bins: Number of equal-width bins for ECE / reliability.
        topk: Top-k accuracies to compute; k must be `<= K`; k=1 is skipped
            because it equals accuracy.
        class_names: Length-K class names; defaults to `class_0 ... class_{K-1}`.
        top_confusion_pairs: How many of the most common confusion pairs to return.

    Returns:
        A JSON-serializable dict; per-class metrics are in `per_class`, and the
        confusion matrices are in `confusion_matrix` and
        `confusion_matrix_normalized`.
    """
    logits = _as_numpy(logits).astype(np.float64)
    targets = _as_numpy(targets).astype(np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, K), got shape {logits.shape}")
    if targets.ndim != 1 or len(targets) != len(logits):
        raise ValueError("targets must be 1-D and match logits length")
    n, k = logits.shape
    if n == 0:
        raise ValueError("cannot compute metrics on an empty dataset")
    if targets.min() < 0 or targets.max() >= k:
        raise ValueError(f"targets must be in [0, {k}), got [{targets.min()}, {targets.max()}]")

    names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"class_names must have length {k}, got {len(names)}")

    probs = softmax(logits)
    predictions = probs.argmax(axis=1)
    labels = np.arange(k)

    accuracy = float(accuracy_score(targets, predictions))

    topk_acc: dict[str, float] = {}
    for value in topk:
        value = int(value)
        if value <= 1 or value > k:
            continue
        topk_acc[str(value)] = float(
            top_k_accuracy_score(targets, probs, k=value, labels=labels)
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    # For single-label multi-class, micro-P = micro-R = micro-F1 = accuracy;
    # compute it explicitly so the identity sits next to macro/weighted values.
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="weighted", zero_division=0
    )

    per_class = [
        {
            "class_id": int(i),
            "class_name": names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(k)
    ]

    matrix = confusion_matrix(targets, predictions, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )

    # Multi-class ROC-AUC: one-vs-rest. Macro averages the K binary AUCs;
    # micro pools all class predictions into one binary problem.
    if k == 2:
        roc_macro = roc_micro = float(roc_auc_score(targets, probs[:, 1]))
        roc_per_class = [roc_macro, roc_macro]
    else:
        roc_macro = float(
            roc_auc_score(targets, probs, multi_class="ovr", average="macro", labels=labels)
        )
        roc_micro = float(
            roc_auc_score(targets, probs, multi_class="ovr", average="micro", labels=labels)
        )
        roc_per_class = []
        for i in range(k):
            binary = (targets == i).astype(np.int64)
            if binary.min() == binary.max():
                roc_per_class.append(float("nan"))
            else:
                roc_per_class.append(float(roc_auc_score(binary, probs[:, i])))

    # PR-AUC / average precision: one-vs-rest per class, then macro average.
    ap_per_class: list[float] = []
    for i in range(k):
        binary = (targets == i).astype(np.int64)
        if binary.min() == binary.max():
            ap_per_class.append(float("nan"))
        else:
            ap_per_class.append(float(average_precision_score(binary, probs[:, i])))
    finite_ap = [value for value in ap_per_class if np.isfinite(value)]
    ap_macro = float(np.mean(finite_ap)) if finite_ap else float("nan")

    # Hard-label NLL and multi-class Brier; neither equals the label-smoothed
    # training loss.
    nll = float(log_loss(targets, probs, labels=labels))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), targets] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    ece, reliability = expected_calibration_error(probs, targets, n_bins=n_bins)

    return {
        "n": int(n),
        "num_classes": int(k),
        "class_names": names,
        "accuracy": accuracy,
        "top_k_accuracy": topk_acc,
        "precision_micro": float(micro_p),
        "recall_micro": float(micro_r),
        "f1_micro": float(micro_f1),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_p),
        "recall_weighted": float(weighted_r),
        "f1_weighted": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "top_confusions": top_confusions(matrix, names, top_n=top_confusion_pairs),
        "roc_auc_ovr_macro": roc_macro,
        "roc_auc_ovr_micro": roc_micro,
        "roc_auc_per_class": roc_per_class,
        "average_precision_macro": ap_macro,
        "average_precision_per_class": ap_per_class,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "reliability": reliability,
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test for two models on the same samples (is accuracy different?).

    Definitions:
        n_a_correct_b_wrong = A correct and B wrong (b)
        n_a_wrong_b_correct = A wrong and B correct (c)
    Only discordant samples are used; when b+c is small, use the exact binomial
    distribution, otherwise the chi-square approximation is also reported.
    """
    correct_a = _as_numpy(correct_a).astype(bool)
    correct_b = _as_numpy(correct_b).astype(bool)
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length")

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c
    n = len(correct_a)
    diff = float(correct_a.mean() - correct_b.mean())

    if discordant == 0:
        return {
            "n": n,
            "accuracy_a": float(correct_a.mean()),
            "accuracy_b": float(correct_b.mean()),
            "diff": diff,
            "n_a_correct_b_wrong": 0,
            "n_a_wrong_b_correct": 0,
            "discordant": 0,
            "statistic": 0.0,
            "p_value_exact": 1.0,
            "p_value_chi2": 1.0,
        }

    p_exact = float(binomtest(min(b, c), discordant, 0.5, alternative="two-sided").pvalue)
    statistic = float((abs(b - c) - 1) ** 2 / discordant)
    p_chi2 = float(chi2.sf(statistic, df=1))
    return {
        "n": n,
        "accuracy_a": float(correct_a.mean()),
        "accuracy_b": float(correct_b.mean()),
        "diff": diff,
        "n_a_correct_b_wrong": b,
        "n_a_wrong_b_correct": c,
        "discordant": discordant,
        "statistic": statistic,
        "p_value_exact": p_exact,
        "p_value_chi2": p_chi2,
    }


def paired_bootstrap_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap percentile confidence interval for the accuracy difference (A - B).

    Each resample uses the same sample indices for A and B, preserving the
    correlation between their predictions; this is more correct than checking
    whether two independent CIs overlap when comparing two models.
    """
    correct_a = _as_numpy(correct_a).astype(np.float64)
    correct_b = _as_numpy(correct_b).astype(np.float64)
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    n = len(correct_a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()

    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "diff": float(correct_a.mean() - correct_b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
    }


def wilson_interval(correct: np.ndarray, z: float = 1.96) -> dict:
    """Wilson 95% interval for a single accuracy (better than the normal approximation near 0/1)."""
    correct = _as_numpy(correct).astype(bool)
    n = len(correct)
    if n == 0:
        raise ValueError("cannot compute an interval on zero samples")
    k = int(correct.sum())
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "accuracy": float(p),
        "low": float(max(0.0, center - half)),
        "high": float(min(1.0, center + half)),
        "n": n,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni multiple-comparison correction, preserving input order."""
    values = [float(p) for p in p_values]
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted
