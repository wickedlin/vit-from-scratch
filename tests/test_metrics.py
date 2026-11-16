"""Tests for src/vit_from_scratch/metrics.py (S3-3.7).

Run from the project root:
    python tests/test_metrics.py

These are tests for evaluation utilities, not models/data; the focus is on
the "aggregate first, then compute" convention.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.metrics import (  # noqa: E402
    compute_metrics,
    expected_calibration_error,
    holm_adjust,
    mcnemar_test,
    paired_bootstrap_ci,
    softmax,
    top_confusions,
    wilson_interval,
)


def logits_from_predictions(predictions: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Build logits whose argmax is exactly predictions."""
    logits = np.full((len(predictions), num_classes), -10.0, dtype=np.float64)
    logits[np.arange(len(predictions)), predictions] = 10.0
    return logits


class TestSoftmax(unittest.TestCase):
    def test_matches_torch_and_sums_to_one(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(7, 5))
        probs = softmax(logits)
        ref = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=-1).numpy()
        np.testing.assert_allclose(probs, ref, atol=1e-12)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(7), atol=1e-12)

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            softmax(np.zeros(5))


class TestPerfectPredictions(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([0, 1, 2, 0, 1, 2])
        logits = np.full((len(self.targets), 3), -20.0)
        logits[np.arange(len(self.targets)), self.targets] = 20.0
        self.metrics = compute_metrics(
            logits, self.targets, class_names=["a", "b", "c"]
        )

    def test_accuracy_and_topk(self):
        self.assertAlmostEqual(self.metrics["accuracy"], 1.0)
        self.assertAlmostEqual(self.metrics["top_k_accuracy"]["3"], 1.0)

    def test_per_class_and_macro(self):
        self.assertAlmostEqual(self.metrics["f1_macro"], 1.0)
        self.assertAlmostEqual(self.metrics["precision_macro"], 1.0)
        self.assertAlmostEqual(self.metrics["recall_macro"], 1.0)
        for row in self.metrics["per_class"]:
            self.assertAlmostEqual(row["f1"], 1.0)
            self.assertEqual(row["support"], 2)

    def test_probability_metrics(self):
        self.assertLess(self.metrics["nll"], 1e-6)
        self.assertLess(self.metrics["brier"], 1e-6)
        self.assertLess(self.metrics["ece"], 1e-6)
        self.assertAlmostEqual(self.metrics["roc_auc_ovr_macro"], 1.0)
        self.assertAlmostEqual(self.metrics["average_precision_macro"], 1.0)


class TestKnownConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([0, 0, 0, 1, 1, 2])
        self.predictions = np.array([0, 0, 1, 1, 2, 2])
        self.metrics = compute_metrics(
            logits_from_predictions(self.predictions), self.targets
        )

    def test_accuracy_and_confusion(self):
        self.assertAlmostEqual(self.metrics["accuracy"], 4 / 6)
        self.assertEqual(
            self.metrics["confusion_matrix"],
            [[2, 1, 0], [0, 1, 1], [0, 0, 1]],
        )
        np.testing.assert_allclose(
            self.metrics["confusion_matrix_normalized"],
            [[2 / 3, 1 / 3, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
        )

    def test_per_class_precision_recall_f1(self):
        rows = self.metrics["per_class"]
        expected = [
            (1.0, 2 / 3, 0.8, 3),
            (0.5, 0.5, 0.5, 2),
            (0.5, 1.0, 2 / 3, 1),
        ]
        for row, (precision, recall, f1, support) in zip(rows, expected):
            self.assertAlmostEqual(row["precision"], precision)
            self.assertAlmostEqual(row["recall"], recall)
            self.assertAlmostEqual(row["f1"], f1)
            self.assertEqual(row["support"], support)

    def test_macro_weighted_and_micro(self):
        self.assertAlmostEqual(self.metrics["precision_macro"], 2 / 3)
        self.assertAlmostEqual(self.metrics["recall_macro"], 13 / 18)
        self.assertAlmostEqual(self.metrics["f1_macro"], 59 / 90)
        self.assertAlmostEqual(self.metrics["f1_weighted"], 61 / 90)
        # Identity for single-label multiclass: micro-F1 = accuracy.
        self.assertAlmostEqual(self.metrics["f1_micro"], self.metrics["accuracy"])
        self.assertAlmostEqual(self.metrics["f1_micro"], 4 / 6)

    def test_top_confusions(self):
        pairs = self.metrics["top_confusions"]
        self.assertEqual(len(pairs), 2)
        # Each error pair occurs once; true=1->pred=2 has a higher rate, so it ranks first.
        self.assertEqual(
            (pairs[0]["true_id"], pairs[0]["pred_id"], pairs[0]["rate"]),
            (1, 2, 0.5),
        )
        self.assertEqual(
            (pairs[1]["true_id"], pairs[1]["pred_id"], pairs[1]["rate"]),
            (0, 1, 1 / 3),
        )


class TestCalibration(unittest.TestCase):
    def test_ece_known_value(self):
        probs = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.90, 0.05, 0.05],
                [0.20, 0.20, 0.60],
                [0.20, 0.20, 0.60],
            ]
        )
        targets = np.array([0, 1, 2, 0])
        # Two bins: [0,0.5) empty; [0.5,1] holds 4 samples, conf 0.75, acc 0.5.
        ece, rows = expected_calibration_error(probs, targets, n_bins=2)
        self.assertAlmostEqual(ece, 0.25)
        self.assertEqual(rows[0]["count"], 0)
        self.assertEqual(rows[1]["count"], 4)
        self.assertAlmostEqual(rows[1]["confidence"], 0.75)
        self.assertAlmostEqual(rows[1]["accuracy"], 0.5)
        self.assertAlmostEqual(rows[1]["gap"], 0.25)

    def test_ece_is_zero_for_perfectly_calibrated_bins(self):
        # One bin with conf=0.5 and exactly half correct -> gap=0.
        probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        targets = np.array([0, 0, 1, 1])
        ece, _ = expected_calibration_error(probs, targets, n_bins=1)
        self.assertAlmostEqual(ece, 0.0)


class TestMcNemar(unittest.TestCase):
    def test_balanced_discordant_pairs(self):
        correct_a = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        correct_b = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
        result = mcnemar_test(correct_a, correct_b)
        self.assertEqual(result["n_a_correct_b_wrong"], 2)
        self.assertEqual(result["n_a_wrong_b_correct"], 2)
        self.assertAlmostEqual(result["p_value_exact"], 1.0)
        self.assertAlmostEqual(result["diff"], 0.0)

    def test_one_sided_discordance_is_significant(self):
        correct_a = np.array([1] * 10 + [0] * 10, dtype=bool)
        correct_b = np.array([0] * 10 + [0] * 10, dtype=bool)
        result = mcnemar_test(correct_a, correct_b)
        self.assertEqual(result["n_a_correct_b_wrong"], 10)
        self.assertEqual(result["n_a_wrong_b_correct"], 0)
        self.assertLess(result["p_value_exact"], 0.01)
        self.assertLess(result["p_value_chi2"], 0.01)

    def test_identical_models(self):
        correct = np.array([1, 0, 1, 1, 0], dtype=bool)
        result = mcnemar_test(correct, correct)
        self.assertEqual(result["discordant"], 0)
        self.assertAlmostEqual(result["p_value_exact"], 1.0)


class TestBootstrapAndIntervals(unittest.TestCase):
    def test_paired_bootstrap_identical_models(self):
        correct = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
        result = paired_bootstrap_ci(correct, correct, n_boot=200, seed=0)
        self.assertAlmostEqual(result["diff"], 0.0)
        self.assertAlmostEqual(result["ci_low"], 0.0)
        self.assertAlmostEqual(result["ci_high"], 0.0)

    def test_paired_bootstrap_points_in_right_direction(self):
        correct_a = np.array([1] * 90 + [0] * 10, dtype=bool)
        correct_b = np.array([0] * 90 + [0] * 10, dtype=bool)
        result = paired_bootstrap_ci(correct_a, correct_b, n_boot=300, seed=0)
        self.assertAlmostEqual(result["diff"], 0.9)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertGreaterEqual(result["ci_high"], result["ci_low"])

    def test_wilson_interval_known_center(self):
        correct = np.array([1] * 500 + [0] * 500, dtype=bool)
        result = wilson_interval(correct)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertLess(result["low"], 0.5)
        self.assertGreater(result["high"], 0.5)
        self.assertEqual(result["n"], 1000)

    def test_holm_adjust(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


class TestAggregationLesson(unittest.TestCase):
    def test_macro_f1_is_not_batch_average(self):
        """Averaging per-batch macro-F1 differs from computing it on the full set."""
        t1 = np.array([0, 0, 1])
        p1 = np.array([0, 0, 1])
        t2 = np.array([2, 2, 2])
        p2 = np.array([0, 1, 2])
        labels = [0, 1, 2]
        batch1 = precision_recall_fscore_support(
            t1, p1, labels=labels, average="macro", zero_division=0
        )[2]
        batch2 = precision_recall_fscore_support(
            t2, p2, labels=labels, average="macro", zero_division=0
        )[2]
        batch_average = (batch1 + batch2) / 2

        targets = np.concatenate([t1, t2])
        predictions = np.concatenate([p1, p2])
        full = compute_metrics(logits_from_predictions(predictions), targets)["f1_macro"]

        self.assertAlmostEqual(full, 59 / 90)
        self.assertGreater(abs(full - batch_average), 0.2)
        self.assertNotAlmostEqual(full, batch_average, places=3)


class TestValidation(unittest.TestCase):
    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.zeros((3, 2)), np.array([0, 1]))

    def test_rejects_out_of_range_targets(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.zeros((2, 2)), np.array([0, 2]))

    def test_rejects_wrong_class_name_count(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.zeros((2, 2)), np.array([0, 1]), class_names=["only"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
