"""Tests for src/vit_from_scratch/data.py (S3-3.1, CIFAR-10 pipeline).

Run from the project root:
    python tests/test_data.py -v

The learner writes data.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.data import (  # noqa: E402
    CIFAR10_MEAN,
    CIFAR10_STD,
    build_dataloaders,
    build_datasets,
    build_transforms,
    stratified_split_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("VIT_DATA_ROOT", str(PROJECT_ROOT / "data")))

_DATASETS = None


def get_datasets():
    """Build the default split once and reuse it across test classes."""
    global _DATASETS
    if _DATASETS is None:
        _DATASETS = build_datasets(
            data_root=DATA_ROOT, val_ratio=0.1, seed=42, aug="basic"
        )
    return _DATASETS


def random_pil(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))


class TestTransforms(unittest.TestCase):
    def test_output_shape_dtype(self):
        x = build_transforms(train=False)(random_pil(0))
        self.assertEqual(tuple(x.shape), (3, 32, 32))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x).all())

    def test_train_no_aug_equals_eval(self):
        """With aug='none', the train view should match the eval view exactly, in order."""
        img = random_pil(1)
        train_x = build_transforms(train=True, aug="none")(img)
        val_x = build_transforms(train=False)(img)
        torch.testing.assert_close(train_x, val_x, atol=0, rtol=0)

    def test_normalization_stats(self):
        """After normalization, training pixels should have per-channel mean~=0 and std~=1."""
        ds = CIFAR10(DATA_ROOT, train=True, download=True,
                     transform=build_transforms(train=False))
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:2048].tolist()
        imgs = torch.stack([ds[i][0] for i in idx])

        mean = imgs.mean(dim=(0, 2, 3))
        std = imgs.std(dim=(0, 2, 3), unbiased=False)
        torch.testing.assert_close(mean, torch.zeros(3), atol=0.03, rtol=0)
        torch.testing.assert_close(std, torch.ones(3), atol=0.04, rtol=0)

        # The normalization constants should match the data too (allow tiny 1/255 quantization error).
        # torchvision's CIFAR10.data is already in HWC layout: (N, 32, 32, 3).
        raw = torch.from_numpy(np.asarray(ds.data[idx], dtype=np.float32) / 255.0)
        raw_mean = raw.mean(dim=(0, 1, 2))
        raw_std = raw.std(dim=(0, 1, 2), unbiased=False)
        torch.testing.assert_close(raw_mean, torch.tensor(CIFAR10_MEAN), atol=0.02, rtol=0)
        torch.testing.assert_close(raw_std, torch.tensor(CIFAR10_STD), atol=0.02, rtol=0)

    def test_basic_aug_is_stochastic(self):
        """Two train-augmentation samples should differ (same image, different RNG state)."""
        tf = build_transforms(train=True, aug="basic")
        diffs = 0
        for i in range(16):
            img = random_pil(i)
            torch.manual_seed(0)
            a = tf(img)
            torch.manual_seed(1)
            b = tf(img)
            if not torch.allclose(a, b):
                diffs += 1
        self.assertGreater(diffs, 0)

    def test_randaugment_works(self):
        tf = build_transforms(train=True, aug="randaugment")
        x = tf(random_pil(2))
        self.assertEqual(tuple(x.shape), (3, 32, 32))
        self.assertTrue(torch.isfinite(x).all())

    def test_invalid_aug_raises(self):
        with self.assertRaises(ValueError):
            build_transforms(train=True, aug="strong")


class TestSplit(unittest.TestCase):
    def test_sizes(self):
        train_set, val_set, test_set = get_datasets()
        self.assertEqual(len(train_set), 45_000)
        self.assertEqual(len(val_set), 5_000)
        self.assertEqual(len(test_set), 10_000)

    def test_disjoint_and_covering(self):
        train_set, val_set, _ = get_datasets()
        train_idx = set(train_set.indices)
        val_idx = set(val_set.indices)
        self.assertTrue(train_idx.isdisjoint(val_idx))
        self.assertEqual(train_idx | val_idx, set(range(50_000)))

    def test_class_balance(self):
        """After the stratified split, train/val per-class counts should each be fixed."""
        train_set, val_set, _ = get_datasets()
        targets = np.asarray(train_set.dataset.targets)

        train_counts = np.bincount(targets[train_set.indices].astype(np.int64), minlength=10)
        val_counts = np.bincount(targets[val_set.indices].astype(np.int64), minlength=10)
        self.assertEqual(train_counts.tolist(), [4_500] * 10)
        self.assertEqual(val_counts.tolist(), [500] * 10)

    def test_reproducible(self):
        _, val_a = stratified_split_indices(np.arange(50_000) % 10, 0.1, seed=42)
        _, val_b = stratified_split_indices(np.arange(50_000) % 10, 0.1, seed=42)
        _, val_c = stratified_split_indices(np.arange(50_000) % 10, 0.1, seed=7)
        self.assertTrue(np.array_equal(val_a, val_b))
        self.assertFalse(np.array_equal(val_a, val_c))

    def test_invalid_ratio_raises(self):
        with self.assertRaises(ValueError):
            stratified_split_indices(np.arange(10) % 2, 1.5, seed=0)


class TestDataLoader(unittest.TestCase):
    def test_train_batch_shapes(self):
        train_set, _, _ = get_datasets()
        loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        self.assertEqual(tuple(images.shape), (32, 3, 32, 32))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(tuple(labels.shape), (32,))
        self.assertEqual(labels.dtype, torch.int64)
        self.assertTrue(int(labels.min()) >= 0 and int(labels.max()) < 10)

    def test_eval_transforms_are_deterministic(self):
        _, val_set, test_set = get_datasets()
        for ds, i in ((val_set, 0), (test_set, 0)):
            a, la = ds[i]
            b, lb = ds[i]
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(la, lb)

    def test_shuffle_is_reproducible(self):
        """With the same seed, DataLoader batch order and augmentation randomness should match."""
        train_set, _, _ = get_datasets()

        def make_loader():
            g = torch.Generator().manual_seed(123)
            return DataLoader(train_set, batch_size=8, shuffle=True,
                              num_workers=0, generator=g)

        torch.manual_seed(0)
        images_a, labels_a = next(iter(make_loader()))
        torch.manual_seed(0)
        images_b, labels_b = next(iter(make_loader()))
        torch.testing.assert_close(images_a, images_b, atol=0, rtol=0)
        torch.testing.assert_close(labels_a, labels_b, atol=0, rtol=0)

    def test_build_dataloaders(self):
        loaders = build_dataloaders(
            data_root=DATA_ROOT, batch_size=16, num_workers=0, aug="basic"
        )
        self.assertEqual(set(loaders), {"train", "val", "test"})
        images, labels = next(iter(loaders["val"]))
        self.assertEqual(tuple(images.shape), (16, 3, 32, 32))
        self.assertEqual(tuple(labels.shape), (16,))
        self.assertEqual(len(loaders["train"]), (45_000 + 15) // 16)
        self.assertEqual(len(loaders["val"]), (5_000 + 15) // 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
