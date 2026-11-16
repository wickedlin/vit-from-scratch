"""CIFAR-10 data pipeline (S3-3.1).

Data flow:
    CIFAR10(train=True, 50,000 images) --stratified split--> 45,000 train / 5,000 val
    CIFAR10(train=False, 10,000 images) --------------------> test (used only for final evaluation)
    Train: RandomCrop + RandomHorizontalFlip (+ RandAugment) -> ToTensor -> Normalize
    Val/test: ToTensor -> Normalize
    DataLoader output: images (B, 3, 32, 32) float32 + labels (B,) int64
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transforms(train: bool, aug: str = "basic") -> transforms.Compose:
    """Build the CIFAR-10 transform pipeline.

    Args:
        train: True returns the training view (may include random augmentation);
            False returns the deterministic validation/test view.
        aug: Training augmentation scheme; only used when train=True:
             "none"        ToTensor + Normalize only;
             "basic"       RandomCrop(32, padding=4) + HorizontalFlip;
             "randaugment" basic + RandAugment(num_ops=2, magnitude=9).

    Returns:
        transforms.Compose: input PIL.Image (32, 32, 3) uint8,
        output torch.Tensor (3, 32, 32) float32 with per-channel mean 0 and std 1.
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    if not train:
        return transforms.Compose([transforms.ToTensor(), normalize])

    if aug == "none":
        aug_ops: list = []
    elif aug == "basic":
        aug_ops = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    elif aug == "randaugment":
        aug_ops = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ]
    else:
        raise ValueError(
            f"unknown aug={aug!r}; expected one of 'none', 'basic', 'randaugment'"
        )

    return transforms.Compose([*aug_ops, transforms.ToTensor(), normalize])


def stratified_split_indices(
    targets: Sequence[int] | np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training indices by class, preserving the class ratio.

    Args:
        targets: Length-N class labels (CIFAR10.targets).
        val_ratio: Validation fraction, must satisfy 0 < val_ratio < 1.
        seed: Seed for the NumPy Generator; only controls the split and does
            not touch the global RNG.

    Returns:
        (train_indices, val_indices): both sorted np.int64 arrays, disjoint,
        whose union is range(N). About val_ratio of each class goes to validation.
    """
    targets = np.asarray(targets)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    rng = np.random.default_rng(seed)
    val_parts = []
    for cls in np.unique(targets):
        cls_idx = np.flatnonzero(targets == cls)
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_ratio))
        val_parts.append(cls_idx[:n_val])

    val_indices = np.sort(np.concatenate(val_parts))
    train_indices = np.setdiff1d(np.arange(len(targets)), val_indices)
    return train_indices, val_indices


def _seed_worker(worker_id: int) -> None:
    """Seed the Python/NumPy RNGs of each DataLoader worker.

    PyTorch sets each worker's torch seed automatically (base_seed + worker_id),
    but random and numpy are not inherited; if transforms or the data pipeline
    use them, missing this seeding makes two runs differ.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    """Create and seed a CPU Generator for DataLoader shuffle and worker seeds."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_datasets(
    data_root: str | Path = "data",
    val_ratio: float = 0.1,
    seed: int = 42,
    aug: str = "basic",
) -> tuple[Subset, Subset, Dataset]:
    """Download/load CIFAR-10 and build train/val/test dataset views.

    Train and validation views come from the same train=True split but use
    different transforms; therefore two CIFAR10(train=True) objects are created,
    each holding about 150 MB in memory.

    Returns:
        (train_subset, val_subset, test_set); the first two are Subsets,
        test_set is the original CIFAR10(train=False).
    """
    root = Path(data_root)
    train_base = CIFAR10(
        root, train=True, download=True, transform=build_transforms(True, aug)
    )
    val_base = CIFAR10(
        root, train=True, download=True, transform=build_transforms(False)
    )
    test_set = CIFAR10(
        root, train=False, download=True, transform=build_transforms(False)
    )

    train_indices, val_indices = stratified_split_indices(
        train_base.targets, val_ratio=val_ratio, seed=seed
    )
    return Subset(train_base, train_indices), Subset(val_base, val_indices), test_set


def build_dataloaders(
    data_root: str | Path = "data",
    val_ratio: float = 0.1,
    seed: int = 42,
    aug: str = "basic",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_seed: int | None = None,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders.

    Args:
        seed: Data-split seed (which samples go to train/val), also the default
            for shuffle_seed.
        shuffle_seed: Seed for training-loader shuffle and worker seeds. Defaults
            to seed; training scripts pass config.seed so the augmentation
            sequence follows the training seed while the split stays at seed=42,
            making runs with different training seeds comparable.
        batch_size: Samples per batch; the S3 baseline uses 32.
        num_workers: Number of worker processes. On Windows, 0 is recommended;
            macOS/Linux can use 2-4.
        pin_memory: Only meaningful on CUDA; pass device.type == "cuda".

    Returns:
        {"train": DataLoader(shuffle=True, generator=shuffle_seed),
         "val": DataLoader(shuffle=False, generator=seed+1),
         "test": DataLoader(shuffle=False, generator=seed+2)}

        All three loaders use worker_init_fn: when num_workers > 0, each worker's
        torch/numpy/random RNG is determined by "DataLoader generator base seed +
        worker_id", so the same seed gives the same batch order and augmentation
        sequence.
    """
    if shuffle_seed is None:
        shuffle_seed = seed
    train_set, val_set, test_set = build_datasets(
        data_root=data_root, val_ratio=val_ratio, seed=seed, aug=aug
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
    )
    return {
        "train": DataLoader(
            train_set,
            shuffle=True,
            generator=_make_generator(shuffle_seed),
            drop_last=False,
            **common,
        ),
        "val": DataLoader(
            val_set,
            shuffle=False,
            generator=_make_generator(seed + 1),
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            test_set,
            shuffle=False,
            generator=_make_generator(seed + 2),
            drop_last=False,
            **common,
        ),
    }
