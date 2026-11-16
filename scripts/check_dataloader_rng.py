"""Verify how num_workers affects batch ordering and the augmentation RNG stream.

Usage (from the project root):
    python scripts/check_dataloader_rng.py

Expected:
- The order is identical across all three runs (the sampler lives in the main
  process and is seeded by the generator).
- The aug values differ across runs (each worker's torch RNG is seeded from
  base_seed + worker_id; with num_workers=0 the main-process global RNG is used).

Note: this script only observes the mechanism; it is not part of training.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    """Return (index, one random augmentation value) for each sample."""

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int):
        # Simulate RandomCrop/Flip: consume torch's default RNG
        rand_aug = torch.randint(0, 1000, (1,)).item()
        return index, rand_aug


def run(num_workers: int, seed: int = 0):
    # Main-process global RNG: this is what augmentation uses when num_workers=0
    torch.manual_seed(seed)
    # DataLoader-specific generator: determines shuffle order and feeds workers' base_seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        ToyDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    order, aug = [], []
    for indices, rand_values in loader:
        order.extend(indices.tolist())
        aug.extend(rand_values.tolist())
    return order, aug


if __name__ == "__main__":
    base_order = None
    for nw in (0, 2, 4):
        order, aug = run(nw)
        if base_order is None:
            base_order = order
        print(f"num_workers={nw}")
        print(f"  order: {order}")
        print(f"  aug  : {aug}")
        print(f"  order same as num_workers=0: {order == base_order}")
