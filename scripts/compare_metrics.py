"""Compare two metrics.jsonl files (S3-3.4 reproducibility check).

Example:
    python scripts/compare_metrics.py experiments/repro_a/metrics.jsonl \
        experiments/repro_b/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr")


def load(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two metrics.jsonl files")
    parser.add_argument("a", type=str)
    parser.add_argument("b", type=str)
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    if len(a) != len(b):
        raise SystemExit(f"epoch count mismatch: {len(a)} vs {len(b)}")

    for key in KEYS:
        diffs = [abs(x[key] - y[key]) for x, y in zip(a, b)]
        print(f"{key:>10}: max abs diff = {max(diffs):.3e}")

    identical = all(
        x[key] == y[key] for x, y in zip(a, b) for key in KEYS
    )
    print("identical:", identical)
    if not identical:
        first = next(
            i
            for i, (x, y) in enumerate(zip(a, b))
            if any(x[key] != y[key] for key in KEYS)
        )
        print(f"first differing epoch: {first}")


if __name__ == "__main__":
    main()
