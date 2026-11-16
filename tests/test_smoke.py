"""Minimal smoke test for the project skeleton.

Run from the project root:
    python tests/test_smoke.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vit_from_scratch  # noqa: E402
from vit_from_scratch.utils import get_device  # noqa: E402


class TestSmoke(unittest.TestCase):
    def test_package_import(self):
        self.assertTrue(vit_from_scratch.__version__)

    def test_torch_import(self):
        self.assertTrue(torch.__version__)

    def test_device_priority(self):
        self.assertIn(get_device().type, ("cuda", "mps", "cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
