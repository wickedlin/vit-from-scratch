"""Environment self-check for the ViT course.

Run:
    conda activate nlp
    python scripts/check_env.py

Checks:
1. Python / PyTorch / torchvision versions
2. Available device: cuda > mps > cpu
3. A 4096x4096 fp32 matmul benchmark
4. A small autograd step to verify backward works on the selected device
"""

from __future__ import annotations

import platform
import time

import torch
import torchvision


def get_device() -> torch.device:
    """Return the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued GPU work finishes, so timing is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    print(f"python      : {platform.python_version()}")
    print(f"platform    : {platform.platform()}")
    print(f"torch       : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")

    device = get_device()
    print(f"device      : {device}")

    if device.type == "cuda":
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
        print(f"cuda        : {torch.version.cuda}")
    elif device.type == "mps":
        print("gpu         : Apple Silicon (MPS)")
    else:
        print("gpu         : none (CPU only)")

    # 4096x4096 fp32 matmul benchmark
    n = 4096
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    synchronize(device)
    start = time.perf_counter()
    c = a @ b
    synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"matmul      : {n}x{n} fp32 in {elapsed * 1000:.1f} ms")
    print(f"checksum    : {c.mean().item():.4f}")

    # autograd smoke test
    x = torch.randn(8, 16, device=device, requires_grad=True)
    w = torch.randn(16, 16, device=device, requires_grad=True)
    loss = (x @ w).pow(2).mean()
    loss.backward()
    print(f"autograd    : loss={loss.item():.4f}, x.grad_norm={x.grad.norm().item():.4f}")
    print("status      : OK")


if __name__ == "__main__":
    main()
