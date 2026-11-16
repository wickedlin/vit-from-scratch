"""Profile ViT parameters, thop MACs, and training peak memory.

Examples:
    python scripts/profile_vit.py --patch-size 4 --dim 192 --depth 6 --heads 3 --mlp-dim 768 --batch-size 64 --memory
    python scripts/profile_vit.py --patch-size 8 --dim 384 --depth 12 --heads 6 --mlp-dim 1536 --batch-size 128 --memory

Notes:
- thop reports MACs (multiply-accumulate operations), not FLOPs. 1 MAC = 2 FLOPs.
- thop hooks nn.Linear / nn.Conv2d; it does not count torch.matmul inside attention.
- thop's parameter count misses custom modules and bare nn.Parameter tensors,
  so use sum(p.numel()) as the source of truth for parameters.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vit_from_scratch.vit import ViT  # noqa: E402


def build_model(args: argparse.Namespace, device: str = "cpu") -> ViT:
    return ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=0.0,
        emb_dropout=0.0,
        drop_path=0.0,
        pool=args.pool,
        pos_embed="learned",
        pre_norm=True,
    ).to(device)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_macs(args: argparse.Namespace) -> tuple[int, int]:
    from thop import profile

    model = build_model(args).eval()
    x = torch.randn(1, 3, args.image_size, args.image_size)
    macs, thop_params = profile(model, inputs=(x,), verbose=False)
    return int(macs), int(thop_params)


def probe_memory(args: argparse.Namespace) -> float | None:
    if not torch.cuda.is_available():
        return None

    torch.cuda.empty_cache()
    gc.collect()

    model = build_model(args, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device="cuda")
    y = torch.randint(0, args.num_classes, (args.batch_size,), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    del model, optimizer, x, y, logits, loss
    torch.cuda.empty_cache()
    gc.collect()
    return peak_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ViT params / MACs / memory")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--pool", choices=["cls", "mean"], default="cls")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--memory",
        action="store_true",
        help="run one train step and report peak CUDA memory",
    )
    args = parser.parse_args()

    model = build_model(args)
    n_actual = count_params(model)
    n_patches = model.num_patches
    n_tokens = model.num_tokens
    patch_dim = 3 * args.patch_size ** 2

    macs, n_thop = profile_macs(args)

    print(
        f"config : image={args.image_size} patch={args.patch_size} dim={args.dim} "
        f"depth={args.depth} heads={args.heads} mlp={args.mlp_dim} pool={args.pool}"
    )
    print(
        f"shapes : N={n_patches}, T={n_tokens}, C*P^2={patch_dim}, D={args.dim}"
    )
    print(
        f"params : actual={n_actual:,}  thop={n_thop:,}  diff={n_actual - n_thop:,}"
    )
    print(f"MACs   : thop={macs:,} (batch=1 forward)")
    print("         thop counts Linear/Conv only; attention torch.matmul is not counted.")

    if args.memory:
        peak = probe_memory(args)
        if peak is None:
            print("memory : CUDA not available, skipped")
        else:
            print(
                f"memory : peak={peak:,.0f} MB "
                f"(batch={args.batch_size}, fp32, AdamW step)"
            )


if __name__ == "__main__":
    main()
