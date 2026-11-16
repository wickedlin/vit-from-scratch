"""Trace tensor shapes through vit-pytorch's ViT.

Run:
    conda activate nlp
    python scripts/shape_trace.py
"""

from __future__ import annotations

import torch
from vit_pytorch import ViT

IMAGE_SIZE = 224
NUM_CLASSES = 1000
DIM = 768
DEPTH = 12
HEADS = 12
DIM_HEAD = 64
MLP_DIM = 3072


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(patch_size: int) -> ViT:
    return ViT(
        image_size=IMAGE_SIZE,
        patch_size=patch_size,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        pool="cls",
        channels=3,
        dim_head=DIM_HEAD,
        dropout=0.0,
        emb_dropout=0.0,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def register_shape_hook(module: torch.nn.Module, name: str, rows: list[tuple[str, object]]):
    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            rows.append((name, tuple(output.shape)))
        else:
            rows.append((name, type(output).__name__))

    return module.register_forward_hook(hook)


def trace(patch_size: int, detailed: bool, device: torch.device) -> None:
    model = build_model(patch_size).to(device)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    num_patches = (IMAGE_SIZE // patch_size) ** 2
    print(f"\n=== patch_size={patch_size} ===")
    print(f"num_patches              : {num_patches}")
    print(f"tokens (with cls)        : {num_patches + 1}")
    print(f"cls_token                : {tuple(model.cls_token.shape)}")
    print(f"pos_embedding            : {tuple(model.pos_embedding.shape)}")
    print(f"parameters               : {parameter_count(model):,}")

    rows: list[tuple[str, object]] = []
    handles = []

    if detailed:
        block = model.transformer.layers[0]
        handles += [
            register_shape_hook(model.to_patch_embedding, "patch_embedding", rows),
            register_shape_hook(block[0].norm, "attn.norm", rows),
            register_shape_hook(block[0].to_qkv, "attn.qkv_linear", rows),
            register_shape_hook(block[0].attend, "attn.softmax", rows),
            register_shape_hook(block[0].to_out, "attn.out", rows),
            register_shape_hook(block[1].net[0], "ff.norm", rows),
            register_shape_hook(block[1].net[1], "ff.linear1", rows),
            register_shape_hook(block[1].net[4], "ff.linear2", rows),
            register_shape_hook(model.transformer, "transformer", rows),
            register_shape_hook(model.mlp_head, "mlp_head", rows),
        ]
    else:
        block = model.transformer.layers[0]
        handles += [register_shape_hook(block[0].attend, "attn.softmax", rows)]

    with torch.no_grad():
        logits = model(x)

    for handle in handles:
        handle.remove()

    for name, shape in rows:
        print(f"{name:<24} : {shape}")
    print(f"logits                   : {tuple(logits.shape)}")

    del model, logits, x
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    torch.manual_seed(0)
    device = get_device()
    print(f"device: {device}")
    trace(patch_size=16, detailed=True, device=device)
    trace(patch_size=8, detailed=False, device=device)


if __name__ == "__main__":
    main()
