"""Tests for src/vit_from_scratch/vit.py (PatchEmbed).

Run from the project root:
    python tests/test_vit.py

The learner writes vit.py by hand; this file is generated scaffolding.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_from_scratch.vit import (  # noqa: E402
    ClassToken,
    LearnedPositionEmbedding,
    PatchEmbed,
    SinusoidalPositionEmbedding,
    ViT,
    build_sincos_pos_embed,
    pool_tokens,
)


def vit_param_formula(
    dim: int,
    mlp_dim: int,
    depth: int,
    num_classes: int,
    in_channels: int,
    patch_size: int,
    num_patches: int,
    pool: str,
    pos_embed: str,
) -> int:
    """Hand-derived parameter count for our ViT (qkv_bias=False)."""
    num_tokens = num_patches + (1 if pool == "cls" else 0)
    patch_embed = dim * in_channels * patch_size ** 2 + dim
    cls_token = dim if pool == "cls" else 0
    pos = num_tokens * dim if pos_embed == "learned" else 0
    block = 4 * dim ** 2 + 2 * dim * mlp_dim + 6 * dim + mlp_dim
    final_norm = 2 * dim
    head = dim * num_classes + num_classes
    return patch_embed + cls_token + pos + depth * block + final_norm + head


class TestPatchEmbed(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shape(self):
        model = PatchEmbed(image_size=32, patch_size=4, in_channels=3, dim=192)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)

        self.assertEqual(out.shape, (2, 64, 192))
        self.assertEqual(model.num_patches, 64)
        self.assertEqual(model.grid_size, (8, 8))

    def test_sequence_length_formula(self):
        # N = (H / P) * (W / P) = H * W / P^2 for a square input.
        for image_size, patch_size in [(32, 4), (32, 8), (224, 16), (224, 14)]:
            model = PatchEmbed(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=3,
                dim=8,
            )
            expected = (image_size // patch_size) ** 2
            x = torch.randn(1, 3, image_size, image_size)

            self.assertEqual(model.num_patches, expected)
            self.assertEqual(model(x).shape, (1, expected, 8))

    def test_conv_matches_manual_reshape(self):
        # Same weights in both parameterisations must give the same tokens.
        conv_model = PatchEmbed(32, 4, 3, 16, use_conv=True)
        manual_model = PatchEmbed(32, 4, 3, 16, use_conv=False)
        with torch.no_grad():
            manual_model.proj.weight.copy_(conv_model.proj.weight.reshape(16, -1))
            manual_model.proj.bias.copy_(conv_model.proj.bias)

        x = torch.randn(2, 3, 32, 32)

        self.assertTrue(
            torch.allclose(conv_model(x), manual_model(x), atol=1e-6)
        )

    def test_single_patch_matches_linear_projection(self):
        # One 4x4 patch: the token must equal W_flat @ patch + b.
        dim = 8
        model = PatchEmbed(image_size=4, patch_size=4, in_channels=3, dim=dim)
        x = torch.randn(1, 3, 4, 4)

        out = model(x)
        weight = model.proj.weight.reshape(dim, -1)
        expected = weight @ x.reshape(-1) + model.proj.bias

        self.assertTrue(torch.allclose(out[0, 0], expected, atol=1e-6))

    def test_parameter_count(self):
        # Parameters: D * C * P^2 + D, independent of image size.
        model = PatchEmbed(32, 4, 3, 192)
        expected = 192 * 3 * 4 ** 2 + 192

        self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_non_divisible_patch_size_raises(self):
        with self.assertRaises(ValueError):
            PatchEmbed(image_size=30, patch_size=4, in_channels=3, dim=8)

    def test_wrong_input_size_raises(self):
        model = PatchEmbed(image_size=32, patch_size=4, in_channels=3, dim=8)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 16, 16))

    def test_gradients_flow(self):
        model = PatchEmbed(32, 4, 3, 16)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)

        model(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)


class TestClassToken(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shape_and_patch_preserved(self):
        model = ClassToken(16)
        x = torch.randn(2, 5, 16)
        out = model(x)

        self.assertEqual(out.shape, (2, 6, 16))
        self.assertTrue(torch.allclose(out[:, 1:], x))

    def test_cls_identical_across_batch(self):
        model = ClassToken(16)
        out = model(torch.randn(4, 5, 16))

        # One learnable token broadcast to every sample.
        for b in range(1, 4):
            self.assertTrue(torch.allclose(out[b, 0], out[0, 0]))

    def test_parameter_count(self):
        model = ClassToken(16)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 16)

    def test_gradients_flow(self):
        model = ClassToken(16)
        x = torch.randn(2, 5, 16, requires_grad=True)

        model(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.token.grad)
        self.assertTrue(torch.isfinite(model.token.grad).all())


class TestLearnedPositionEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shape_and_addition(self):
        model = LearnedPositionEmbedding(num_tokens=65, dim=16)
        x = torch.randn(2, 65, 16)
        out = model(x)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.allclose(out, x + model.pos_embed))

    def test_parameter_count(self):
        model = LearnedPositionEmbedding(num_tokens=65, dim=16)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 65 * 16)

    def test_wrong_token_count_raises(self):
        model = LearnedPositionEmbedding(num_tokens=65, dim=16)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 64, 16))

    def test_gradient_reaches_position_table(self):
        model = LearnedPositionEmbedding(num_tokens=65, dim=16)
        model(torch.randn(2, 65, 16)).sum().backward()

        self.assertIsNotNone(model.pos_embed.grad)
        self.assertTrue(torch.isfinite(model.pos_embed.grad).all())


class TestSinusoidalPositionEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shape_and_bounds(self):
        pe = build_sincos_pos_embed(65, 16)

        self.assertEqual(pe.shape, (65, 16))
        self.assertGreaterEqual(pe.min().item(), -1.0)
        self.assertLessEqual(pe.max().item(), 1.0)

    def test_first_token_pattern(self):
        pe = build_sincos_pos_embed(65, 16)

        # pos=0 -> sin(0)=0 for even dims, cos(0)=1 for odd dims.
        self.assertTrue(torch.allclose(pe[0, 0::2], torch.zeros(8)))
        self.assertTrue(torch.allclose(pe[0, 1::2], torch.ones(8)))

    def test_matches_naive_reference(self):
        num_tokens, dim = 65, 16
        pe = build_sincos_pos_embed(num_tokens, dim)

        ref = torch.zeros(num_tokens, dim, dtype=torch.float64)
        for pos in range(num_tokens):
            for i in range(dim // 2):
                ref[pos, 2 * i] = math.sin(pos / 10000 ** (2 * i / dim))
                ref[pos, 2 * i + 1] = math.cos(pos / 10000 ** (2 * i / dim))

        self.assertTrue(torch.allclose(pe.double(), ref, atol=1e-5))

    def test_odd_dim_raises(self):
        with self.assertRaises(ValueError):
            build_sincos_pos_embed(65, 15)

    def test_module_has_no_parameters(self):
        model = SinusoidalPositionEmbedding(num_tokens=65, dim=16)

        self.assertEqual(len(list(model.parameters())), 0)
        # Non-persistent buffer: deterministic values are not saved in state_dict.
        self.assertEqual(len(model.state_dict()), 0)

    def test_module_forward_and_wrong_token_count(self):
        model = SinusoidalPositionEmbedding(num_tokens=65, dim=16)
        x = torch.randn(2, 65, 16)

        self.assertTrue(torch.allclose(model(x), x + model.pos_embed))

        with self.assertRaises(ValueError):
            model(torch.randn(2, 64, 16))


class TestPoolTokens(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_cls_pool(self):
        x = torch.randn(2, 65, 16)
        self.assertTrue(torch.allclose(pool_tokens(x, "cls"), x[:, 0]))

    def test_mean_pool(self):
        x = torch.randn(2, 65, 16)
        self.assertTrue(torch.allclose(pool_tokens(x, "mean"), x.mean(dim=1)))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            pool_tokens(torch.randn(2, 65, 16), "max")


class TestViT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shape_and_defaults(self):
        model = ViT()
        out = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(out.shape, (2, 10))
        self.assertEqual(model.num_patches, 64)
        self.assertEqual(model.num_tokens, 65)
        self.assertTrue(model.use_cls)

    def test_parameter_count(self):
        model = ViT()
        expected = vit_param_formula(192, 768, 6, 10, 3, 4, 64, "cls", "learned")

        self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
        self.assertEqual(expected, 2_690_122)

    def test_parameter_count_variants(self):
        for pool in ("cls", "mean"):
            for pos_embed in ("learned", "sincos", "none"):
                model = ViT(depth=2, pool=pool, pos_embed=pos_embed)
                expected = vit_param_formula(
                    192, 768, 2, 10, 3, 4, 64, pool, pos_embed
                )
                self.assertEqual(
                    sum(p.numel() for p in model.parameters()),
                    expected,
                    f"pool={pool}, pos_embed={pos_embed}",
                )

    def test_manual_forward_matches(self):
        model = ViT(depth=2).eval()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            ref = model.patch_embed(x)
            ref = model.cls_token(ref)
            ref = model.pos_embedding(ref)
            ref = model.blocks(ref)
            ref = model.norm(ref)
            ref = pool_tokens(ref, model.pool)
            ref = model.head(ref)
            out = model(x)

        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_pool_mean_uses_no_cls(self):
        model = ViT(depth=1, pool="mean")

        self.assertFalse(model.use_cls)
        self.assertEqual(model.num_tokens, 64)
        self.assertIsInstance(model.cls_token, nn.Identity)

    def test_pos_embed_variants_forward(self):
        for pos_embed in ("learned", "sincos", "none"):
            model = ViT(depth=1, pos_embed=pos_embed)
            self.assertEqual(
                model(torch.randn(1, 3, 32, 32)).shape, (1, 10), pos_embed
            )

    def test_gradients_flow(self):
        model = ViT(depth=2)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)

        model(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_pre_norm_switch(self):
        model = ViT(depth=1, pre_norm=False)
        self.assertEqual(model(torch.randn(1, 3, 32, 32)).shape, (1, 10))

    def test_invalid_pool_raises(self):
        with self.assertRaises(ValueError):
            ViT(pool="max")

    def test_invalid_pos_embed_raises(self):
        with self.assertRaises(ValueError):
            ViT(pos_embed="rope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
