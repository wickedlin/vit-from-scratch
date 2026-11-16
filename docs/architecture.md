# Architecture and Tensor Shapes / 架构与张量形状

[English](#english) · [中文](#中文)

This document is the map between the code and the tensors. Read it with the source files open. The numbers use the CIFAR-10 baseline configuration: `image=32`, `patch=4`, `dim=192`, `depth=12`, `heads=3`, `mlp_dim=768`, `num_classes=10`.

本文是代码与张量之间的地图，建议对照源码阅读。数值采用 CIFAR-10 基线配置：`image=32`、`patch=4`、`dim=192`、`depth=12`、`heads=3`、`mlp_dim=768`、`num_classes=10`。

---

## English

### Module dependency graph

```text
utils.py
  └── get_device(), set_seed()

attention.py
  └── scaled_dot_product_attention(), MultiHeadSelfAttention

transformer.py
  ├── LayerNorm, MLP, DropPath
  └── TransformerBlock (uses MultiHeadSelfAttention)

vit.py
  ├── PatchEmbed
  ├── ClassToken, LearnedPositionEmbedding, SinusoidalPositionEmbedding
  ├── pool_tokens
  └── ViT (uses TransformerBlock)

data.py        -> build_datasets(), build_dataloaders()
optim.py       -> build_optimizer(), build_scheduler(), label_smoothing_loss(), clip_gradients()
amp.py         -> resolve_amp_dtype(), GradScaler, backward_microbatch(), optimizer_step()
train.py       -> TrainConfig, train_one_epoch(), evaluate(), fit(), checkpoints
metrics.py     -> compute_metrics(), mcnemar_test(), paired_bootstrap_ci(), wilson_interval()

swin.py
  ├── window_partition / window_reverse / shift_window / unshift_window
  ├── compute_shifted_window_mask
  ├── PatchMerging, RelativePositionBias, WindowAttention
  ├── SwinTransformerBlock, SwinStage, PatchEmbed2D
  └── SwinTransformer

mae.py
  ├── random_masking, unshuffle_tokens, mae_loss, patchify, unpatchify
  ├── MAEEncoder, MAEDecoder, MaskedAutoencoder
  └── load_mae_encoder_weights()
```

### ViT forward pass

| Step | Operation | Shape |
|---|---|---|
| 1 | Input image | `(B, 3, 32, 32)` |
| 2 | `PatchEmbed` Conv2d `k=4, s=4, D=192` | `(B, 192, 8, 8)` |
| 3 | `flatten(2).transpose(1,2)` | `(B, 64, 192)` |
| 4 | Prepend cls token | `(B, 65, 192)` |
| 5 | Add learned position embedding | `(B, 65, 192)` |
| 6 | Per block: `LN1` | `(B, 65, 192)` |
| 7 | `to_qkv` (no bias) | `(B, 65, 576)` |
| 8 | Split into Q/K/V, then heads | `(B, 3, 65, 64)` each |
| 9 | `QKᵀ / sqrt(64)` | `(B, 3, 65, 65)` |
| 10 | Softmax + `@ V` | `(B, 3, 65, 64)` |
| 11 | Merge heads + `to_out` | `(B, 65, 192)` |
| 12 | Residual add | `(B, 65, 192)` |
| 13 | `LN2` + MLP hidden | `(B, 65, 768)` |
| 14 | MLP output + residual | `(B, 65, 192)` |
| 15 | After 12 blocks: final `LN` | `(B, 65, 192)` |
| 16 | cls pooling | `(B, 192)` |
| 17 | Linear head | `(B, 10)` |

Two details that cause most bugs:

- The attention score tensor is `(B, h, N, N)`. The softmax axis is the **last** axis (keys), not the head axis.
- `reshape` after attention must first move the head axis back: `(B,h,N,d) → (B,N,h,d) → (B,N,h*d)`.

### Parameter accounting (ViT baseline)

| Component | Formula | Count |
|---|---|---:|
| Patch projection (Conv2d, bias) | `D·C·P² + D = 192·3·16 + 192` | 9,408 |
| cls token | `D` | 192 |
| Learned position table | `(N+1)·D = 65·192` | 12,480 |
| Per block, attention | `3D² + D² + D = 4D² + D` | 147,648 |
| Per block, MLP | `2·D·M + M + D = 2·192·768 + 768 + 192` | 295,680 |
| Per block, two LayerNorms | `4D` | 768 |
| Per block total | `4D² + 2DM + 6D + M` | 444,288 |
| 12 blocks | `12 × 444,288` | 5,331,456 |
| Final LayerNorm | `2D` | 384 |
| Classification head | `D·K + K = 192·10 + 10` | 1,930 |
| **Total** | | **5,355,850** |

`thop` reports `345,638,784` MACs while the analytical count is `365,107,584`. The difference is `12 × 2T²D = 12 × 2 × 65² × 192`, because `thop` only hooks `nn.Linear`/`nn.Conv2d` and does not count the two attention `torch.matmul` calls. This is a useful lesson: **never trust a profiler without knowing what it hooks.**

### Data pipeline

```text
CIFAR10 train=True (50,000)
        │
        ├── stratified_split_indices(seed=42, val_ratio=0.1)
        │       ├── train indices (45,000) -> train transform
        │       └── val indices   (5,000) -> deterministic transform
        │
CIFAR10 train=False (10,000) ----------> deterministic transform -> test

Train transform:
  RandomCrop(32, padding=4, reflect) -> RandomHorizontalFlip
  [optional RandAugment(num_ops=2, magnitude=9)]
  -> ToTensor -> Normalize(CIFAR10_MEAN, CIFAR10_STD)

DataLoader:
  train: shuffle=True, generator seeded by shuffle_seed
  val/test: shuffle=False
  worker_init_fn seeds numpy/random with base_seed + worker_id
```

Normalization constants are computed from the original training images **before augmentation**. The split is stratified so every class contributes the same fraction to validation.

### Training loop

```text
for each epoch:
    for each micro-batch:
        read current LR
        forward under autocast
        loss = criterion(logits, labels)
        scaled_loss.backward() with loss_weight = 1 / group_size
        accumulate metrics on device
        if end of accumulation group or last micro-batch:
            unscale -> clip -> scaler.step -> scaler.update
            scheduler.step -> zero_grad

    evaluate on validation split
    write metrics.jsonl
    save last.pt; save best.pt if val accuracy improves
```

The loop reads the LR **before** the optimizer step, because `scheduler.step()` advances the LR to the next update. Reading it after the step would report the wrong LR (and 0 for the last epoch).

### Swin forward pass (CIFAR configuration)

| Step | Operation | Shape |
|---|---|---|
| 1 | Input | `(B, 3, 32, 32)` |
| 2 | `PatchEmbed2D` p4, dim 64 | `(B, 8, 8, 64)` |
| 3 | Stage 1: 2 blocks, window 4, shift 2 | `(B, 8, 8, 64)` |
| 4 | PatchMerging | `(B, 4, 4, 128)` |
| 5 | Stage 2: 2 blocks, window 2, shift 1 | `(B, 4, 4, 128)` |
| 6 | PatchMerging | `(B, 2, 2, 256)` |
| 7 | Stage 3: 6 blocks, window 2, shift 0 | `(B, 2, 2, 256)` |
| 8 | Final LN + mean over tokens | `(B, 256)` |
| 9 | Head | `(B, 10)` |

Parameters: 5,407,366. Analytical MACs: about 33.1M per image, roughly 1/11 of the ViT baseline.

Why the last stage has `shift=0`: the stage-3 resolution is `2×2` and the window size is `2`, so there is exactly one window. Shifting a single window and masking it would produce no useful cross-window attention, so the implementation disables the shift.

### MAE forward pass

| Step | Operation | Shape |
|---|---|---|
| 1 | Input | `(B, 3, 32, 32)` |
| 2 | `patchify` target | `(B, 64, 48)` |
| 3 | PatchEmbed + position | `(B, 64, 192)` |
| 4 | `random_masking(mask_ratio=0.75)` | `x_keep (B,16,192)`, `mask (B,64)`, `ids_restore (B,64)` |
| 5 | Encoder: 12 blocks on visible tokens only | `(B, 16, 192)` |
| 6 | `enc_to_dec` | `(B, 16, 128)` |
| 7 | Append 48 mask tokens | `(B, 64, 128)` |
| 8 | `unshuffle_tokens` + decoder position | `(B, 64, 128)` |
| 9 | Decoder: 4 blocks | `(B, 64, 128)` |
| 10 | Linear prediction | `(B, 64, 48)` |
| 11 | `mae_loss` on masked patches only | scalar |

Parameters: 6,184,560 total = encoder 5,353,536 + decoder 831,024. The fine-tuned ViT has 5,355,850 parameters: the encoder is copied, a cls token and a classification head are added, and the decoder is discarded.

Two details that matter:

- The encoder operates on **16 visible tokens**, not 64. This is the main compute saving of MAE.
- Position embeddings are looked up per patch index. After masking, gather the rows with `ids_keep`; the decoder adds position embeddings **after** restoring the original order.

### Where to look when something breaks

| Symptom | First place to check |
|---|---|
| Shape error in attention | `attention.py`: softmax axis, `transpose(1,2)` / `reshape` |
| Pre-LN/Post-LN mismatch | `transformer.py`: residual placement and DropPath scope |
| Patch embedding reference mismatch | `vit.py`: flatten order `(C,P,P)` vs. `(P,P,C)` |
| Data leakage / odd accuracy | `data.py`: split, normalization statistics, transform application |
| Loss not decreasing | `train.py`: zero_grad, scheduler order, LR logging, accumulation weights |
| Metrics disagree with a library | `metrics.py`: aggregate first, then compute; float64 softmax |
| Swin mask wrong | `swin.py`: `compute_shifted_window_mask`, `window_reverse` then `unshift_window` |
| MAE loss suspicious | `mae.py`: `mask`, `ids_restore`, `norm_pix_loss`, loss only on masked patches |

---

## 中文

### 模块依赖图

```text
utils.py
  └── get_device(), set_seed()

attention.py
  └── scaled_dot_product_attention(), MultiHeadSelfAttention

transformer.py
  ├── LayerNorm, MLP, DropPath
  └── TransformerBlock（使用 MultiHeadSelfAttention）

vit.py
  ├── PatchEmbed
  ├── ClassToken, LearnedPositionEmbedding, SinusoidalPositionEmbedding
  ├── pool_tokens
  └── ViT（使用 TransformerBlock）

data.py        -> build_datasets(), build_dataloaders()
optim.py       -> build_optimizer(), build_scheduler(), label_smoothing_loss(), clip_gradients()
amp.py         -> resolve_amp_dtype(), GradScaler, backward_microbatch(), optimizer_step()
train.py       -> TrainConfig, train_one_epoch(), evaluate(), fit(), checkpoint
metrics.py     -> compute_metrics(), mcnemar_test(), paired_bootstrap_ci(), wilson_interval()

swin.py
  ├── window_partition / window_reverse / shift_window / unshift_window
  ├── compute_shifted_window_mask
  ├── PatchMerging, RelativePositionBias, WindowAttention
  ├── SwinTransformerBlock, SwinStage, PatchEmbed2D
  └── SwinTransformer

mae.py
  ├── random_masking, unshuffle_tokens, mae_loss, patchify, unpatchify
  ├── MAEEncoder, MAEDecoder, MaskedAutoencoder
  └── load_mae_encoder_weights()
```

### ViT 前向

| 步骤 | 操作 | Shape |
|---|---|---|
| 1 | 输入图像 | `(B, 3, 32, 32)` |
| 2 | `PatchEmbed` Conv2d `k=4, s=4, D=192` | `(B, 192, 8, 8)` |
| 3 | `flatten(2).transpose(1,2)` | `(B, 64, 192)` |
| 4 | 前置 cls token | `(B, 65, 192)` |
| 5 | 加可学习位置编码 | `(B, 65, 192)` |
| 6 | 每个 block：`LN1` | `(B, 65, 192)` |
| 7 | `to_qkv`（无 bias） | `(B, 65, 576)` |
| 8 | 拆 Q/K/V 并分头 | 各 `(B, 3, 65, 64)` |
| 9 | `QKᵀ / sqrt(64)` | `(B, 3, 65, 65)` |
| 10 | Softmax + `@ V` | `(B, 3, 65, 64)` |
| 11 | 合并头 + `to_out` | `(B, 65, 192)` |
| 12 | 残差相加 | `(B, 65, 192)` |
| 13 | `LN2` + MLP 隐层 | `(B, 65, 768)` |
| 14 | MLP 输出 + 残差 | `(B, 65, 192)` |
| 15 | 12 个 block 后：final `LN` | `(B, 65, 192)` |
| 16 | cls pooling | `(B, 192)` |
| 17 | 线性分类头 | `(B, 10)` |

最容易出错的两个细节：

- attention score 是 `(B, h, N, N)`；softmax 轴是**最后一维**（key），不是 head 维。
- attention 后的 `reshape` 必须先把 head 维移回去：`(B,h,N,d) → (B,N,h,d) → (B,N,h*d)`。

### 参数账（ViT 基线）

| 组件 | 公式 | 数量 |
|---|---|---:|
| Patch 投影（Conv2d，含 bias） | `D·C·P² + D = 192·3·16 + 192` | 9,408 |
| cls token | `D` | 192 |
| 可学习位置表 | `(N+1)·D = 65·192` | 12,480 |
| 每 block attention | `3D² + D² + D = 4D² + D` | 147,648 |
| 每 block MLP | `2·D·M + M + D = 2·192·768 + 768 + 192` | 295,680 |
| 每 block 两个 LayerNorm | `4D` | 768 |
| 每 block 合计 | `4D² + 2DM + 6D + M` | 444,288 |
| 12 blocks | `12 × 444,288` | 5,331,456 |
| Final LayerNorm | `2D` | 384 |
| 分类头 | `D·K + K = 192·10 + 10` | 1,930 |
| **合计** | | **5,355,850** |

`thop` 报告 `345,638,784` MACs，而解析计算是 `365,107,584`。差额是 `12 × 2T²D = 12 × 2 × 65² × 192`，因为 `thop` 只 hook `nn.Linear`/`nn.Conv2d`，没有统计 attention 里的两次 `torch.matmul`。这是一条重要经验：**不知道 profiler hook 了什么，就不要相信它的数字。**

### 数据管线

```text
CIFAR10 train=True (50,000)
        │
        ├── stratified_split_indices(seed=42, val_ratio=0.1)
        │       ├── train indices (45,000) -> 训练 transform
        │       └── val indices   (5,000) -> 确定性 transform
        │
CIFAR10 train=False (10,000) ----------> 确定性 transform -> test

训练 transform：
  RandomCrop(32, padding=4, reflect) -> RandomHorizontalFlip
  [可选 RandAugment(num_ops=2, magnitude=9)]
  -> ToTensor -> Normalize(CIFAR10_MEAN, CIFAR10_STD)

DataLoader：
  train: shuffle=True, generator 由 shuffle_seed 播种
  val/test: shuffle=False
  worker_init_fn 用 base_seed + worker_id 给 numpy/random 播种
```

归一化常数来自**增强前**的原始训练图像。划分按类别分层，保证每个类别进入验证集的比例一致。

### 训练循环

```text
for each epoch:
    for each micro-batch:
        读取当前 LR
        autocast 下前向
        loss = criterion(logits, labels)
        scaled_loss.backward()，loss_weight = 1 / group_size
        在 device 上累加指标
        若到达累积组末尾或最后一个 micro-batch：
            unscale -> clip -> scaler.step -> scaler.update
            scheduler.step -> zero_grad

    在验证集评估
    写 metrics.jsonl
    保存 last.pt；val acc 提升时保存 best.pt
```

循环在 optimizer step **之前**读取 LR，因为 `scheduler.step()` 会把 LR 推到下一次更新。step 之后再读会记错 LR（最后一个 epoch 会读到 0）。

### Swin 前向（CIFAR 配置）

| 步骤 | 操作 | Shape |
|---|---|---|
| 1 | 输入 | `(B, 3, 32, 32)` |
| 2 | `PatchEmbed2D` p4，dim 64 | `(B, 8, 8, 64)` |
| 3 | Stage 1：2 blocks，window 4，shift 2 | `(B, 8, 8, 64)` |
| 4 | PatchMerging | `(B, 4, 4, 128)` |
| 5 | Stage 2：2 blocks，window 2，shift 1 | `(B, 4, 4, 128)` |
| 6 | PatchMerging | `(B, 2, 2, 256)` |
| 7 | Stage 3：6 blocks，window 2，shift 0 | `(B, 2, 2, 256)` |
| 8 | Final LN + 对 token 求平均 | `(B, 256)` |
| 9 | 分类头 | `(B, 10)` |

参数量：5,407,366。解析 MACs：约 33.1M/图，约为 ViT 基线的 1/11。

最后一个 stage 为什么 `shift=0`：stage 3 的分辨率是 `2×2`，窗口大小是 `2`，只有一个窗口。对单窗口做 shift 再 mask 不会产生有用的跨窗口注意力，因此实现关闭 shift。

### MAE 前向

| 步骤 | 操作 | Shape |
|---|---|---|
| 1 | 输入 | `(B, 3, 32, 32)` |
| 2 | `patchify` 目标 | `(B, 64, 48)` |
| 3 | PatchEmbed + 位置编码 | `(B, 64, 192)` |
| 4 | `random_masking(mask_ratio=0.75)` | `x_keep (B,16,192)`、`mask (B,64)`、`ids_restore (B,64)` |
| 5 | Encoder：12 blocks，只处理可见 token | `(B, 16, 192)` |
| 6 | `enc_to_dec` | `(B, 16, 128)` |
| 7 | 拼接 48 个 mask token | `(B, 64, 128)` |
| 8 | `unshuffle_tokens` + decoder 位置编码 | `(B, 64, 128)` |
| 9 | Decoder：4 blocks | `(B, 64, 128)` |
| 10 | 线性预测 | `(B, 64, 48)` |
| 11 | `mae_loss` 只算 masked patch | 标量 |

参数量：总计 6,184,560 = encoder 5,353,536 + decoder 831,024。微调后的 ViT 有 5,355,850 参数：encoder 被复制，加上 cls token 与分类头，decoder 丢弃。

两个关键细节：

- Encoder 只处理 **16 个可见 token**，不是 64 个。这是 MAE 的主要计算节省。
- 位置编码按 patch 下标查表。mask 后要按 `ids_keep` 取对应行；decoder 在还原原始顺序**之后**再加位置编码。

### 出问题时先看哪里

| 症状 | 首先检查 |
|---|---|
| attention shape 报错 | `attention.py`：softmax 轴、`transpose(1,2)` / `reshape` |
| Pre-LN/Post-LN 不一致 | `transformer.py`：残差位置与 DropPath 作用范围 |
| patch embedding 与参考不一致 | `vit.py`：flatten 顺序 `(C,P,P)` vs. `(P,P,C)` |
| 数据泄漏 / 精度异常 | `data.py`：split、归一化统计量、transform 应用位置 |
| loss 不下降 | `train.py`：zero_grad、scheduler 顺序、LR 记录、累积权重 |
| 指标与库不一致 | `metrics.py`：先聚合再算；float64 softmax |
| Swin mask 错误 | `swin.py`：`compute_shifted_window_mask`、先 `window_reverse` 再 `unshift_window` |
| MAE loss 可疑 | `mae.py`：`mask`、`ids_restore`、`norm_pix_loss`、只算 masked |
