# Curriculum / 学习路径

[English](#english) · [中文](#中文)

This document is the course map. It states what you should be able to **do** after each stage, which files to read, and how to verify yourself. The Q&A files go deeper into mechanisms and misconceptions.

本文是课程地图：说明每个阶段结束后你应当**能做什么**、该读哪些文件、如何自检。更深入的机制与常见误解见对应的 Q&A 文件。

---

## English

### Suggested pace

The course was designed for 9–11 weeks at 12–15 hours per week. The stages are sequential: each one assumes the previous implementation and tests pass. If you already know Transformers, compress S1–S2; do not skip S3, because most real failures happen in data, optimization, and evaluation rather than in the model definition.

| Stage | Suggested time | Theme | Main deliverable |
|---|---|---|---|
| S0 | 0.5 week | Environment, devices, shapes | `check_env.py`, `shape_trace.py`, device abstraction |
| S1 | 1.5–2 weeks | Attention and Transformer block | `attention.py`, `transformer.py` + tests |
| S2 | 1 week | Assemble ViT | `vit.py` + numerical alignment with a reference |
| S3 | 1.5–2 weeks | Train on CIFAR-10 | `data.py`, `optim.py`, `amp.py`, `train.py`, `metrics.py` + a reproducible baseline |
| S4 | 1.5 weeks | Swin Transformer | `swin.py` + a trained model and comparison |
| S5 | 2 weeks | MAE | `mae.py` + pretraining and fine-tuning |
| S6 | 1 week | Research judgment | Written analyses, ablation interpretation, diagnosis |

### S0 — Environment, device abstraction, and shape literacy

**Goal:** make the project runnable and reproducible on your machine, and learn to trace tensor shapes before writing model code.

**Topics**
- Python environment, PyTorch/torchvision versions, CUDA/MPS/CPU detection.
- Seed control and why `torch.manual_seed` alone is not enough for a data pipeline.
- Shape notation: `B` batch, `C` channels, `H/W` spatial size, `P` patch size, `N` token count, `D` token width.
- Parameter count vs. computation (MACs/FLOPs) vs. activation memory.

**Hands-on**
- Run `python scripts/check_env.py`.
- Run `python scripts/shape_trace.py` and read every shape it prints.
- Explain why a 224×224 image with patch size 16 gives 196 patches plus one cls token.

**Acceptance**
- You can state the shape of every intermediate tensor in a ViT forward pass.
- You can explain why a 4096×4096 matmul benchmark needs warm-up and synchronization.
- You can distinguish parameters, MACs, and peak memory.

**Files:** `src/vit_from_scratch/utils.py`, `scripts/check_env.py`, `scripts/shape_trace.py`, plus shape notes you write yourself.

### S1 — Attention and Transformer block

**Goal:** derive and implement scaled dot-product attention, multi-head attention, LayerNorm, and a Pre-LN/Post-LN Transformer block.

**Topics**
- `scores = QKᵀ / sqrt(d_k)`, softmax over the key axis, `weights @ V`.
- The variance argument for `1/sqrt(d_k)`: if `q·k` has variance `d_k`, scaling makes logits variance 1 and prevents softmax saturation.
- Masks: `True = keep`; fill disallowed positions with `-inf` **before** softmax.
- Multi-head split/merge: `(B,N,D) → (B,N,h,d_head) → (B,h,N,d_head)`.
- LayerNorm over the last dimension, biased variance, `eps` inside the square root.
- Pre-LN `x + F(LN(x))` vs. Post-LN `LN(x + F(x))`; why Pre-LN trains more stably.
- MLP/GELU, residual connections, DropPath, and why DropPath wraps only the branch.

**Hands-on**
- Implement a naive SDPA and a vectorized SDPA; compare with `F.scaled_dot_product_attention` numerically and by gradients.
- Implement MHSA and verify that `heads=1` matches a single-head implementation.
- Run the Pre-LN/Post-LN gradient-norm experiment in `scripts/compare_norm_placement.py`.

**Acceptance**
- Attention tests pass, including mask and causal cases.
- You can write the MHSA shape chain from memory.
- You can explain why `mask` must be applied before softmax, not by multiplying the probabilities by zero.
- You can explain why `x + DropPath(f(x))` is correct and `DropPath(x + f(x))` is not.

**Files:** `src/vit_from_scratch/attention.py`, `src/vit_from_scratch/transformer.py`, `tests/test_attention.py`, `tests/test_transformer.py`, `tests/test_block_variants.py`.

### S2 — Assemble ViT

**Goal:** turn patches into tokens, add cls/position embeddings, stack blocks, and match a reference implementation numerically.

**Topics**
- Patch embedding as `Conv2d(kernel_size=P, stride=P)`; equivalence with reshape + Linear.
- Sequence length `N = (H/P)(W/P)`; token width `D` vs. patch vector length `C·P²`.
- cls token, learned absolute position embeddings, fixed sinusoidal embeddings, cls vs. mean pooling.
- Parameter accounting: patch projection, cls token, position table, each block, final LN, head.
- Weight initialization with truncated normal.
- Reference alignment: patch flatten order `(C,P,P)` vs. `(P,P,C)`; `permute` vs. `reshape`.

**Hands-on**
- Implement `PatchEmbed` twice (`use_conv=True/False`) and verify `allclose`.
- Implement `ViT.forward` from a blank file.
- Copy weights into `vit-pytorch` and verify logits match.

**Acceptance**
- A `(2,3,32,32)` input produces `(2,10)` logits.
- The parameter count matches the hand-derived formula within 1%.
- You can explain the three places where patch order matters when copying reference weights.

**Files:** `src/vit_from_scratch/vit.py`, `tests/test_vit.py`, `tests/test_vit_reference.py`, `scripts/profile_vit.py`.

### S3 — Train on CIFAR-10

**Goal:** build a correct data pipeline and training loop, reproduce a baseline, diagnose curves, and evaluate beyond accuracy.

**Topics**
- Stratified train/val split, normalization statistics from the training set only, augmentation.
- DataLoader randomness: sampler order vs. worker RNG; `base_seed + worker_id`.
- AdamW vs. Adam + L2; decoupled weight decay; no decay for 1-D parameters.
- Warmup + cosine schedule; step counting with gradient accumulation.
- Label smoothing, gradient clipping order, AMP (`bf16`/`fp16`), GradScaler.
- Reproducibility: seeds, deterministic cuDNN, identical `metrics.jsonl`.
- Debugging: 1% overfitting test, learning-rate scan, curve diagnosis.
- Evaluation: accuracy, macro/micro/weighted F1, top-k, ROC-AUC/PR-AUC, NLL/Brier/ECE, confusion matrix, McNemar, bootstrap CI.

**Hands-on**
- Run `scripts/overfit_small.py` to near-100% train accuracy on 1% of the data.
- Run `scripts/lr_scan.py` and explain the underfit / stable / unstable regions.
- Train the 100-epoch baseline to ≥80% test accuracy.
- Run `scripts/eval_metrics.py` on several checkpoints.

**Acceptance**
- Two runs with the same seed produce identical metrics.
- The baseline reaches ≥80% test accuracy and is reproducible.
- You can explain why averaging per-batch F1 is not the dataset F1.
- You can explain why McNemar/bootstrap on one test set do not measure seed variance.

**Files:** `data.py`, `optim.py`, `amp.py`, `train.py`, `metrics.py`, `scripts/overfit_small.py`, `scripts/lr_scan.py`, `scripts/eval_checkpoint.py`, `scripts/eval_metrics.py`.

### S4 — Swin Transformer

**Goal:** implement window attention, shifted windows, PatchMerging, and relative position bias; train and compare with ViT.

**Topics**
- Motivation: global attention is `O(N²)`; images have local structure; hierarchical features help dense prediction.
- `window_partition` / `window_reverse` and their inverse relationship.
- Cyclic shift, re-partition, attention mask, reverse, unshift.
- Why the mask only blocks wrapped regions, not every non-adjacent token.
- PatchMerging: `2×2` neighborhood, `4C → 2C`, spatial size halved.
- Relative position bias: `(2M−1)²` table per attention layer, shared across windows, not across blocks.
- Complexity accounting: token count `64 → 16 → 4`, windows, single-window stage.

**Hands-on**
- Verify partition/reverse with random tensors.
- Build the shifted-window mask two ways: the sliced `img_mask` method and a brute-force source-coordinate method.
- Train Swin on CIFAR-10 and compare with ViT using the S3 protocol.

**Acceptance**
- Partition/reverse and shift/unshift are exact inverses.
- The mask test matches a brute-force implementation.
- You can explain why Swin matches ViT accuracy at much lower MACs on this setup.
- You can explain why the Swin vs. ViT difference is not statistically significant here.

**Files:** `src/vit_from_scratch/swin.py`, `tests/test_swin.py`, `tests/test_swin_stage.py`, `tests/test_swin_block.py`, `tests/test_swin_model.py`, `scripts/train_swin.py`, `scripts/bench_swin.py`.

### S5 — MAE

**Goal:** implement masked autoencoding, pretrain on unlabeled CIFAR-10 images, fine-tune the encoder, and compare with training from scratch.

**Topics**
- Self-supervision: reconstruct masked patches; labels are not used during pretraining.
- High mask ratio (75%): images are redundant, so a low ratio makes reconstruction too easy.
- Asymmetric encoder/decoder: encoder sees only visible tokens; decoder sees visible encodings + mask tokens.
- Loss only on masked patches; `norm_pix_loss` and its per-patch normalization.
- `gather` / `scatter` / `argsort` shuffle and restore; position embeddings must be gathered with `ids_keep`.
- Encoder-to-ViT weight transfer: patch embed, patch position rows, blocks, final LN; cls token and head stay new.

**Hands-on**
- Verify masking/gather/restore is invertible.
- Verify visible positions receive zero gradient.
- Pretrain 100 epochs and fine-tune 100 epochs.
- Compare with the S3 baseline using McNemar and bootstrap.

**Acceptance**
- The MAE forward pass passes shape, gradient, and small-batch overfitting tests.
- The fine-tuned model is evaluated on the same split and protocol as the baseline.
- You can explain the +9.31 pp gain without appealing to "more capacity".
- You can state the limitations: single seed, CIFAR scale, same-source data.

**Files:** `src/vit_from_scratch/mae.py`, `tests/test_mae.py`, `tests/test_mae_model.py`, `tests/test_mae_finetune.py`, `scripts/pretrain_mae.py`, `scripts/finetune_mae.py`.

### S6 — Research judgment

**Goal:** move from "I can implement it" to "I can decide whether a claim, metric, or diagnosis is justified."

**Topics**
- Claim ladder: observation → comparison → causal claim → general claim; threats to validity.
- Metric/decision fit: selective prediction needs risk–coverage, not ROC-AUC; calibration needs reliability/ECE, not only NLL.
- Architecture vs. pretraining method: MAE is a pretraining method, not an architecture; deployment latency is a hard constraint.
- Diagnosis: two hypotheses + a minimal single-variable experiment.
- AI-code verification: tiny-subset overfit, reference comparison, data/eval audit, diff/bisect.

**Acceptance**
- You can answer the five scenario questions in `docs/qa/s6-judgment.md` with explicit assumptions and evidence.
- You can critique the claims in `docs/experiments.md` and name at least three threats to validity.

**Files:** `docs/experiments.md`, `docs/qa/s6-judgment.md`, `docs/qa/s6-module-exam.md`; optional extensions: add positional-embedding or pooling ablations.

---

## 中文

### 建议节奏

课程按 9–11 周、每周 12–15 小时设计。阶段是串行的：每一阶段都假设前一阶段的实现与测试已经通过。如果你已经熟悉 Transformer，可以压缩 S1–S2；不要跳过 S3，因为真实项目中的失败大多发生在数据、优化和评估，而不是模型定义。

| 阶段 | 建议时间 | 主题 | 主要产出 |
|---|---|---|---|
| S0 | 0.5 周 | 环境、设备、形状 | `check_env.py`、`shape_trace.py`、设备抽象 |
| S1 | 1.5–2 周 | Attention 与 Transformer block | `attention.py`、`transformer.py` + 测试 |
| S2 | 1 周 | 组装 ViT | `vit.py` + 与参考实现的数值对齐 |
| S3 | 1.5–2 周 | 在 CIFAR-10 上训练 | `data.py`、`optim.py`、`amp.py`、`train.py`、`metrics.py` + 可复现基线 |
| S4 | 1.5 周 | Swin Transformer | `swin.py` + 训练与对比 |
| S5 | 2 周 | MAE | `mae.py` + 预训练与微调 |
| S6 | 1 周 | 研究判断力 | 书面分析、消融解读、故障诊断 |

### S0 — 环境、设备抽象与形状素养

**目标：** 让项目在你的机器上可运行、可复现，并在写模型代码前学会追踪张量 shape。

**知识点**
- Python 环境、PyTorch/torchvision 版本、CUDA/MPS/CPU 检测。
- 随机种子控制；为什么只设 `torch.manual_seed` 对数据管线不够。
- Shape 记号：`B` batch、`C` 通道、`H/W` 空间尺寸、`P` patch 大小、`N` token 数、`D` token 宽度。
- 参数量 vs. 计算量（MACs/FLOPs）vs. 激活显存。

**动手**
- 运行 `python scripts/check_env.py`。
- 运行 `python scripts/shape_trace.py`，读懂它打印的每个 shape。
- 解释 224×224、patch 16 为什么得到 196 个 patch，再加 1 个 cls token。

**验收**
- 能说出 ViT 前向中每个中间张量的 shape。
- 能解释 4096×4096 matmul benchmark 为什么需要 warm-up 与同步。
- 能区分参数量、MACs 与峰值显存。

**文件：** `src/vit_from_scratch/utils.py`、`scripts/check_env.py`、`scripts/shape_trace.py`，以及你自己写的 shape 笔记。

### S1 — Attention 与 Transformer block

**目标：** 推导并实现 scaled dot-product attention、多头注意力、LayerNorm，以及 Pre-LN/Post-LN Transformer block。

**知识点**
- `scores = QKᵀ / sqrt(d_k)`、在 key 维做 softmax、`weights @ V`。
- `1/sqrt(d_k)` 的方差论证：若 `q·k` 方差为 `d_k`，缩放后 logits 方差为 1，避免 softmax 饱和。
- Mask：`True = keep`；在 softmax **之前**把不允许的位置填 `-inf`。
- 多头 split/merge：`(B,N,D) → (B,N,h,d_head) → (B,h,N,d_head)`。
- 在最后一维做 LayerNorm、biased variance、`eps` 放在平方根内。
- Pre-LN `x + F(LN(x))` vs. Post-LN `LN(x + F(x))`；Pre-LN 为什么更稳定。
- MLP/GELU、残差、DropPath，以及为什么 DropPath 只包住分支。

**动手**
- 实现 naive SDPA 与向量化 SDPA，与 `F.scaled_dot_product_attention` 做数值与梯度对照。
- 实现 MHSA，并验证 `heads=1` 时与单头实现一致。
- 运行 `scripts/compare_norm_placement.py` 的 Pre-LN/Post-LN 梯度范数实验。

**验收**
- Attention 测试通过，包括 mask 与 causal。
- 能默写 MHSA 的 shape 链。
- 能解释 mask 为什么必须在 softmax 前，而不是把概率乘 0。
- 能解释 `x + DropPath(f(x))` 为什么正确，`DropPath(x + f(x))` 为什么错误。

**文件：** `src/vit_from_scratch/attention.py`、`src/vit_from_scratch/transformer.py`、`tests/test_attention.py`、`tests/test_transformer.py`、`tests/test_block_variants.py`。

### S2 — 组装 ViT

**目标：** 把 patch 变成 token，加入 cls/位置编码，堆叠 block，并与参考实现做数值对齐。

**知识点**
- Patch embedding 等价于 `Conv2d(kernel_size=P, stride=P)`；与 reshape + Linear 等价。
- 序列长度 `N = (H/P)(W/P)`；token 宽度 `D` vs. patch 向量长度 `C·P²`。
- cls token、可学习绝对位置编码、固定正弦编码、cls vs. mean pooling。
- 参数账：patch 投影、cls token、位置表、每个 block、final LN、分类头。
- 截断正态初始化。
- 参考对齐：patch flatten 顺序 `(C,P,P)` vs. `(P,P,C)`；`permute` vs. `reshape`。

**动手**
- 用 `use_conv=True/False` 实现两遍 `PatchEmbed`，验证 `allclose`。
- 从空白文件实现 `ViT.forward`。
- 把权重复制进 `vit-pytorch`，验证 logits 一致。

**验收**
- `(2,3,32,32)` 输入得到 `(2,10)` logits。
- 参数量与手推公式误差 <1%。
- 能解释复制参考权重时 patch 顺序在哪三处起作用。

**文件：** `src/vit_from_scratch/vit.py`、`tests/test_vit.py`、`tests/test_vit_reference.py`、`scripts/profile_vit.py`。

### S3 — 在 CIFAR-10 上训练

**目标：** 构建正确的数据管线与训练循环，复现基线，诊断曲线，并做超越 accuracy 的评估。

**知识点**
- 分层 train/val 划分、只用训练集算归一化统计量、数据增强。
- DataLoader 随机性：sampler 顺序 vs. worker RNG；`base_seed + worker_id`。
- AdamW vs. Adam + L2；解耦 weight decay；1-D 参数不做 decay。
- Warmup + cosine；梯度累积下的 step 计数。
- Label smoothing、梯度裁剪顺序、AMP（`bf16`/`fp16`）、GradScaler。
- 可复现性：seed、确定性 cuDNN、两次 `metrics.jsonl` 完全一致。
- 调试：1% 过拟合测试、LR 扫描、曲线诊断。
- 评估：accuracy、macro/micro/weighted F1、top-k、ROC-AUC/PR-AUC、NLL/Brier/ECE、混淆矩阵、McNemar、bootstrap CI。

**动手**
- 运行 `scripts/overfit_small.py`，在 1% 数据上把 train acc 训到接近 100%。
- 运行 `scripts/lr_scan.py`，解释 underfit / 稳定 / 不稳定区域。
- 训练 100-epoch 基线到 test acc ≥80%。
- 在多个 checkpoint 上运行 `scripts/eval_metrics.py`。

**验收**
- 同 seed 两次运行指标完全一致。
- 基线 test acc ≥80% 且可复现。
- 能解释为什么按 batch 平均 F1 不等于全量 F1。
- 能解释为什么单个 test set 上的 McNemar/bootstrap 不衡量 seed 方差。

**文件：** `data.py`、`optim.py`、`amp.py`、`train.py`、`metrics.py`、`scripts/overfit_small.py`、`scripts/lr_scan.py`、`scripts/eval_checkpoint.py`、`scripts/eval_metrics.py`。

### S4 — Swin Transformer

**目标：** 实现窗口注意力、shifted window、PatchMerging 与相对位置偏置；训练并与 ViT 对比。

**知识点**
- 动机：全局注意力是 `O(N²)`；图像有局部结构；层级特征有利于稠密预测。
- `window_partition` / `window_reverse` 及其互逆关系。
- 循环位移、重新切窗、attention mask、反变换、unshift。
- 为什么 mask 只屏蔽 wrap 拼出的块，而不是所有非相邻 token。
- PatchMerging：`2×2` 邻域、`4C → 2C`、空间尺寸减半。
- 相对位置偏置：每个 attention 层一张 `(2M−1)²` 表，窗口间共享，block 间不共享。
- 计算量账本：token 数 `64 → 16 → 4`、窗口数、单窗口 stage。

**动手**
- 用随机张量验证 partition/reverse。
- 用两种方法构造 shifted-window mask：切片 `img_mask` 法与暴力来源坐标法。
- 在 CIFAR-10 上训练 Swin，并用 S3 协议与 ViT 比较。

**验收**
- Partition/reverse 与 shift/unshift 精确互逆。
- Mask 测试与暴力实现一致。
- 能解释 Swin 为什么在此设置下用低得多的 MACs 达到同一精度。
- 能解释这里的 Swin vs. ViT 差异为什么不显著。

**文件：** `src/vit_from_scratch/swin.py`、`tests/test_swin.py`、`tests/test_swin_stage.py`、`tests/test_swin_block.py`、`tests/test_swin_model.py`、`scripts/train_swin.py`、`scripts/bench_swin.py`。

### S5 — MAE

**目标：** 实现 masked autoencoding，在无标签 CIFAR-10 图像上预训练，微调 encoder，并与从零训练比较。

**知识点**
- 自监督：重建被遮住的 patch；预训练不使用标签。
- 高 mask ratio（75%）：图像冗余，低比例会让重建太容易。
- 非对称 encoder/decoder：encoder 只看可见 token；decoder 看可见编码 + mask token。
- Loss 只算 masked patch；`norm_pix_loss` 与逐 patch 归一化。
- `gather` / `scatter` / `argsort` 的打乱与还原；位置编码要按 `ids_keep` 取行。
- Encoder → ViT 权重迁移：patch embed、patch 位置行、blocks、final LN；cls token 与 head 保持新初始化。

**动手**
- 验证 masking/gather/restore 可逆。
- 验证可见位置梯度为 0。
- 预训练 100 epoch，微调 100 epoch。
- 用 McNemar 与 bootstrap 与 S3 基线比较。

**验收**
- MAE forward 通过 shape、梯度与小批量过拟合测试。
- 微调模型在与基线相同的 split 与协议上评估。
- 能在不诉诸“容量更大”的前提下解释 +9.31 pp 的增益。
- 能说出局限：单 seed、CIFAR 规模、同源数据。

**文件：** `src/vit_from_scratch/mae.py`、`tests/test_mae.py`、`tests/test_mae_model.py`、`tests/test_mae_finetune.py`、`scripts/pretrain_mae.py`、`scripts/finetune_mae.py`。

### S6 — 研究判断力

**目标：** 从“我能实现”进阶到“我能判断一个 claim、指标或诊断是否站得住脚”。

**知识点**
- Claim ladder：观察 → 比较 → 因果 → 一般化；以及 threats to validity。
- 指标与决策匹配：selective prediction 需要 risk–coverage，不是 ROC-AUC；校准需要 reliability/ECE，不只是 NLL。
- 架构 vs. 预训练方法：MAE 是预训练方法，不是架构；部署延迟是硬约束。
- 诊断：两个假设 + 最小单变量实验。
- AI 代码验证：tiny-subset overfit、reference 对照、data/eval audit、diff/bisect。

**验收**
- 能对 `docs/qa/s6-judgment.md` 的五个场景题给出带假设与证据的回答。
- 能批评 `docs/experiments.md` 中的 claim，并说出至少三个 threats to validity。

**文件：** `docs/experiments.md`、`docs/qa/s6-judgment.md`、`docs/qa/s6-module-exam.md`；可选扩展：位置编码或 pooling 消融。
