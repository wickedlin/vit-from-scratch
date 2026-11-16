# Unit-by-Unit Learning Plan / 逐单元学习计划

[English](#english) · [中文](#中文)

This file turns the stage-level curriculum into a unit-by-unit checklist. Each row answers four questions: **what to do**, **which Q&A to answer**, **what to produce**, and **how to know you passed**. Unit IDs follow the original course numbering: `S1-1.1`, `S1-1.2`, and so on. A unit may reference several Q&A items; answer them after the implementation or experiment, not before.

本文把阶段级课程拆成逐单元清单。每一行回答四件事：**做什么**、**回答哪些问答**、**产出什么**、**怎么算通过**。单元编号沿用原课程：`S1-1.1`、`S1-1.2` 等。一个单元可能对应多道问答；先做实现或实验，再回答问题。

---

## English

### How to use this plan

1. Pick one unit. Read its "Do" column and the linked source file.
2. Implement or run the unit. Do not move on if the tests fail.
3. Answer the listed Q&A items in your own words; check the model answer only afterward.
4. Produce the deliverable. A unit is "passed" only when the deliverable exists and you can explain it.

The Q&A links point to the question IDs in [`docs/qa/`](qa/). The stage-level goals are in [`docs/curriculum.md`](curriculum.md); the tensor shapes are in [`docs/architecture.md`](architecture.md).

### S0 — Environment, devices, and shapes

| Unit | Do | Answer | Deliverable / pass |
|---|---|---|---|
| S0-0.1 | Create the Python environment; install PyTorch, torchvision, and `requirements.txt`; run `python scripts/check_env.py`; explain device priority, matmul warm-up/sync, and seeding. | Q0.1, Q0.2, Q0.6 | `check_env.py` prints a valid device and autograd passes; you can explain the output. |
| S0-0.2 | Run the minimal `vit-pytorch` forward; change patch 16→8; compute sequence length and parameter change. | Q0.3, Q0.4 | Your numbers match the formula: 196→784 patches, 197→785 tokens, and the parameter accounting. |
| S0-0.3 | Run `python scripts/shape_trace.py`; write the full shape chain from image to logits. | Q0.7 | You can draw or recite `(B,C,H,W) → (B,N,D) → (B,T,D) → (B,K)` without looking. |
| S0-0.4 | Check the package layout, `pyproject.toml`, `utils.py`, seeds, and smoke test; run `pip install -e .`. | Q0.1, Q0.6 | `test_smoke.py` passes 3/3; imports work from any directory after editable install. |

### S1 — Attention and the Transformer block

| Unit | Do | Answer | Deliverable / pass |
|---|---|---|---|
| S1-1.1 | Read *Attention Is All You Need* §3.2; derive `QKᵀ/√d_k`, the variance argument for scaling, softmax, and mask placement. | Q1.1, Q1.2, Q1.4 | A written derivation with shapes; you can explain why mask goes before softmax. |
| S1-1.2 | Implement naive and vectorized SDPA; compare with `F.scaled_dot_product_attention`; add mask and causal tests. | Q1.3, Q1.5, Q1.12 | `tests/test_attention.py` passes; `allclose` against the reference; mask/causal cases pass. |
| S1-1.3 | Derive and implement multi-head attention: split/merge, `dim_head`, `inner_dim`, output projection; verify `heads=1` equivalence. | Q1.6, Q1.7 | `tests/test_attention.py` MHSA tests pass; you can write the shape chain and parameter count from memory. |
| S1-1.4 | Derive and implement LayerNorm; compare Pre-LN vs. Post-LN; run `scripts/compare_norm_placement.py`. | Q1.8, Q1.9 | LayerNorm tests pass; the gradient-norm experiment shows the expected trend; you can explain the final LN. |
| S1-1.5 | Implement MLP/GELU, residual connections, DropPath, and `TransformerBlock`; verify the `pre_norm` switch. | Q1.10, Q1.11 | `test_transformer.py` and `test_block_variants.py` pass; DropPath wraps only the branch. |
| S1-1.6 | Study numerics: stable softmax, `-inf` masking, all-masked rows, bf16/fp16 differences. | Q1.12 | You can explain the NaN case and why bf16 does not need loss scaling while fp16 does. |

### S2 — Assembling ViT

| Unit | Do | Answer | Deliverable / pass |
|---|---|---|---|
| S2-2.1 | Implement patch embedding twice (`Conv2d` and reshape+Linear); verify equivalence; derive `N=(H/P)(W/P)`. | Q2.1, Q2.2 | `tests/test_vit.py` patch tests pass; both paths give `allclose` outputs. |
| S2-2.2 | Add cls token, learned/sinusoidal/none position embeddings, and cls/mean pooling; account for parameters. | Q2.3, Q2.4, Q2.5 | Patch/position/cls tests pass; you can compare the three position schemes and two pooling schemes. |
| S2-2.3 | Assemble the full ViT forward and initialization; derive the parameter formula. | Q2.6, Q2.7, Q2.9, Q2.11 | `(2,3,32,32)→(2,10)`; parameter count matches the formula; default and baseline configs are distinguished. |
| S2-2.4 | Copy weights into `vit-pytorch` and verify numerical alignment; handle patch flatten order and the reference patch LNs. | Q2.8 | `tests/test_vit_reference.py` passes 4/4; you can explain `permute` vs. `reshape`. |
| S2-2.5 | Hand-compute parameters, MACs, and peak memory; compare with `thop` and the memory probe. | Q2.6, Q0.4, Q0.5 | The hand accounts match within 1%; you can explain the `thop` difference. |

### S3 — Training and evaluation on CIFAR-10

| Unit | Do | Answer | Deliverable / pass |
|---|---|---|---|
| S3-3.1 | Build the data pipeline: stratified split, normalization, augmentation, DataLoader workers. | Q3.1, Q3.2, Q3.3, Q3.4 | `tests/test_data.py` passes; you can explain leakage, crop range, and worker RNG. |
| S3-3.2 | Implement AdamW parameter groups, warmup + cosine, label smoothing, and gradient clipping. | Q3.5, Q3.6, Q3.7, Q3.8 | `tests/test_optim.py` passes; you can explain the update order and step counting. |
| S3-3.3 | Add AMP: dtype resolution, GradScaler, micro-batch backward, optimizer step; benchmark on your device. | Q3.9 | `tests/test_amp.py` passes; you have ms/step and memory numbers; you can say when AMP helps. |
| S3-3.4 | Build the training loop: accumulation, scheduler, checkpointing, JSONL logs, seed control. | Q3.10, Q3.15, Q3.16 | `tests/test_train.py` passes; two same-seed runs produce identical metrics. |
| S3-3.5 | Debug: 1% overfitting test, LR scan, abnormal-curve diagnosis. | Q3.11, Q3.12 | Tiny-subset train accuracy reaches ~99%; LR summary explains underfit/stable/unstable. |
| S3-3.6 | Train the 100-epoch baseline; run regularization and the 30-epoch structural ablation. | Use [`docs/experiments.md`](experiments.md) and Q3.10/Q3.12 | Baseline test accuracy ≥80%; complete experiment tables; you can state what each comparison supports. |
| S3-3.7 | Implement metrics and error analysis: P/R/F1, confusion matrix, top-k, ROC/PR, NLL/Brier/ECE, McNemar, bootstrap. | Q3.13, Q3.14 | `tests/test_metrics.py` passes; `metrics_full.json` and four figures are produced; you can explain each metric. |

### S4 — Swin Transformer

| Unit | Do | Answer | Deliverable / pass |
|---|---|---|---|
| S4-4.1 | Explain the motivation: global `O(N²)` attention, locality, hierarchy, data hunger. | Q4.1 | You can compare ViT and Swin on cost, receptive field, and inductive bias. |
| S4-4.2 | Implement `window_partition`/`window_reverse`, shift/unshift, and the shifted-window mask. | Q4.2, Q4.3, Q4.4 | `tests/test_swin.py` passes; the mask matches the brute-force implementation. |
| S4-4.3 | Implement PatchMerging, relative position bias, and the stage structure. | Q4.5, Q4.6, Q4.7 | `tests/test_swin_stage.py` passes; parameter accounts are correct; you can explain stage vs. block. |
| S4-4.4 | Assemble `WindowAttention`, `SwinTransformerBlock`, and the full model; run shape/gradient tests. | Q4.10 | `test_swin_block.py` and `test_swin_model.py` pass; you did not copy repository code. |
| S4-4.5 | Train Swin on CIFAR-10; compare with ViT under the same protocol; report efficiency and significance. | Q4.8, Q4.9 | 100-epoch run completes; same split/metrics; you can interpret the McNemar/bootstrap result. |

### S5 — Masked Autoencoder

| Unit | Do | Answer | Deliverable / pass |
|---|---|---|---|
| S5-5.1 | Explain MAE motivation, the 75% mask ratio, and the data/token budget. | Q5.1, Q5.2 | You can compute CIFAR 64→16 and STL-10 144→36 visible tokens. |
| S5-5.2 | Implement random masking, gather/restore, masked-only loss, and `norm_pix_loss`. | Q5.3, Q5.4, Q5.5, Q5.6, Q5.10 | `tests/test_mae.py` passes; visible positions have zero gradient; norm/affine tests pass. |
| S5-5.3 | Assemble encoder/decoder and the full MAE forward; check shapes, gradients, and tiny-batch overfitting. | Q5.7, Q5.11 | `tests/test_mae_model.py` passes; parameter count is 6,184,560. |
| S5-5.4 | Run pretraining and fine-tuning; implement the encoder→ViT weight mapping. | Q5.8 | Smoke runs are reproducible; `test_mae_finetune.py` passes; checkpoints reload. |
| S5-5.5 | Run the formal 100+100 epoch experiment; compare with the S3 baseline under the same protocol. | Q5.9 | Same split/metrics; McNemar/bootstrap reported; you can explain the gain without "more capacity". |

### S6 — Judgment and extension

| Unit | Do | Answer | Deliverable / pass |
|---|---|---|---|
| S6-6.1 | Module oral exam + code review/fault diagnosis (A+B), or the five-scenario judgment interview. | [`s6-module-exam.md`](qa/s6-module-exam.md) A1–A8; [`s6-judgment.md`](qa/s6-judgment.md) Q6.1–Q6.5 | Whiteboard/data-flow explanations; find and explain bugs; state assumptions and failure modes. |
| S6-6.2 | Run or interpret ablations: patch size, position encoding, Pre/Post-LN, depth/width, pooling. | Q6.1 (claim scope); [`docs/experiments.md`](experiments.md) | At least four ablation groups; table + conclusion; you can name the threats to validity. |
| S6-6.3 | Code review against a reference implementation; profile time/memory; find one real issue. | Q6.4, Q6.5 | A written review and at least one fix or documented performance finding. |
| S6-6.4 | Give a 20-minute talk or write a blog post; answer five follow-up questions. | Q6.1–Q6.5 | Talk/blog draft + follow-up notes. |

---

## 中文

### 怎么用这份计划

1. 选一个单元，读它的“做什么”列和对应源码。
2. 实现或运行这个单元。测试不通过就不要进入下一个单元。
3. 用自己的话回答列出的问答；答完再看参考答案。
4. 产出交付物。只有交付物存在、且你能解释它，这个单元才算通过。

问答链接指向 [`docs/qa/`](qa/) 中的题号。阶段目标见 [`docs/curriculum.md`](curriculum.md)；张量 shape 见 [`docs/architecture.md`](architecture.md)。

### S0 — 环境、设备与形状

| 单元 | 做什么 | 回答 | 交付物 / 通过标准 |
|---|---|---|---|
| S0-0.1 | 建 Python 环境，安装 PyTorch、torchvision 与 `requirements.txt`；运行 `python scripts/check_env.py`；解释设备优先级、matmul warm-up/同步与播种。 | Q0.1、Q0.2、Q0.6 | `check_env.py` 打印有效设备且 autograd 通过；能解释输出。 |
| S0-0.2 | 跑 `vit-pytorch` 最小前向；把 patch 16 改成 8；算序列长度与参数变化。 | Q0.3、Q0.4 | 数字与公式一致：196→784 patch、197→785 token，以及参数账。 |
| S0-0.3 | 运行 `python scripts/shape_trace.py`；写出从图像到 logits 的完整 shape 链。 | Q0.7 | 不看代码能画出/背出 `(B,C,H,W) → (B,N,D) → (B,T,D) → (B,K)`。 |
| S0-0.4 | 检查包布局、`pyproject.toml`、`utils.py`、随机种子与 smoke test；运行 `pip install -e .`。 | Q0.1、Q0.6 | `test_smoke.py` 3/3 通过；editable 安装后任意目录都能 import。 |

### S1 — Attention 与 Transformer block

| 单元 | 做什么 | 回答 | 交付物 / 通过标准 |
|---|---|---|---|
| S1-1.1 | 读 *Attention Is All You Need* §3.2；推导 `QKᵀ/√d_k`、缩放的方差论证、softmax 与 mask 位置。 | Q1.1、Q1.2、Q1.4 | 写出带 shape 的推导；能解释 mask 为什么在 softmax 前。 |
| S1-1.2 | 实现 naive 与向量化 SDPA；与 `F.scaled_dot_product_attention` 对照；补 mask 与 causal 测试。 | Q1.3、Q1.5、Q1.12 | `tests/test_attention.py` 通过；与参考 `allclose`；mask/causal 用例通过。 |
| S1-1.3 | 推导并实现多头注意力：split/merge、`dim_head`、`inner_dim`、输出投影；验证 `heads=1` 等价。 | Q1.6、Q1.7 | MHSA 测试通过；能默写 shape 链与参数量。 |
| S1-1.4 | 推导并实现 LayerNorm；比较 Pre-LN 与 Post-LN；运行 `scripts/compare_norm_placement.py`。 | Q1.8、Q1.9 | LayerNorm 测试通过；梯度范数实验趋势正确；能解释 final LN。 |
| S1-1.5 | 实现 MLP/GELU、残差、DropPath 与 `TransformerBlock`；验证 `pre_norm` 开关。 | Q1.10、Q1.11 | `test_transformer.py` 与 `test_block_variants.py` 通过；DropPath 只包分支。 |
| S1-1.6 | 学习数值问题：稳定 softmax、`-inf` mask、整行 mask、bf16/fp16 差异。 | Q1.12 | 能解释 NaN 场景，以及 bf16 不需要 loss scaling、fp16 需要的原因。 |

### S2 — 组装 ViT

| 单元 | 做什么 | 回答 | 交付物 / 通过标准 |
|---|---|---|---|
| S2-2.1 | 用两种方式实现 patch embedding（`Conv2d` 与 reshape+Linear）；验证等价；推导 `N=(H/P)(W/P)`。 | Q2.1、Q2.2 | patch 测试通过；两条路径输出 `allclose`。 |
| S2-2.2 | 加入 cls token、learned/sincos/none 位置编码、cls/mean pooling；算参数账。 | Q2.3、Q2.4、Q2.5 | patch/position/cls 测试通过；能比较三种位置编码与两种 pooling。 |
| S2-2.3 | 组装完整 ViT forward 与初始化；推导参数公式。 | Q2.6、Q2.7、Q2.9、Q2.11 | `(2,3,32,32)→(2,10)`；参数量与公式一致；能区分默认配置与基线配置。 |
| S2-2.4 | 把权重复制进 `vit-pytorch` 并做数值对齐；处理 patch flatten 顺序与参考实现的 patch LN。 | Q2.8 | `tests/test_vit_reference.py` 4/4 通过；能解释 `permute` vs. `reshape`。 |
| S2-2.5 | 手算参数、MACs、峰值显存；与 `thop` 和显存探针对账。 | Q2.6、Q0.4、Q0.5 | 手算与工具差 <1%；能解释 `thop` 差额。 |

### S3 — CIFAR-10 训练与评估

| 单元 | 做什么 | 回答 | 交付物 / 通过标准 |
|---|---|---|---|
| S3-3.1 | 构建数据管线：分层划分、归一化、增强、DataLoader worker。 | Q3.1、Q3.2、Q3.3、Q3.4 | `tests/test_data.py` 通过；能解释泄漏、裁剪范围、worker RNG。 |
| S3-3.2 | 实现 AdamW 参数分组、warmup + cosine、label smoothing、梯度裁剪。 | Q3.5、Q3.6、Q3.7、Q3.8 | `tests/test_optim.py` 通过；能解释更新顺序与 step 口径。 |
| S3-3.3 | 加 AMP：dtype 解析、GradScaler、micro-batch 反传、optimizer step；在设备上 benchmark。 | Q3.9 | `tests/test_amp.py` 通过；有 ms/step 与显存数字；能说 AMP 何时有效。 |
| S3-3.4 | 构建训练循环：梯度累积、scheduler、checkpoint、JSONL 日志、seed 控制。 | Q3.10、Q3.15、Q3.16 | `tests/test_train.py` 通过；同 seed 两次运行指标一致。 |
| S3-3.5 | 调试：1% 过拟合测试、LR 扫描、异常曲线诊断。 | Q3.11、Q3.12 | 小样本 train acc 接近 99%；LR summary 能解释 underfit/稳定/不稳定。 |
| S3-3.6 | 训练 100-epoch 基线；跑正则化与 30-epoch 结构消融。 | 参考 [`docs/experiments.md`](experiments.md) 与 Q3.10/Q3.12 | 基线 test acc ≥80%；实验表完整；能说明每个对比支持什么。 |
| S3-3.7 | 实现指标与误差分析：P/R/F1、混淆矩阵、top-k、ROC/PR、NLL/Brier/ECE、McNemar、bootstrap。 | Q3.13、Q3.14 | `tests/test_metrics.py` 通过；生成 `metrics_full.json` 与四张图；能解释每个指标。 |

### S4 — Swin Transformer

| 单元 | 做什么 | 回答 | 交付物 / 通过标准 |
|---|---|---|---|
| S4-4.1 | 解释动机：全局 `O(N²)` attention、局部性、层级、数据需求。 | Q4.1 | 能从成本、感受野、归纳偏置三方面比较 ViT 与 Swin。 |
| S4-4.2 | 实现 `window_partition`/`window_reverse`、shift/unshift、shifted-window mask。 | Q4.2、Q4.3、Q4.4 | `tests/test_swin.py` 通过；mask 与暴力实现一致。 |
| S4-4.3 | 实现 PatchMerging、相对位置偏置与 stage 结构。 | Q4.5、Q4.6、Q4.7 | `tests/test_swin_stage.py` 通过；参数账正确；能解释 stage vs. block。 |
| S4-4.4 | 组装 `WindowAttention`、`SwinTransformerBlock` 与完整模型；跑 shape/梯度测试。 | Q4.10 | `test_swin_block.py` 与 `test_swin_model.py` 通过；没有复制仓库代码。 |
| S4-4.5 | 在 CIFAR-10 上训练 Swin；用同协议与 ViT 对比；报告效率与显著性。 | Q4.8、Q4.9 | 100-epoch run 完成；同 split/指标；能解读 McNemar/bootstrap。 |

### S5 — Masked Autoencoder

| 单元 | 做什么 | 回答 | 交付物 / 通过标准 |
|---|---|---|---|
| S5-5.1 | 解释 MAE 动机、75% mask ratio、数据与 token 预算。 | Q5.1、Q5.2 | 能算出 CIFAR 64→16、STL-10 144→36 个可见 token。 |
| S5-5.2 | 实现随机 masking、gather/restore、masked-only loss、`norm_pix_loss`。 | Q5.3、Q5.4、Q5.5、Q5.6、Q5.10 | `tests/test_mae.py` 通过；可见位置梯度为 0；norm/仿射测试通过。 |
| S5-5.3 | 组装 encoder/decoder 与完整 MAE forward；检查 shape、梯度与小批量过拟合。 | Q5.7、Q5.11 | `tests/test_mae_model.py` 通过；参数量 6,184,560。 |
| S5-5.4 | 跑预训练与微调；实现 encoder→ViT 权重映射。 | Q5.8 | smoke 可复现；`test_mae_finetune.py` 通过；checkpoint 可重载。 |
| S5-5.5 | 跑正式 100+100 epoch 实验；用同协议与 S3 基线比较。 | Q5.9 | 同 split/指标；报告 McNemar/bootstrap；能不用“容量更大”解释增益。 |

### S6 — 判断力与扩展

| 单元 | 做什么 | 回答 | 交付物 / 通过标准 |
|---|---|---|---|
| S6-6.1 | 模块口试 + 代码审查/故障诊断（A+B），或五场景判断力面试。 | [`s6-module-exam.md`](qa/s6-module-exam.md) A1–A8；[`s6-judgment.md`](qa/s6-judgment.md) Q6.1–Q6.5 | 白板/数据流讲解；找出并解释 bug；明确假设与失败模式。 |
| S6-6.2 | 运行或解读消融：patch size、位置编码、Pre/Post-LN、深度/宽度、pooling。 | Q6.1（claim 范围）；[`docs/experiments.md`](experiments.md) | 至少 4 组消融；表 + 结论；能说出效度威胁。 |
| S6-6.3 | 对照参考实现做代码 review；profile 时间/显存；找至少 1 个真实问题。 | Q6.4、Q6.5 | 书面 review 与至少 1 个修复或性能发现。 |
| S6-6.4 | 做 20 分钟讲解或写技术博客；回答 5 个追问。 | Q6.1–Q6.5 | 讲稿/博客草稿 + 追问笔记。 |
