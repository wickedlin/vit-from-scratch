# Project Map and Review Checklist / 项目清单与审查清单

[English](#english) · [中文](#中文)

This file is the inventory of the open-source teaching project. It answers three questions: **what is included**, **why each file is here**, and **how to verify that the project is readable and runnable on Windows and macOS**.

本文是开源教学项目的文件清单，回答三个问题：**包含什么**、**每个文件为什么在这里**、**如何验证它在 Windows 与 macOS 上可读且可运行**。

---

## English

### Included files

#### Package and project configuration

| File | Stage | Purpose |
|---|---|---|
| `README.md` | — | Bilingual project overview, results, quick start, structure, platform notes |
| `LICENSE` | — | MIT license |
| `pyproject.toml` | — | `src` layout, editable install, package metadata |
| `requirements.txt` | — | Course dependencies (PyTorch installed separately per platform) |
| `.gitignore` | — | Excludes datasets, weights, experiments, caches, IDE files |
| `.gitattributes` | — | Line-ending normalization and binary markers |

#### Source (`src/vit_from_scratch/`)

| File | Stage | Purpose | Key public API |
|---|---|---|---|
| `utils.py` | S0 | Device priority and seeding | `get_device`, `set_seed` |
| `attention.py` | S1 | Naive/vectorized SDPA and MHSA | `scaled_dot_product_attention`, `MultiHeadSelfAttention` |
| `transformer.py` | S1 | LayerNorm, MLP, DropPath, block | `LayerNorm`, `MLP`, `DropPath`, `TransformerBlock` |
| `vit.py` | S2 | Patch embedding, cls/position embeddings, ViT | `PatchEmbed`, `ViT`, `pool_tokens` |
| `data.py` | S3 | CIFAR-10 download, stratified split, transforms | `build_datasets`, `build_dataloaders` |
| `optim.py` | S3 | AdamW groups, schedule, label smoothing, clipping | `build_optimizer`, `build_scheduler`, `label_smoothing_loss`, `clip_gradients` |
| `amp.py` | S3 | AMP dtype, GradScaler, micro-batch loop | `resolve_amp_dtype`, `backward_microbatch`, `optimizer_step` |
| `train.py` | S3 | Training/eval loop, checkpoints, logs, CLI | `TrainConfig`, `train_one_epoch`, `evaluate`, `fit` |
| `metrics.py` | S3 | Classification metrics and statistical tests | `compute_metrics`, `mcnemar_test`, `paired_bootstrap_ci` |
| `swin.py` | S4 | Window/shifted attention, PatchMerging, Swin | `window_partition`, `compute_shifted_window_mask`, `SwinTransformer` |
| `mae.py` | S5 | Masking, encoder/decoder, reconstruction loss | `random_masking`, `mae_loss`, `MaskedAutoencoder`, `load_mae_encoder_weights` |
| `__init__.py` | — | Package version | `__version__` |

#### Tests (`tests/`)

| File | Covers |
|---|---|
| `test_smoke.py` | Package import, torch import, device priority |
| `test_attention.py` | SDPA vs. reference, masks, causal, gradients, MHSA |
| `test_transformer.py` | LayerNorm, MLP, DropPath, block variants, gradients |
| `test_block_variants.py` | Pre-LN/Post-LN switch and DropPath placement |
| `test_vit.py` | Patch embedding equivalence, cls/pos embeddings, pooling, parameter formula, forward |
| `test_vit_reference.py` | Numerical alignment with `vit-pytorch` after weight copying |
| `test_data.py` | Transforms, stratified split, DataLoader reproducibility (downloads CIFAR-10) |
| `test_optim.py` | Weight-decay groups, warmup/cosine, label smoothing, clipping contract |
| `test_amp.py` | AMP mode resolution, GradScaler, accumulation; CUDA tests skip elsewhere |
| `test_train.py` | Training loop, LR reporting, accumulation, checkpointing, reproducibility |
| `test_metrics.py` | Metric definitions, ECE, McNemar, bootstrap, Holm |
| `test_swin.py` | Partition/reverse, shift/unshift, shifted-window mask vs. brute force |
| `test_swin_stage.py` | PatchMerging, relative position bias, shape and gradient behavior |
| `test_swin_block.py` | Window attention masks, block data flow, parameter counts |
| `test_swin_model.py` | Full Swin forward, parameter count, invalid configurations |
| `test_mae.py` | Random masking, gather/restore, masked-only loss, `norm_pix_loss` |
| `test_mae_model.py` | MAE forward, shapes, gradients, tiny-batch overfitting |
| `test_mae_finetune.py` | Encoder-to-ViT weight transfer and preserved new parameters |

All tests use `unittest` and insert `src` into `sys.path`, so they run from the repository root without installation. They are also compatible with `pytest` if you prefer it.

#### Scripts (`scripts/`)

| File | Stage | Purpose |
|---|---|---|
| `check_env.py` | S0 | Python/PyTorch versions, device, matmul benchmark, autograd smoke |
| `shape_trace.py` | S0 | Shape trace through `vit-pytorch` |
| `compare_norm_placement.py` | S1 | Pre-LN vs. Post-LN gradient norms and high-LR training |
| `profile_vit.py` | S2 | Parameter count, `thop` MACs, optional CUDA memory probe |
| `overfit_small.py` | S3 | 1% overfitting diagnostic |
| `lr_scan.py` | S3 | Learning-rate sweep on a subset |
| `check_dataloader_rng.py` | S3 | Batch order vs. augmentation randomness across workers |
| `compare_metrics.py` | S3 | Reproducibility diff of two `metrics.jsonl` files |
| `cifar10_samples.py` | S3 | Contact sheet of CIFAR-10 samples |
| `plot_metrics.py` | S3 | Loss/accuracy/LR curves from `metrics.jsonl` |
| `eval_checkpoint.py` | S3 | Loss/accuracy on val or test |
| `eval_metrics.py` | S3 | Full metric suite, figures, McNemar, bootstrap |
| `bench_amp.py` | S3 | CUDA-only fp32/bf16/fp16 benchmark |
| `train_swin.py` | S4 | Train Swin on CIFAR-10 |
| `bench_swin.py` | S4 | Swin step-time benchmark |
| `pretrain_mae.py` | S5 | MAE pretraining |
| `finetune_mae.py` | S5 | MAE encoder → ViT fine-tuning |

Scripts that import `vit_from_scratch` include a `sys.path` bootstrap, so they run from the repository root even without `pip install -e .`. The editable install is still recommended.

#### Documentation (`docs/`)

| File | Contents |
|---|---|
| `curriculum.md` | Stage goals, topics, hands-on tasks, acceptance criteria, suggested pace |
| `architecture.md` | Module dependency graph, tensor shapes, parameter accounting, data flow, training loop, failure lookup table |
| `experiments.md` | Protocol, all result tables, figures, statistical comparisons, threats to validity |
| `project-map.md` | This file |
| `qa/s0-environment.md` | Environment, devices, shapes, parameters vs. compute |
| `qa/s1-attention.md` | SDPA, scaling, masks, MHSA, LayerNorm, Pre/Post-LN, DropPath |
| `qa/s2-vit.md` | Patch embedding, cls/position embeddings, pooling, parameter counts, reference alignment |
| `qa/s3-training.md` | Data pipeline, optimization, AMP, debugging, reproducibility, metrics, statistics |
| `qa/s4-swin.md` | Window partition, shifted windows, masks, PatchMerging, relative bias, complexity |
| `qa/s5-mae.md` | Masking, asymmetric encoder/decoder, loss, `norm_pix_loss`, fine-tuning, results |
| `qa/s6-judgment.md` | Claim ladder, selective prediction, architecture vs. pretraining, diagnosis, AI-code verification |
| `qa/s6-module-exam.md` | Eight-module oral exam: shape flow, formula/pseudocode, reason/cost, failure mode for each module |
| `assets/*.png` | Curated result figures referenced by the README and experiment document |

### Deliberately excluded

| Path in the private course workspace | Why it is excluded |
|---|---|
| `data/` | CIFAR-10 is downloaded by torchvision on first run |
| `experiments/` | Run outputs, metrics, and checkpoints are local artifacts; weights can be hundreds of MB |
| `reports/` | Internal session notes and preliminary plots; the public conclusions are rewritten in `docs/experiments.md` |
| `temp/` | Scratch evaluation-curve gallery, not part of the course |
| `.idea/`, `.claude/`, `.DS_Store` | IDE, tool, and OS caches |
| `CLAUDE.md`, `PROGRESS.md` | Private teaching-session state, not learner-facing documentation |
| `scripts/patch_macho_zerofill.py` | A macOS 27/scipy wheel workaround, not part of the ViT curriculum |
| `*.pt`, `*.pth`, `*.ckpt` | Model weights |
| `*.tar.gz` | Dataset archives |
| `experiments/**/predictions_*.npz` | Large evaluation artifacts; regenerate with `eval_metrics.py` |

### Readability and runnability checklist

Use this checklist when reviewing or publishing the repository.

#### Reading

- [x] Every source file has a module docstring and English comments; no Chinese remains in `src/`, `tests/`, or `scripts/`.
- [x] Core APIs have docstrings with shapes, argument meanings, return values, and raised errors.
- [x] `README.md`, `docs/curriculum.md`, `docs/architecture.md`, `docs/experiments.md`, `docs/project-map.md`, and every `docs/qa/*.md` contain both English and Chinese.
- [x] Every stage has a Q&A document with questions and answers.
- [x] The architecture document traces the tensor shapes from input to logits for ViT, Swin, and MAE.
- [x] Results in the README match `docs/experiments.md` and the original run records.

#### Running

- [x] `pip install -e .` works with the `src` layout.
- [x] `python scripts/check_env.py` works on CUDA, MPS, and CPU.
- [x] `python -m unittest discover -s tests -v` passes on macOS; CUDA-only tests skip automatically.
- [x] Tests do not depend on pre-existing checkpoints or experiment outputs.
- [x] CIFAR-10 downloads automatically; `data/` is gitignored.
- [x] Training creates its output directory and writes `config.json`, `model_config.json`, `metrics.jsonl`, `best.pt`, `last.pt`.
- [x] Checkpoints only contain `state_dict` tensors and load with `map_location`.
- [x] CLI commands in the README are single-line and work in both bash and Windows shells.
- [x] No script requires a hard-coded absolute path.

#### Windows/macOS compatibility

- [x] Device selection is `cuda → mps → cpu`; no hard-coded `.cuda()`.
- [x] `pin_memory` is enabled only on CUDA.
- [x] `--amp auto` falls back to fp32 on CPU/MPS.
- [x] All multiprocessing entry points use `if __name__ == "__main__"`.
- [x] DataLoader supports `num_workers=0`, the safest Windows setting.
- [x] Paths use `pathlib`; no separator assumptions.
- [x] All file writes specify `encoding="utf-8"`.
- [x] CUDA-only tools exit or skip cleanly when CUDA is unavailable.
- [x] `scipy`, `scikit-learn`, and `pillow` are listed explicitly because tests and metrics import them.

### Verification log (2026-11-16)

Environment: macOS 27.0 arm64, Python 3.10.22, PyTorch 2.14.1, torchvision 0.29.2, Apple MPS.

| Check | Command | Result |
|---|---|---|
| Static language check | `grep -rnP '[\\x{4e00}-\\x{9fff}]' src tests scripts` | No matches |
| Compile all Python files | `python -m py_compile src/vit_from_scratch/*.py tests/*.py scripts/*.py` | Passed |
| Full test suite | `VIT_DATA_ROOT=<existing CIFAR-10> python -m unittest discover -s tests -v` | 274 tests passed, 5 CUDA-only skipped |
| Wheel build | `python -m pip wheel . --no-deps` | `vit_from_scratch-1.0.0` built |
| ViT end-to-end | 1 epoch, batch 128, MPS | Train/val metrics written, `best.pt`/`last.pt` created |
| Swin end-to-end | 1 epoch, batch 128, MPS | 5,407,366 params; metrics written |
| MAE pretraining smoke | 1 step, CPU | Loss computed; checkpoint written |
| MAE fine-tuning smoke | 1 epoch, 16 samples, CPU | Encoder weights loaded; metrics written |
| Evaluation reproduction | MAE checkpoint, test split | acc 0.9087, macro F1 0.9087, AUC 0.9838, NLL 0.3976, ECE 0.0585 |
| Multi-model evaluation | MAE + Swin checkpoints, val split | Metrics, McNemar, bootstrap, and four figures generated |
| Environment check | `python scripts/check_env.py` | Device `mps`, matmul and autograd OK |
| Script CLI loading | `--help` for all scripts | All 17 scripts load |

The original private course workspace was not modified; `git status` there remains clean.

### Suggested review order

1. Read `README.md` (English section).
2. Read `docs/curriculum.md` to see the intended learning sequence.
3. Open `docs/architecture.md` next to `src/vit_from_scratch/vit.py`.
4. Run the quick tests and a 1-epoch smoke run.
5. Read one Q&A file per stage and check that each answer points to code or evidence.
6. Read `docs/experiments.md`, then challenge the claims with the threats-to-validity list.

---

## 中文

### 纳入的文件

#### 打包与项目配置

| 文件 | 阶段 | 用途 |
|---|---|---|
| `README.md` | — | 双语项目概览、结果、快速开始、结构、平台说明 |
| `LICENSE` | — | MIT 许可证 |
| `pyproject.toml` | — | `src` 布局、editable 安装、包元数据 |
| `requirements.txt` | — | 课程依赖（PyTorch 按平台单独安装） |
| `.gitignore` | — | 排除数据集、权重、实验、缓存、IDE 文件 |
| `.gitattributes` | — | 行尾规范化与二进制标记 |

#### 源码（`src/vit_from_scratch/`）

| 文件 | 阶段 | 用途 | 主要公开接口 |
|---|---|---|---|
| `utils.py` | S0 | 设备优先级与随机种子 | `get_device`、`set_seed` |
| `attention.py` | S1 | naive/向量化 SDPA 与 MHSA | `scaled_dot_product_attention`、`MultiHeadSelfAttention` |
| `transformer.py` | S1 | LayerNorm、MLP、DropPath、block | `LayerNorm`、`MLP`、`DropPath`、`TransformerBlock` |
| `vit.py` | S2 | Patch embedding、cls/位置编码、ViT | `PatchEmbed`、`ViT`、`pool_tokens` |
| `data.py` | S3 | CIFAR-10 下载、分层划分、transform | `build_datasets`、`build_dataloaders` |
| `optim.py` | S3 | AdamW 分组、调度、label smoothing、裁剪 | `build_optimizer`、`build_scheduler`、`label_smoothing_loss`、`clip_gradients` |
| `amp.py` | S3 | AMP dtype、GradScaler、micro-batch 循环 | `resolve_amp_dtype`、`backward_microbatch`、`optimizer_step` |
| `train.py` | S3 | 训练/评估循环、checkpoint、日志、CLI | `TrainConfig`、`train_one_epoch`、`evaluate`、`fit` |
| `metrics.py` | S3 | 分类指标与统计检验 | `compute_metrics`、`mcnemar_test`、`paired_bootstrap_ci` |
| `swin.py` | S4 | 窗口/shifted attention、PatchMerging、Swin | `window_partition`、`compute_shifted_window_mask`、`SwinTransformer` |
| `mae.py` | S5 | masking、encoder/decoder、重建损失 | `random_masking`、`mae_loss`、`MaskedAutoencoder`、`load_mae_encoder_weights` |
| `__init__.py` | — | 包版本 | `__version__` |

#### 测试（`tests/`）

| 文件 | 覆盖内容 |
|---|---|
| `test_smoke.py` | 包导入、torch 导入、设备优先级 |
| `test_attention.py` | SDPA 与参考实现、mask、causal、梯度、MHSA |
| `test_transformer.py` | LayerNorm、MLP、DropPath、block 变体、梯度 |
| `test_block_variants.py` | Pre-LN/Post-LN 开关与 DropPath 位置 |
| `test_vit.py` | patch embedding 等价性、cls/位置编码、pooling、参数公式、前向 |
| `test_vit_reference.py` | 复制权重后与 `vit-pytorch` 的数值对齐 |
| `test_data.py` | transform、分层划分、DataLoader 可复现性（会下载 CIFAR-10） |
| `test_optim.py` | weight decay 分组、warmup/cosine、label smoothing、裁剪契约 |
| `test_amp.py` | AMP 模式解析、GradScaler、梯度累积；CUDA 测试在其他平台跳过 |
| `test_train.py` | 训练循环、LR 记录、累积、checkpoint、可复现性 |
| `test_metrics.py` | 指标定义、ECE、McNemar、bootstrap、Holm |
| `test_swin.py` | partition/reverse、shift/unshift、shifted-window mask 与暴力对照 |
| `test_swin_stage.py` | PatchMerging、相对位置偏置、shape 与梯度行为 |
| `test_swin_block.py` | 窗口 attention mask、block 数据流、参数量 |
| `test_swin_model.py` | 完整 Swin 前向、参数量、非法配置 |
| `test_mae.py` | 随机 masking、gather/restore、masked-only loss、`norm_pix_loss` |
| `test_mae_model.py` | MAE 前向、shape、梯度、小批量过拟合 |
| `test_mae_finetune.py` | encoder → ViT 权重迁移与新参数保持 |

所有测试使用 `unittest`，并把 `src` 插入 `sys.path`，因此无需安装即可从仓库根目录运行。若你偏好 `pytest`，这些测试也兼容。

#### 脚本（`scripts/`）

| 文件 | 阶段 | 用途 |
|---|---|---|
| `check_env.py` | S0 | Python/PyTorch 版本、设备、matmul benchmark、autograd smoke |
| `shape_trace.py` | S0 | 追踪 `vit-pytorch` 的 shape |
| `compare_norm_placement.py` | S1 | Pre-LN vs. Post-LN 梯度范数与高 LR 训练 |
| `profile_vit.py` | S2 | 参数量、`thop` MACs、可选 CUDA 显存探针 |
| `overfit_small.py` | S3 | 1% 过拟合诊断 |
| `lr_scan.py` | S3 | 子集上的学习率扫描 |
| `check_dataloader_rng.py` | S3 | 不同 worker 下的取批顺序与增强随机性 |
| `compare_metrics.py` | S3 | 两个 `metrics.jsonl` 的可复现性 diff |
| `cifar10_samples.py` | S3 | CIFAR-10 样本 contact sheet |
| `plot_metrics.py` | S3 | 从 `metrics.jsonl` 画 loss/acc/LR 曲线 |
| `eval_checkpoint.py` | S3 | 在 val 或 test 上算 loss/accuracy |
| `eval_metrics.py` | S3 | 完整指标、图、McNemar、bootstrap |
| `bench_amp.py` | S3 | 仅 CUDA 的 fp32/bf16/fp16 benchmark |
| `train_swin.py` | S4 | 在 CIFAR-10 上训练 Swin |
| `bench_swin.py` | S4 | Swin step time benchmark |
| `pretrain_mae.py` | S5 | MAE 预训练 |
| `finetune_mae.py` | S5 | MAE encoder → ViT 微调 |

导入 `vit_from_scratch` 的脚本都带 `sys.path` bootstrap，因此即使没有 `pip install -e .`，也能从仓库根目录运行。仍推荐 editable 安装。

#### 文档（`docs/`）

| 文件 | 内容 |
|---|---|
| `curriculum.md` | 阶段目标、知识点、动手任务、验收标准、建议节奏 |
| `architecture.md` | 模块依赖图、张量 shape、参数账、数据流、训练循环、故障排查表 |
| `experiments.md` | 协议、全部结果表、图、统计比较、效度威胁 |
| `project-map.md` | 本文件 |
| `qa/s0-environment.md` | 环境、设备、shape、参数量 vs. 计算量 |
| `qa/s1-attention.md` | SDPA、缩放、mask、MHSA、LayerNorm、Pre/Post-LN、DropPath |
| `qa/s2-vit.md` | patch embedding、cls/位置编码、pooling、参数账、参考对齐 |
| `qa/s3-training.md` | 数据管线、优化、AMP、调试、可复现性、指标、统计 |
| `qa/s4-swin.md` | 窗口切分、shifted window、mask、PatchMerging、相对偏置、复杂度 |
| `qa/s5-mae.md` | masking、非对称 encoder/decoder、loss、`norm_pix_loss`、微调、结果 |
| `qa/s6-judgment.md` | claim ladder、selective prediction、架构 vs. 预训练、诊断、AI 代码验证 |
| `qa/s6-module-exam.md` | 八模块口试样卷：每个模块的 shape 流、公式/伪代码、理由/代价、失败模式 |
| `assets/*.png` | README 与实验文档引用的精选结果图 |

### 有意排除的内容

| 私有课程仓库中的路径 | 排除原因 |
|---|---|
| `data/` | CIFAR-10 由 torchvision 首次运行时自动下载 |
| `experiments/` | 运行产物、指标与 checkpoint 是本地文件；权重可达数百 MB |
| `reports/` | 内部会话笔记与初步图；公开结论已重写进 `docs/experiments.md` |
| `temp/` | 评估曲线草稿画廊，不属于课程 |
| `.idea/`、`.claude/`、`.DS_Store` | IDE、工具与系统缓存 |
| `CLAUDE.md`、`PROGRESS.md` | 私有教学会话状态，不是面向学习者的文档 |
| `scripts/patch_macho_zerofill.py` | macOS 27/scipy wheel 的环境补丁，不属于 ViT 课程 |
| `*.pt`、`*.pth`、`*.ckpt` | 模型权重 |
| `*.tar.gz` | 数据集压缩包 |
| `experiments/**/predictions_*.npz` | 大型评估产物；可用 `eval_metrics.py` 重新生成 |

### 可读性与可运行性清单

审查或发布仓库时逐项检查。

#### 阅读

- [x] 每个源码文件都有模块 docstring，注释为英文；`src/`、`tests/`、`scripts/` 中不再有中文。
- [x] 核心 API 的 docstring 写明 shape、参数含义、返回值与异常。
- [x] `README.md`、`docs/curriculum.md`、`docs/architecture.md`、`docs/experiments.md`、`docs/project-map.md` 与每个 `docs/qa/*.md` 都包含中英文。
- [x] 每个阶段都有问答文档，且包含问题与答案。
- [x] 架构文档追踪了 ViT、Swin、MAE 从输入到 logits 的张量 shape。
- [x] README 的结果与 `docs/experiments.md` 及原始运行记录一致。

#### 运行

- [x] `pip install -e .` 在 `src` 布局下可用。
- [x] `python scripts/check_env.py` 在 CUDA、MPS、CPU 上可用。
- [x] `python -m unittest discover -s tests -v` 在 macOS 上通过；仅 CUDA 的测试自动跳过。
- [x] 测试不依赖预先存在的 checkpoint 或实验产物。
- [x] CIFAR-10 自动下载；`data/` 被 gitignore。
- [x] 训练会自动创建输出目录，并写 `config.json`、`model_config.json`、`metrics.jsonl`、`best.pt`、`last.pt`。
- [x] checkpoint 只包含 `state_dict` 张量，并用 `map_location` 加载。
- [x] README 中的 CLI 命令是单行，bash 与 Windows shell 都能运行。
- [x] 没有脚本依赖硬编码绝对路径。

#### Windows/macOS 兼容

- [x] 设备选择为 `cuda → mps → cpu`，没有硬编码 `.cuda()`。
- [x] `pin_memory` 只在 CUDA 上开启。
- [x] `--amp auto` 在 CPU/MPS 上退回 fp32。
- [x] 所有多进程入口都有 `if __name__ == "__main__"`。
- [x] DataLoader 支持 `num_workers=0`，这是 Windows 上最安全的设置。
- [x] 路径使用 `pathlib`，不假设分隔符。
- [x] 所有文件写入都指定 `encoding="utf-8"`。
- [x] 仅 CUDA 的工具在无 CUDA 时正常退出或跳过。
- [x] 显式列出 `scipy`、`scikit-learn`、`pillow`，因为测试与指标会直接导入它们。

### 验证记录（2026-11-16）

环境：macOS 27.0 arm64、Python 3.10.22、PyTorch 2.14.1、torchvision 0.29.2、Apple MPS。

| 检查 | 命令 | 结果 |
|---|---|---|
| 代码语言静态检查 | `grep -rnP '[\\x{4e00}-\\x{9fff}]' src tests scripts` | 无匹配 |
| 编译全部 Python 文件 | `python -m py_compile src/vit_from_scratch/*.py tests/*.py scripts/*.py` | 通过 |
| 完整测试 | `VIT_DATA_ROOT=<已有 CIFAR-10> python -m unittest discover -s tests -v` | 274 项通过，5 项 CUDA 专用跳过 |
| Wheel 构建 | `python -m pip wheel . --no-deps` | 构建出 `vit_from_scratch-1.0.0` |
| ViT 端到端 | 1 epoch、batch 128、MPS | 写入 train/val 指标与 `best.pt`/`last.pt` |
| Swin 端到端 | 1 epoch、batch 128、MPS | 5,407,366 参数；指标写入 |
| MAE 预训练 smoke | 1 step、CPU | 算出 loss；写入 checkpoint |
| MAE 微调 smoke | 1 epoch、16 样本、CPU | 成功加载 encoder 权重；指标写入 |
| 评估复现 | MAE checkpoint、test split | acc 0.9087、macro F1 0.9087、AUC 0.9838、NLL 0.3976、ECE 0.0585 |
| 多模型评估 | MAE + Swin checkpoint、val split | 生成指标、McNemar、bootstrap 与四张图 |
| 环境检查 | `python scripts/check_env.py` | 设备 `mps`，matmul 与 autograd 正常 |
| 脚本 CLI 加载 | 所有脚本 `--help` | 17 个脚本全部可加载 |

原私有课程仓库未被修改；其 `git status` 仍然干净。

### 建议审查顺序

1. 读 `README.md` 的中文部分。
2. 读 `docs/curriculum.md`，理解预期学习顺序。
3. 把 `docs/architecture.md` 与 `src/vit_from_scratch/vit.py` 对照阅读。
4. 跑快速测试与 1-epoch smoke。
5. 每个阶段读一份 Q&A，检查每个答案是否指向代码或证据。
6. 读 `docs/experiments.md`，用效度威胁清单挑战其中的 claim。
