# S6 — Research Judgment / 研究判断力

[English](#english) · [中文](#中文)

This stage is not about writing more model code. It is about deciding whether a claim, a metric, or a diagnosis is justified. The five scenarios below have no single "correct" answer; the goal is to state assumptions, evidence, and failure modes explicitly. The companion [module oral exam](s6-module-exam.md) covers the eight implementation modules.

这一阶段不是继续写模型代码，而是判断一个 claim、指标或诊断是否站得住脚。下面五个场景没有唯一“正确答案”；目标是明确说出假设、证据与失败模式。配套的[模块口试样卷](s6-module-exam.md)覆盖八个实现模块。

---

## English

### Q6.1 "MAE generally beats training from scratch — our +9.31 pp experiment proves it." Do you agree?

**Answer.** No. The experiment supports a much narrower claim:

> On CIFAR-10, using the same 45,000 training images, this MAE pretraining recipe (100 epochs, mask 0.75, `norm_pix_loss`) followed by 100 epochs of fine-tuning beats this specific from-scratch ViT baseline by 9.31 pp on the 10,000-image test split (McNemar p = 2.92e-141, bootstrap 95% CI [+8.59, +10.03] pp).

That is a **comparison** claim under a specific protocol, not a general or causal claim. To move up the claim ladder, you need to address:

- **Seed variance**: one run per configuration; the CI covers test-sample noise conditional on the two trained models, not initialization/data-order variance.
- **Baseline strength**: the from-scratch baseline is a 100-epoch ViT-tiny with a modest recipe. A stronger baseline (longer training, better augmentation, more tuning) could shrink the gap.
- **Compute fairness**: MAE gets 100 pretraining epochs plus 100 fine-tuning epochs. The from-scratch baseline gets 100 epochs. Match total compute before claiming a method advantage.
- **Data overlap / external validity**: the unlabeled pretraining images are the same CIFAR-10 training images. The result does not establish transfer from a different or larger unlabeled pool.
- **Scale**: 32×32 images, 64 tokens, 16 visible tokens. ImageNet-scale conclusions require ImageNet-scale experiments.
- **Metric scope**: accuracy is one endpoint; calibration, robustness, and downstream transfer are different claims.

**Claim ladder.**

```text
Observation:  MAE fine-tune test acc = 0.9087; baseline = 0.8156.
Comparison:   On this split and protocol, MAE is 9.31 pp higher.
Causal:       The gain is caused by the pretrained representation, not by extra
              capacity (params are equal), but compute and tuning differ.
General:      MAE is a better pretraining method for vision in general.
```

The experiment supports levels 1–2, partially supports level 3 with caveats, and does **not** support level 4.

**Common mistake.** Treating a statistically significant McNemar p-value as evidence of general superiority. McNemar answers "are these two trained models different on this test set?", not "is this method better in general?".

### Q6.2 You need a classifier that can abstain and send uncertain cases to a human. Which metric do you optimize?

**Answer.** Use **risk–coverage** and **selective accuracy**, not ROC-AUC alone.

- Sort samples by a confidence score (e.g., max softmax probability, margin, or an ensemble disagreement score).
- For each coverage level `c` (fraction of samples the model answers), compute selective accuracy on the covered set and selective risk (error rate).
- Plot risk vs. coverage. Compare with random abstention and with a simple baseline.
- Report the operating point you will deploy: coverage, selective accuracy, error rate among reviewed samples, and the human review budget.
- If calibration matters, report a reliability diagram, ECE at several bin counts, NLL, and Brier. ECE is bin-sensitive; NLL/Brier are proper scoring rules.

ROC-AUC measures ranking over all thresholds and treats false positives and false negatives symmetrically. It does not tell you the error rate at the abstention threshold you will actually use. If the cost of a wrong answer is much higher than the cost of abstaining, optimize the risk–coverage trade-off at the chosen operating point.

**Common mistake.** Reporting "AUC 0.99, so the system is safe to deploy with abstention". AUC can be high while the selective risk at 80% coverage is unacceptable. Another mistake: choosing the threshold on the test set; use a validation set and report the test result once.

### Q6.3 A team must hit 20 ms latency on an edge device. They say "use MAE". What is wrong with the question, and how do you decide?

**Answer.** MAE is a **pretraining method**, not a deployable architecture. You can pretrain with MAE and then deploy a ViT, a Swin, or another encoder; the decoder is discarded. The real question is: which architecture and input configuration meet the **hard** 20 ms constraint on the target hardware, and what accuracy do you lose?

Decision procedure:

1. Fix the deployment constraints: latency percentile (p50/p99), batch size (often 1), resolution, memory, supported operators, quantization.
2. Measure latency on the target device, not from theoretical MACs. MACs are a proxy; memory bandwidth, kernel launch overhead, and operator support dominate at small batch.
3. Compare candidates: ViT (global `O(N²)` attention), Swin (windowed `O(N)` attention, hierarchy), a hybrid/CNN baseline, and different patch sizes/resolutions.
4. Measure accuracy under the same pretraining/fine-tuning budget, then apply quantization/pruning and re-measure both latency and accuracy.
5. State the failure risk: accuracy drop under quantization, unsupported operators, thermal throttling, p99 latency spikes.

A defensible claim is: "On device X, batch 1, resolution 224, Swin with window 7 meets p99 20 ms and loses Y pp accuracy relative to ViT; MAE pretraining improves accuracy by Z pp at the same latency." "Use MAE" alone is not an answer.

**Common mistake.** Treating 20 ms as a soft target or assuming MACs predict latency. Another mistake: evaluating only average latency and ignoring p99/thermal behavior.

### Q6.4 Validation accuracy dropped from 0.82 to 0.76 after a code change. Give two hypotheses and a minimal experiment for each.

**Answer.** Two hypotheses, each with a single-variable experiment:

**H1: The data/evaluation path changed (split, normalization, labels, transform, metric).**
- Experiment: freeze the model and checkpoint; evaluate the same checkpoint with the old and new data pipeline on the same 1,000 samples. Compare logits, predictions, and accuracy. If they differ, the bug is in data/eval, not training.
- Also: verify the normalization constants, class-to-index mapping, and that the test split is not accidentally used for training or model selection.

**H2: The optimization/training path changed (LR schedule, accumulation, AMP, seed, loss).**
- Experiment: run a tiny-subset overfit test (e.g., 450 images) with the new code. If it cannot reach near-100% training accuracy, inspect the training loop. If it can, compare a short run (2 epochs) against the previous commit with the same seed and diff `metrics.jsonl` epoch by epoch.
- Also: log the LR actually used, gradient norm, and loss scale; check that `scheduler.step()` and `optimizer.step()` happen at the same cadence as before.

If both pass, use `git bisect` or a diff of configs to isolate the change, and compare intermediate activations or parameter norms at initialization. The minimal experiment must change **one** variable at a time; otherwise the drop cannot be attributed.

**Common mistake.** Immediately tuning hyperparameters to recover accuracy. That can mask the bug and make the result unreproducible. Another mistake: changing several things at once and then guessing.

### Q6.5 An AI-generated training script reports 90% accuracy. How do you verify it before believing the result?

**Answer.** Treat the number as a claim to audit, not a result.

1. **Tiny-subset overfit**: train on 1% of the data with augmentation/regularization off. It should reach near-100% training accuracy. If not, the pipeline/training loop is broken.
2. **Reference comparison**: compare the model or a component against a known implementation on random tensors and copied weights. Check shapes, logits, and gradients.
3. **Data/eval audit**: verify the split (no train/test overlap), label alignment, normalization statistics, transform application, and that the test set was not used for tuning or checkpoint selection.
4. **Baseline sanity**: run a trivial baseline (majority class, logistic regression, small CNN) and check that the reported number is plausible.
5. **Diff/bisect**: compare against the previous working commit; bisect the change if the number moved unexpectedly.
6. **Seed and stability**: repeat with at least 3 seeds; check variance. A single 90% run is weak evidence.
7. **Metric audit**: independently recompute accuracy/F1 from saved predictions; check class mapping and aggregation (micro vs. macro).
8. **Leakage checks**: duplicated images across splits, test-time augmentation applied during training, normalization computed on test, checkpoint selected on test.

Only after these checks should you report the number, and you should report the protocol and limitations with it.

**Common mistake.** Trusting a high number because the code "looks reasonable". Another mistake: checking only the final accuracy and not the data/eval path; most inflated results come from leakage or a broken split, not from the model.

### The four frameworks in one page

| Framework | Core question | What to write down |
|---|---|---|
| Claim ladder + threats | What exactly does the evidence support? | Observation / comparison / causal / general; seeds, baseline strength, external validity, data overlap, compute |
| Metric–decision fit | Does the metric match the decision? | Operating point, coverage, selective risk, calibration, cost of errors vs. abstention |
| Architecture vs. method | What is being deployed, and what is a hard constraint? | Pretraining method vs. architecture; latency/memory/operator constraints; measured on target hardware |
| Diagnosis + AI-code verification | What is the smallest experiment that distinguishes hypotheses? | Two hypotheses, one variable per experiment, tiny-subset/reference/audit/bisect |

---

## 中文

### Q6.1 “MAE 普遍优于从零训练——我们的 +9.31 pp 实验证明了这一点。”你同意吗？

**答案.** 不同意。这个实验支持的 claim 窄得多：

> 在 CIFAR-10 上，使用同一批 45,000 张训练图，这个 MAE 预训练配方（100 epoch、mask 0.75、`norm_pix_loss`）加 100 epoch 微调，在 10,000 张 test split 上比这个特定的从零训练 ViT 基线高 9.31 pp（McNemar p = 2.92e-141，bootstrap 95% CI [+8.59, +10.03] pp）。

这是一个特定协议下的**比较** claim，不是一般化或因果 claim。要往上走，需要处理：

- **Seed 方差**：每个配置一次运行；CI 覆盖的是条件于两个已训练模型的 test 样本噪声，不是初始化/数据顺序方差。
- **基线强度**：从零基线是 100 epoch 的 ViT-tiny，配方不算强。更强基线（更长训练、更好增强、更多调参）可能缩小差距。
- **算力公平**：MAE 得到 100 epoch 预训练 + 100 epoch 微调；从零基线只有 100 epoch。要声称方法优势，先匹配总算力。
- **数据重叠 / 外部效度**：无标签预训练图就是同一批 CIFAR-10 训练图。不能据此声称能从不同或更大的无标签池迁移。
- **规模**：32×32、64 token、16 个可见 token。ImageNet 规模的结论需要 ImageNet 规模实验。
- **指标范围**：accuracy 只是一个终点；校准、鲁棒性、下游迁移是不同 claim。

**Claim ladder.**

```text
观察：      MAE 微调 test acc = 0.9087；基线 = 0.8156。
比较：      在这个 split 与协议下，MAE 高 9.31 pp。
因果：      增益来自预训练表示，而不是额外容量（参数量相同），
            但算力与调参不同。
一般化：    MAE 是普遍更好的视觉预训练方法。
```

实验支持第 1–2 层，第 3 层需要在 caveat 下部分支持，**不**支持第 4 层。

**常见误解.** 把 McNemar 的显著 p 值当成普遍优越的证据。McNemar 回答的是“这两个已训练模型在这个 test set 上是否不同”，不是“这个方法是否普遍更好”。

### Q6.2 需要一个能弃权、把不确定样本交给人的分类器。应该优化什么指标？

**答案.** 用 **risk–coverage** 与 **selective accuracy**，不能只看 ROC-AUC。

- 按置信度给样本排序（如 max softmax、margin、ensemble disagreement）。
- 对每个 coverage 水平 `c`（模型回答的样本比例），计算 covered set 上的 selective accuracy 与 selective risk（错误率）。
- 画 risk vs. coverage，与随机弃权及简单基线比较。
- 报告部署工作点：coverage、selective accuracy、被复核样本中的错误率、人工复核预算。
- 如果关心校准，报告 reliability diagram、多个分箱数的 ECE、NLL、Brier。ECE 对分箱敏感；NLL/Brier 是 proper scoring rule。

ROC-AUC 衡量所有阈值上的排序，并把假阳性与假阴性对称对待。它不告诉你实际弃权阈值下的错误率。如果错答成本远高于弃权成本，应在选定工作点优化 risk–coverage 取舍。

**常见误解.** 报告“AUC 0.99，所以可以安全地用弃权部署”。AUC 可以很高，而 80% coverage 下的 selective risk 仍不可接受。另一个错误：在 test set 上选阈值；应在 validation 上选，test 只报告一次。

### Q6.3 团队要在边缘设备上达到 20 ms 延迟，说“用 MAE”。这个问题哪里不对？如何决策？

**答案.** MAE 是**预训练方法**，不是可部署架构。可以用 MAE 预训练，然后部署 ViT、Swin 或其他 encoder；decoder 丢弃。真正的问题是：在目标硬件上，哪种架构与输入配置能满足**硬性** 20 ms 约束，以及会损失多少精度？

决策流程：

1. 固定部署约束：延迟分位数（p50/p99）、batch size（常为 1）、分辨率、显存、支持的算子、量化。
2. 在目标设备上实测延迟，不要用理论 MACs。MACs 只是代理；小 batch 下，内存带宽、kernel launch 开销、算子支持往往主导。
3. 比较候选：ViT（全局 `O(N²)` attention）、Swin（窗口 `O(N)` attention、层级）、混合/CNN 基线，以及不同 patch/分辨率。
4. 在相同预训练/微调预算下测精度，再做量化/剪枝，重新测延迟与精度。
5. 说明失败风险：量化掉点、算子不支持、热降频、p99 延迟尖峰。

站得住脚的 claim 是：“在设备 X、batch 1、分辨率 224 下，窗口 7 的 Swin 满足 p99 20 ms，相对 ViT 掉 Y pp 精度；MAE 预训练在相同延迟下提升 Z pp。” 只说“用 MAE”不是答案。

**常见误解.** 把 20 ms 当软目标，或用 MACs 预测延迟。另一个错误：只看平均延迟，忽略 p99/热行为。

### Q6.4 改代码后验证准确率从 0.82 掉到 0.76。给出两个假设，以及每个假设的最小实验。

**答案.** 两个假设，各配一个单变量实验：

**H1：数据/评估路径变了（split、归一化、标签、transform、指标）。**
- 实验：固定模型与 checkpoint；用旧数据管线与新数据管线在同一批 1,000 个样本上评估。比较 logits、预测与准确率。若不同，bug 在数据/评估，不在训练。
- 另外：核对归一化常数、类别到下标的映射，确认 test split 没有意外参与训练或模型选择。

**H2：优化/训练路径变了（LR schedule、累积、AMP、seed、loss）。**
- 实验：用新代码跑 tiny-subset 过拟合（如 450 张图）。如果连训练集都到不了接近 100%，检查训练循环。如果能，用同一 seed 跑 2 epoch，与上一个 commit 逐 epoch diff `metrics.jsonl`。
- 另外：记录实际使用的 LR、梯度范数、loss scale；确认 `scheduler.step()` 与 `optimizer.step()` 的节奏与之前一致。

若两者都通过，用 `git bisect` 或配置 diff 定位改动，并比较初始化时的中间激活或参数范数。最小实验必须一次只改**一个**变量，否则无法归因。

**常见误解.** 立刻调超参把精度“调回来”。这可能掩盖 bug，并让结果不可复现。另一个错误：一次改很多地方再猜。

### Q6.5 一个 AI 生成的训练脚本报告 90% 准确率。相信之前如何验证？

**答案.** 把数字当成待审计的 claim，而不是结果。

1. **Tiny-subset 过拟合**：用 1% 数据、关闭增强/正则训练，train acc 应接近 100%。如果不行，管线/训练循环有 bug。
2. **参考对照**：在随机张量、复制权重下，把模型或组件与已知实现比较。检查 shape、logits、梯度。
3. **数据/评估审计**：核对 split（无 train/test 重叠）、标签对齐、归一化统计量、transform 应用位置，确认 test 没有参与调参或 checkpoint 选择。
4. **基线 sanity**：跑一个平凡基线（多数类、逻辑回归、小 CNN），检查报告数字是否合理。
5. **Diff/bisect**：与上一个可用 commit 对比；数字异常变化时 bisect 改动。
6. **Seed 与稳定性**：至少 3 个 seed，检查方差。单次 90% 是弱证据。
7. **指标审计**：从保存的预测独立重算 accuracy/F1；检查类别映射与聚合方式（micro vs. macro）。
8. **泄漏检查**：跨 split 重复图、训练时误用 test-time augmentation、在 test 上算归一化、用 test 选 checkpoint。

完成这些检查后再报告数字，并同时报告协议与局限。

**常见误解.** 因为代码“看起来合理”就相信高数字。另一个错误：只检查最终 accuracy，不查数据/评估路径；大多数虚高结果来自泄漏或 split 错误，而不是模型。

### 四个框架一页总结

| 框架 | 核心问题 | 要写下的内容 |
|---|---|---|
| Claim ladder + threats | 证据到底支持什么？ | 观察 / 比较 / 因果 / 一般化；seed、基线强度、外部效度、数据重叠、算力 |
| Metric–decision fit | 指标是否匹配决策？ | 工作点、coverage、selective risk、校准、错答 vs. 弃权成本 |
| Architecture vs. method | 部署的是什么，什么是硬约束？ | 预训练方法 vs. 架构；延迟/显存/算子约束；在目标硬件实测 |
| Diagnosis + AI-code verification | 区分假设的最小实验是什么？ | 两个假设、一次一个变量、tiny-subset/reference/audit/bisect |
