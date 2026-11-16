# S3 — Training, Debugging, and Evaluation / 训练、调试与评估

[English](#english) · [中文](#中文)

---

## English

### Q3.1 Why use a stratified train/validation split?

**Answer.** CIFAR-10 has 50,000 training images, 5,000 per class. `stratified_split_indices` shuffles each class separately and takes about 10% for validation, giving 4,500 train / 500 validation images per class (45,000 / 5,000 total). This keeps the class distribution identical in both splits, so validation accuracy is not affected by random class imbalance. The test split (10,000 images) is never used for training or model selection.

**Common mistake.** Splitting the concatenated dataset without stratification. With 10 classes and a 10% validation fraction, random splits can still be imbalanced by chance, especially on smaller datasets. Another mistake: tuning on the test split.

### Q3.2 When are normalization statistics computed, and why does the timing matter?

**Answer.** The constants are computed from the **original training images before augmentation** and then applied to train, validation, and test. Using validation or test images to compute mean/std leaks information about the evaluation distribution into training. Applying normalization after augmentation would make the statistics depend on random crops and flips, and the validation transform would not match the training transform's distribution. The repository uses the standard CIFAR-10 constants `mean=(0.4914,0.4822,0.4465)`, `std=(0.2470,0.2435,0.2616)`, and the data test verifies them against the raw training set.

**Common mistake.** Calling `Normalize` before `ToTensor`, or recomputing mean/std on the augmented training batches. Also: using the test set "just to check" the constants.

### Q3.3 What is the actual translation range of `RandomCrop(32, padding=4)`?

**Answer.** The image is padded to 40×40, then a 32×32 crop is taken. The crop's top-left corner can move from 0 to 8 in the padded image, which corresponds to an offset from −4 to +4 relative to the original image. The **maximum shift is ±4 pixels**; the **total span is 8 pixels**. Do not confuse the two. With `padding_mode="reflect"`, the border is mirrored; with the default constant mode, the border is zero (black), which is absent at test time because the test transform has no crop.

**Common mistake.** Saying "up to 8 pixels of shift". The crop can move by 8 positions in total, but relative to the original origin the maximum offset is 4.

### Q3.4 How does DataLoader randomness work with multiple workers?

**Answer.** Two different random processes are involved:

1. **Batch order** is decided by the sampler in the main process, seeded by the DataLoader `generator`. Changing `num_workers` does not change the order when the generator seed is fixed.
2. **Augmentation randomness** happens inside workers. PyTorch gives each worker a torch seed `base_seed + worker_id`; `_seed_worker` additionally seeds NumPy and Python with that value. With `num_workers=0`, transforms use the main process's global RNG, and `worker_init_fn` is not called.

`scripts/check_dataloader_rng.py` demonstrates this: with the same generator seed, `num_workers=0/2/4` produce the same batch order but different augmentation values.

**Common mistake.** Believing workers "race" for random numbers. They have independent, deterministically seeded RNGs. Another mistake: expecting the augmentation stream to be identical when `num_workers` changes; it is not, because worker seeds differ.

### Q3.5 What is the difference between AdamW and Adam + L2?

**Answer.** Adam + L2 adds `λθ` to the gradient, which is then divided by `sqrt(v̂)`, so the effective decay is `lr · λθ / sqrt(v̂)` and varies per parameter. AdamW decouples the decay: `θ ← θ − lr · (m̂/(sqrt(v̂)+eps)) − lr · λθ`. The decay is applied directly to the parameter, independent of the adaptive denominator. This is the "decoupled weight decay" in the AdamW paper. In this repository, matrices get `weight_decay=0.05`; biases and LayerNorm `gamma/beta` get 0. Decaying 1-D parameters is not an effective regularizer and can shrink the scale/shift that LayerNorm needs.

**Common mistake.** Saying "AdamW is just Adam with L2". The difference is where the decay enters. Another mistake: applying weight decay to LayerNorm parameters and then compensating with a larger learning rate.

### Q3.6 How do warmup, cosine decay, and step counting work?

**Answer.** For the first `warmup_steps` optimizer steps, the LR rises linearly from `base_lr/warmup_steps` to `base_lr`; then it follows a cosine curve to `min_lr`. The schedule is stepped once per **optimizer step**, not per micro-batch. With `N` training samples, batch `B`, and accumulation `A`:

```text
micro_per_epoch = ceil(N / B)
opt_per_epoch   = ceil(micro_per_epoch / A)
total_steps     = epochs * opt_per_epoch
warmup_steps    = warmup_epochs * opt_per_epoch
```

The scheduler is not reset per epoch. Warmup controls how quickly the peak LR is reached; it does **not** make an unsafe peak LR safe. In the course experiment, `lr=3e-3, warmup=4` only delayed the instability from epoch 1 to epochs 3–4; the peak LR was still too high, and the fix was to lower it to `1e-3`.

**Common mistake.** Calling `scheduler.step()` inside the micro-batch loop, or resetting the scheduler every epoch. Another mistake: thinking warmup solves divergence instead of only delaying it.

### Q3.7 What does label smoothing do, and why does it not change the argmax?

**Answer.** With smoothing `ε` and `K` classes, the target distribution is `q = (1−ε)·onehot + ε/K`. The loss is

```text
L = −(1−ε)·log p_y − ε·mean_k(log p_k)
```

which is equivalent to cross-entropy against `q`. At inference, `argmax_k p_k` is unchanged because the same `ε/K` is added to every class's target; the ordering of the logits/probabilities is unaffected. Label smoothing reduces overconfidence, changes the calibration target, and puts a positive floor under the loss, so the model cannot drive NLL to zero. The repository supports `smoothing=0` for the overfitting diagnostic and `0.1` for the baseline.

**Common mistake.** Saying label smoothing "changes the decision boundary" or "changes the argmax". It changes the training target and confidence calibration, not the class ranking at inference. Another mistake: using label-smoothed loss as if it were the hard-label NLL when reporting NLL; `eval_metrics.py` reports both `loss_smoothed` and `loss_hard`.

### Q3.8 What does gradient clipping actually control?

**Answer.** `clip_gradients` computes the global L2 norm over all parameter gradients and, if it exceeds `max_norm`, rescales all gradients by `max_norm / norm`. It controls the **global norm**, not the direction of the update or the balance between layers. It cannot fix Post-LN's per-layer Jacobian scaling or a bad LR; it only prevents extremely large steps. With AMP, call `scaler.unscale_(optimizer)` first, otherwise the norm is measured on scaled gradients and the effective threshold is `max_norm / S`. In this repository, `optimizer_step` performs unscale → clip → `scaler.step` → `scaler.update`.

**Common mistake.** Thinking clipping makes Post-LN as stable as Pre-LN. Clipping bounds the norm; it does not repair the gradient path. Another mistake: clipping before unscaling in fp16 AMP.

### Q3.9 When does AMP help, and what is the difference between bf16 and fp16?

**Answer.** bf16 has 8 exponent bits and 7 mantissa bits; fp16 has 5 exponent bits and 10 mantissa bits. bf16 has a wide range but coarse precision; fp16 has finer precision but overflows/underflows sooner and needs `GradScaler`. AMP helps when the workload is **compute-bound**: larger batches, larger matrices, enough work per kernel launch. It can hurt when the workload is **launch-bound**, because autocast adds cast kernels and the GPU is dominated by launch overhead. In the course run at batch 32, fp32/bf16/fp16 were 42.19/46.19/44.62 ms per step; at batch 128 they were 142.28/79.03/76.15 ms. The larger batch became compute-bound and AMP paid off. On MPS/CPU, `--amp auto` stays fp32.

**Common mistake.** Assuming AMP always speeds up training. Measure `ms/step` and peak memory at your actual batch size, and check whether the GPU is launch-bound.

### Q3.10 What makes a training run reproducible, and what are the limits?

**Answer.** The repository combines: `set_seed` for Python/NumPy/PyTorch, deterministic cuDNN flags on CUDA, a seeded DataLoader generator, `worker_init_fn` for worker RNGs, fixed data splits, and deterministic evaluation. Two runs with the same seed produce identical `metrics.jsonl` values; `scripts/compare_metrics.py` verifies this. Limits: some CUDA kernels are nondeterministic, different hardware/driver versions can change numerics, and changing the number of workers changes the augmentation stream. Reproducibility is "same code, same seed, same environment", not "same number on every machine".

**Common mistake.** Comparing two runs with different seeds and calling the difference a bug. Another mistake: assuming `torch.manual_seed` alone is enough; data workers need their own seeding.

### Q3.11 What does the 1% overfitting diagnostic prove?

**Answer.** It samples about 450 training images (45 per class), disables augmentation, weight decay, label smoothing, and dropout, and trains on the same images it evaluates. If the model cannot reach near-100% training accuracy on 450 images, the bug is likely in the pipeline or training loop (label mismatch, wrong normalization, missing `zero_grad`, a scheduler that decays to zero too early), not in model capacity. In the course run it reached 99.11% train/eval accuracy at epoch 133. This proves the pipeline **can learn**; it does not prove generalization.

**Common mistake.** Treating a high 1% accuracy as evidence that the full model generalizes. The evaluation set is the training set. Another mistake: leaving augmentation on, which makes the diagnostic much harder to interpret.

### Q3.12 How do you distinguish underfitting, optimization instability, and divergence from curves?

**Answer.**

- **Underfitting**: training and validation loss decrease smoothly and are still decreasing at the end; accuracy is low but stable. Typical causes: LR too small, too few epochs, model too small.
- **Instability**: loss rises or oscillates while the LR is high, then partially recovers as the schedule decays. Typical cause: peak LR too large.
- **Divergence**: loss becomes NaN/inf or grows without bound. Typical causes: LR far too large, numerical overflow, bad loss.

In the LR scan, `3e-4` was best, `1e-3` was stable, and `3e-3`/`1e-2` showed a loss increase at epoch 0→1 followed by slow recovery — instability, not underfitting. A warmup=4 run only delayed the instability; the peak LR was still too high.

**Common mistake.** Calling every bad curve "underfitting". Check whether the loss is still decreasing smoothly or whether it jumped when the LR peaked.

### Q3.13 Explain accuracy, micro/macro/weighted F1, ROC-AUC, PR-AUC, NLL/Brier, and ECE.

**Answer.**

- **Accuracy** is the fraction of correct predictions. For single-label multi-class, it equals micro-F1 (micro-P = micro-R = micro-F1 = accuracy).
- **Macro-F1** averages per-class F1 equally; **weighted-F1** weights by class support. Macro is sensitive to rare classes; weighted is dominated by frequent classes.
- **ROC-AUC (OvR)** treats each class as one-vs-rest. Macro averages the per-class AUCs; micro pools all class decisions. AUC measures ranking quality across thresholds, not calibration.
- **PR-AUC / average precision** focuses on the positive class and is more informative when positives are rare; its baseline is the prevalence, not 0.5.
- **NLL** and **Brier** are proper scoring rules: they reward both discrimination and calibration. **ECE** is a binned summary of |accuracy − confidence| and depends on the number and width of bins. ECE and NLL can rank models differently.
- All of these are nonlinear over the dataset: collect all logits/labels first, then compute. Do not average per-batch F1 or ECE.

**Common mistake.** Using ROC-AUC as a selective-prediction metric, or assuming a lower ECE always means a better model. Also: reporting micro-F1 as if it were a different number from accuracy.

### Q3.14 What do McNemar and bootstrap tell you, and what do they not tell you?

**Answer.** McNemar's test uses the two models' paired predictions on the **same test samples**. It counts `b` (A correct, B wrong) and `c` (A wrong, B correct), and tests whether the discordant counts are symmetric. The paired bootstrap resamples test samples, computes the accuracy difference for each resample, and reports a percentile interval. Both are conditional on the two trained models and quantify test-sample uncertainty. They do **not** capture seed-to-seed variance from initialization, data order, or augmentation. When comparing more than two models, apply Holm correction to the p-values.

**Common mistake.** Saying "the 95% CI excludes zero, therefore the method is generally better". The CI is about this test set and these two trained models. Another mistake: treating McNemar as a test of seed robustness.

### Q3.15 Why does the training loop read the LR before `optimizer_step`?

**Answer.** `optimizer_step` calls `scheduler.step()` after `optimizer.step()`. If you read `optimizer.param_groups[0]["lr"]` after that, you see the LR for the **next** update. At the end of training, there may be no next update, so the last epoch would be logged as LR 0. The loop saves `step_lr` at the start of each micro-batch and records `last_lr` when the optimizer step happens; `fit` uses `train_metrics["lr"]`. `tests/test_train.py` has a regression test for exactly this off-by-one.

**Common mistake.** Reading the LR after `optimizer_step` and interpreting the final 0 as a scheduler bug or as "the model stopped learning".

### Q3.16 How does gradient accumulation change the optimization?

**Answer.** Each micro-batch computes its own forward/backward; gradients accumulate. The loss for each micro-batch is weighted by `1 / group_size`, where `group_size` is the number of micro-batches in the current accumulation group (the last group may be smaller). The optimizer and scheduler step once per group, not once per micro-batch. With batch 32 and accumulation 4, the effective batch is 128, but the forward passes are still separate: dropout masks, augmentation, and normalization statistics are computed per micro-batch. This is close to, but not bit-identical with, a single batch of 128.

**Common mistake.** Stepping the optimizer every micro-batch, or forgetting to scale the loss and thus multiplying the effective LR by the accumulation factor.

---

## 中文

### Q3.1 为什么要按类别分层划分 train/validation？

**答案.** CIFAR-10 有 50,000 张训练图，每类 5,000 张。`stratified_split_indices` 对每个类别单独打乱，取约 10% 做验证，得到每类 4,500 train / 500 validation（总计 45,000 / 5,000）。这样两个 split 的类别分布一致，验证准确率不会被随机的类别不平衡影响。test split（10,000 张）从不参与训练或模型选择。

**常见误解.** 把拼接后的数据集直接随机划分。10 类、10% 验证时，随机划分仍可能偶然不平衡，数据更小时更明显。另一个错误：在 test split 上调参。

### Q3.2 归一化统计量在什么时候计算？为什么时机重要？

**答案.** 常数来自**增强前的原始训练图像**，然后应用到 train/validation/test。用验证或测试图像计算 mean/std 会把评估分布的信息泄漏进训练。在增强后计算会让统计量依赖随机裁剪/翻转，而且验证 transform 的分布与训练 transform 不一致。仓库使用标准 CIFAR-10 常数 `mean=(0.4914,0.4822,0.4465)`、`std=(0.2470,0.2435,0.2616)`，数据测试会用原始训练集核对它们。

**常见误解.** 在 `ToTensor` 之前调用 `Normalize`；或在增强后的训练 batch 上重新计算 mean/std。另一个错误：“只是用 test 集检查一下”常数。

### Q3.3 `RandomCrop(32, padding=4)` 的实际平移范围是多少？

**答案.** 图像先 padding 到 40×40，再裁 32×32。裁剪框左上角在 padded 图中可从 0 移到 8，对应相对原图原点的偏移 −4 到 +4。**最大偏移是 ±4 像素**；**总跨度是 8 像素**。不要混淆两者。`padding_mode="reflect"` 镜像边界；默认 constant 是零（黑边），测试时没有裁剪，所以不会出现黑边。

**常见误解.** 说“最多平移 8 像素”。裁剪位置总共有 8 个跨度，但相对原图原点的最大偏移是 4。

### Q3.4 多 worker 的 DataLoader 随机性如何工作？

**答案.** 涉及两个不同的随机过程：

1. **取批顺序**由主进程的 sampler 决定，使用 DataLoader `generator` 播种。generator seed 固定时，改变 `num_workers` 不会改变顺序。
2. **增强随机性**发生在 worker 内。PyTorch 给每个 worker 的 torch seed 是 `base_seed + worker_id`；`_seed_worker` 再用该值给 NumPy 与 Python 播种。`num_workers=0` 时，transform 使用主进程全局 RNG，`worker_init_fn` 不会被调用。

`scripts/check_dataloader_rng.py` 演示了这一点：同一个 generator seed 下，`num_workers=0/2/4` 的取批顺序相同，但增强值不同。

**常见误解.** 以为 worker “抢随机数”。它们有独立、确定播种的 RNG。另一个误解：期待改变 `num_workers` 后增强序列完全一致；不会，因为 worker seed 不同。

### Q3.5 AdamW 与 Adam + L2 有什么区别？

**答案.** Adam + L2 把 `λθ` 加到梯度上，随后被 `sqrt(v̂)` 除，所以有效衰减是 `lr · λθ / sqrt(v̂)`，随参数变化。AdamW 解耦衰减：`θ ← θ − lr · (m̂/(sqrt(v̂)+eps)) − lr · λθ`。衰减直接作用于参数，与自适应分母无关。这就是 AdamW 论文里的 decoupled weight decay。本仓库中矩阵权重 `weight_decay=0.05`；bias 与 LayerNorm `gamma/beta` 为 0。对 1-D 参数做 decay 不是有效正则，还可能缩小 LayerNorm 需要的缩放/平移。

**常见误解.** 说“AdamW 就是带 L2 的 Adam”。差别在衰减进入的位置。另一个错误：对 LayerNorm 参数做 weight decay，再用更大的 LR 补偿。

### Q3.6 warmup、cosine 与 step 计数如何工作？

**答案.** 前 `warmup_steps` 个 optimizer step，LR 从 `base_lr/warmup_steps` 线性升到 `base_lr`；之后按 cosine 降到 `min_lr`。调度器每个 **optimizer step** 前进一次，不是每个 micro-batch。设训练样本 `N`、batch `B`、累积 `A`：

```text
micro_per_epoch = ceil(N / B)
opt_per_epoch   = ceil(micro_per_epoch / A)
total_steps     = epochs * opt_per_epoch
warmup_steps    = warmup_epochs * opt_per_epoch
```

调度器不按 epoch 重置。warmup 控制到达峰值 LR 的速度，**不能**让不安全的峰值 LR 变安全。课程实验中 `lr=3e-3, warmup=4` 只把不稳定从 epoch 1 推迟到 epoch 3–4；峰值 LR 仍然过高，修复是降到 `1e-3`。

**常见误解.** 在 micro-batch 循环里调用 `scheduler.step()`，或每个 epoch 重置调度器。另一个误解：以为 warmup 能解决发散，而不只是推迟它。

### Q3.7 Label smoothing 做什么？为什么它不改变 argmax？

**答案.** 设平滑系数 `ε`、类别数 `K`，目标分布是 `q = (1−ε)·onehot + ε/K`。Loss 为

```text
L = −(1−ε)·log p_y − ε·mean_k(log p_k)
```

等价于对 `q` 做交叉熵。推理时 `argmax_k p_k` 不变，因为每个类的目标都加了相同的 `ε/K`，logits/概率排序不受影响。Label smoothing 降低过度自信、改变校准目标，并给 loss 一个正的下界，所以 NLL 不能降到 0。仓库在过拟合诊断中用 `smoothing=0`，基线用 `0.1`。

**常见误解.** 说 label smoothing “改变决策边界”或“改变 argmax”。它改变训练目标与置信度校准，不改变推理时的类别排序。另一个错误：把 label-smoothed loss 当作硬标签 NLL 报告；`eval_metrics.py` 同时报告 `loss_smoothed` 与 `loss_hard`。

### Q3.8 梯度裁剪到底控制什么？

**答案.** `clip_gradients` 计算所有参数梯度的全局 L2 范数，若超过 `max_norm`，把所有梯度按 `max_norm / norm` 缩放。它控制的是**全局范数**，不是更新方向，也不是层间平衡。它不能修复 Post-LN 的逐层 Jacobian 缩放或错误的 LR；只能防止极端大的 step。AMP 下必须先 `scaler.unscale_(optimizer)`，否则在缩放后的梯度上测范数，有效阈值变成 `max_norm / S`。本仓库的 `optimizer_step` 顺序是 unscale → clip → `scaler.step` → `scaler.update`。

**常见误解.** 以为裁剪能让 Post-LN 和 Pre-LN 一样稳。裁剪约束范数，不修复梯度路径。另一个错误：fp16 AMP 下在 unscale 之前裁剪。

### Q3.9 AMP 什么时候有效？bf16 与 fp16 有什么区别？

**答案.** bf16 有 8 位指数、7 位尾数；fp16 有 5 位指数、10 位尾数。bf16 范围宽但精度粗；fp16 精度细但更容易溢出/下溢，需要 `GradScaler`。AMP 在 **compute-bound** 时有效：更大的 batch、更大的矩阵、每次 kernel launch 有足够多的工作。**launch-bound** 时可能更慢，因为 autocast 插入 cast kernel，GPU 主要被 launch 开销占据。课程中 batch 32 时 fp32/bf16/fp16 为 42.19/46.19/44.62 ms/step；batch 128 时为 142.28/79.03/76.15 ms。大 batch 变成 compute-bound，AMP 才获益。MPS/CPU 上 `--amp auto` 保持 fp32。

**常见误解.** 以为 AMP 总能加速。要在实际 batch size 下测 `ms/step` 与峰值显存，并判断是否 launch-bound。

### Q3.10 训练可复现靠什么？局限是什么？

**答案.** 仓库组合了：`set_seed` 设置 Python/NumPy/PyTorch，CUDA 上确定性 cuDNN 标志，带种子的 DataLoader generator，`worker_init_fn` 给 worker RNG 播种，固定数据划分，确定性评估。同 seed 两次运行会产生完全相同的 `metrics.jsonl`；`scripts/compare_metrics.py` 可验证。局限：部分 CUDA kernel 非确定；不同硬件/驱动会改变数值；改变 worker 数会改变增强流。可复现性是“同代码、同 seed、同环境”，不是“每台机器同一个数”。

**常见误解.** 比较不同 seed 的两次运行，把差异当 bug。另一个错误：以为只设 `torch.manual_seed` 就够；数据 worker 需要单独播种。

### Q3.11 1% 过拟合诊断证明了什么？

**答案.** 它抽取约 450 张训练图（每类 45 张），关闭增强、weight decay、label smoothing、dropout，并在同一批图上训练与评估。如果模型连 450 张图都记不住，bug 很可能在数据管线或训练循环（标签错位、归一化错、忘记 `zero_grad`、scheduler 过早衰减到 0），而不是模型容量。课程中在第 133 epoch 达到 99.11% train/eval accuracy。它证明管线**能学习**；不证明泛化。

**常见误解.** 把 1% 上的高准确率当作完整模型泛化的证据。评估集就是训练集。另一个错误：忘记关闭增强，导致诊断难以解释。

### Q3.12 如何从曲线区分 underfit、优化不稳定与发散？

**答案.**

- **Underfit**：train/val loss 平滑下降，结束时仍在下降；准确率低但稳定。常见原因：LR 太小、epoch 太少、模型太小。
- **不稳定**：LR 高时 loss 上升或震荡，随后随 schedule 衰减部分恢复。常见原因：峰值 LR 太大。
- **发散**：loss 变成 NaN/inf 或无界增长。常见原因：LR 极大、数值溢出、loss 错误。

LR 扫描中 `3e-4` 最好，`1e-3` 稳定，`3e-3`/`1e-2` 在 epoch 0→1 loss 上升后缓慢恢复——这是不稳定，不是 underfit。warmup=4 只推迟不稳定；峰值 LR 仍过高。

**常见误解.** 把所有坏曲线都叫 underfit。要看 loss 是平滑继续下降，还是在 LR 峰值时跳变。

### Q3.13 解释 accuracy、micro/macro/weighted F1、ROC-AUC、PR-AUC、NLL/Brier、ECE。

**答案.**

- **Accuracy** 是正确预测比例。单标签多分类中等于 micro-F1（micro-P = micro-R = micro-F1 = accuracy）。
- **Macro-F1** 对每类 F1 等权平均；**weighted-F1** 按类 support 加权。Macro 对稀有类敏感；weighted 被高频类主导。
- **ROC-AUC（OvR）** 把每类当作 one-vs-rest。Macro 平均各类 AUC；micro 把所有类别决策拼在一起。AUC 衡量跨阈值的排序质量，不是校准。
- **PR-AUC / average precision** 关注正类，正类稀少时更有信息量；基线是 prevalence，不是 0.5。
- **NLL** 与 **Brier** 是 proper scoring rule：同时奖励区分度与校准。**ECE** 是 |accuracy − confidence| 的分箱摘要，依赖箱数与箱宽。ECE 与 NLL 可能给出不同的模型排序。
- 这些都是数据集上的非线性指标：先收集全部 logits/labels，再计算。不要按 batch 平均 F1 或 ECE。

**常见误解.** 用 ROC-AUC 做 selective prediction；或认为 ECE 更低一定模型更好。另一个错误：把 micro-F1 当成与 accuracy 不同的数字报告。

### Q3.14 McNemar 与 bootstrap 告诉你什么？不告诉你什么？

**答案.** McNemar 使用两个模型在**同一批 test 样本**上的配对预测，统计 `b`（A 对 B 错）与 `c`（A 错 B 对），检验不一致计数是否对称。配对 bootstrap 重采样 test 样本，计算每次的 accuracy 差，给出百分位区间。两者都条件于两个已训练模型，量化的是 test 样本不确定性。它们**不**覆盖初始化、数据顺序、增强带来的 seed 间方差。比较两个以上模型时，对 p 值做 Holm 校正。

**常见误解.** 说“95% CI 不含 0，所以方法普遍更好”。CI 只关于这个 test set 与这两个已训练模型。另一个错误：把 McNemar 当成 seed 稳健性检验。

### Q3.15 为什么训练循环在 `optimizer_step` 之前读 LR？

**答案.** `optimizer_step` 在 `optimizer.step()` 之后调用 `scheduler.step()`。若之后读 `optimizer.param_groups[0]["lr"]`，看到的是**下一次**更新的 LR。训练结束时可能没有下一次更新，最后一个 epoch 会被记为 LR 0。循环在每个 micro-batch 开头保存 `step_lr`，在 optimizer step 发生时记录 `last_lr`；`fit` 使用 `train_metrics["lr"]`。`tests/test_train.py` 有专门针对这个 off-by-one 的回归测试。

**常见误解.** 在 `optimizer_step` 之后读 LR，把最后的 0 当成 scheduler bug 或“模型停止学习”。

### Q3.16 梯度累积如何改变优化？

**答案.** 每个 micro-batch 各自前向/反向，梯度累加。每个 micro-batch 的 loss 按 `1 / group_size` 加权，`group_size` 是当前累积组的 micro-batch 数（最后一组可能更小）。optimizer 与 scheduler 每组只 step 一次，不是每个 micro-batch。batch 32、累积 4 时有效 batch 是 128，但前向仍是分开的：dropout mask、增强、归一化统计量按 micro-batch 计算。这接近单个 batch 128，但不是逐位相同。

**常见误解.** 每个 micro-batch 都 step optimizer；或忘记缩放 loss，导致有效 LR 乘以累积倍数。
