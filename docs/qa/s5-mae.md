# S5 — Masked Autoencoders / 掩码自编码器

[English](#english) · [中文](#中文)

---

## English

### Q5.1 What is MAE, and why does masked reconstruction work for images?

**Answer.** MAE is a self-supervised pretraining method: mask a large fraction of image patches, encode only the visible patches, and train a decoder to reconstruct the masked pixels. No labels are used. Images are highly redundant in space, so a model that can predict missing regions from context learns useful visual structure. Unlike language, where masked tokens are discrete and predictions are categorical, image patches are continuous and the loss is a regression (MSE), which is why MAE predicts normalized pixel values per patch.

**Common mistake.** Calling MAE an architecture. It is a **pretraining method**: the encoder is a ViT-like Transformer, and after pretraining you discard the decoder and fine-tune the encoder as a classifier. Another mistake: thinking MAE uses labels during pretraining; it does not.

### Q5.2 Why mask 75% of the patches?

**Answer.** Masking too few patches makes the task too easy: neighboring pixels are highly correlated, so the model can interpolate or copy from visible patches without learning semantics. A high ratio (the paper uses 75%) removes most of the local evidence and forces the model to use global context. On CIFAR-10, `32×32` with patch 4 gives 64 tokens; 75% masking leaves 16 visible tokens. On STL-10 `96×96` with patch 8, 144 tokens leave 36 visible. The ratio is a difficulty/control knob, not a universal constant.

**Common mistake.** Assuming a higher mask ratio is always better. Too high a ratio leaves too few visible tokens to encode useful context; the right ratio depends on image redundancy and token count.

### Q5.3 Why is the encoder/decoder asymmetric?

**Answer.** The encoder sees only the 16 visible tokens, so its attention cost is proportional to `K = N(1−r)` rather than `N`. The decoder is lightweight (4 blocks, width 128 in this implementation) and sees all `N` slots: the encoded visible tokens plus learnable mask tokens, restored to original patch order. The asymmetry saves compute and prevents the decoder from becoming a second full encoder. After pretraining, the decoder is discarded; only the encoder is transferred to the classifier.

**Common mistake.** Sending all 64 tokens, including mask tokens, through the encoder. That removes the main compute benefit and differs from the MAE design. Another mistake: thinking the decoder is used at inference; it is not.

### Q5.4 Why compute the loss only on masked patches?

**Answer.** If the loss included visible patches, the model could learn a trivial identity/copy function for those positions and still reduce the loss, without learning to reconstruct missing content. Restricting the loss to masked patches forces the model to predict the unknown from context. In the implementation, `mae_loss` multiplies the per-patch MSE by `mask` and divides by `mask.sum()`, so visible positions receive zero gradient. The `norm_pix_loss` option changes the **target values** (per-patch normalization) but does not change **which patches** contribute; the two choices are orthogonal.

**Common mistake.** Confusing `norm_pix_loss` with the masked-only restriction. Normalization changes the regression target; the mask decides where the loss is computed. Another mistake: forgetting that the mask must be in original patch order when computing the loss.

### Q5.5 How does `random_masking` shuffle and restore tokens?

**Answer.** For `x` of shape `(B,N,D)` and `len_keep = floor(N(1−r))`:

```text
noise      = rand(B,N)
ids_shuffle = argsort(noise, dim=1)      # random permutation per row
ids_restore = argsort(ids_shuffle, dim=1) # inverse permutation
ids_keep    = ids_shuffle[:, :len_keep]
x_keep      = gather(x, dim=1, index=ids_keep expanded over D)
mask        = ones(B,N); mask.scatter_(1, ids_keep, 0)
```

`ids_restore` maps shuffled positions back to original patch indices. The decoder concatenates visible encodings and mask tokens in shuffled order, then applies `unshuffle_tokens` with `ids_restore`. `gather` requires the index tensor to have the same shape as the output, so `ids_keep` is expanded across the feature dimension.

**Common mistake.** Using `ids_shuffle` where `ids_restore` is required (or vice versa). Another mistake: passing a `(B,K)` index to `gather` without expanding it to `(B,K,D)`, or gathering along the wrong dimension.

### Q5.6 How do position embeddings interact with masking?

**Answer.** The encoder adds its position table `(1,N,D)` to all patch tokens before masking, then gathers rows with `ids_keep`, so each visible token keeps the position of its original patch. The decoder adds its own position table `(1,N,D_dec)` **after** restoring the original patch order, so each slot receives the position of the patch it represents. Adding positions after masking without gathering would misalign the visible tokens; adding decoder positions before `unshuffle` would assign positions in shuffled order.

**Common mistake.** Masking first and then adding the full position table, or forgetting `ids_keep` when selecting position rows. Another mistake: using the encoder position table for the decoder when the widths differ.

### Q5.7 What are the MAE shapes and parameter counts?

**Answer.** For CIFAR-10 `32×32`, patch 4, encoder width 192, depth 12, decoder width 128, depth 4, mask 0.75:

```text
image            (B,3,32,32)
patchify target  (B,64,48)
patch tokens     (B,64,192) + pos
random_masking   x_keep (B,16,192), mask (B,64), ids_restore (B,64)
encoder          (B,16,192)
enc_to_dec       (B,16,128)
+ mask tokens    (B,64,128)
unshuffle + pos  (B,64,128)
decoder          (B,64,128)
pred             (B,64,48)
loss             scalar
```

Parameters: 6,184,560 = encoder 5,353,536 + decoder 831,024. The encoder has no cls token; fine-tuning adds one plus a classification head, giving 5,355,850 parameters.

**Common mistake.** Reporting the pretraining parameter count as the fine-tuned model's parameter count. The decoder is discarded, and cls/head are added.

### Q5.8 How are MAE encoder weights transferred to ViT?

**Answer.** `load_mae_encoder_weights` copies the patch embedding, the patch rows of the position table (`pos_embed[:, 1:]` when cls is used), every Transformer block, and the final LayerNorm. It leaves the cls token, the cls position row, and the classification head at their new initialization. The decoder is not used. It also checks patch count, patch embedding shape, depth, and position table shape before copying.

**Common mistake.** Copying the decoder or the mask token into the ViT, or overwriting the new cls/head. Another mistake: copying the whole position table when the ViT has a cls row, which shifts every patch position by one.

### Q5.9 What does the MAE result actually show?

**Answer.** MAE pretraining (100 epochs) + fine-tuning (100 epochs) reaches 0.9087 test accuracy versus 0.8156 for the ViT trained from scratch: +9.31 pp. McNemar b=1196, c=265, p=2.92e-141; paired bootstrap 95% CI [+8.59, +10.03] pp. The fine-tuned model has the same 5,355,850 parameters as the baseline, so the gain is not from extra capacity. It comes from the pretrained representation and initialization. Limits: one seed, the unlabeled data are the same 45k CIFAR-10 training images, and the scale is small; this is not evidence that MAE universally beats supervised training.

**Common mistake.** Explaining the gain as "the model has more parameters" or "the model is more expressive". Parameter count is unchanged; capacity is not the mechanism. Another mistake: treating the McNemar/bootstrap interval as seed variance.

### Q5.10 What exactly does `norm_pix_loss` do?

**Answer.** For each target patch with `D_p = P·P·C` values, it computes the patch mean and variance and replaces the target with `(x − mean) / sqrt(var + 1e-6)`. The prediction is compared to this normalized target with MSE. This removes per-patch brightness/contrast, so the model cannot reduce the loss by only matching low-frequency intensity, and it balances patches with different scales. A constant patch becomes all zeros. It does **not** make the data Gaussian, and it does not change which patches are masked.

**Common mistake.** Saying `norm_pix_loss` "normalizes the distribution to N(0,1)". Per-patch standardization gives zero mean and unit variance but does not change the shape of the distribution. Another mistake: normalizing the prediction instead of the target in a way that breaks the loss scale.

### Q5.11 What are the common MAE failure modes?

**Answer.**

- `mask_ratio` outside `(0,1)`, or `len_keep < 1`; the code raises.
- `gather` index shape wrong, or gathering the wrong dimension.
- Using `ids_shuffle` instead of `ids_restore` when restoring order.
- Position embeddings not gathered with `ids_keep`, or decoder positions added before unshuffle.
- Loss computed over all patches instead of masked patches.
- Mask and prediction on different devices; `mae_loss` checks this.
- Forgetting that the encoder output has `K` tokens while the decoder expects `N` slots.
- Fine-tuning with the decoder still attached, or copying the wrong position rows.

**Common mistake.** Testing only the loss value. Check the mask sum, the visible-position gradient (should be zero), and the reconstruction after unshuffle; these catch most silent errors.

---

## 中文

### Q5.1 MAE 是什么？为什么 masked reconstruction 对图像有效？

**答案.** MAE 是一种自监督预训练方法：遮住大部分图像 patch，只编码可见 patch，再训练 decoder 重建被遮住的像素。预训练不使用标签。图像在空间上高度冗余，能根据上下文预测缺失区域的模型会学到有用的视觉结构。与语言不同，语言中被 mask 的 token 是离散的、预测是分类；图像 patch 是连续的，loss 是回归（MSE），所以 MAE 预测的是逐 patch 的像素值。

**常见误解.** 把 MAE 当成一种架构。它是一种**预训练方法**：encoder 是 ViT 类 Transformer，预训练后丢弃 decoder，把 encoder 微调成分类器。另一个错误：以为 MAE 预训练使用标签；它不使用。

### Q5.2 为什么 mask 75% 的 patch？

**答案.** mask 太少会让任务太简单：相邻像素高度相关，模型可以直接插值或从可见 patch 复制，不需要学语义。高比例（论文用 75%）去掉大部分局部证据，迫使模型使用全局上下文。CIFAR-10 `32×32`、patch 4 时有 64 个 token；75% mask 后剩 16 个可见 token。STL-10 `96×96`、patch 8 时 144 个 token，剩 36 个。比例是难度/控制旋钮，不是普适常数。

**常见误解.** 认为 mask 比例越高越好。比例过高会留下太少可见 token，无法编码有用上下文；合适比例取决于图像冗余度与 token 数。

### Q5.3 为什么 encoder/decoder 是非对称的？

**答案.** Encoder 只处理 16 个可见 token，attention 成本与 `K = N(1−r)` 成正比，而不是 `N`。Decoder 很轻（本实现 4 个 block、宽度 128），处理全部 `N` 个槽位：可见 token 的编码加上可学习 mask token，再还原成原始 patch 顺序。非对称设计节省计算，也避免 decoder 变成第二个完整 encoder。预训练后 decoder 丢弃，只把 encoder 迁移到分类器。

**常见误解.** 把全部 64 个 token（包括 mask token）都送进 encoder。这会去掉主要计算收益，也偏离 MAE 设计。另一个错误：以为 decoder 在推理时使用；它不使用。

### Q5.4 为什么 loss 只算 masked patch？

**答案.** 如果 loss 包含可见 patch，模型可以学到恒等/复制函数来降低这些位置的 loss，而不学习重建缺失内容。只在 masked patch 上算 loss，迫使模型从上下文预测未知。实现中 `mae_loss` 把逐 patch MSE 乘以 `mask` 再除以 `mask.sum()`，所以可见位置梯度为 0。`norm_pix_loss` 改变的是**目标值**（逐 patch 归一化），不改变**哪些 patch** 参与 loss；两者正交。

**常见误解.** 把 `norm_pix_loss` 与“只算 masked”混为一谈。归一化改变回归目标；mask 决定 loss 计算位置。另一个错误：计算 loss 时忘记 mask 是原始 patch 顺序。

### Q5.5 `random_masking` 如何打乱与还原 token？

**答案.** 对形状 `(B,N,D)` 的 `x`，`len_keep = floor(N(1−r))`：

```text
noise       = rand(B,N)
ids_shuffle = argsort(noise, dim=1)       # 每行一个随机排列
ids_restore = argsort(ids_shuffle, dim=1) # 逆排列
ids_keep    = ids_shuffle[:, :len_keep]
x_keep      = gather(x, dim=1, index=ids_keep 在 D 上 expand)
mask        = ones(B,N); mask.scatter_(1, ids_keep, 0)
```

`ids_restore` 把打乱后的位置映射回原始 patch 下标。Decoder 按打乱顺序拼接可见编码与 mask token，再用 `ids_restore` 调 `unshuffle_tokens`。`gather` 要求 index 与输出同形，所以 `ids_keep` 要在特征维上 expand。

**常见误解.** 在需要 `ids_restore` 的地方用了 `ids_shuffle`（或反过来）。另一个错误：把 `(B,K)` 的 index 直接传给 `gather` 而不 expand 到 `(B,K,D)`，或在错误的维度上 gather。

### Q5.6 位置编码与 masking 如何交互？

**答案.** Encoder 在 mask 前把位置表 `(1,N,D)` 加到所有 patch token 上，再用 `ids_keep` 取行，因此每个可见 token 保留其原始 patch 的位置。Decoder 在还原原始 patch 顺序**之后**再加自己的位置表 `(1,N,D_dec)`，因此每个槽位得到它代表的 patch 位置。mask 后不加 gather 直接加位置表会让可见 token 位置错位；在 `unshuffle` 之前加 decoder 位置表会按打乱顺序分配位置。

**常见误解.** 先 mask 再加完整位置表；或选择位置行时忘记 `ids_keep`。另一个错误：decoder 宽度不同却使用 encoder 的位置表。

### Q5.7 MAE 的 shape 与参数量是多少？

**答案.** CIFAR-10 `32×32`、patch 4、encoder 宽 192/深 12、decoder 宽 128/深 4、mask 0.75：

```text
image            (B,3,32,32)
patchify target  (B,64,48)
patch tokens     (B,64,192) + pos
random_masking   x_keep (B,16,192)、mask (B,64)、ids_restore (B,64)
encoder          (B,16,192)
enc_to_dec       (B,16,128)
+ mask tokens    (B,64,128)
unshuffle + pos  (B,64,128)
decoder          (B,64,128)
pred             (B,64,48)
loss             标量
```

参数量：6,184,560 = encoder 5,353,536 + decoder 831,024。Encoder 没有 cls token；微调时加 cls 与分类头，得到 5,355,850 参数。

**常见误解.** 把预训练参数量当成微调模型的参数量。Decoder 被丢弃，cls/head 是新加的。

### Q5.8 MAE encoder 权重如何迁移到 ViT？

**答案.** `load_mae_encoder_weights` 复制 patch embedding、位置表的 patch 行（有 cls 时 `pos_embed[:, 1:]`）、所有 Transformer block、final LayerNorm。它保留 cls token、cls 位置行与分类头的新初始化。Decoder 不使用。复制前还会检查 patch 数、patch embedding shape、深度、位置表 shape。

**常见误解.** 把 decoder 或 mask token 复制进 ViT，或覆盖新的 cls/head。另一个错误：ViT 有 cls 行时复制整个位置表，导致每个 patch 位置错一位。

### Q5.9 MAE 的结果到底说明了什么？

**答案.** MAE 预训练（100 epoch）+ 微调（100 epoch）达到 0.9087 test accuracy，从零训练 ViT 是 0.8156，+9.31 pp。McNemar b=1196、c=265、p=2.92e-141；配对 bootstrap 95% CI [+8.59, +10.03] pp。微调模型与基线参数量相同（5,355,850），所以增益不是来自额外容量，而是来自预训练表示与初始化。局限：单 seed；无标签数据就是同一批 45k CIFAR-10 训练图；规模小。这不是 MAE 普遍优于监督训练的证据。

**常见误解.** 用“参数更多”或“表达力更强”解释增益。参数量没变，容量不是机制。另一个错误：把 McNemar/bootstrap 区间当成 seed 方差。

### Q5.10 `norm_pix_loss` 具体做什么？

**答案.** 对每个目标 patch 的 `D_p = P·P·C` 个值，计算 patch 均值与方差，把目标替换为 `(x − mean) / sqrt(var + 1e-6)`。预测与这个归一化目标做 MSE。这去掉了逐 patch 的亮度/对比度，模型不能只靠匹配低频强度降低 loss，也平衡了不同尺度的 patch。常值 patch 会变成全 0。它**不**把数据变成高斯分布，也不改变哪些 patch 被 mask。

**常见误解.** 说 `norm_pix_loss` “把分布归一化到 N(0,1)”。逐 patch 标准化只保证零均值/单位方差，不改变分布形状。另一个错误：归一化预测而不是目标，破坏 loss 尺度。

### Q5.11 MAE 的常见失败模式有哪些？

**答案.**

- `mask_ratio` 不在 `(0,1)`，或 `len_keep < 1`；代码会报错。
- `gather` index shape 错，或在错误维度 gather。
- 还原顺序时用了 `ids_shuffle` 而不是 `ids_restore`。
- 位置编码没有按 `ids_keep` 取行，或 decoder 位置编码在 unshuffle 之前加。
- loss 算在所有 patch 上而不是 masked patch 上。
- mask 与预测在不同设备；`mae_loss` 会检查。
- 忘记 encoder 输出是 `K` 个 token，而 decoder 需要 `N` 个槽位。
- 微调时 decoder 还挂着，或复制了错误的位置行。

**常见误解.** 只看 loss 数值。检查 mask 总和、可见位置梯度（应为 0）、unshuffle 后的重建结果；这些能抓住大多数静默错误。
