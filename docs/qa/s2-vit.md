# S2 — Assembling the Vision Transformer / 组装 Vision Transformer

[English](#english) · [中文](#中文)

---

## English

### Q2.1 Why is patch embedding equivalent to a convolution, and when does the equivalence break?

**Answer.** A `Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P, padding=0)` slides a `P×P` window over non-overlapping patches. Each output position is a dot product between the flattened patch `(C,P,P)` and one row of the kernel, plus a bias. That is exactly a linear projection of each patch to `D` dimensions, with the same weights shared across all positions. The equivalence requires `kernel_size == stride == P` and `padding=0`, plus `H` and `W` divisible by `P`. If `stride < kernel`, patches overlap and the token grid changes; if `stride > kernel`, some pixels are skipped. Both break the "partition the image" interpretation.

**Common mistake.** Saying "convolution is different because it has locality" without noticing that a non-overlapping `P×P` convolution is just a per-patch linear layer. The locality is in the patch definition, not in the projection.

### Q2.2 Derive the token count. What is the difference between `N`, `T`, `D`, and `C·P²`?

**Answer.** For an image `H×W` and patch `P`, the grid is `(H/P) × (W/P)`, so the number of patches is `N = (H/P)(W/P)`. With a cls token the sequence length is `T = N + 1`. `C·P²` is the number of scalar values in one patch (the input width of the patch projection). `D` is the output width of the token after projection. For CIFAR-10 with `32×32`, `P=4`, `C=3`, `D=192`: `N=64`, `T=65`, `C·P²=48`, `D=192`.

**Common mistake.** Confusing `D` with `C·P²`. The projection maps 48 inputs to 192 outputs; it does not add information, it changes the feature space and capacity. Another mistake: calling `T` the number of patches when it includes cls.

### Q2.3 What is the cls token, and why prepend it?

**Answer.** The cls token is a learnable vector of shape `(1,1,D)`, expanded to `(B,1,D)` and concatenated before the patch tokens. After the blocks, its output is used as the image representation. It is shared across the batch and learned end-to-end. Prepending is a convention tied to the position-embedding table and to `pool_tokens` taking index 0; appending would also be mathematically valid if the position table and pooling index were adjusted consistently.

**Common mistake.** Thinking the cls token is a separate parameter per image. There is one vector, broadcast across the batch. Also: assuming the cls token has a special architectural role beyond being a learnable summary token; with mean pooling, it is optional.

### Q2.4 Compare learned, sinusoidal, and no position embeddings.

**Answer.** Self-attention is permutation-equivariant: without position information, shuffling tokens gives the same set of outputs. Learned absolute embeddings use a table `(1,T,D)` with `T·D` parameters; sinusoidal embeddings are a fixed buffer with no parameters; `none` is only useful as an ablation or when positions are injected elsewhere. The ViT paper found learned and sinusoidal embeddings perform similarly at scale; learned tables are simple but tied to the training resolution. Changing resolution requires interpolation or regeneration. For the baseline, the learned table costs `65×192=12,480` parameters.

**Common mistake.** Saying sinusoidal embeddings are "better because they generalize to any length" without noting that 1-D sinusoidal encodings do not encode 2-D image structure explicitly, and that resolution changes still require care.

### Q2.5 Compare cls pooling, mean pooling, and their parameter costs.

**Answer.** `cls` pooling takes token 0; `mean` averages all tokens (in this implementation, including cls if it is present). cls pooling adds the cls token parameter `D`; mean pooling adds none. Both feed the same `D→K` head. Mean pooling can work well without a cls token and may be more robust when the cls token is undertrained; cls pooling gives the model a dedicated summary slot. The repository supports both via `pool="cls"` and `pool="mean"`, and the original course listed cls+mean concatenation as a possible ablation, but it is not implemented.

**Common mistake.** Saying mean pooling is "free" while forgetting that the head input width is still `D`; or comparing cls vs. mean without retraining, since the representations are not interchangeable.

### Q2.6 Derive the ViT parameter count.

**Answer.** For width `D`, MLP width `M`, depth `L`, `K` classes, patch count `N`, and a learned position table:

```text
patch projection:   D·C·P² + D
cls token:          D                       (if pool="cls")
position table:     (N + 1)·D               (if learned and cls)
per block:
  attention:        3D² + D² + D = 4D² + D  (qkv_bias=False)
  MLP:              2DM + M + D
  two LayerNorms:   2·2D = 4D
  total:            4D² + 2DM + 6D + M
final LayerNorm:    2D
head:               D·K + K
```

For the baseline (`D=192`, `M=768`, `L=12`, `K=10`, `N=64`): patch 9,408 + cls 192 + position 12,480 + 12×444,288 + final 384 + head 1,930 = **5,355,850**. The class default `ViT(depth=6)` gives **2,690,122**; the S3 CLI default overrides depth to 12.

**Common mistake.** Forgetting the LayerNorm parameters (4D per block, 2D final), the head bias, or the position table. Another mistake: using `D²` for the patch projection when the input width is `C·P²`.

### Q2.7 How is the model initialized, and why not initialize sublayers to zero?

**Answer.** `_init_weights` applies truncated normal (`std=0.02`) to Linear/Conv weights and zeros their biases; the cls token and position tables also use truncated normal. Zero-initializing a whole sublayer would make its output zero, so its parameters receive zero gradient and the block temporarily reduces to the residual identity. Symmetry is preserved, not broken, which is exactly why learning stalls. Small random initialization breaks symmetry while keeping activations in a reasonable range.

**Common mistake.** "Zero init loses symmetry." It preserves symmetry. Also: initializing LayerNorm `gamma=1, beta=0` does **not** make LayerNorm an identity; it still normalizes and only skips the affine rescale/shift.

### Q2.8 Why does the reference alignment need special handling of patch weights?

**Answer.** `vit-pytorch` flattens each patch in `(P,P,C)` order (the `Rearrange` order), while this implementation's convolution flattens in `(C,P,P)` order (PyTorch Conv2d weight layout). To copy a Conv2d weight `(D,C,P,P)` into a reference Linear weight `(D,P·P·C)`, permute to `(D,P,P,C)` and then flatten. A plain `reshape` without the permutation produces the wrong element order. The reference also inserts two patch-embedding LayerNorms that are not in the ViT paper; the test replaces them with `nn.Identity()` before copying. The rest (cls, position, blocks, final LN, head) maps directly.

**Common mistake.** Using `reshape` on a non-contiguous permuted tensor and assuming it copies in the intended order. `reshape` may silently copy; `permute` changes the logical axis order but not the data. Always verify with a numerical `allclose` test.

### Q2.9 Why is there a final LayerNorm before the classification head?

**Answer.** The blocks are Pre-LN: each block normalizes its sublayer inputs, but the residual stream itself is not normalized. The final LayerNorm puts the output features on a consistent scale before pooling and the linear head, which stabilizes training and makes the head's initialization less sensitive to depth. This matches the standard Pre-LN Transformer design.

**Common mistake.** Removing the final LN because "the last block already has a LN" — the last block's LN2 is inside the MLP branch, not on the residual stream.

### Q2.10 What inductive bias does ViT have compared with a CNN?

**Answer.** A CNN bakes in locality, translation equivariance, and weight sharing across positions. ViT's patch projection is local, but attention is global from the first layer, and there is no explicit 2-D neighborhood prior beyond position embeddings. As a result, ViT is often more data-hungry and relies on augmentation, regularization, and scale; DeiT showed that strong training recipes and distillation can close much of the gap on smaller datasets. Swin reintroduces locality and hierarchy through windows and patch merging.

**Common mistake.** Saying ViT has "no inductive bias". It has patch locality and position embeddings; it lacks the strong convolutional priors. Another mistake: assuming global attention automatically means better accuracy on small data.

### Q2.11 Why does `ViT()` default to depth 6 while the training CLI defaults to depth 12?

**Answer.** The class default is a small configuration for shape tests and quick experiments: 2,690,122 parameters with `D=192`, `M=768`, `L=6`. The S3 baseline uses `L=12` for more capacity: 5,355,850 parameters. The CLI passes `--depth 12` explicitly, and the run writes `model_config.json` so evaluation reconstructs the same architecture. This is why `eval_checkpoint.py` reads the model config instead of guessing.

**Common mistake.** Loading a depth-12 checkpoint into a default `ViT()` and getting a state-dict size mismatch. Always use the saved `model_config.json` (the evaluation scripts do).

---

## 中文

### Q2.1 为什么 patch embedding 等价于卷积？等价条件是什么？

**答案.** `Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P, padding=0)` 在图像上以不重叠的 `P×P` 窗口滑动。每个输出位置是展平 patch `(C,P,P)` 与卷积核一行的点积再加 bias。这正是一个把每个 patch 线性投影到 `D` 维的层，且所有位置共享权重。等价条件：`kernel_size == stride == P`、`padding=0`，且 `H/W` 能被 `P` 整除。若 `stride < kernel`，patch 重叠，token 网格改变；若 `stride > kernel`，会漏像素。两者都破坏“划分图像”的解释。

**常见误解.** 说“卷积有局部性所以不同”，却没注意到不重叠的 `P×P` 卷积就是逐 patch 线性层。局部性来自 patch 的定义，不来自投影本身。

### Q2.2 推导 token 数。`N`、`T`、`D`、`C·P²` 有什么区别？

**答案.** 对 `H×W` 图像与 patch `P`，网格是 `(H/P) × (W/P)`，patch 数 `N = (H/P)(W/P)`。有 cls token 时序列长度 `T = N + 1`。`C·P²` 是一个 patch 的标量数（patch 投影的输入宽度）。`D` 是投影后 token 的输出宽度。CIFAR-10 `32×32`、`P=4`、`C=3`、`D=192` 时：`N=64`、`T=65`、`C·P²=48`、`D=192`。

**常见误解.** 把 `D` 与 `C·P²` 混为一谈。投影把 48 维输入映射到 192 维输出，不增加信息，只改变特征空间与容量。另一个错误：把 `T` 说成 patch 数，忘了 cls。

### Q2.3 cls token 是什么？为什么要前置？

**答案.** cls token 是形状 `(1,1,D)` 的可学习向量，扩展成 `(B,1,D)` 后拼在 patch token 前面。经过所有 block 后，它的输出作为图像表示。它在 batch 间共享，端到端学习。前置是一个约定，与位置编码表的第 0 行、`pool_tokens` 取 index 0 绑定；如果同步调整位置表与 pooling 下标，追加在末尾在数学上也成立。

**常见误解.** 以为每张图有独立的 cls 参数。实际只有一个向量，在 batch 维广播。另一个误解：认为 cls token 有超出“可学习摘要 token”的特殊架构作用；用 mean pooling 时它可以不存在。

### Q2.4 比较可学习、正弦、无位置编码。

**答案.** 自注意力是置换等变的：没有位置信息时，打乱 token 会得到同一组输出。可学习绝对位置编码用 `(1,T,D)` 的表，有 `T·D` 个参数；正弦编码是固定 buffer，无参数；`none` 只适合做消融，或在别处注入位置。ViT 论文发现大规模下可学习与正弦编码效果相近；可学习表简单，但与训练分辨率绑定，换分辨率需要插值或重新生成。基线中可学习表占 `65×192=12,480` 参数。

**常见误解.** 说正弦编码“因为能泛化到任意长度所以更好”，却忽略 1D 正弦编码并不显式编码 2D 图像结构，换分辨率仍需处理。

### Q2.5 比较 cls pooling、mean pooling 与参数代价。

**答案.** `cls` pooling 取 token 0；`mean` 对所有 token 求平均（本实现中若含 cls 也一并平均）。cls pooling 增加 cls token 参数 `D`；mean pooling 不增加参数。两者接同一个 `D→K` 分类头。mean pooling 在没有 cls token 时也能工作，且在 cls token 训练不足时可能更稳；cls pooling 给模型一个专门的摘要槽位。仓库通过 `pool="cls"` 与 `pool="mean"` 支持两者；原课程把 cls+mean 拼接列为可能的消融，但没有实现。

**常见误解.** 说 mean pooling “免费”却忘了 head 输入宽度仍是 `D`；或不重新训练就比较 cls 与 mean，二者表示不可互换。

### Q2.6 推导 ViT 参数量。

**答案.** 设宽度 `D`、MLP 宽度 `M`、深度 `L`、类别数 `K`、patch 数 `N`、可学习位置表：

```text
patch 投影:   D·C·P² + D
cls token:    D                       （pool="cls" 时）
位置表:       (N + 1)·D               （learned 且含 cls 时）
每个 block：
  attention:  3D² + D² + D = 4D² + D  （qkv_bias=False）
  MLP:        2DM + M + D
  两个 LN:     2·2D = 4D
  合计:        4D² + 2DM + 6D + M
final LN:     2D
分类头:       D·K + K
```

基线（`D=192`、`M=768`、`L=12`、`K=10`、`N=64`）：patch 9,408 + cls 192 + 位置 12,480 + 12×444,288 + final 384 + head 1,930 = **5,355,850**。类默认 `ViT(depth=6)` 是 **2,690,122**；S3 的 CLI 默认把 depth 覆盖为 12。

**常见误解.** 漏掉 LayerNorm 参数（每 block 4D，final 2D）、head bias 或位置表；或 patch 投影误用 `D²`，而输入宽度应是 `C·P²`。

### Q2.7 模型如何初始化？为什么子层不能初始化为 0？

**答案.** `_init_weights` 对 Linear/Conv 权重做截断正态（`std=0.02`），bias 置 0；cls token 与位置表也用截断正态。若整个子层初始化为 0，其输出为 0，参数拿到零梯度，block 暂时退化为残差恒等。对称性被**保持**而不是被打破，所以学习停滞。小随机初始化既打破对称，又把激活保持在合理范围。

**常见误解.** “全零初始化损失对称性。”它保持对称性。另一个误解：LayerNorm 的 `gamma=1, beta=0` 不是恒等映射；它仍然归一化，只是不做仿射缩放/平移。

### Q2.8 为什么参考对齐时 patch 权重需要特殊处理？

**答案.** `vit-pytorch` 按 `(P,P,C)` 顺序展平 patch（`Rearrange` 的顺序），而本实现的卷积按 `(C,P,P)` 顺序展平（PyTorch Conv2d 权重布局）。把 Conv2d 权重 `(D,C,P,P)` 复制到参考 Linear 权重 `(D,P·P·C)` 时，要先 permute 成 `(D,P,P,C)` 再展平。不做 permute 直接 `reshape` 会得到错误的元素顺序。参考实现还插入了两个 ViT 论文中没有的 patch embedding LayerNorm；测试在复制前把它们替换成 `nn.Identity()`。其余（cls、位置、blocks、final LN、head）直接映射。

**常见误解.** 对非连续的 permute 结果直接 `reshape`，并假设它按预期顺序复制。`reshape` 可能默默复制；`permute` 只改变逻辑轴序，不搬数据。永远用数值 `allclose` 验证。

### Q2.9 为什么分类头前要有一个 final LayerNorm？

**答案.** 这些 block 是 Pre-LN：每个 block 归一化的是子层输入，残差流本身没有归一化。final LayerNorm 在 pooling 与线性头之前把输出特征放到一致尺度，稳定训练，并降低 head 初始化对深度的敏感性。这符合标准 Pre-LN Transformer 设计。

**常见误解.** 因为“最后一个 block 已经有 LN”就删掉 final LN——最后一个 block 的 LN2 在 MLP 分支内部，不在残差流上。

### Q2.10 ViT 与 CNN 相比有什么归纳偏置？

**答案.** CNN 内置局部性、平移等变性与跨位置权重共享。ViT 的 patch 投影是局部的，但从第一层开始 attention 就是全局的，除位置编码外没有显式的 2D 邻域先验。因此 ViT 通常更吃数据，依赖增强、正则与规模；DeiT 表明强训练配方与蒸馏能在较小数据上缩小差距。Swin 通过窗口与 patch merging 重新引入局部性与层级。

**常见误解.** 说 ViT “没有归纳偏置”。它有 patch 局部性与位置编码，只是缺少强卷积先验。另一个误解：全局 attention 自动意味着小数据上更准。

### Q2.11 为什么 `ViT()` 默认 depth 6，而训练 CLI 默认 depth 12？

**答案.** 类默认是小配置，便于 shape 测试与快速实验：`D=192`、`M=768`、`L=6` 时 2,690,122 参数。S3 基线用 `L=12` 换取容量：5,355,850 参数。CLI 显式传 `--depth 12`，并把 `model_config.json` 写进 run 目录，评估时据此重建同一架构。这就是 `eval_checkpoint.py` 读取模型配置而不是猜参数的原因。

**常见误解.** 把 depth-12 checkpoint 加载进默认 `ViT()`，得到 state_dict 尺寸不匹配。始终使用保存的 `model_config.json`（评估脚本已经这样做）。
