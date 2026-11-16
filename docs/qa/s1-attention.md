# S1 — Attention and the Transformer Block / Attention 与 Transformer Block

[English](#english) · [中文](#中文)

---

## English

### Q1.1 Derive scaled dot-product attention and give the shapes.

**Answer.** For a sequence of `N` tokens with width `D`, project to queries, keys, and values:

```text
Q = X W_Q,  K = X W_K,  V = X W_V
scores = Q Kᵀ / sqrt(d_k)          # (B, h, N, M)
A = softmax(scores, dim=-1)        # each query row sums to 1 over keys
out = A V                          # (B, h, N, d_v)
```

With `h` heads, `Q` and `K` are `(B,h,N,d_k)` and `(B,h,M,d_k)`; `V` is `(B,h,M,d_v)`. Self-attention has `M=N`. The softmax is over the **key** axis, so each query distributes one unit of weight across the keys.

**Common mistake.** Writing the score shape as `(B,N,N)` while forgetting the head axis, or softmaxing over `dim=1` (heads) instead of `dim=-1` (keys).

### Q1.2 Why divide by `sqrt(d_k)`?

**Answer.** Assume the components of `q` and `k` are independent with mean 0 and variance 1. Their dot product is a sum of `d_k` terms, so its variance is `d_k`. Without scaling, the logits grow with `sqrt(d_k)`; large logits make softmax nearly one-hot, the gradient through softmax becomes tiny, and training saturates. Dividing by `sqrt(d_k)` makes the logit variance about 1, keeping the softmax in a responsive range.

**Common mistake.** Saying "we divide to make the numbers smaller" without the variance argument, or claiming the scale removes the dependence on `d_k` entirely. It normalizes the variance, not the information content.

### Q1.3 Why is the softmax axis the last one?

**Answer.** `scores` has shape `(B, h, N, M)`: for each batch, head, and query token, there are `M` key logits. The normalization must be over the key axis (`M`, the last axis), so each query's attention weights sum to 1. In this implementation `scaled_dot_product_attention` calls `torch.softmax(scores, dim=-1)`.

**Common mistake.** Using `dim=1`, which normalizes across heads. The tensor is still valid, so the code runs, but the semantics are wrong and tests fail in confusing ways.

### Q1.4 Why apply the mask before softmax instead of multiplying the probabilities by zero afterward?

**Answer.** Setting disallowed logits to `-inf` before softmax gives `exp(-inf)=0`, so those positions receive exactly zero weight and the remaining weights are renormalized to sum to 1. If you softmax first and then multiply some probabilities by zero, the remaining weights sum to less than 1; the output is no longer a convex combination of values, and the blocked positions still influenced the softmax denominator and the gradients.

**Common mistake.** Believing softmax can output `-inf`. Softmax outputs are in `(0,1)`; the `-inf` is an input to softmax. Another failure mode: mask an entire row, which makes every logit `-inf`; then `exp(-inf)/sum(exp(-inf)) = 0/0 = NaN`. Leave at least one allowed key per query.

### Q1.5 What does `mask=True` mean in this code?

**Answer.** `True` means **keep** (allowed to attend); `False` means mask out. The implementation uses `scores.masked_fill(~mask, -inf)`. This convention is documented in the docstring and tested. The causal mask is built as a lower-triangular boolean matrix with the diagonal included.

**Common mistake.** Reversing the convention between PyTorch's `attn_mask` (often `True = masked`) and this repository's `True = keep`. Always read the docstring or the `masked_fill` line.

### Q1.6 How does multi-head attention split and merge, and what is the parameter count?

**Answer.** A fused `to_qkv` linear maps `D → 3D` (for equal head widths). Split it into three `(B,N,D)` tensors, reshape each to `(B,N,h,d_head)`, then transpose to `(B,h,N,d_head)`. After SDPA, transpose back to `(B,N,h,d_head)`, reshape to `(B,N,D)`, and apply `to_out` (`D → D`). With `qkv_bias=False`, the parameters are `3D²` for Q/K/V plus `D² + D` for the output projection: `4D² + D`. For `D=192`, that is `147,648` per block.

**Common mistake.** Reshaping `(B,h,N,d)` directly to `(B,N,D)` without moving the head axis back first. That interleaves heads and tokens incorrectly.

### Q1.7 Why use multiple heads instead of one wide attention?

**Answer.** Multiple heads let the model learn different relations in different subspaces at the same total width. Each head has `d_head = D/h`; the concatenation followed by `W_o` mixes the heads. With `h=1` the implementation is mathematically equivalent to single-head attention of width `D`. More heads are not automatically "more expressive": they reduce the per-head width and add a structural prior about which subspaces to separate.

**Common mistake.** Claiming each head sees a different part of the input or that heads are independent after `W_o`. They share the same input tokens, and `W_o` mixes them.

### Q1.8 Derive LayerNorm and explain the biased variance.

**Answer.** For each token `x ∈ R^D`:

```text
mu = mean(x)
var = mean((x - mu)^2)          # biased: divide by D
x_hat = (x - mu) / sqrt(var + eps)
y = gamma * x_hat + beta
```

`gamma` and `beta` are learnable `D`-vectors (2D parameters). Biased variance is used to match `nn.LayerNorm`; with unbiased variance the normalized output variance would be `(D-1)/D`, not 1. `eps` is inside the square root to avoid division by zero. LayerNorm is applied per token, not across the batch.

**Common mistake.** Using `unbiased=True` and expecting numerical agreement with `nn.LayerNorm`; or normalizing over the batch dimension, which is BatchNorm.

### Q1.9 What is the difference between Pre-LN and Post-LN, and why is Pre-LN more stable?

**Answer.**

```text
Pre-LN:   x = x + F(LN(x))
Post-LN:  x = LN(x + F(x))
```

In Pre-LN the residual path is a pure identity from input to output, so gradients can flow directly through the block without passing through a LayerNorm Jacobian. In Post-LN every residual sum is normalized before the next block; at initialization the gradient can be amplified or attenuated across depth, and deep Post-LN models are more sensitive to the learning rate and often need warm-up. Pre-LN models usually add a final LayerNorm before the head. The experiment in `scripts/compare_norm_placement.py` shows the gradient-norm drift growing with depth for Post-LN and staying flatter for Pre-LN, and at a high learning rate Post-LN fails to fit while Pre-LN still reduces the loss.

**Common mistake.** Writing Pre-LN as `x + LN(F(x))`. The LN belongs on the **input** of the sublayer, not on its output. Another mistake: thinking Pre-LN eliminates the need for a final LN; the final LN is still needed before pooling/classification.

### Q1.10 What does DropPath do, and where should it be applied?

**Answer.** DropPath (stochastic depth) drops an entire residual branch per sample during training and rescales the kept branch by `1/keep_prob` so the expected value is unchanged. It should wrap only the branch: `x + DropPath(F(x))`. It must not wrap the residual sum `DropPath(x + F(x))`, because that would scale or zero the identity path and destroy the gradient highway. At evaluation time it is the identity. `drop_prob=1.0` is invalid: `keep_prob=0` would divide by zero. The code rejects it.

**Common mistake.** Thinking DropPath is ordinary dropout over elements. It is a per-sample, per-branch mask. Another mistake: testing it with `drop_path=0`, where it is `nn.Identity` and a wrong placement is invisible; use a deterministic stand-in such as `_Half` or a fixed RNG.

### Q1.11 Why do residual connections matter, and what happens if a sublayer is initialized to zero?

**Answer.** The residual path gives gradients a direct route to earlier layers and makes it easy for a block to learn an identity function. If a sublayer's output is exactly zero, its own parameters receive zero gradient (`∂L/∂W = δ · inputᵀ = 0`) for that branch, but the residual path still carries the input and gradients to the rest of the network. The model temporarily behaves like a shallower network; it does not "lose symmetry" in the sense of breaking it — symmetry is **preserved**, which is why all-zero initialization is a bad idea for breaking symmetry.

**Common mistake.** Saying "all-zero initialization destroys symmetry". The correct statement is: it preserves symmetry and gives the sublayer zero gradient, so the sublayer cannot learn to break it.

### Q1.12 What numerical issues arise in attention, and how do bf16/fp16 differ?

**Answer.** Softmax can overflow/underflow if implemented naively; the stable form subtracts the row maximum before exponentiation (`softmax` in PyTorch already does this). Masked rows can produce `0/0 = NaN`. fp16 has 5 exponent bits and 10 mantissa bits, so it overflows around 65,504 and has limited precision; bf16 has 8 exponent bits and 7 mantissa bits, so it has fp32-like range but coarser precision. GradScaler is needed for fp16 because small gradients underflow; bf16 does not need loss scaling. In this repository, `--amp auto` uses bf16/fp16 only on CUDA and stays fp32 on MPS/CPU.

**Common mistake.** Assuming bf16 and fp16 are interchangeable, or applying `GradScaler` to bf16 as if it needed scaling. Also: masking every key for a query, which is a data/mask bug that shows up as NaN, not as a precision issue.

---

## 中文

### Q1.1 推导 scaled dot-product attention，并写出 shape。

**答案.** 对 `N` 个宽度为 `D` 的 token，先投影出 query、key、value：

```text
Q = X W_Q,  K = X W_K,  V = X W_V
scores = Q Kᵀ / sqrt(d_k)          # (B, h, N, M)
A = softmax(scores, dim=-1)        # 每个 query 行在 key 上求和为 1
out = A V                          # (B, h, N, d_v)
```

有 `h` 个头时，`Q`、`K` 是 `(B,h,N,d_k)` 与 `(B,h,M,d_k)`，`V` 是 `(B,h,M,d_v)`。自注意力中 `M=N`。softmax 在**key** 轴做，因此每个 query 把 1 单位权重分配给所有 key。

**常见误解.** 写 score shape 时漏掉 head 维，写成 `(B,N,N)`；或在 `dim=1`（head 维）而不是 `dim=-1`（key 维）做 softmax。

### Q1.2 为什么除以 `sqrt(d_k)`？

**答案.** 假设 `q`、`k` 的各分量独立、均值 0、方差 1。它们的点积是 `d_k` 项之和，方差为 `d_k`。不缩放时 logits 随 `sqrt(d_k)` 增长；logits 过大会让 softmax 接近 one-hot，经过 softmax 的梯度变得很小，训练饱和。除以 `sqrt(d_k)` 后 logits 方差约为 1，softmax 保持在有响应的区间。

**常见误解.** 只说“把数变小”，不给出方差论证；或声称缩放完全消除了对 `d_k` 的依赖。它归一化的是方差，不是信息量。

### Q1.3 为什么 softmax 轴是最后一维？

**答案.** `scores` 的形状是 `(B, h, N, M)`：对每个 batch、head、query token，有 `M` 个 key logits。归一化必须在 key 轴（`M`，最后一维）上做，这样每个 query 的 attention 权重和为 1。本实现的 `scaled_dot_product_attention` 调用 `torch.softmax(scores, dim=-1)`。

**常见误解.** 用 `dim=1`，在 head 维上归一化。张量仍然合法，代码不会报错，但语义错了，测试会以令人困惑的方式失败。

### Q1.4 为什么 mask 要在 softmax 之前，而不是之后把概率乘 0？

**答案.** 在 softmax 前把不允许的 logits 设为 `-inf`，`exp(-inf)=0`，这些位置权重精确为 0，剩余权重重新归一化到和为 1。如果先 softmax 再把部分概率乘 0，剩余权重和小于 1；输出不再是 value 的凸组合，而且被屏蔽位置仍然影响 softmax 分母与梯度。

**常见误解.** 以为 softmax 能输出 `-inf`。softmax 输出在 `(0,1)`；`-inf` 是 softmax 的输入。另一种失败模式：把某一整行都 mask 掉，所有 logits 都是 `-inf`，`exp(-inf)/sum(exp(-inf)) = 0/0 = NaN`。每个 query 至少要留一个允许的 key。

### Q1.5 本代码中 `mask=True` 表示什么？

**答案.** `True` 表示**保留**（允许 attend），`False` 表示屏蔽。实现使用 `scores.masked_fill(~mask, -inf)`。这个约定写在 docstring 里并有测试覆盖。causal mask 是包含对角线的下三角布尔矩阵。

**常见误解.** 把 PyTorch `attn_mask`（通常是 `True = masked`）与本仓库的 `True = keep` 混用。永远先看 docstring 或 `masked_fill` 那一行。

### Q1.6 多头注意力如何 split/merge？参数量是多少？

**答案.** 融合的 `to_qkv` 线性层把 `D → 3D`（等宽头）。拆成三个 `(B,N,D)`，各自 reshape 成 `(B,N,h,d_head)`，再 transpose 成 `(B,h,N,d_head)`。SDPA 后 transpose 回 `(B,N,h,d_head)`，reshape 成 `(B,N,D)`，再过一个 `to_out`（`D → D`）。`qkv_bias=False` 时参数为 Q/K/V 的 `3D²` 加输出投影的 `D² + D`，即 `4D² + D`。`D=192` 时每个 block 是 `147,648`。

**常见误解.** 把 `(B,h,N,d)` 直接 reshape 成 `(B,N,D)`，没有先把 head 轴移回去。这样会把 head 与 token 交错，语义错误。

### Q1.7 为什么用多头而不是一个更宽的单头注意力？

**答案.** 多头让模型在相同总宽度下、在不同子空间学习不同关系。每个头宽度 `d_head = D/h`；拼接后再由 `W_o` 混合各头。`h=1` 时实现与宽度 `D` 的单头注意力数学等价。头更多不自动等于“表达力更强”：它降低了每头宽度，并加入了“应该分离哪些子空间”的结构先验。

**常见误解.** 声称每个头看到输入的不同部分，或认为 `W_o` 之后各头仍独立。它们输入相同的 token，`W_o` 会混合它们。

### Q1.8 推导 LayerNorm，并解释为什么用 biased variance。

**答案.** 对每个 token `x ∈ R^D`：

```text
mu = mean(x)
var = mean((x - mu)^2)          # biased：除以 D
x_hat = (x - mu) / sqrt(var + eps)
y = gamma * x_hat + beta
```

`gamma`、`beta` 是可学习的 `D` 维向量（2D 参数）。使用 biased variance 是为了与 `nn.LayerNorm` 一致；若用 unbiased variance，归一化后的输出方差会是 `(D-1)/D`，不是 1。`eps` 放在平方根内，避免除零。LayerNorm 按 token 做，不跨 batch。

**常见误解.** 用 `unbiased=True` 却期待与 `nn.LayerNorm` 数值一致；或在 batch 维上归一化，那是 BatchNorm。

### Q1.9 Pre-LN 与 Post-LN 有什么区别？为什么 Pre-LN 更稳定？

**答案.**

```text
Pre-LN:   x = x + F(LN(x))
Post-LN:  x = LN(x + F(x))
```

Pre-LN 的残差路径是从输入到输出的纯恒等映射，梯度可以不经 LayerNorm 的 Jacobian 直接流过 block。Post-LN 中每个残差和都要先归一化再进入下一层；初始化时梯度可能随深度被放大或衰减，深层 Post-LN 对学习率更敏感，通常需要 warm-up。Pre-LN 模型通常在分类头前加一个 final LayerNorm。`scripts/compare_norm_placement.py` 的实验显示：Post-LN 的梯度范数随深度漂移更明显，Pre-LN 更平稳；在高学习率下 Post-LN 无法拟合，而 Pre-LN 仍能降低 loss。

**常见误解.** 把 Pre-LN 写成 `x + LN(F(x))`。LN 在子层的**输入**上，不在输出上。另一个误解：以为 Pre-LN 不需要 final LN；分类/pooling 前仍然需要。

### Q1.10 DropPath 做什么？应该加在哪里？

**答案.** DropPath（stochastic depth）在训练时按样本丢弃整个残差分支，并把保留的分支按 `1/keep_prob` 缩放，使期望不变。它只应包住分支：`x + DropPath(F(x))`。不能包住残差和 `DropPath(x + F(x))`，因为那会缩放或清零恒等路径，破坏梯度高速通道。评估时它是恒等映射。`drop_prob=1.0` 非法：`keep_prob=0` 会除零，代码会拒绝。

**常见误解.** 把 DropPath 当成逐元素 dropout。它是按样本、按分支的 mask。另一个错误：用 `drop_path=0` 测试它，此时它是 `nn.Identity`，放错位置也看不出来；应使用 `_Half` 这类确定性替身或固定 RNG。

### Q1.11 残差连接为什么重要？如果子层初始化为 0 会怎样？

**答案.** 残差路径给梯度一条直达浅层的通路，也让 block 容易学到恒等映射。如果子层输出精确为 0，该分支参数拿到零梯度（`∂L/∂W = δ · inputᵀ = 0`），但残差路径仍把输入与梯度传下去，模型暂时退化为更浅的网络；它不是“损失了对称性”，而是**保持了对称性**，所以全零初始化无法打破对称，子层也学不动。

**常见误解.** 说“全零初始化导致对称性损失”。正确说法是：它保持对称性、给子层零梯度，因此子层无法学会打破对称。

### Q1.12 Attention 中有哪些数值问题？bf16 与 fp16 有什么区别？

**答案.** 朴素 softmax 会溢出/下溢；稳定实现先减去行最大值再取指数（PyTorch 的 softmax 已经这样做）。整行被 mask 会产生 `0/0 = NaN`。fp16 有 5 位指数、10 位尾数，约 65,504 溢出，精度有限；bf16 有 8 位指数、7 位尾数，范围接近 fp32，但精度更粗。fp16 需要 GradScaler 防止小梯度下溢；bf16 不需要 loss scaling。本仓库 `--amp auto` 只在 CUDA 上用 bf16/fp16，MPS/CPU 保持 fp32。

**常见误解.** 把 bf16 与 fp16 当成可互换；或给 bf16 套 GradScaler。另一个问题：某个 query 的所有 key 都被 mask，这是 mask 数据 bug，表现为 NaN，不是精度问题。
