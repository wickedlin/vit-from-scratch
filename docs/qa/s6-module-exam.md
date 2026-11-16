# S6 Appendix — Eight-Module Oral Exam / 八模块口试样卷

[English](#english) · [中文](#中文)

This appendix reconstructs the S6 "module oral exam" format: eight modules, four prompts each — shape/data flow, formula or pseudocode, reason and cost, and a failure mode or boundary case. The answers are model answers, not a memorization script; the goal is to explain the mechanism and its limits.

本附录还原 S6 的“模块口试”形式：8 个模块，每个 4 题——shape/数据流、公式或伪代码、理由与代价、失败模式或边界。答案是参考答案，不是背诵脚本；目标是解释机制与边界。

---

## English

### A1 — Scaled dot-product attention

**A1-1. Shape / data flow.** `Q (B,h,N,d_k)`, `K (B,h,M,d_k)`, `V (B,h,M,d_v)` → `scores (B,h,N,M)` → softmax over the last axis → `weights @ V` → `(B,h,N,d_v)`.

**A1-2. Formula / pseudocode.**
```text
scores = Q @ K.transpose(-2,-1) / sqrt(d_k)
scores = scores.masked_fill(~mask, -inf)   # True = keep
A = softmax(scores, dim=-1)
out = A @ V
```

**A1-3. Reason / cost.** The `1/sqrt(d_k)` keeps the logit variance near 1, preventing softmax saturation and vanishing gradients. Time is `O(N·M·d_k + N·M·d_v)`; the score matrix costs `O(N·M)` memory, which is why long sequences are expensive.

**A1-4. Failure mode / boundary.** Softmaxing the head axis instead of the key axis is a silent semantic bug. Masking an entire query row produces `0/0 = NaN`. Applying the mask after softmax leaves the remaining weights unnormalized.

### A2 — Multi-head self-attention

**A2-1. Shape / data flow.** `(B,N,D)` → `to_qkv` → `(B,N,3D)` → `chunk(3)` → three `(B,N,D)` → reshape `(B,N,h,d_head)` → transpose `(B,h,N,d_head)` → SDPA → `(B,h,N,d_head)` → transpose `(B,N,h,d_head)` → reshape `(B,N,D)` → `to_out` → `(B,N,D)`.

**A2-2. Formula / parameter count.** `d_head = D/h`, `inner_dim = h·d_head`. With `qkv_bias=False`: `3D²` for Q/K/V and `D²+D` for the output projection, total `4D²+D`. For `D=192`, that is 147,648 per block.

**A2-3. Reason / cost.** Heads learn different relations in different subspaces at the same total width; `W_o` mixes them. Attention cost is `O(N²D)` and projection cost is `O(ND²)`. More heads reduce `d_head`; they are not automatically "more expressive".

**A2-4. Failure mode / boundary.** Reshaping `(B,h,N,d)` directly to `(B,N,D)` interleaves heads and tokens. `D % h != 0` must raise. Using the wrong chunk order swaps Q/K/V roles.

### A3 — Pre-LN vs. Post-LN

**A3-1. Formulas.**
```text
Pre-LN:   x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
Post-LN:  x = LN1(x + Attn(x));  x = LN2(x + MLP(x))
```

**A3-2. Gradient path.** Pre-LN has a pure identity residual path, so gradients reach earlier layers without passing through a LayerNorm Jacobian. Post-LN normalizes every residual sum, and its gradient scale can drift with depth; deep Post-LN models are more sensitive to LR and often need warm-up. Pre-LN models add a final LN before the head.

**A3-3. Reason / cost.** Pre-LN is the standard stable default for deep Transformers. Each block has two LayerNorms (`4D` parameters); the final LN adds `2D`. Post-LN is not "wrong"; it is a different trade-off that can work with careful initialization/warm-up.

**A3-4. Failure mode / boundary.** Writing Pre-LN as `x + LN(F(x))` puts LN on the output instead of the input. Wrapping `DropPath` around the residual sum destroys the identity path. Using unbiased variance breaks numerical agreement with `nn.LayerNorm`. Forgetting the final LN leaves the residual stream unnormalized before pooling.

### A4 — Patch embedding, cls token, and position embeddings

**A4-1. Shapes.** `(B,C,H,W)` → patches `(B,N,C·P²)` → projection `(B,N,D)` → prepend cls `(B,N+1,D)` → add position `(B,N+1,D)`, where `N=(H/P)(W/P)`.

**A4-2. Formula / pseudocode.** A `Conv2d(C,D,kernel_size=P,stride=P,padding=0)` is equivalent to reshaping the image into non-overlapping patches and applying `Linear(C·P²,D)` to each patch. The position table is `(1,N+1,D)` for learned embeddings; cls is a `(1,1,D)` learnable vector broadcast over the batch.

**A4-3. Reason / cost.** Patch projection costs `D·C·P² + D`; the position table costs `(N+1)·D`; cls costs `D`. Learned positions are simple but tied to resolution; sinusoidal positions are parameter-free but still need care when resolution changes. The cls token gives a dedicated summary slot; mean pooling is an alternative.

**A4-4. Failure mode / boundary.** `H` or `W` not divisible by `P` changes the token grid. The conv flatten order is `(C,P,P)`, while `vit-pytorch` uses `(P,P,C)`; copying weights without a permutation is wrong. A position table sized for `N` tokens fails when cls is added. Changing resolution requires interpolation or a new table.

### A5 — ViT forward and pooling

**A5-1. Shape chain.** `(B,C,H,W)` → `(B,N,D)` → `(B,T,D)` with `T=N+1` → blocks → final LN → pool → `(B,D)` → head → `(B,K)`.

**A5-2. Formula / parameter count.** Baseline: 5,355,850. Per block `4D² + 2DM + 6D + M = 444,288`; final LN `2D`; head `D·K+K`; patch projection, cls, and position table as in A4. The analytical MAC count is 365,107,584; `thop` reports 345,638,784 because it does not hook the attention matmuls.

**A5-3. Reason / cost.** The final LN stabilizes the residual stream before pooling. cls pooling uses token 0; mean pooling averages all tokens. The head is a single linear layer; the representation work happens in the blocks.

**A5-4. Failure mode / boundary.** Loading a depth-12 checkpoint into default `ViT(depth=6)` gives a size mismatch; read `model_config.json`. Using cls pooling without a cls token, or mean pooling while expecting cls semantics, silently changes the representation. Omitting the final LN changes the head input scale.

### A6 — Swin windows, shifts, and masks

**A6-1. Shapes.** `(B,H,W,C)` → `window_partition` → `(B·nH·nW, M², C)` → window attention → `window_reverse` → `(B,H,W,C)`. Shift and unshift keep the shape.

**A6-2. Formula / pseudocode.**
```text
h = LN(x)
if shift > 0: h = roll(h, (-s,-s)); mask = shifted_window_mask(H,W,M,s)
windows = partition(h, M)
windows = window_attention(windows, mask)
h = reverse(windows, M, H, W)
if shift > 0: h = roll(h, (+s,+s))
x = x + DropPath(h); x = x + DropPath(MLP(LN(x)))
```

**A6-3. Reason / cost.** Window attention is linear in the number of tokens when `M` is fixed. Shifting lets information cross window boundaries; the mask prevents attention between regions that the cyclic roll falsely placed in the same window. The cost is local attention plus the mask construction.

**A6-4. Failure mode / boundary.** The mask blocks only wrap-created mixed regions, not all non-adjacent tokens. It must be applied before softmax and built on the score device. The inverse order is `window_reverse` then `unshift_window`. A single-window stage must use `shift=0`; otherwise the mask builder rejects it.

### A7 — PatchMerging, relative position bias, and stages

**A7-1. Shapes.** `(B,H,W,C)` → 2×2 neighborhoods → concat `(B,H/2,W/2,4C)` → `LayerNorm(4C)` → `Linear(4C→2C)` → `(B,H/2,W/2,2C)`. Relative bias output: `(h,M²,M²)`.

**A7-2. Formula / parameter count.** PatchMerging: `8C` LN parameters + `8C²` linear parameters. Relative bias: table `((2M−1)²,h)` plus a non-parameter index buffer `(M²,M²)`. A stage is a sequence of blocks at one resolution; PatchMerging connects stages.

**A7-3. Reason / cost.** PatchMerging builds a CNN-like hierarchy and reduces spatial size by 4 while doubling channels. Relative bias encodes within-window geometry and is shared across windows in the same attention layer, not across blocks. Stage 3 of the CIFAR config has one window and six blocks, so its total cost is the highest.

**A7-4. Failure mode / boundary.** Odd `H/W` cannot be merged. A different quadrant order is a channel permutation; matching a reference requires permuting norm weights and reduction input columns. Sharing one bias table across blocks is wrong. Saying "tokens halve" is wrong: `H/W` halve, so tokens quarter.

### A8 — MAE masking, decoder, and loss

**A8-1. Shapes.** `(B,C,H,W)` → target `(B,N,P²C)`; encoder input `(B,N,D)`; masking keeps `x_keep (B,K,D)`, `mask (B,N)`, `ids_restore (B,N)`; decoder input `(B,N,D_dec)`; prediction `(B,N,P²C)`.

**A8-2. Pseudocode.**
```text
noise = rand(B,N); ids_shuffle = argsort(noise); ids_restore = argsort(ids_shuffle)
ids_keep = ids_shuffle[:, :K]
x_keep = gather(x, ids_keep expanded over D)
mask = ones(B,N); mask.scatter_(1, ids_keep, 0)
enc = encoder(x_keep)
dec = enc_to_dec(enc)
dec = cat([dec, mask_token.expand(B,N-K,-1)], dim=1)
dec = gather(dec, ids_restore expanded over D_dec) + decoder_pos
pred = decoder(dec)
loss = mean_over_masked((pred - target_norm)^2)
```

**A8-3. Reason / cost.** A high mask ratio makes reconstruction nontrivial. The encoder sees only `K` tokens; the decoder is lightweight and discarded after pretraining. `norm_pix_loss` normalizes each target patch; the loss is still computed only on masked patches.

**A8-4. Failure mode / boundary.** Using `ids_shuffle` instead of `ids_restore` scrambles order. `gather` indices must be expanded over the feature dimension. Encoder position rows must be gathered with `ids_keep`; decoder positions are added after unshuffle. Computing loss on visible patches creates a copy shortcut. An all-masked sample is invalid.

---

## 中文

### A1 — Scaled dot-product attention

**A1-1. Shape / 数据流.** `Q (B,h,N,d_k)`、`K (B,h,M,d_k)`、`V (B,h,M,d_v)` → `scores (B,h,N,M)` → 在最后一维 softmax → `weights @ V` → `(B,h,N,d_v)`。

**A1-2. 公式 / 伪代码.**
```text
scores = Q @ K.transpose(-2,-1) / sqrt(d_k)
scores = scores.masked_fill(~mask, -inf)   # True = keep
A = softmax(scores, dim=-1)
out = A @ V
```

**A1-3. 理由 / 代价.** `1/sqrt(d_k)` 让 logit 方差接近 1，避免 softmax 饱和与梯度消失。时间是 `O(N·M·d_k + N·M·d_v)`；score 矩阵占 `O(N·M)` 显存，所以长序列很贵。

**A1-4. 失败模式 / 边界.** 在 head 轴而不是 key 轴做 softmax 是静默语义错误。整行 mask 会产生 `0/0 = NaN`。softmax 之后才 mask 会让剩余权重不归一化。

### A2 — Multi-head self-attention

**A2-1. Shape / 数据流.** `(B,N,D)` → `to_qkv` → `(B,N,3D)` → `chunk(3)` → 三个 `(B,N,D)` → reshape `(B,N,h,d_head)` → transpose `(B,h,N,d_head)` → SDPA → `(B,h,N,d_head)` → transpose `(B,N,h,d_head)` → reshape `(B,N,D)` → `to_out` → `(B,N,D)`。

**A2-2. 公式 / 参数量.** `d_head = D/h`、`inner_dim = h·d_head`。`qkv_bias=False` 时 Q/K/V 为 `3D²`，输出投影为 `D²+D`，合计 `4D²+D`。`D=192` 时每个 block 是 147,648。

**A2-3. 理由 / 代价.** 各头在相同总宽度下、在不同子空间学习不同关系；`W_o` 负责混合。Attention 成本 `O(N²D)`，投影成本 `O(ND²)`。头更多会降低 `d_head`，不自动等于“表达力更强”。

**A2-4. 失败模式 / 边界.** 把 `(B,h,N,d)` 直接 reshape 成 `(B,N,D)` 会把 head 与 token 交错。`D % h != 0` 必须报错。`chunk` 顺序错会交换 Q/K/V 角色。

### A3 — Pre-LN vs. Post-LN

**A3-1. 公式.**
```text
Pre-LN:   x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
Post-LN:  x = LN1(x + Attn(x));  x = LN2(x + MLP(x))
```

**A3-2. 梯度路径.** Pre-LN 有纯恒等残差路径，梯度到浅层不必经过 LayerNorm 的 Jacobian。Post-LN 对每个残差和做归一化，梯度尺度可能随深度漂移；深层 Post-LN 对 LR 更敏感，常需 warm-up。Pre-LN 模型在分类头前加 final LN。

**A3-3. 理由 / 代价.** Pre-LN 是深层 Transformer 稳定训练的默认选择。每个 block 有两个 LayerNorm（`4D` 参数）；final LN 再加 `2D`。Post-LN 不是“错”，只是需要更小心初始化/warm-up 的另一种取舍。

**A3-4. 失败模式 / 边界.** 把 Pre-LN 写成 `x + LN(F(x))`，LN 放到了输出而不是输入。把 DropPath 套在残差和上会破坏恒等路径。用 unbiased variance 会与 `nn.LayerNorm` 数值不一致。忘记 final LN 会让 pooling 前的残差流未归一化。

### A4 — Patch embedding、cls token 与位置编码

**A4-1. Shape.** `(B,C,H,W)` → patch `(B,N,C·P²)` → 投影 `(B,N,D)` → 前置 cls `(B,N+1,D)` → 加位置 `(B,N+1,D)`，其中 `N=(H/P)(W/P)`。

**A4-2. 公式 / 伪代码.** `Conv2d(C,D,kernel_size=P,stride=P,padding=0)` 等价于把图像切成不重叠 patch，再对每个 patch 做 `Linear(C·P²,D)`。可学习位置表是 `(1,N+1,D)`；cls 是 `(1,1,D)` 的可学习向量，在 batch 维广播。

**A4-3. 理由 / 代价.** Patch 投影 `D·C·P² + D`；位置表 `(N+1)·D`；cls `D`。可学习位置简单但与分辨率绑定；正弦位置无参数，但换分辨率仍需处理。cls token 提供专门摘要槽位；mean pooling 是替代方案。

**A4-4. 失败模式 / 边界.** `H/W` 不能被 `P` 整除会改变 token 网格。Conv 的 flatten 顺序是 `(C,P,P)`，`vit-pytorch` 是 `(P,P,C)`；不做 permutation 直接复制权重会错。位置表按 `N` 大小构造时，加 cls 会尺寸不匹配。换分辨率需要插值或重建表。

### A5 — ViT forward 与 pooling

**A5-1. Shape 链.** `(B,C,H,W)` → `(B,N,D)` → `(B,T,D)`（`T=N+1`）→ blocks → final LN → pool → `(B,D)` → head → `(B,K)`。

**A5-2. 公式 / 参数量.** 基线 5,355,850。每 block `4D² + 2DM + 6D + M = 444,288`；final LN `2D`；head `D·K+K`；patch 投影、cls、位置表同 A4。解析 MACs 365,107,584；`thop` 报 345,638,784，因为它不 hook attention matmul。

**A5-3. 理由 / 代价.** final LN 在 pooling 前稳定残差流。cls pooling 取 token 0；mean pooling 对所有 token 求平均。分类头只是一层线性层；表示学习发生在 blocks 里。

**A5-4. 失败模式 / 边界.** 把 depth-12 checkpoint 加载进默认 `ViT(depth=6)` 会尺寸不匹配；要读 `model_config.json`。没有 cls token 却用 cls pooling，或期待 mean pooling 具有 cls 语义，会静默改变表示。省略 final LN 会改变 head 输入尺度。

### A6 — Swin 窗口、shift 与 mask

**A6-1. Shape.** `(B,H,W,C)` → `window_partition` → `(B·nH·nW, M², C)` → 窗口 attention → `window_reverse` → `(B,H,W,C)`。shift/unshift 不改变 shape。

**A6-2. 公式 / 伪代码.**
```text
h = LN(x)
if shift > 0: h = roll(h, (-s,-s)); mask = shifted_window_mask(H,W,M,s)
windows = partition(h, M)
windows = window_attention(windows, mask)
h = reverse(windows, M, H, W)
if shift > 0: h = roll(h, (+s,+s))
x = x + DropPath(h); x = x + DropPath(MLP(LN(x)))
```

**A6-3. 理由 / 代价.** 窗口固定时，窗口 attention 对 token 数线性。Shift 让信息跨窗口；mask 阻止循环 roll 把远距离区域错误地放进同一窗口。代价是局部 attention 加 mask 构造。

**A6-4. 失败模式 / 边界.** Mask 只屏蔽 wrap 拼出的混合区域，不屏蔽所有非相邻 token。Mask 必须在 softmax 前、且在 score 所在设备上构造。逆变换顺序是 `window_reverse` 然后 `unshift_window`。单窗口 stage 必须 `shift=0`；否则 mask 构造会拒绝。

### A7 — PatchMerging、相对位置偏置与 stage

**A7-1. Shape.** `(B,H,W,C)` → 2×2 邻域 → 拼接 `(B,H/2,W/2,4C)` → `LayerNorm(4C)` → `Linear(4C→2C)` → `(B,H/2,W/2,2C)`。相对偏置输出 `(h,M²,M²)`。

**A7-2. 公式 / 参数量.** PatchMerging：`8C` 个 LN 参数 + `8C²` 个线性参数。相对偏置：表 `((2M−1)²,h)` 加非参数索引 buffer `(M²,M²)`。Stage 是同一分辨率下的一组 block；PatchMerging 连接 stage。

**A7-3. 理由 / 代价.** PatchMerging 构建类 CNN 层级，空间尺寸降 4 倍、通道加倍。相对偏置编码窗口内几何关系，在同一 attention 层的所有窗口间共享，不跨 block 共享。CIFAR 配置的 stage 3 只有一个窗口、6 个 block，总成本最高。

**A7-4. 失败模式 / 边界.** 奇数 `H/W` 无法 merge。不同象限顺序是通道置换；要与参考对齐，需同步置换 norm 权重与 reduction 输入列。跨 block 共享一张偏置表是错的。说“token 减半”是错的：`H/W` 减半，token 数变 1/4。

### A8 — MAE masking、decoder 与 loss

**A8-1. Shape.** `(B,C,H,W)` → target `(B,N,P²C)`；encoder 输入 `(B,N,D)`；masking 后 `x_keep (B,K,D)`、`mask (B,N)`、`ids_restore (B,N)`；decoder 输入 `(B,N,D_dec)`；预测 `(B,N,P²C)`。

**A8-2. 伪代码.**
```text
noise = rand(B,N); ids_shuffle = argsort(noise); ids_restore = argsort(ids_shuffle)
ids_keep = ids_shuffle[:, :K]
x_keep = gather(x, ids_keep expanded over D)
mask = ones(B,N); mask.scatter_(1, ids_keep, 0)
enc = encoder(x_keep)
dec = enc_to_dec(enc)
dec = cat([dec, mask_token.expand(B,N-K,-1)], dim=1)
dec = gather(dec, ids_restore expanded over D_dec) + decoder_pos
pred = decoder(dec)
loss = mean_over_masked((pred - target_norm)^2)
```

**A8-3. 理由 / 代价.** 高 mask ratio 让重建不平凡。Encoder 只处理 `K` 个 token；decoder 很轻，预训练后丢弃。`norm_pix_loss` 对每个目标 patch 归一化；loss 仍只算 masked patch。

**A8-4. 失败模式 / 边界.** 用 `ids_shuffle` 而不是 `ids_restore` 会打乱顺序。`gather` 的 index 要在特征维 expand。Encoder 位置行要按 `ids_keep` gather；decoder 位置在 unshuffle 之后加。在可见 patch 上算 loss 会制造复制捷径。整样本全 mask 非法。
