# S4 — Swin Transformer / Swin Transformer

[English](#english) · [中文](#中文)

---

## English

### Q4.1 Why introduce Swin instead of using global ViT attention everywhere?

**Answer.** Global self-attention costs `O(N²)` in the number of tokens. At high resolution, `N` grows quadratically with image side length, so attention becomes the bottleneck. Swin makes three changes: (1) compute attention **inside local windows**, which is linear in `N` when the window size is fixed; (2) **shift** the windows between blocks so information can cross window boundaries; (3) **merge** neighboring patches between stages to build a hierarchy, like a CNN. The shifted-window design keeps the cost local while recovering cross-window interaction.

**Common mistake.** Saying Swin "removes global attention" and therefore loses long-range modeling. Shifted windows plus hierarchy propagate information across the image over several blocks; it is a different trade-off, not a strict limitation.

### Q4.2 What are the shapes and the ordering convention of `window_partition` and `window_reverse`?

**Answer.** For `x` of shape `(B,H,W,C)` and window size `M`, with `nH=H/M` and `nW=W/M`, `window_partition` returns `(B·nH·nW, M·M, C)`. The order is batch first, then window row `i`, then window column `j`. `window_reverse` reshapes back to `(B,H,W,C)` using the same convention. They are exact inverses when the shapes are consistent. For the stage-1 CIFAR feature map `8×8`, `M=4`: 4 windows, each with 16 tokens.

**Common mistake.** Using a different window order in partition and reverse. The test compares the two functions on random tensors; a mismatch produces scrambled spatial layout that still has the right shape.

### Q4.3 What does the shifted window do, and what is the correct inverse order?

**Answer.** `shift_window` applies `torch.roll(x, shifts=(-s,-s), dims=(1,2))`, a cyclic shift up and left. After partition and attention, `window_reverse` restores the feature map, then `unshift_window` applies `(+s,+s)`. The forward composition is `shift ∘ partition`; the inverse must be `reverse ∘ unshift`, not the other way around. The cyclic shift leaves the window shape unchanged, so every window is still `M×M`; the wrap-around creates windows that mix regions that were far apart in the original image, which is why a mask is needed.

**Common mistake.** Reversing the order (`unshift` before `window_reverse`), or applying `roll` to a 3-D window tensor and rolling the token/channel axis instead of the spatial axes. The code validates the 4-D shape to catch this.

### Q4.4 Which token pairs does the shifted-window mask block?

**Answer.** It blocks pairs that end up in the same shifted window but came from different, non-adjacent regions of the original image due to cyclic wrap. It does **not** block every non-adjacent pair: tokens that are genuinely adjacent across an original window boundary are allowed to attend. The implementation assigns a region id to each pixel with nine slices of an `H×W` mask, partitions that mask into windows, and allows attention only when two tokens have the same id. For `H=W=8`, `M=4`, `s=2`, window 0 has no wrap, window 1 wraps horizontally, window 2 vertically, and window 3 both ways. When the feature map has a single window (`H<=M` or `W<=M`), the shift is set to 0 because there is no cross-window interaction to recover.

**Common mistake.** Saying the mask blocks "all non-adjacent tokens" or "the four corners". It blocks only the wrap-created mixed regions. Another mistake: applying the mask after softmax, or forgetting to build it on the same device as the attention scores (the original implementation hit an MPS device mismatch here).

### Q4.5 How does PatchMerging work, and what does it cost?

**Answer.** PatchMerging takes non-overlapping `2×2` neighborhoods, concatenates their channels (`4C`), applies `LayerNorm(4C)`, and projects `4C → 2C` with a bias-free Linear. The output has half the height and width and twice the channels: `(B,H,W,C) → (B,H/2,W/2,2C)`. Parameters: LayerNorm has `2·4C = 8C`; the Linear has `4C·2C = 8C²`. For `C=64`: 512 + 32,768 = 33,280. The four quadrants are concatenated in a fixed order. Changing that order is a channel permutation; to match a reference exactly, you must also permute the norm weights and the reduction input columns.

**Common mistake.** Saying PatchMerging halves the token count but keeps the channel width, or forgetting the LayerNorm parameters. Another mistake: using a different quadrant order and expecting weight compatibility with `timm`.

### Q4.6 What is the relative position bias, and what is its sharing scope?

**Answer.** For a window of `M×M` tokens there are `(2M−1)²` possible relative offsets. `RelativePositionBias` stores a learnable table `((2M−1)², num_heads)` and a fixed index buffer `(M², M²)` that maps each token pair to its offset row. The output is `(num_heads, M², M²)`, added to the attention logits before softmax. Each **attention layer** owns one table; all windows processed by that layer share it, but different blocks do not share it. The index buffer is not a parameter and is not saved in `state_dict` (`persistent=False`).

**Common mistake.** Saying all blocks in a stage share one table. Each block's attention layer has its own. Another mistake: treating the index buffer as a learned parameter.

### Q4.7 What is the difference between a stage and a block?

**Answer.** A **block** is one Swin Transformer block: LayerNorm → W-MSA or SW-MSA → residual → LayerNorm → MLP → residual. A **stage** is a sequence of blocks operating at the same resolution and channel width. Stages are connected by PatchMerging. In one forward pass, all stages run sequentially and are trained jointly; there is no separate "stage training". Within a stage, even-indexed blocks use W-MSA and odd-indexed blocks use SW-MSA. The last stage has a single window, so its shift is 0.

**Common mistake.** Thinking stages are independent models, or that PatchMerging happens inside every block. It happens only at stage transitions.

### Q4.8 How does the compute scale across stages?

**Answer.** For the CIFAR configuration, patch embedding produces `8×8` tokens (64). Stage 1 keeps `8×8`; PatchMerging gives `4×4` (16); the next gives `2×2` (4). Window sizes are 4/2/2, so stages have 4/4/1 windows. Per block, the projection and MLP terms scale with the token count, while the window-attention term scales with the number of windows times `M²`; when the token count shrinks, the channel width doubles, so the per-token cost rises. The net effect is that per-block cost is roughly constant across stages, and stage 3 has the highest total because it has 6 blocks. The measured cost is about 33.1M MACs per image, versus 365.1M for the ViT baseline.

**Common mistake.** Saying "tokens decrease by 4×, so compute decreases by 4×". The channel width doubles at each merge, and the number of blocks differs by stage.

### Q4.9 What does the Swin vs. ViT comparison actually show?

**Answer.** Swin reaches 0.8201 test accuracy versus 0.8156 for ViT, a +0.45 pp difference. McNemar exact p = 0.267 and the paired bootstrap 95% CI is [−1.26, +0.34] pp, so the accuracy difference is not statistically significant. Swin uses about 33.1M MACs per image versus 365.1M, roughly 1/11. The justified claim is: **Swin is competitive and much cheaper on this setup**, not "Swin is more accurate". Swin also has better AUC (0.9651 vs. 0.9525) and NLL (0.6520 vs. 0.7323) in this run, but a single seed cannot establish a general ranking.

**Common mistake.** Reporting the +0.45 pp as a win without the confidence interval. Another mistake: generalizing a CIFAR-sized 3-stage Swin result to the official Swin-T/ImageNet setting.

### Q4.10 What are the common Swin failure modes?

**Answer.**

- `H` or `W` not divisible by the window size; the code raises a `ValueError`.
- `shift_size >= window_size`; the code rejects it.
- Using `shift_size > 0` with a single window; the mask builder rejects it, and `SwinTransformer` sets shift to 0 in that case.
- Applying the mask after softmax, or using the wrong mask convention (`True = keep` here).
- Building the mask on CPU while attention runs on MPS/CUDA; pass `device=x.device`.
- Reversing `window_reverse` and `unshift_window`.
- Sharing `RelativePositionBias` across blocks, or treating its index buffer as a parameter.

**Common mistake.** Debugging only on CPU. The MPS device bug appeared only when the model was moved to the GPU; CPU tests passed. Always run at least one smoke test on the target device.

---

## 中文

### Q4.1 为什么引入 Swin，而不是到处用全局 ViT attention？

**答案.** 全局自注意力的成本是 token 数 `N` 的 `O(N²)`。高分辨率下 `N` 随图像边长平方增长，attention 成为瓶颈。Swin 做了三件事：(1) 在**局部窗口**内算 attention，窗口固定时对 `N` 线性；(2) 在 block 之间**平移窗口**，让信息跨窗口流动；(3) 在 stage 之间**合并**相邻 patch，像 CNN 一样构建层级。Shifted window 在保持局部成本的同时恢复跨窗口交互。

**常见误解.** 说 Swin “去掉了全局注意力”，因此失去长程建模。Shifted window 加层级会在多个 block 中把信息传播到全图；这是不同的取舍，不是严格的限制。

### Q4.2 `window_partition` 与 `window_reverse` 的 shape 和顺序约定是什么？

**答案.** 对形状 `(B,H,W,C)` 的 `x`、窗口大小 `M`，设 `nH=H/M`、`nW=W/M`，`window_partition` 返回 `(B·nH·nW, M·M, C)`。顺序是 batch 优先，然后窗口行 `i`，最后窗口列 `j`。`window_reverse` 用同一约定还原成 `(B,H,W,C)`。形状一致时二者精确互逆。CIFAR stage 1 特征图 `8×8`、`M=4`：4 个窗口，每个 16 个 token。

**常见误解.** partition 与 reverse 使用不同的窗口顺序。测试会在随机张量上对比两个函数；顺序不一致会得到 shape 正确但空间布局错乱的结果。

### Q4.3 Shifted window 做什么？正确的反变换顺序是什么？

**答案.** `shift_window` 执行 `torch.roll(x, shifts=(-s,-s), dims=(1,2))`，即向上、向左循环平移。partition 与 attention 后，`window_reverse` 还原特征图，再 `unshift_window` 执行 `(+s,+s)`。前向复合是 `shift ∘ partition`；逆变换必须是 `reverse ∘ unshift`，不能反过来。循环平移不改变窗口形状，所以每个窗口仍是 `M×M`；wrap 会把原图中相距很远的区域拼进同一个窗口，因此需要 mask。

**常见误解.** 反变换顺序颠倒（先 `unshift` 再 `window_reverse`）；或对 3-D 窗口张量做 `roll`，滚动了 token/通道轴而不是空间轴。代码会校验 4-D shape 来拦截这类错误。

### Q4.4 Shifted-window mask 屏蔽哪些 token 对？

**答案.** 它屏蔽的是：由于循环平移，落在同一个 shifted window 内、但在原图中来自不同且不相邻区域的 token 对。它**不**屏蔽所有非相邻 token：跨原窗口边界但空间上真正相邻的 token 是允许 attend 的。实现用 `H×W` mask 的九个切片给每个像素分配区域 id，再把 mask 切窗，只有 id 相同的 token 对允许 attention。`H=W=8`、`M=4`、`s=2` 时，窗口 0 无 wrap，窗口 1 横向 wrap，窗口 2 纵向 wrap，窗口 3 双向 wrap。特征图只有一个窗口（`H<=M` 或 `W<=M`）时，shift 设为 0，因为没有跨窗口交互需要恢复。

**常见误解.** 说 mask 屏蔽“所有非相邻 token”或“四个角”。它只屏蔽 wrap 拼出的混合区域。另一个错误：在 softmax 之后才 mask；或忘记在与 attention score 相同的设备上构造 mask（原实现就是在这里遇到 MPS 设备不匹配）。

### Q4.5 PatchMerging 如何工作？代价是多少？

**答案.** PatchMerging 取不重叠的 `2×2` 邻域，拼接通道（`4C`），做 `LayerNorm(4C)`，再用无 bias 的 Linear 投影 `4C → 2C`。输出高宽减半、通道加倍：`(B,H,W,C) → (B,H/2,W/2,2C)`。参数：LayerNorm 有 `2·4C = 8C`；Linear 有 `4C·2C = 8C²`。`C=64` 时：512 + 32,768 = 33,280。四个象限按固定顺序拼接。改变顺序等价于通道置换；要与参考实现精确对齐，必须同步置换 norm 权重与 reduction 的输入列。

**常见误解.** 说 PatchMerging 让 token 数减半但通道宽度不变；或漏掉 LayerNorm 参数。另一个错误：使用不同的象限顺序却期待与 `timm` 权重兼容。

### Q4.6 相对位置偏置是什么？共享范围如何？

**答案.** 对 `M×M` 的窗口，共有 `(2M−1)²` 种相对偏移。`RelativePositionBias` 存一张可学习表 `((2M−1)², num_heads)`，以及固定的索引 buffer `(M², M²)`，把每对 token 映射到偏移行。输出是 `(num_heads, M², M²)`，在 softmax 前加到 attention logits 上。每个 **attention 层**拥有一张表；该层处理的所有窗口共享它，但不同 block 不共享。索引 buffer 不是参数，也不写入 `state_dict`（`persistent=False`）。

**常见误解.** 说同一 stage 内所有 block 共享一张表。每个 block 的 attention 层各有一张。另一个错误：把索引 buffer 当成可学习参数。

### Q4.7 Stage 与 block 有什么区别？

**答案.** **Block** 是一个 Swin Transformer block：LayerNorm → W-MSA 或 SW-MSA → 残差 → LayerNorm → MLP → 残差。**Stage** 是同一分辨率、同一通道宽度下的一组 block。Stage 之间由 PatchMerging 连接。一次前向中所有 stage 依次经过并联合训练，不存在“分 stage 训练”。Stage 内偶数 block 用 W-MSA，奇数 block 用 SW-MSA。最后一个 stage 只有一个窗口，所以 shift=0。

**常见误解.** 认为 stage 是独立模型，或以为每个 block 内部都做 PatchMerging。PatchMerging 只发生在 stage 过渡处。

### Q4.8 各 stage 的计算量如何变化？

**答案.** CIFAR 配置中，patch embedding 产生 `8×8` token（64）。Stage 1 保持 `8×8`；PatchMerging 后是 `4×4`（16）；再一次是 `2×2`（4）。窗口大小 4/2/2，因此各 stage 有 4/4/1 个窗口。每个 block 的投影与 MLP 项随 token 数缩放，窗口 attention 项随窗口数乘 `M²` 缩放；token 数减少时通道宽度加倍，所以每 token 成本上升。净效果是每 block 成本大致不变，而 stage 3 有 6 个 block，总成本最高。实测约 33.1M MACs/图，ViT 基线是 365.1M。

**常见误解.** 说“token 数减少 4 倍，所以计算量减少 4 倍”。每次 merge 通道宽度加倍，且各 stage 的 block 数不同。

### Q4.9 Swin vs. ViT 的对比到底说明了什么？

**答案.** Swin test accuracy 0.8201，ViT 0.8156，差 +0.45 pp。McNemar 精确 p = 0.267，配对 bootstrap 95% CI [−1.26, +0.34] pp，所以精度差异不显著。Swin 单图约 33.1M MACs，ViT 365.1M，约 1/11。合理的 claim 是：**在这个设置下 Swin 有竞争力且便宜得多**，不是“Swin 更准”。Swin 在这次运行中 AUC（0.9651 vs. 0.9525）与 NLL（0.6520 vs. 0.7323）也更好，但单 seed 不能建立普遍排序。

**常见误解.** 只报 +0.45 pp 的“胜利”，不给置信区间。另一个错误：把 CIFAR 规模的三 stage Swin 结果推广到官方 Swin-T/ImageNet 设置。

### Q4.10 Swin 的常见失败模式有哪些？

**答案.**

- `H` 或 `W` 不能被窗口大小整除；代码会抛 `ValueError`。
- `shift_size >= window_size`；代码会拒绝。
- 单窗口时使用 `shift_size > 0`；mask 构造函数会拒绝，`SwinTransformer` 在这种情况下把 shift 设为 0。
- 在 softmax 之后 mask，或 mask 约定反了（本仓库 `True = keep`）。
- mask 在 CPU 上构造，而 attention 在 MPS/CUDA 上；要传 `device=x.device`。
- `window_reverse` 与 `unshift_window` 顺序颠倒。
- 让多个 block 共享 `RelativePositionBias`，或把索引 buffer 当参数。

**常见误解.** 只在 CPU 上调试。MPS 设备 bug 只在把模型移到 GPU 后出现，CPU 测试全过。始终在目标设备上跑至少一次 smoke test。
