# S0 — Environment, Devices, and Shape Literacy / 环境、设备与形状素养

[English](#english) · [中文](#中文)

---

## English

### Q0.1 Why does `get_device()` prefer CUDA, then MPS, then CPU?

**Answer.** The goal is to use the fastest available accelerator without changing the model code. CUDA is the most mature training path (AMP, `pin_memory`, deterministic cuDNN, profilers). Apple MPS is the native GPU path on Apple Silicon and is sufficient for this course, but some operations and AMP combinations are less mature. CPU is the universal fallback. Every script obtains a `torch.device` from this function and moves tensors/models to it; no code hard-codes `.cuda()`.

**Common mistake.** Treating MPS as "CUDA with a different name" and forcing `torch.cuda.amp` or `pin_memory=True`. In this repository, `--amp auto` stays fp32 on MPS/CPU, and `pin_memory` is enabled only when `device.type == "cuda"`.

### Q0.2 Why does a GPU matmul benchmark need warm-up and synchronization?

**Answer.** GPU work is asynchronous: a kernel launch returns before the computation finishes. Without `torch.cuda.synchronize()` / `torch.mps.synchronize()`, you measure launch overhead, not compute time. The first call is also slower because cuBLAS/cuDNN initialize kernels and allocate workspaces. A reliable benchmark therefore does (1) a few warm-up iterations, (2) synchronization before starting the timer, and (3) synchronization before stopping it. The lesson is that **warm-up and synchronization can change the number by orders of magnitude**, so an unsynchronized first-call measurement is not a benchmark.

**Common mistake.** Reporting the first timed call, or using `time.perf_counter()` around an asynchronous operation without synchronizing. The result looks fast but is meaningless.

### Q0.3 For a 224×224 image with patch 16, how many tokens are there? What happens with patch 8?

**Answer.** The grid is `(224/16) × (224/16) = 14 × 14 = 196` patches. With a cls token, the sequence length is `T = 197`. With patch 8, the grid is `28 × 28 = 784`, so `T = 785`. The patch projection shrinks (each patch has `8² = 64` pixels instead of `16² = 256`), but the position table grows from `197D` to `785D`. In the `vit-pytorch` reference with `D=768`, 1000 classes, the total went from 86,543,080 to 86,551,144 parameters, a net `+8,064`. The accounting is: position table `+(785−197)×768 = +451,584`; patch projection `−768×3×(256−64) = −442,368`; the reference's first patch-embedding LayerNorm shrinks by `−2×(768−192) = −1,152`; net `+8,064`. The important point is not the exact cross-term but the direction: **more patches → more tokens → bigger position table and more attention compute**.

**Common mistake.** Saying "patch 8 has fewer parameters because the kernel is smaller" without accounting for the position table and the `N²` attention cost.

### Q0.4 What is the difference between parameters, MACs/FLOPs, and activation memory?

**Answer.**

- **Parameters** are learnable tensors: weights and biases. They determine model size and optimizer memory.
- **MACs** (multiply-accumulate operations) count the arithmetic for one forward pass. One MAC is usually counted as two FLOPs. They determine compute cost, not storage.
- **Activation memory** is the memory needed to store intermediate tensors for the backward pass. It scales with batch size, sequence length, and depth. Optimizer state (AdamW keeps two moments per parameter) adds to training memory.

For the ViT baseline: 5,355,850 parameters, 365,107,584 analytical MACs per image, and about 437 MB peak training memory at batch 32. These are three different quantities; a model with few parameters can still be expensive if it has long sequences.

**Common mistake.** Using "model size" to mean all three. A profiler's parameter count and its MAC count can also differ from the analytical values because of what it hooks (see Q0.5).

### Q0.5 Why does `thop` report fewer MACs than the analytical count?

**Answer.** `thop` hooks `nn.Linear` and `nn.Conv2d`. It does not see the two `torch.matmul` calls inside attention (`QKᵀ` and `weights @ V`). For the ViT baseline, the analytical total is 365,107,584 MACs and `thop` reports 345,638,784, a difference of 19,468,800. That difference is exactly `12 blocks × 2 × T² × D = 12 × 2 × 65² × 192 = 19,468,800`, the uncounted attention matmuls. A profiler is only useful when you know its hook set.

**Common mistake.** Treating `thop` as ground truth or comparing two models profiled with different hook coverage.

### Q0.6 Why seed Python, NumPy, and PyTorch separately?

**Answer.** They are independent RNGs. `random` drives some Python-side choices, NumPy drives array-based sampling and some transforms, and PyTorch drives tensor initialization and `torch.rand`. `set_seed` calls all three, plus `torch.cuda.manual_seed_all` when CUDA is present. DataLoader workers get a fourth layer: `worker_init_fn` seeds NumPy and Python with `base_seed + worker_id`, while PyTorch sets the worker's torch seed automatically. Missing any layer makes runs differ.

**Common mistake.** Setting only `torch.manual_seed(0)` and expecting identical data augmentation across runs, especially with `num_workers > 0`.

### Q0.7 What does "shape literacy" mean, and why learn it before writing model code?

**Answer.** It means being able to write the shape of every intermediate tensor without running the code: `(B,C,H,W) → (B,N,D) → (B,T,D) → (B,h,T,d_head) → (B,T,D) → (B,K)`. It lets you catch transposition errors, derive parameter counts, reason about memory, and debug a reference mismatch. Almost every bug in this course was first diagnosed as a shape question: softmax on the wrong axis, missing token dimension, or `reshape` after attention merging heads without moving the head axis back.

**Common mistake.** Memorizing layer names instead of shapes. Names change between libraries; shape rules do not.

---

## 中文

### Q0.1 为什么 `get_device()` 按 CUDA → MPS → CPU 选择？

**答案.** 目标是在不改模型代码的前提下使用最快的可用加速器。CUDA 的训练生态最成熟（AMP、`pin_memory`、确定性 cuDNN、profiler）。Apple MPS 是 Apple Silicon 的原生 GPU 路径，对本课程足够，但部分算子与 AMP 组合不如 CUDA 成熟。CPU 是通用兜底。所有脚本都从这个函数拿到 `torch.device`，再把张量/模型搬过去；没有代码硬编码 `.cuda()`。

**常见误解.** 把 MPS 当成“换了名字的 CUDA”，强行使用 `torch.cuda.amp` 或 `pin_memory=True`。本仓库中 `--amp auto` 在 MPS/CPU 上保持 fp32，只有 `device.type == "cuda"` 才开启 `pin_memory`。

### Q0.2 为什么 GPU matmul benchmark 需要 warm-up 与同步？

**答案.** GPU 工作是异步的：kernel launch 在计算完成前就返回。没有 `torch.cuda.synchronize()` / `torch.mps.synchronize()`，测到的是 launch 开销，不是计算时间。首次调用还会因为 cuBLAS/cuDNN 初始化 kernel、分配 workspace 而更慢。可靠 benchmark 因此要：(1) 先跑几次 warm-up；(2) 开始计时前同步；(3) 结束计时前同步。核心结论是：**warm-up 与同步会让数字差几个数量级**，所以未同步的首次调用不能当作 benchmark。

**常见误解.** 报告第一次计时的结果，或在异步操作外包 `time.perf_counter()` 却不同步。数字看起来很快，但没有意义。

### Q0.3 224×224、patch 16 有多少 token？换成 patch 8 会怎样？

**答案.** 网格是 `(224/16) × (224/16) = 14 × 14 = 196` 个 patch。加上 cls token 后序列长度 `T = 197`。patch 8 时网格是 `28 × 28 = 784`，`T = 785`。Patch 投影会变小（每个 patch 从 `16² = 256` 个像素变成 `8² = 64` 个），但位置表从 `197D` 增长到 `785D`。在 `D=768`、1000 类的 `vit-pytorch` 参考中，总参数从 86,543,080 变为 86,551,144，净 `+8,064`。账本是：位置表 `+(785−197)×768 = +451,584`；patch 投影 `−768×3×(256−64) = −442,368`；参考实现第一个 patch embedding LayerNorm 缩小 `−2×(768−192) = −1,152`；净 `+8,064`。重点不是精确的交叉项，而是方向：**patch 越多 → token 越多 → 位置表更大、attention 计算更贵**。

**常见误解.** 只说“patch 8 的卷积核更小所以参数更少”，却不考虑位置表与 `N²` 的 attention 成本。

### Q0.4 参数量、MACs/FLOPs、激活显存有什么区别？

**答案.**

- **参数量**是可学习张量（权重与 bias），决定模型大小与优化器显存。
- **MACs** 是一次前向的乘加次数。通常 1 MAC 记作 2 FLOPs。它决定计算成本，不是存储。
- **激活显存**是反向传播需要保存的中间张量。它随 batch size、序列长度、深度增长。AdamW 还为每个参数保存两个动量，增加训练显存。

ViT 基线：5,355,850 参数、单图 365,107,584 解析 MACs、batch 32 时约 437 MB 峰值训练显存。这是三个不同的量；参数少的模型如果序列很长，仍然可能很贵。

**常见误解.** 用“模型大小”同时指代三者。profiler 的参数计数与 MAC 计数也可能与解析值不同，取决于它 hook 了什么（见 Q0.5）。

### Q0.5 为什么 `thop` 报告的 MACs 比解析值少？

**答案.** `thop` 只 hook `nn.Linear` 与 `nn.Conv2d`，看不到 attention 里的两次 `torch.matmul`（`QKᵀ` 与 `weights @ V`）。ViT 基线的解析总量是 365,107,584 MACs，`thop` 报告 345,638,784，差 19,468,800，正好是 `12 × 2 × T² × D = 12 × 2 × 65² × 192 = 19,468,800`，即未统计的 attention matmul。只有知道 profiler 的 hook 集合，它的数字才有用。

**常见误解.** 把 `thop` 当 ground truth，或比较两个用不同 hook 覆盖范围 profile 的模型。

### Q0.6 为什么要分别给 Python、NumPy、PyTorch 设种子？

**答案.** 它们是独立的 RNG。`random` 驱动部分 Python 侧选择，NumPy 驱动数组采样与部分 transform，PyTorch 驱动张量初始化与 `torch.rand`。`set_seed` 三者都设，并在 CUDA 存在时调用 `torch.cuda.manual_seed_all`。DataLoader worker 还有第四层：`worker_init_fn` 用 `base_seed + worker_id` 给 NumPy 与 Python 播种，PyTorch 自动设置 worker 的 torch seed。少设任何一层，两次运行就会不同。

**常见误解.** 只设 `torch.manual_seed(0)`，却期待两次运行的数据增强完全一致，尤其是 `num_workers > 0` 时。

### Q0.7 什么是“形状素养”，为什么写模型代码前要先学？

**答案.** 它指不运行代码也能写出每个中间张量的 shape：`(B,C,H,W) → (B,N,D) → (B,T,D) → (B,h,T,d_head) → (B,T,D) → (B,K)`。它能帮你抓转置错误、推参数账、估显存、定位与参考实现的差异。本课程几乎所有 bug 首先都是一个 shape 问题：softmax 轴错、漏 token 维、attention 合并 head 后没把 head 轴移回去就 `reshape`。

**常见误解.** 背层名而不背 shape 规则。层名随库变化，shape 规则不变。
