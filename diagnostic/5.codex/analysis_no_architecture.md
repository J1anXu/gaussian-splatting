# 不改整体训练架构时的提速空间分析

## 结论

在当前 `nurips26_improve_2` 分支上，如果 **不改整体训练架构**，训练速度仍然有继续提升的空间，但更现实的预期是：

- 工程型优化：`10%~25%`
- 如果接受一定的质量/速度 tradeoff：还可以再多拿一截
- 想要再来一次“量级变化”，基本还是要动架构

当前版本真正的大头已经不是 culling，而是：

1. `packed_sparse_adam`
2. `gather + H2D`
3. 两阶段 `render`
4. `densify_stats / densify`

## 当前瓶颈判断

从现有代码路径看，主要耗时集中在下面几段：

- CPU Adam：
  - `/home/jian/gaussian-splatting/pipeline_grad_sync.py`
  - `/home/jian/gaussian-splatting/scene/gaussian_model.py`
- `gather + H2D`：
  - `/home/jian/gaussian-splatting/train.py`
  - `/home/jian/gaussian-splatting/scene/gaussian_model.py`
- 两阶段渲染：
  - `Phase1` no-grad render
  - `Phase2` grad render + backward
- `densify_stats / densify / prune`

换句话说，现在最主要的问题不是“看不见哪些点”，而是“每轮要搬多少数据、渲多少次、更新多少参数”。

## 最值得做的方向

### 1. H2D 真正做成独立流重叠

当前实现里，D2H 已经有单独的通信流，但 H2D 还没有完全对称地独立出来。

这意味着现在还没有完全实现：

- 上一块 D2H
- 当前块 render/backward
- 下一块 H2D

三者同时重叠。

这条路径的优点是：

- 不改训练语义
- 不改 offloading 大框架
- 直接针对当前稳定版本里的主路径开销

这是我认为当前最值得优先尝试的 pipeline 优化。

### 2. 重新评估 block 大小

当前 outdoor block split size 是：

- `SPLIT_SIZE_OUTDOOR = 800000`

如果 block 太碎，会显著放大固定成本：

- `pre_gather`
- H2D 次数
- render 次数
- async Adam 提交次数
- merge 次数

在 4090 这类显卡上，更大的 block 往往可能更划算，只要显存还顶得住。

建议认真扫一轮：

- `800k`
- `1.0M`
- `1.2M`

这是“不改架构”前提下非常值得做的参数实验。

### 3. 更激进地跳过小块

当前已经有：

- `SKIP_SMALL_BLOCK_THRESH = 0.05`

如果小块很多，它们会带来明显的：

- 调度税
- 搬运税
- render 税

适当提高这个阈值，可能会直接减少：

- 每轮 block 数
- gather 次数
- H2D 次数
- grad worker 排队数

这条通常能带来比较直接的速度改善，但需要同时观察质量回退。

### 4. 控制 densify 和点数增长

很多成本本质上都是按点数线性增长的：

- H2D
- Adam
- render
- backward
- densify_stats

所以控点数增长，本身就是一种全局提速。

更现实的调法包括：

- 提前停止 densify
- 拉大 densification interval
- 更保守地 densify
- 更激进地 prune
- 给后段点数增长设更强约束

如果后段总点数能明显压下来，很多项会一起下降。

### 5. 压 `f_rest` 负担

在当前 packed 表达里，高阶 SH 的 `f_rest` 占了非常大的 payload。

这意味着每轮很多成本，其实都在为 `f_rest` 交税：

- gather
- H2D
- D2H
- packed Adam

如果不改整体架构，最有潜力的算法侧优化之一就是：

- 延后 SH degree 提升
- 更长时间只训练低阶 SH
- 降低 `f_rest` 的训练活跃度
- 让高阶 SH 的更新更稀疏

这条不是纯工程优化，但非常可能是“不改大架构时最有价值的算法刀”。

## 还可以做，但优先级次一些

### 6. 继续抠 CPU Adam，但别指望它单独救全局

当前 `CPU_ADAM_OMP_THREADS = 24` 已经比较像甜点区。

这意味着 Adam 侧还能继续做的，多半是：

- OMP 绑核/亲和性
- 减少额外 pack/unpack
- 减少 worker 被小任务打断

但如果只是不改算法、不改数据流，Adam 再往上抠，收益大概率不如前面几项。

## 不建议再大投入的方向

### 1. 继续深挖 frustum culling

它现在已经不是主矛盾了。

### 2. 再做 hot/cold 常驻实验

这条已经在当前结构里验证过，收益和复杂度不成正比。

### 3. 继续找 debug/sync 级别的小点

这类收益已经基本拿完了，不太可能再有明显提升。

## 我建议的实施顺序

如果我们坚持“不改整体训练架构”，我建议按这个顺序推进：

1. `H2D stream` 重叠
2. 扫 `SPLIT_SIZE_OUTDOOR`
3. 扫 `SKIP_SMALL_BLOCK_THRESH`
4. 调 `densify / prune` 策略，控制点数增长
5. 设计一版 `f_rest` 降负担方案

## 一句话总结

不改整体架构，仍然有继续提速的空间，但重心已经变成：

- 减少每轮固定调度和搬运成本
- 减少 block 数
- 减少点数增长
- 减少高阶 SH 的 payload

而不是继续在 culling 上深挖。
