# Codex 诊断与后续优化路线

## 1. 目标

这份文档的目标不是再做零散猜测，而是把当前 pipeline 的真实大头和后续最值得改的方向排清楚。后面我们可以按这里的顺序，一项一项做 A/B。

当前判断：

- 还有明显优化空间
- 但剩下最大的空间已经不在小修小补，而在架构层
- 继续只抠 `frustum_culling` 或小同步点，收益会越来越小

---

## 2. 当前版本的真实瓶颈

基于最近几份 `781-800` 窗口 trace，当前代表性结果可参考：

- `timeline/trace_nurips26_improve_bicycle_0408_1351.json`
- `timeline/trace_nurips26_improve_2_bicycle_0408_1402.json`
- `timeline/trace_nurips26_improve_2_bicycle_0408_1420.json`

其中 `0408_1420` 这一版按 iter 聚合后的均值大致是：

| 项目 | 平均每 iter 耗时 |
|---|---:|
| `packed_sparse_adam` | 48.2 ms |
| `render_grad` | 22.0 ms |
| `backward` | 16.8 ms |
| `h2d_nograd` | 14.6 ms |
| `gather_nograd` | 10.7 ms |
| `h2d_grad` | 8.9 ms |
| `densify_stats` | 15.5 ms |
| `d2h_kick` | 2.9 ms |
| `frustum_culling` | 5.1 ms |
| `densify_and_prune` | 13.7 ms |
| 总计 | 171.5 ms |
| 去掉 `densify_and_prune` | 157.8 ms |

这说明当前最主要的瓶颈不是 culling，而是：

1. CPU Adam
2. 两阶段 offload 的数据搬运
3. 每个 block 两次 render 的整体成本

---

## 3. 当前 pipeline 的结构性代价

你的训练之所以“还是非常慢”，根本原因是它同时背了三层税：

### 3.1 参数主存放在 CPU

所以每轮都要交：

- CPU gather
- H2D
- D2H
- CPU Adam

只要主参数仍在 CPU，这部分税就不会消失，只能想办法减少体积、减少频率、或者让它们更好地 overlap。

### 3.2 两阶段训练

当前每个有效 block 都会经历：

1. `Phase1` no-grad render
2. `Phase2` grad render + backward

这意味着 render 本身天然要付两次。

### 3.3 每个 Gaussian 的 packed row 很重

当前 packed layout 在 `scene/gaussian_model.py` 里是：

- `xyz = 3`
- `f_dc = 3`
- `f_rest = 45`
- `scaling = 3`
- `rotation = 4`
- `opacity = 1`

总计 `59` 维。

其中最关键的是：

- `f_rest = 45/59 = 76.3%`

也就是说，你现在大多数 gather / H2D / D2H / Adam 带宽，都砸在高阶 SH 上。

---

## 4. 已经验证过有效的改动

### 4.1 关闭训练态的重同步 debug 检查

这类小同步点可以清掉，但它已经不是主矛盾。

### 4.2 `Phase1 -> Phase2` 局部复用

这个方向是成立的。`Top-K` 复用确实能打到重复 H2D。

但也已经看到一个现实约束：

- 稍微贪心一点，显存就会明显上升

所以这条线仍然能继续调，但不太像后面最大的收益来源。

### 4.3 `densify_and_prune` 分摊

这个改动更像“削峰”，不是“抬高稳态吞吐”的主解。

---

## 5. 还存在的大优化空间

下面按优先级和收益来排。

### 5.1 第一优先级：Adam 路径

当前 `packed_sparse_adam` 仍然是最大的单项。

现状：

- 代码在 `submodules/diff-gaussian-rasterization/cpu_adam.cpp`
- `packed_sparse_adam()` 的 OMP 线程数写死成了 `num_threads(16)`
- 当前机器 `nproc = 32`

这意味着至少有两个明确机会：

#### A. 低风险版本：让线程数可调

先做：

- `16 / 24 / 32` 线程 A/B
- 或改成 `omp_get_max_threads()`
- 或把线程数做成环境变量 / config 开关

潜在收益：

- 如果当前没有打满内存带宽，这一步可能能直接拿到一段白捡的加速

#### B. 中长期版本：重做优化器

真正的大刀是：

- GPU sparse Adam
- 或 hybrid optimizer

目标是让“当前 visible subset 的更新”尽量在 GPU 上完成，而不是每轮都回 CPU 做完整的 sparse Adam。

这是后面最像“量级提升”的方向之一。

---

### 5.2 第二优先级：属性分层，不要每轮都完整搬 59 维

这是我现在最看重的一条。

因为一旦把 `f_rest` 从热路径里拿掉一部分，收益会同时落在：

- `gather_nograd`
- `h2d_nograd`
- `h2d_grad`
- `d2h_kick`
- `packed_sparse_adam`

可行方向：

#### A. 热/冷参数分层

热参数：

- `xyz`
- `f_dc`
- `scaling`
- `rotation`
- `opacity`

冷参数：

- `f_rest`

#### B. `f_rest` 降频更新

例如：

- 几何参数每轮更新
- `f_rest` 每 `k` 轮更新一次

#### C. `f_rest` 低精度

例如：

- `f_rest` 梯度 D2H 用 fp16
- `f_rest` 的 Adam moments 用低精度

#### D. 训练日程更保守地升 SH

你当前 `oneupSHdegree()` 是每 1000 iter 升一级。对完整训练 wall-clock 来说，还可以进一步保守一些：

- 更晚进入 full SH
- keep_training / 后段 profiling 时只验证主路径，不一定每次都用最重配置

---

### 5.3 第三优先级：真正的双通信流

你现在已经把 D2H 放到了 comm stream，这很好。

但 H2D 还在默认流里。

所以还没做到真正的三段 overlap：

- 当前块 `render/backward`
- 上一块 `D2H`
- 下一块 `H2D`

建议方向：

- 增加 `h2d_stream`
- `d2h_stream` 和 `h2d_stream` 分开
- `default_stream` 只做 render/backward

这不是最大奖励项，但还是值得做，因为现在 H2D 仍然占了很大一块。

---

### 5.4 第四优先级：减少 block 数

这个方向很容易被忽略，但其实很重要。

从 trace 看，当前每个 iter 平均大约会处理：

- `~5.5` 个 block 的 `gather_nograd`
- `~5.5` 个 block 的 `render_nograd`
- `~5.5` 个 block 的 `render_grad`

也就是说，现在很多开销不是只和点数相关，而是和“块数”相关。

当前 outdoor 的：

- `SPLIT_SIZE_OUTDOOR = 800000`

这对 4090 来说可能偏保守。

所以很值得做一轮：

- `SPLIT_SIZE = 0.8M / 1.0M / 1.2M / 1.6M`

看：

- 平均块数
- 显存峰值
- iter time
- 质量变化

如果块数下降，很多 per-block 固定税都会跟着降：

- gather
- H2D
- render 启动开销
- backward 启动开销
- Adam 调度开销

这是一个很有希望的“低工程量、高潜力”方向。

---

### 5.5 第五优先级：让当前 packed GPU subset 保持 packed，不再拆 6 份又拼回去

现在的路径本质上是：

1. CPU 侧 packed
2. H2D 到 GPU
3. GPU 上拆成 6 份属性 tensor
4. backward 后再把 6 份 grad `cat` 回 packed grad
5. D2H 回 CPU

这套 pack/unpack 有明显税：

- H2D 后的 `clone`
- backward 后的 `torch.cat`
- 额外显存峰值

如果将来要做一刀更重的重构，我最看好的方向之一是：

- GPU subset 也保持 packed
- render/backward 尽量直接消费 packed 或半 packed 结构
- D2H 直接回传 packed grad

这条会比较难，但如果做通，能同时改善多段开销。

---

### 5.6 第六优先级：block 级粗筛

现在每个 block 都会进入点级 frustum culling。

而你其实已经有 block 的空间边界：

- `block_bounds`

所以有一个中等风险、但可能有用的方向：

- 先做 block-level AABB / sphere frustum test
- 完全不相交的 block 直接整块跳过
- 只对候选 block 做点级 culling

这条不一定是最大收益，因为当前 `frustum_culling` 不是主瓶颈。

但如果它能减少进入后续 Phase1 / Phase2 的 block 数，就会产生放大效应。

---

## 6. 现在不建议优先投入的方向

### 6.1 继续抠 frustum culling 细节

原因：

- 它不是当前最大瓶颈
- Gaussian-aware culling 更像“选择策略变化”，不一定更快
- 还会带来更复杂的状态维护

### 6.2 继续激进扩大 `Phase1 -> Phase2` retain

原因：

- 你已经遇到显存暴涨
- 这个方向收益递减很快
- 不像主架构改动那样能持续放大

### 6.3 继续围绕 densify 做小修补

原因：

- 现在它更多是峰值问题
- 不是当前稳态吞吐的最大来源

---

## 7. 很值得立刻确认的两个事实

### 7.1 `HALF` 目前并没有真正进入当前主 pipeline

`config.py` 里有：

- `HALF = False`

但当前主训练路径里，`pipeline_grad_sync.py` 并没有接 `grad_offload` 那套半精度 D2H 逻辑。

也就是说：

- 你现在看到的主路径 D2H 仍然是 FP32

这给了我们一个很清楚的小实验方向：

- 只把 D2H 的 packed grad 改成 fp16
- worker 线程里再 cast 回 fp32 做 Adam

这条不是最大的收益项，但应该是比较干净的实验点。

### 7.2 `packed_sparse_adam` 的线程数固定为 16

这是一个非常具体、非常适合先做的 A/B 点。

---

## 8. 我建议的实施顺序

下面这个顺序是按“尽量先拿到大收益，同时降低返工风险”排的。

### P0：先做低风险、大概率有结果的实验

1. `packed_sparse_adam` 线程数改成可调
2. 做 `16 / 24 / 32` 线程对比
3. 做 `SPLIT_SIZE` sweep：`0.8M / 1.0M / 1.2M / 1.6M`
4. 评估 `SKIP_SMALL_BLOCK_THRESH = 0.05 / 0.1 / 0.15`

### P1：pipeline 再挖一刀

1. 加 `h2d_stream`
2. 做 H2D / D2H 双流 overlap
3. 视情况尝试 D2H fp16

### P2：开始打真正的大头

1. 设计热/冷参数分层
2. 把 `f_rest` 从每轮完整热路径里拿掉一部分
3. 评估：
   - 降频更新
   - 低精度
   - 分层 H2D

### P3：高收益重构

1. hybrid / GPU sparse Adam
2. GPU subset packed 化
3. 视收益再考虑更激进的算法改动

---

## 9. 我现在最看好的三刀

如果只选三个最值得继续往下挖的方向，我会选：

1. `packed_sparse_adam` 线程数与实现优化
2. `f_rest` 分层，减少热路径 76% 的 payload
3. 减少 block 数，尤其是重新评估 `SPLIT_SIZE`

这三条相比之下，更像能把整体 wall-clock 明显往下压的方向。

---

## 10. 下一步建议

最合适的起点不是直接上大重构，而是从下面两项里挑一个先做：

1. 先改 `packed_sparse_adam` 线程数可调，做一轮严格 A/B
2. 先做 `SPLIT_SIZE` 扫描，看看 4090 上更大的 block 是否更划算

我个人更推荐先做第 1 个，因为：

- 代码改动小
- 结果很快能看出来
- 即使收益一般，也能帮我们判断 CPU Adam 到底是“线程没吃满”还是“已经碰到内存带宽墙”

