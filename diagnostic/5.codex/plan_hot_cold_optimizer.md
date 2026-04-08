# Hot/Cold Optimizer 改造计划

## 1. 目标

我们不做“全量参数常驻 GPU”的大爆改，而是走一条更符合当前 offloading pipeline 的路线：

- **GPU hot state**：`xyz + scaling + rotation + opacity`
- **CPU cold state**：`features_dc + features_rest`

最终目标：

1. 几何相关状态常驻 GPU
2. frustum culling 在 GPU 上完成
3. 热参数在 GPU 上更新
4. 冷参数仍在 CPU 上更新
5. 每轮只为颜色参数付 gather/H2D/D2H 的税

---

## 2. 为什么选这条路线

这条路线比“全量 GPU Adam”更适合当前代码结构，原因是：

1. 当前最大的 CPU 单项是 `packed_sparse_adam`
2. 但颜色参数占总 payload 的绝大部分，不适合直接全搬到 GPU
3. renderer 每轮都一定要用到几何、尺度、旋转、透明度
4. frustum culling 本身也天然更适合读 GPU 热状态

所以这条路线的核心思想是：

- **把真正热的东西放 GPU**
- **把真正重的颜色仍放 CPU**

---

## 3. 拆分方案

### 3.1 GPU hot state

第一版建议放这些：

- `_xyz_hot_gpu`
- `_scaling_hot_gpu`
- `_rotation_hot_gpu`
- `_opacity_hot_gpu`

以及对应 optimizer state：

- `_hot_exp_avg_gpu`
- `_hot_exp_avg_sq_gpu`

说明：

- 我建议把 `opacity` 也归到 hot，而不是只放 `xyz/scaling/rotation`
- 因为前向每次都要用它，而且 prune 也依赖它

### 3.2 CPU cold state

第一版保留在 CPU：

- `_features_dc`
- `_features_rest`
- 对应 CPU optimizer state

### 3.3 CPU shadow / 结构同步

需要保留 CPU 上的 hot shadow，但它不一定每轮强一致。

同步时机只放在这些结构性事件前后：

1. `densify_and_prune`
2. `split_in_half`
3. `save/checkpoint`
4. 需要 CPU 侧完整几何状态的 debug / 导出

也就是说：

- **正常训练 iter 内，GPU hot master 为准**
- **结构变更时，再把 hot state 同步回 CPU**

---

## 4. 最终训练数据流

最终希望的每 iter 流程如下。

### 4.1 前向前

1. GPU hot state 做 culling
2. 得到 `visible_indices`
3. CPU 只按 `visible_indices` gather 冷颜色参数
4. 只把冷颜色子集 H2D 到 GPU

### 4.2 前向与反向

1. GPU 上用：
   - hot 几何状态
   - cold 颜色子集
2. render / backward

### 4.3 更新

1. GPU optimizer 更新 hot state
2. cold color grad 子集 D2H 回 CPU
3. CPU optimizer 更新 cold colors

### 4.4 结构变更

在 densify / prune / split 触发前：

1. 先把 GPU hot master 同步回 CPU
2. 在 CPU 做结构性修改
3. 重建 hot/cold state
4. 继续训练

---

## 5. 实施顺序

我们不要一步到位，而是分成几个里程碑。

### P0：加观测，不改语义

目标：

- 先把新旧路径后面要比较的指标补齐

要做的事：

1. 给 `flush_last()` 外面单独加等待时间 trace
2. 给未来 hot/cold 路径预留独立事件名：
   - `gather_cold`
   - `h2d_cold`
   - `gpu_hot_adam`
   - `d2h_cold_grad`

验收：

- 不改训练结果
- timeline 能分清楚 CPU Adam 尾巴到底暴露了多少

---

### P1：先做 hot/cold 数据结构拆分，但训练逻辑先不变

目标：

- 先把结构拆干净，不立刻改优化器

要做的事：

1. 在 `GaussianModel` 中明确 hot/cold slice
2. 新增：
   - hot attrs 访问器
   - cold attrs 访问器
   - hot/cold pack/unpack helper
3. 让现有 packed 路径仍能工作
4. 不改训练语义，只完成结构分层

涉及文件：

- `scene/gaussian_model.py`
- `config.py`

验收：

- 训练结果不变
- 代码层面已经能清楚区分 hot 和 cold

---

### P2：让 GPU hot state 常驻，但先不改优化器

目标：

- 先把热状态常驻 GPU 路打通
- 先吃掉一部分几何 H2D 和 culling 成本

要做的事：

1. 为每个 submodel 创建 GPU hot master
2. `frustum_culling` 改成优先读 hot GPU state
3. 前向阶段不再从 CPU gather/h2d 几何 hot attrs
4. 只为 cold color 子集做 gather/H2D

第一版可以先继续：

- hot grad D2H 回 CPU
- hot update 仍在 CPU 做

说明：

- 这一版不是最终目标
- 但它能先验证 hot/cold 分离后的前向路径是否正确

涉及文件：

- `train.py`
- `scene/gaussian_model.py`
- `utils/camera_utils.py`

验收：

- `h2d_*` 中几何部分明显下降
- culling 能稳定在 GPU hot state 上工作

---

### P3：引入 GPU hot optimizer

目标：

- 把热参数的更新从 CPU 搬到 GPU

要做的事：

1. 新增一个 CUDA kernel：
   - `packed_sparse_hot_adam_cuda`
   - 或更通用的 `packed_sparse_adam_cuda`
2. 输入：
   - hot visible subset
   - hot grads
   - full hot moments
   - `visible_indices`
   - hot `lr_per_col`
3. 输出：
   - updated hot GPU subset
   - updated full GPU hot moments

第一版建议：

- hot state 的 master 在 GPU
- 不做每 iter 的 hot params 回 CPU
- 只在结构变更点同步回 CPU

涉及文件：

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/adam.cu`
- `submodules/diff-gaussian-rasterization/ext.cpp`
- `pipeline_grad_sync.py`
- `scene/gaussian_model.py`

验收：

- `packed_sparse_adam` CPU 时间大幅下降
- 训练不崩
- hot 参数数值与旧路径在短窗口内可对齐

---

### P4：冷参数保留 CPU optimizer

目标：

- 只让颜色参数继续走 CPU 更新

要做的事：

1. backward 后只回传 cold color grad
2. CPU 只对 cold state 做 sparse Adam
3. 从 `pipeline_grad_sync.py` 中拆开：
   - hot GPU optimizer
   - cold CPU optimizer

说明：

- 这一阶段是 hot/cold optimizer 方案真正成立的关键

验收：

- 每 iter D2H 只剩 cold color grad
- CPU Adam 负担明显下降

---

### P5：处理结构变更

目标：

- 让 densify / prune / split 不把 hot/cold 双状态弄乱

要做的事：

1. 在 `densify_and_prune` 前同步 hot GPU -> CPU
2. CPU 侧完成：
   - append
   - prune
   - split
3. 再重建：
   - GPU hot master
   - GPU hot moments
   - CPU cold state
   - CPU cold optimizer state

这是最容易出 bug 的阶段，必须单独做。

验收：

- 不再出现索引错位
- split 后各 block 的 hot/cold 状态一致

---

## 6. 第一阶段不做的事

为了控制风险，第一阶段明确不做：

1. 不做全量参数 GPU 常驻
2. 不把 `features_rest` 搬到 GPU 更新
3. 不立刻做 Gaussian-aware culling
4. 不同时重写 densify 算法

先把主干打通，再决定是否继续扩 GPU 范围。

---

## 7. 风险点

### 7.1 最大风险：索引一致性

因为你现在整个 pipeline 都依赖 `visible_indices` 去对齐：

- hot GPU state
- cold CPU state
- merge 阶段的可见点映射

一旦 densify / prune / split 后没同步好，就会直接错位。

### 7.2 第二风险：结构变更时机

GPU hot state 如果长期不回 CPU，没有问题；
但在结构变更前必须显式同步。

### 7.3 第三风险：收益被颜色路径吃掉

因为颜色仍然占总 payload 大头。

这意味着：

- hot/cold 方案肯定能提速
- 但它不一定一次就解决所有 H2D/D2H 开销

---

## 8. 我建议的第一刀

真正开始改时，不建议先写 GPU Adam kernel。

建议先做：

### 第一刀：P1

先把 `GaussianModel` 的 hot/cold 结构拆出来，但训练逻辑先不变。

理由：

1. 这是后面所有改动的基础
2. 一旦结构没拆清楚，后面会反复返工
3. 这一步本身风险相对最低

### 第二刀：P2

在结构拆干净后，再把：

- GPU hot 常驻
- GPU culling
- cold-only gather/H2D

先打通。

### 第三刀：P3

最后再引入 GPU hot optimizer。

---

## 9. 验证标准

每完成一个阶段，都要至少看这几项：

1. `iter time`
2. `packed_sparse_adam`
3. `gather_nograd`
4. `h2d_nograd`
5. `h2d_grad`
6. `d2h_kick`
7. `max_memory_allocated`
8. `点数 / visible 点数 / block 数`

如果这些指标没有一起看，很容易误判。

---

## 10. 我们接下来怎么做

下一步建议就从 **P1：拆 hot/cold 数据结构** 开始。

也就是说，下一次修改的目标不是“先提速”，而是：

- 先把数据结构改成支持这条路线
- 保持当前训练逻辑还能正常跑

这是后面所有优化能稳落地的前提。

