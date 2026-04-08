# plan_3：基于当前实现与 CLM 思路的训练提速计划（保留 offloading 目标）

## 1. 背景与目标

当前代码采用分块 + CPU offloading，是为了在大场景下控制显存峰值并稳定训练。  
优化目标不是让所有块常驻 GPU，而是：

- 在**不破坏分块卸载策略**前提下提升吞吐（it/s）。
- 降低迭代抖动，尤其是 densify/prune 带来的长尾卡顿。
- 保持训练结果与现有实现一致（或可控偏差）。

---

## 2. 现状结论（代码 + trace）

### 2.1 已经做对的点

- 异步 CPU Adam（后台线程）已接入：`pipeline_grad_sync.py`
- D2H 已在独立通信流执行：`_comm_stream`

### 2.2 主要瓶颈（来自 `trace_nurips26_improve_bicycle_0408_1326.json`）

稳态窗口（iter 681–699）近似每 iter：

- `h2d_nograd` ≈ **15.0 ms**
- `h2d_grad` ≈ **19.2 ms**
- 合计 H2D ≈ **34.3 ms**（仍然很大）
- `packed_sparse_adam` ≈ **42.3 ms**
- `densify_stats` ≈ **13.4 ms**

并且在 `iter=700` 出现 6 次 `densify_and_prune`（单次 200ms+），形成明显卡顿峰值。

### 2.3 关键问题

在 `train.py` 中，Phase1 做完 `h2d_nograd + render_nograd` 后立刻 `deactivate_subset()`；  
Phase2 又对同块执行 `h2d_grad`。这导致同一可见块在单 iter 内重复 H2D。

---

## 3. 约束（必须遵守）

1. **不能把所有 block 放在 GPU 常驻**（违背 offloading 初衷）。
2. 任何提速都需要有显存水位保护，避免 OOM。
3. 改动优先选择“可回退、可开关”的形式（config/flag）。

---

## 4. 优化策略（按优先级）

## P0：不增显存的低风险优化（先做）

1. 关闭训练态的重同步 debug 检查（默认）  
   - 例如 `vi.max().item()` 这类会触发 GPU->CPU 同步的检查，只在 debug 模式打开。
2. densify/prune 分批预算化  
   - 不在某个 iter 集中处理全部 block。  
   - 每 iter 只处理有限个 block（轮转），平滑卡顿峰值。
3. 适当调高 `SKIP_SMALL_BLOCK_THRESH`（可实验）  
   - 直接降低每 iter 参与块数，减少 H2D 次数和后续 CPU Adam 压力。

预期：提升稳定性、减少长尾，不改变 offloading 结构。

## P1：显存受控的“局部复用”，而非全量驻留（核心）

目标：减少重复 H2D，但不破坏卸载。

方案：

- 仅缓存少量 block（Top-K，建议 K=1 起步）跨越 Phase1->Phase2 复用。
- 增加显存阈值门控：超过阈值立刻退回原始路径（立即 deactivate）。
- 其余 block 仍按现有 offload 方式执行。

建议参数：

- `PHASE12_REUSE_TOPK=1`
- `PHASE12_REUSE_MAX_ALLOC_GB`（按卡型设定，例如 75% 显存预算）

预期：在可控显存增量下，吃到部分 `h2d_nograd` 节省（不是全量 15ms，但通常可观）。

## P2：通信双流细化（中风险）

目前 D2H 有通信流，H2D 仍在默认流。  
进一步将 H2D 也迁到独立 `h2d_stream`，形成 D2H/H2D 双向并行机会（硬件允许时收益更明显）。

注意：

- 需要在 forward 前显式 wait h2d event，避免读未完成数据。
- 保留回退开关，防止某些硬件/驱动组合收益不稳定。

## P3：CPU 侧后续优化（可选）

1. CPU Adam 细粒度提前更新（更复杂，后做）
2. `visible_indices` CPU 缓存复用，减少重复 `.to("cpu")`

---

## 5. 实施顺序与验收

### Phase A（1-2 天）

- 落地 P0（debug 同步点开关 + densify 分批）
- 产出前后对比 trace（同场景、同窗口）

验收：

- it/s 提升或至少稳定性更好
- densify 峰值显著下降

### Phase B（2-4 天）

- 落地 P1（K=1 + 显存阈值保护）
- 增加 config 开关，便于 A/B

验收：

- `h2d_nograd` 总时长下降
- 显存峰值受控（无 OOM）
- 训练 loss 曲线与 baseline 一致或差异可解释

### Phase C（2-3 天，可选）

- 落地 P2（h2d_stream）
- 在不同场景做收益复现

---

## 6. 风险与回退

- 风险 1：局部复用导致显存抬升  
  - 处理：阈值门控 + Top-K 从 1 起步 + 一键关闭开关

- 风险 2：流同步错误导致数值异常  
  - 处理：先做严格同步版本，trace 验证后再调激进重叠

- 风险 3：densify 分批影响收敛节奏  
  - 处理：记录 densify 触发统计与最终质量，必要时只做“限峰不降频”

---

## 7. 总结

当前 pipeline 已有良好基础（异步 Adam + D2H 通信流），下一步最大收益点依然是 H2D 路径。  
但必须采用“**显存预算下的局部复用**”而非“全量驻留 GPU”，这样才能同时满足：

- 分块 offloading 的核心目标（控显存）
- 可观的训练吞吐提升（减重复 H2D + 平滑 densify 峰值）

