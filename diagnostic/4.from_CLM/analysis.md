# 基于 CLM 的分块训练性能诊断与改进计划

## 时间线数据（bicycle 场景，iter 681–700）

| 组件 | 平均耗时 | 占比 |
|------|----------|------|
| GPU 渲染+backward | 28.9ms | 20.3% |
| CPU Adam | 50.3ms | 35.5% |
| H2D（两个 phase 合计） | 31.0ms | 21.9% |
| gather（CPU→pinned） | 12.4ms | 8.7% |
| D2H（grad 复制） | 5.2ms | 3.7% |
| 其余开销 | 14.1ms | 9.9% |
| **总计** | **141.9ms** | **~7 it/s** |

GPU 利用率仅 17–26%。主要瓶颈是 Adam 和重复 H2D 阻塞了 GPU 流水线。

---

## 一、代码问题（已确认）

### 问题 1：Adam 在主线程同步执行，阻塞 GPU 流水线【已修复】

**位置：** `pipeline_grad_sync.py::flush_and_prepare()`（原始版本）

**现象：** 时间线中 `h2d_grad block=6` 在 `packed_sparse_adam block=4` 结束后 131μs 才开始。
CPU Adam（~50ms）完全串行占用主线程，GPU 在此期间空转等待下一块 H2D。

**可能的方案：** 用 `ThreadPoolExecutor(max_workers=1)` 将 Adam 提交到后台线程。
新版 `flush_and_prepare` 为每个块创建新的 `torch.cuda.Event()`，worker 线程持有前一块的 event 引用并独立 synchronize。

---

### 问题 2：每个 iter 对同一块做两次 H2D【未修复，优先级高】

**位置：** `train.py` Phase 1 + Phase 2 循环

**现象：**
```python
# Phase 1 (nograd)
submodel.kick_h2d_and_activate(requires_grad=False)
render(...)
submodel.deactivate_subset()  # ← GPU tensor 释放

# Phase 2 (grad)
submodel.kick_h2d_and_activate(requires_grad=True)  # ← 重复传同一块！
render(...)
loss.backward()
```
每个 iter 两个 phase 遍历相同的 valid_ids，导致每块 H2D 执行两次。
时间线中 Phase 1 H2D + Phase 2 H2D 合计 ~31ms，约占 iter 时间 22%。
理论上保留 Phase 1 的 GPU tensor 供 Phase 2 复用可节省约一半 H2D（~15ms）。

**根本原因：** `deactivate_subset()` 在 Phase 1 后立即释放 GPU tensor，Phase 2 无法复用。

---

### 问题 3：D2H 与 H2D 使用同一 CUDA stream，无法重叠【未修复，优先级高】

**位置：** `pipeline_grad_sync.py::kick_async_d2h()` + `gaussian_model.py::kick_h2d_and_activate()`

**现象：**
```python
# kick_async_d2h 中：
staging.copy_(gpu_grad_packed, non_blocking=True)   # D2H，default stream
# kick_h2d_and_activate 中：
self._xyz_gpu.copy_(self._xyz[:n_vis], non_blocking=True)  # H2D，也是 default stream
```
CUDA 同一 stream 严格串行：D2H 必须完成才能开始 H2D。
实测 D2H ~5ms，期间下一块的 H2D 被阻塞，GPU 等待 H2D 完成才能开始下一块 backward。
这 5ms 本可以与 GPU 当前块的 backward 重叠。

---

### 问题 4：`torch.nonzero()` 隐式 D2H sync【潜在，待排查】

**位置：** 未确认，需搜索 `train.py` 和 `gaussian_model.py`

`torch.nonzero()` 在 CUDA tensor 上调用时会触发隐式 `cuda.synchronize()`（因为需要把结果形状传给 CPU）。
CLM 用 `torch.nonzero_static()` + 预计算 count 避免此问题。
如果 Phase 1/2 循环中存在此调用，会在 GPU 计算中间插入不必要的同步点。

---

## 二、CLM 巧思中可轻松移植的部分

### ✅ 移植 1：独立通信流（comm_stream）+ 流优先级

**CLM 对应：** 巧思 3（双流双缓冲）、巧思 8（通信流优先级）

**改动范围：** `pipeline_grad_sync.py` 约 10 行

```python
# 在 PipelinedGradSync.__init__ 中：
self._comm_stream = torch.cuda.Stream(priority=-1)  # 高优先级通信流

# 在 kick_async_d2h 中，D2H 移到 comm_stream：
with torch.cuda.stream(self._comm_stream):
    staging.copy_(gpu_grad_packed, non_blocking=True)
    pin_sub_vf.copy_(sub_visibility_filter, non_blocking=True)
    pin_sub_radii.copy_(sub_radii, non_blocking=True)
    pin_vpt_grad.copy_(sub_viewspace_point_tensor.grad, non_blocking=True)

# event 改为 comm_stream event：
self._d2h_event.record(self._comm_stream)
```

**效果：** D2H（~5ms）与当前块 backward 重叠，D2H 不再占用 default_stream 时间槽。
**风险：低**，只需保证 event.synchronize() 在读 pinned buffer 前调用（现有逻辑已满足）。

---

### ✅ 移植 2：expandable_segments 消除内存碎片

**CLM 对应：** 巧思 13

**改动：** 启动脚本加一个环境变量，零代码修改

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py ...
```

**或在 `train.py` 顶部：**
```python
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

**效果：** 防止动态点数变化（densification/pruning）引起的显存碎片，
避免 `max_memory_reserved` 远大于 `max_memory_allocated` 的情况（CLM 观测碎片率可达 2×）。
**风险：无**，纯环境变量，不影响任何逻辑。

---

### ✅ 移植 3：排查并替换 `torch.nonzero()` → `torch.nonzero_static()`

**CLM 对应：** 巧思 6

**排查方式：**
```bash
grep -n "torch.nonzero\b" train.py
grep -n "torch.nonzero\b" scene/gaussian_model.py
```

**替换模式：**
```python
# 旧（隐式 D2H sync）：
indices = torch.nonzero(mask)

# 新（无 sync，需预知最大数量）：
max_count = mask.shape[0]
indices = torch.nonzero_static(mask, size=max_count)
count = mask.sum().item()  # 仅此处有一次 D2H，但可以与其他操作重叠
```

**效果：** 消除流水线中间的隐式 sync 气泡。
**风险：低**，需要知道最大 nonzero 数（worst case 用 tensor 长度即可）。

---

## 三、值得做但改动较大的部分

### ★★★ 改动 1：消除重复 H2D（保留 Phase 1 GPU tensor 供 Phase 2 复用）

**CLM 对应：** 巧思 1（属性分层）、巧思 3（双缓冲 Overlap）

**核心思路：** Phase 1 nograd 渲染后不立即 `deactivate_subset()`，把 GPU tensor 保留到 Phase 2 使用。
Phase 2 只需 `detach()` + `requires_grad_(True)` 重新挂载梯度，避免重复 PCIe 传输。

**改动位置：**
1. `train.py` Phase 1 循环：去掉 `deactivate_subset()`，改为记录 `retained_submodels`
2. `train.py` Phase 2 循环：跳过 `kick_h2d_and_activate()`，直接对已有 GPU tensor 设置 `requires_grad=True`
3. `gaussian_model.py::deactivate_subset()`：新增可选参数 `keep_gpu=False`
4. Phase 2 结束后统一 `deactivate_subset()`

**预期收益：** 节省约 15ms/iter（Phase 1 H2D 完全消除），GPU 利用率从 20% 提升至 ~35%。

**注意事项：**
- 需考虑 Phase 1 和 Phase 2 的块顺序可能不同（Phase 1 是 all valid_ids，Phase 2 是 grad_order）
- `kick_h2d_and_activate` 里除了 copy 还有 `activate_subset()`，需保留 activate 逻辑
- 内存压力：Phase 1 处理完所有块后 GPU 上同时存在多块的 tensor，显存峰值会上升
  - 缓解方案：Phase 2 按块顺序逐块 `deactivate_subset()` 前一块，严格控制 GPU 上活跃块数 ≤ 2

---

### ★★☆ 改动 2：D2H + 下一块 H2D 的真正重叠（双流流水线）

**CLM 对应：** 巧思 3（1F1B 双缓冲）、巧思 8（通信流优先级）

**现状（移植 1 之后）：**
```
时间轴：
  [backward_k] → [D2H_k on comm_stream, 与 backward 重叠] → [H2D_{k+1}] → [backward_{k+1}]
```

**更进一步（真正重叠）：**
```
时间轴：
  [backward_k] → [H2D_{k+1} on comm_stream] 与 [D2H_k on comm_stream] 同时进行
```

**核心改动：** 将 `kick_h2d_and_activate` 中的 `copy_()` 也移到 `comm_stream`：
```python
# gaussian_model.py::kick_h2d_and_activate 中：
with torch.cuda.stream(comm_stream):
    self._xyz_gpu.copy_(self._xyz[:n_vis], non_blocking=True)
    self._features_dc_gpu.copy_(...)
    # ...
```

**同步模式：**
- H2D 完成后（comm_stream event）→ `forward()` 可以开始（default_stream wait comm_stream event）
- D2H 完成后（另一个 comm_stream event）→ Adam worker 线程 synchronize 后读取 pinned buffer

**注意：** 若要 D2H 和 H2D 真正并行（PCIe 全双工），需要两个独立 stream（`d2h_stream` + `h2d_stream`）。
现代 PCIe Gen4 硬件通常支持 bidirectional DMA。

**预期收益：** 在改动 1 基础上额外节省 5ms D2H 等待。**总预期：140ms → ~50ms（20 it/s，2.7× 加速）。**

---

### ★☆☆ 改动 3：CPU Adam 线程复用与流水线深度扩展

**CLM 对应：** 巧思 5（Early CPU Adam）

**现状（问题 1 修复后）：** Adam 在后台 ThreadPoolExecutor 中异步执行，但每个块的 Adam 仍然完整等待自己的 D2H event 后才开始。

**CLM 的做法（Overlapped CPU Adam §4.2.2）：**
- 预计算每个参数的 finalization time `Lg = max{i | g ∈ S_i}`（该参数最后出现的 micro-batch 编号）
- 一旦某个参数所有 micro-batch 的梯度都已 D2H，立即开始 CPU Adam 更新该参数，不等所有参数就绪
- 用 `signal_tensor_pinned` 实现 GPU→CPU 的细粒度通知

**在我们项目中的类比：**
- 块按 `grad_order` 处理，某些 Gaussian 只在靠前的块中可见
- 这些"早 finalized"的参数可以更早开始 Adam，与后续块的渲染重叠
- 需要 `packed_sparse_adam_step` 支持部分参数更新（现有接口应该支持，因为已经是 sparse 更新）

**实现复杂度：高**，需要离线计算每个参数的 finalization block，并修改 Adam worker 的通知机制。
**建议：** 先完成改动 1 和 2，再评估是否需要此优化。

---

## 四、实施计划

### Phase 0：零风险快捷项（1天内完成）
- [ ] 启用 `expandable_segments:True`（移植 2）
- [ ] 全代码 `grep torch.nonzero`，逐一评估替换（移植 3）
- [ ] 测量当前性能基线（async Adam 版本的实际 it/s）

### Phase 1：独立通信流（2–3天）
- [ ] 在 `PipelinedGradSync.__init__` 中添加 `self._comm_stream = torch.cuda.Stream(priority=-1)`
- [ ] 修改 `kick_async_d2h()`：D2H copy 全部移到 `comm_stream`
- [ ] 修改 event record：`self._d2h_event.record(self._comm_stream)`
- [ ] 测量：D2H 是否与 backward 重叠，GPU 利用率变化

### Phase 2：消除重复 H2D（3–5天）
- [ ] 分析 Phase 1 → Phase 2 的块顺序关系，确认是否存在 grad_order ≠ valid_ids 的情况
- [ ] 修改 Phase 1 循环：不 `deactivate_subset()`，记录 `phase1_retained = {sm_id: submodel}`
- [ ] 修改 Phase 2 循环：复用 Phase 1 的 GPU tensor，仅做 `requires_grad_(True)`
- [ ] 增加严格的内存管控：Phase 2 每处理完一块后立即 deactivate，防止显存超限
- [ ] 测试 densification 后的正确性（pruning 后 visible_indices 会变化）

### Phase 3：PCIe 全双工（选做，1–2天）
- [ ] 拆分为 `d2h_stream` + `h2d_stream` 两个独立通信流（均 priority=-1）
- [ ] H2D 移到 `h2d_stream`，D2H 留在 `d2h_stream`
- [ ] 在 `forward()` 开始前 wait `h2d_stream` 上的 event
- [ ] 验证双向传输是否真正并行（用 nsys 观察 PCIe 利用率）

---

## 五、预期性能收益汇总

| 改动 | 节省时间 | 累计 iter 时间 | it/s |
|------|----------|----------------|------|
| 基线（旧代码） | — | 141.9ms | 7.0 |
| Phase 0：Fix 1 async Adam | ~50ms（移出关键路径） | ~90ms | 11.1 |
| Phase 1：独立 comm_stream | ~5ms D2H 重叠 | ~85ms | 11.8 |
| Phase 2：消除重复 H2D | ~15ms | ~70ms | 14.3 |
| Phase 3：PCIe 全双工 | ~5ms | ~65ms | 15.4 |
| 理想（完美流水线） | — | ~50ms | 20.0 |

**说明：** 理想值基于 GPU 渲染+backward（28.9ms）完全流水线化后的理论下限。
实际受制于 CPU gather（12.4ms/iter）和 densification 周期内 `pack_to_buffer()` 的开销。

---

*分析基于 `trace_nurips26_bicycle.json`（iter 681–700，bicycle 场景，约 5 块/iter）*  
*代码版本：async Adam 已修复（ThreadPoolExecutor），Phase 1–3 待实施*


