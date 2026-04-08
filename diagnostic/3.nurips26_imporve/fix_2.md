# Fix 2: Async Adam — 释放 GIL + ThreadPoolExecutor

## 背景

fix_1 中尝试过 `ThreadPoolExecutor` 异步 Adam，结果 **慢了 14%**（6.81 it/s vs 基线 7.89）。
结论是 "GIL bounce 代价 (~20ms/iter) 远超 adam-h2d overlap 收益 (~8ms/iter)"。

本次修复找到了 GIL 竞争的根因并解决它。

## 根因分析

`packed_sparse_adam` 在 `ext.cpp:33` 的 pybind11 绑定：

```cpp
m.def("packed_sparse_adam", &packed_sparse_adam);  // 默认持有 GIL
```

pybind11 的 `m.def()` **默认不释放 GIL**。当 worker 线程调用此函数时：

1. Worker 获取 GIL → 进入 C++（含 OpenMP 16 线程）→ 整个计算期间 GIL 被锁
2. 主线程需要 GIL 来发射 CUDA 调用（`copy_`, `render`, `backward` 都经过 Python）
3. 主线程被阻塞，等 worker 的 C++ 计算完成才能继续
4. 效果等同于串行，但多了线程切换开销 → 比原来更慢

**关键事实**：`packed_sparse_adam` 是纯 C++ 指针运算 + OpenMP 并行，
不访问任何 Python 对象（PyObject），完全可以安全释放 GIL。

## 修复方案

### 改动 1：释放 GIL（核心，1 行）

**文件**：`submodules/diff-gaussian-rasterization/ext.cpp:33`

```cpp
// 改前
m.def("packed_sparse_adam", &packed_sparse_adam);

// 改后
m.def("packed_sparse_adam", &packed_sparse_adam,
      py::call_guard<py::gil_scoped_release>());
```

需要在文件头确认 `#include <pybind11/pybind11.h>` 已有（通过 `torch/extension.h` 间接包含）。
`py::call_guard` 在 `pybind11` 命名空间，需要 `namespace py = pybind11;` 或直接用 `pybind11::call_guard`。

同时检查 `index_copy` 等其他纯 CPU 函数是否也可以释放 GIL（`frustum_culling_idx` 等）。

### 改动 2：pipeline_grad_sync.py — 用 comm_stream + ThreadPoolExecutor

**2a. 添加 comm_stream 用于 D2H**

```python
# __init__ 中
self._comm_stream = torch.cuda.Stream(priority=-1)  # 高优先级通信流
```

`kick_async_d2h` 中所有 D2H copy 移到 comm_stream：
```python
with torch.cuda.stream(self._comm_stream):
    staging.copy_(gpu_grad_packed, non_blocking=True)
    pin_sub_vf.copy_(sub_visibility_filter, non_blocking=True)
    pin_sub_radii.copy_(sub_radii, non_blocking=True)
    pin_vpt_grad.copy_(sub_viewspace_point_tensor.grad, non_blocking=True)
```

event record 也在 comm_stream 上：
```python
self._d2h_event.record(self._comm_stream)
```

效果：D2H (~5ms) 与 default stream 上的下一个 H2D/render 并行。

**2b. 添加 ThreadPoolExecutor(1) 异步执行 Adam**

```python
from concurrent.futures import ThreadPoolExecutor

# __init__ 中
self._adam_executor = ThreadPoolExecutor(max_workers=1)
self._adam_futures = []
```

`flush_and_prepare` 不再在主线程 sync + 执行 prev adam，
而是提交给 worker：

```python
def _submit_adam(self, event, adam_fn, densify_fn):
    """Worker 线程：等 D2H 完成 → 执行 adam + densify_stats"""
    def _worker():
        event.synchronize()  # 等自己的 D2H event
        adam_fn()
        densify_fn()
    self._adam_futures.append(self._adam_executor.submit(_worker))
```

每个 block 需要独立的 CUDA Event（不能共用，因为 worker 异步消费）：
```python
# flush_and_prepare 中，每次创建新 event
event = torch.cuda.Event()
event.record(self._comm_stream)
self._submit_adam(event, adam_fn, densify_fn)
```

**2c. flush_last 做 iteration barrier**

```python
def flush_last(self):
    """等待所有异步 adam 完成，保证 iter N 的参数更新在 iter N+1 的 pre_gather 之前完成"""
    for fut in self._adam_futures:
        fut.result()  # 阻塞等待
    self._adam_futures.clear()
    self.tracer.flush_gpu_events()
```

**2d. run_deferred_densify 不再需要**

之前 densify_stats 和 adam 是分开 defer 的，现在统一在 worker 中执行。
`train.py` 中 `grad_sync.run_deferred_densify()` 调用可以移除（或变成 no-op）。

### 改动 3：_build_lr_per_col 缓存

`_build_lr_per_col` 在 worker 线程中执行，需要访问 Python 对象（optimizer.param_groups），
会短暂需要 GIL。可以预计算并缓存：

```python
# packed_sparse_adam_step 中
def packed_sparse_adam_step(self, idx, grad_subset, iteration):
    self._packed_adam_step += 1
    # lr_per_col 在主线程预计算好，存在 self 上
    cpu_adam.packed_sparse_adam(...)  # ← 这里才释放 GIL
```

或者在 `flush_and_prepare` 主线程中预计算 lr_per_col 传给闭包：
```python
lr_per_col = sm._build_lr_per_col(iteration)  # 主线程，GIL 安全
# 闭包中直接用 lr_per_col，不再调 _build_lr_per_col
```

## 安全性分析

| 关注点 | 结论 |
|--------|------|
| 线程安全 | 每个 block 的 `_packed`, `_packed_exp_avg`, `_packed_exp_avg_sq` 是独立 tensor，block 间 adam 无共享状态 |
| 数据依赖 | block X 的 adam (iter N) 必须在 block X 的 pre_gather (iter N+1) 之前完成 → `flush_last` barrier 保证 |
| staging buffer | adam 读 `_packed_staging`，D2H 写 `_packed_staging` → event.sync 保证 D2H 完成后才读 |
| GIL-free 安全 | `packed_sparse_adam` 纯 `float*` 指针运算 + OpenMP，不 touch PyObject |
| densify_and_prune | 每 100 iter 一次，在 worker 中执行，会调 `pack_to_buffer` 修改 model 结构。安全因为同一 block 不会被并发访问（worker 单线程，block 按序处理） |
| CUDA event 生命周期 | 每个 block 创建独立 event，闭包持有引用，GC 在 future.result() 后释放 |

## 异步后的时间线

```
当前（同步 Adam）:
GPU:  [render₀][bwd₀][D2H₀]...........[render₁][bwd₁][D2H₁]...........[render₂]
CPU:                        [adam₀][H2D₁]                   [adam₁][H2D₂]
                            ↑ GPU 空等 adam

异步 Adam（GIL 释放后）:
GPU:  [render₀][bwd₀][D2H₀][H2D₁][render₁][bwd₁][D2H₁][H2D₂][render₂][bwd₂]
CPU:                  [adam₀ (worker, GIL-free)]  [adam₁ (worker)]  [adam₂]
                      ↑ 主线程立即发 H2D₁, 不等 adam₀
```

## 预期收益

| 状态 | 关键路径 | iter 时间 | it/s |
|------|----------|-----------|------|
| 当前 | GPU + Adam 串行 | ~120ms | ~8 |
| fix_2 async adam | GPU + H2D（Adam 离开关键路径） | ~60-70ms | ~14-16 |

CPU Adam (~30ms) 完全隐藏在 GPU render+backward (~29ms) + H2D (~15ms) 后面。
**预期 ~2x 加速**。

## 实施步骤

1. `ext.cpp` 加 GIL release → 重新编译 diff-gaussian-rasterization
2. `pipeline_grad_sync.py` 加 comm_stream + ThreadPoolExecutor + per-block event
3. `train.py` 移除 `run_deferred_densify()` 调用（或 no-op）
4. 跑 benchmark iter 301-700 对比 it/s
5. 跑 trace 验证 GPU 空闲间隙是否消除
6. 跑完整训练验证 loss/PSNR 不退化（async 不影响数值，只是执行顺序变化）

## 风险

- **OpenMP + Python 线程**: worker 线程调 `packed_sparse_adam` 会在该线程中 fork OpenMP 线程。
  需确认 OpenMP runtime 能正确处理非主线程的 `omp parallel for`（通常可以，但 libgomp 有已知的初始化问题）。
  如果出问题，备选方案：在 C++ 侧用 `std::thread` 而非 OpenMP。
- **Event 创建开销**: 每 block 每 iter 创建一个 `torch.cuda.Event()`，约 6 events/iter。
  CUDA event 创建很轻（<1us），可忽略。如有担忧可用 event pool。
