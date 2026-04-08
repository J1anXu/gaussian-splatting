# Fix 1: 移除 _packed / _packed_exp_avg / _packed_exp_avg_sq 的 pin_memory

## 问题背景

从 commit `aae1ff6` 起修复了 3DGS 训练范式偏差（densify 动量保留），
训练速度从 ~5.86 it/s 降到更低，回归约 2x。

通过 trace 分析（`trace_nurips26_bicycle.json`），发现 `densify_and_prune`
单次耗时从 ~50ms 暴增到 ~2300ms；`packed_sparse_adam_step` 每步也
从 ~10ms 上升到 ~50ms。

## 根因

`gaussian_model.py` 中三个纯 CPU 张量被错误地分配成了 pinned memory：

```python
# pack_to_buffer()
packed = torch.empty(N, D, dtype=torch.float32, pin_memory=True)   # ← 错误

# _init_packed_adam_state()
new_exp_avg    = torch.zeros(N, D, dtype=torch.float32, pin_memory=True)  # ← 错误
new_exp_avg_sq = torch.zeros(N, D, dtype=torch.float32, pin_memory=True)  # ← 错误
```

**Pinned memory（页锁定内存）的本质**：绕过 CPU L3 cache，直接走 uncached 路径，
专为 DMA（GPU↔CPU）设计。对 CPU 自身的读写反而变慢（no write-combining，无缓存加速）。

只有 `_packed_staging`（DMA 的源 buffer，用于 h2d 传输）才需要 pinned memory。
其他三个张量全部只被 CPU 访问：

| 张量 | 用途 | 需要 pin？ |
|---|---|---|
| `_packed` | CPU 参数 buffer，index_select → staging → DMA | 否 |
| `_packed_staging` | DMA 源（h2d），也用作 d2h 梯度接收 | **是** |
| `_packed_exp_avg` | CPU Adam 一阶动量，纯 CPU 读写 | 否 |
| `_packed_exp_avg_sq` | CPU Adam 二阶动量，纯 CPU 读写 | 否 |

### 影响量化

- `_sync_packed_adam_to_optimizer_state()`：每次 densify 前把 exp_avg/sq 按列 `.clone()` 
  回写到 optimizer.state。pinned memory 上的 `.clone()` = 1.1GB uncached 读 → ~2250ms
- `packed_sparse_adam_step()`：每 iter 读写 exp_avg/sq（140MB × 2）。pinned → ~50ms，
  正常 heap → ~28ms

## 修复

**文件**: `scene/gaussian_model.py`

### 1. `pack_to_buffer()`（约第 791 行）

```python
# 修改前
packed = torch.empty(N, D, dtype=torch.float32, pin_memory=True)

# 修改后
# NOT pinned: _packed is accessed only by CPU (index_select → _packed_staging → DMA).
# Only _packed_staging (the DMA source) needs pin_memory. Pinning _packed causes
# uncached CPU reads/writes in the Adam update and index_select scatter.
packed = torch.empty(N, D, dtype=torch.float32)
```

### 2. `_init_packed_adam_state()`（约第 898-899 行）

```python
# 修改前
new_exp_avg    = torch.zeros(N, D, dtype=torch.float32, pin_memory=True)
new_exp_avg_sq = torch.zeros(N, D, dtype=torch.float32, pin_memory=True)

# 修改后
# NOT pinned: exp_avg/sq are pure CPU state, never DMA'd to GPU.
# pin_memory here causes uncached CPU reads (bypasses L3) making
# _sync_packed_adam_to_optimizer_state and packed_sparse_adam_step very slow.
new_exp_avg    = torch.zeros(N, D, dtype=torch.float32)
new_exp_avg_sq = torch.zeros(N, D, dtype=torch.float32)
```

`_packed_staging`（第 809 行）保持 `pin_memory=True` 不变，它是真正的 DMA buffer。

## 效果

测试环境：4090，bicycle，images_4，4.85M pts，8 blocks，iter 301-700

| 版本 | it/s | 训练时间(700iter) |
|---|---|---|
| baseline（含动量保留但有 pin_memory 错误） | 7.89 | 88s |
| fix（移除错误 pin_memory） | **8.14** | 87s |

提升 **+3.2%**，主要来自 `packed_sparse_adam_step` 从 ~50ms → ~28ms，
以及 densify_and_prune 从 ~2300ms → ~200ms（每 100 iter 一次）。

## 尝试过但失败的优化：Async Adam（ThreadPoolExecutor）

**思路**：将上一个 block 的 adam step 提交给 `ThreadPoolExecutor`，
让主线程立即继续下一个 block 的 h2d，理论上可以 overlap adam(N-1) 和 h2d(N)。

**结果**：6.81 it/s，比基线慢 14%。

**原因**：Python GIL 竞争。worker 线程执行 Python 代码（context manager、
函数调用等）时持有 GIL，阻塞主线程发 h2d/render/backward CUDA 调用。
GIL bounce 的代价（~20ms/iter）远超 adam-h2d overlap 的收益（~8ms/iter）。

结论：同步 adam 在 CUDA stream 层面已经与 GPU render+backward 天然重叠，
不需要额外线程。
