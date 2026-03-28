# Densify 点数增长修复文档

## 问题描述

Block-partitioned 3DGS 训练中，点数增长远低于 vanilla 3DGS（差距 30-40%），导致最终重建质量下降。

## 修复内容

### Fix 1: Packed Adam 动量跨 Densify 保留

**文件**: `scene/gaussian_model.py`

**问题**: `_init_packed_adam_state()` 在每次 `pack_to_buffer()` 时将 `_packed_exp_avg` 和 `_packed_exp_avg_sq` 全部归零。由于 `pack_to_buffer()` 在每次 `densify_and_prune` 后都会被调用（每 100 iter），所有 Gaussian 点的 Adam 一阶/二阶动量每 100 iter 丢失一次。而 vanilla 的 `cat_tensors_to_optimizer` 仅对新增点初始化零动量，老点动量保留。

这导致 opacity 训练不稳定，大量点 opacity 跌落被 prune（ours ~4000/step vs vanilla ~50/step）。

**修复**:
1. 新增 `_sync_packed_adam_to_optimizer_state()` 方法：在 densify 前将 `_packed_exp_avg/sq` 写入 `optimizer.state`，使 `cat_tensors_to_optimizer` 和 `_prune_optimizer` 能正确维护动量
2. 修改 `_init_packed_adam_state()`：从 `optimizer.state` 回填到 packed 格式，保留已有点的动量
3. 在 `densify_and_prune` 中 clone/split 前调用 sync

### Fix 2: SSIM 实现统一

**文件**: `train.py`

**问题**: ours 使用 `diff_gaussian_rasterization_wenqi_tam._C.fusedssim`，vanilla 使用 `fused_ssim_cuda.fusedssim`（来自 `submodules/fused-ssim`）。两个不同的 CUDA kernel 对相同输入产生不同的 SSIM 值和梯度，从 iter 1 开始导致参数分化。

**修复**: 将 `fast_ssim` 替换为 vanilla 的 `fused_ssim`。

### 补偿: DENSIFY_GRAD_SCALE = 0.7

**文件**: `pipeline_grad_sync.py`

Block 分割后，每个 block 的梯度被 `prefix_T`（其他 block 的透射率）缩放。这是 block-decomposed 架构的固有特性，无法通过代码修复消除。使用 `DENSIFY_GRAD_SCALE = 0.7` 降低 densify 阈值来补偿。

## 修复效果

| 阶段 | 修复前差距 | 修复后差距 |
|------|----------|----------|
| 单 block (iter 600-2000) | 10-20% | **1-4%** |
| 多 block (split 后) | 35-40% | **~15%** (DENSIFY_GRAD_SCALE=0.7 补偿) |

## 改动文件清单

| 文件 | 改动类型 |
|------|---------|
| `scene/gaussian_model.py` | 新增 `_sync_packed_adam_to_optimizer_state()`；修改 `_init_packed_adam_state()` |
| `train.py` | `fast_ssim` → `fused_ssim` |
| `pipeline_grad_sync.py` | `DENSIFY_GRAD_SCALE = 0.7` |
