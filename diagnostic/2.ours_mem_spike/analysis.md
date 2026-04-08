# GPU 内存尖刺分析

> 分析时间: 2026-03-26
> 分支: nurips26, commit: 12a029a (CapGS)
> 场景: bicycle (0326_0155.log)
> 训练配置: SPLIT_SIZE=800K, GPU_CACHE_THRESHOLD_GB=1.0, 8 blocks

---

## 1. 现象

训练过程中 GPU reserved memory (RSV) 出现大量尖刺:
- peak_rsv 最大值: **2.02 GB** (iter 29491), 而 peak_alloc 最大仅 **1.35 GB** (iter 20862)
- 稳态阶段 (iter 15000-30000) RSV 连续 iteration 间平均跳变 **0.30 GB**
- RSV 跳变 > 0.5GB 的次数: **5,211 次** / 30K iterations
- fragmentation ratio (peak_rsv / peak_alloc): 平均 1.29x, 最差 **1.79x** (iter 19738)

---

## 2. 内存阶段划分

| 阶段 | Iteration | 点数 | Blocks | peak_alloc | 特征 |
|---|---|---|---|---|---|
| Warmup | 1-500 | 0.05M | 1 | ~0.40 GB | 稳定 |
| Densify 早期 | 500-2200 | 0.05M→0.82M | 1 | →0.80 GB | 每100 iter densify |
| Densify + Split | 2200-10200 | 0.82M→3.96M | 1→8 | →1.24 GB | 6次 block split |
| Densify 晚期 | 10200-15000 | 3.96M→4.64M | 8 | avg 1.05 GB | 最后一次 densify at iter 14900 |
| 稳态 | 15000-30000 | 4.64M | 8 | avg 1.08, max 1.35 GB | RSV 波动最剧烈 |

### Block Split 事件

| Iteration | Blocks 变化 | 点数 | RSV 影响 |
|---|---|---|---|
| 2200 | 1→2 | 0.82M | RSV 从 1.02 降至 0.48 |
| 3700 | 2→3 | 1.56M | RSV 从 0.98 降至 0.46 |
| 3900 | 3→4 | 1.69M | RSV 从 1.09 降至 0.67 |
| 6800 | 4→5 | 2.99M | RSV 从 1.21 降至 0.81 |
| 7100 | 5→7 | 3.12M | RSV 从 1.25 降至 0.41 |
| 10200 | 7→8 | 3.96M | RSV 从 1.26 降至 0.60 |

每次 split 后 RSV 立即下降 0.4~0.8 GB, 因为 block 变小后单次分配尺寸减小, CUDA allocator 复用率提高.

---

## 3. 根因分析

### 根因 1: 8-block pipeline 的 allocate/free 模式与 CUDA caching allocator 冲突 (主因)

每个 iteration 对 8 个 block 依次执行:

```
for each block:
  kick_h2d_and_activate()  → GPU 分配 gpu_packed + 6 个 attribute tensor
  render()                 → rasterizer 分配大量临时 buffer
  backward()               → 反向传播分配梯度 tensor
  deactivate_subset()      → 将 _xyz_gpu 等设为 None, 释放 GPU tensor
```

**问题链条:**
1. `deactivate_subset()` (gaussian_model.py:963) 释放 tensor 后, CUDA caching allocator **不归还给 OS**, 只标记为"可复用"
2. 下一个 block 的 visible 点数不同 (vis% 在 10%~60% 间随相机波动), 分配尺寸不匹配 → 无法复用已释放的 block
3. allocator 被迫申请新的 reserved 内存 → `reserved` 持续膨胀
4. 直到 `reserved - allocated > GPU_CACHE_THRESHOLD_GB (1.0GB)` 时触发 `empty_cache()` (train.py:377-378) 一次性释放
5. 周而复始, 形成**锯齿状 RSV 波形**

**数据佐证:**
- vis% 20~25% 时 mean peak_alloc = 0.94 GB
- vis% 40~45% 时 mean peak_alloc = 1.15 GB
- 同一 iter 内不同 block 的 visible 点数差异导致每次分配尺寸不一致

### 根因 2: Grad phase 的 clone() 与 merge 结果长期驻留叠加

`kick_h2d_and_activate(requires_grad=True)` (gaussian_model.py:1091-1103) 中, 每个 attribute 都 `.clone()`:

```python
gpu_packed = staging.cuda(non_blocking=True)     # H2D 完整 tensor
self._xyz_gpu          = gpu_packed[:, s:e].clone()   # clone 1
self._features_dc_gpu  = gpu_packed[:, s:e]...clone() # clone 2
self._features_rest_gpu = ...clone()                   # clone 3
self._scaling_gpu      = ...clone()                    # clone 4
self._rotation_gpu     = ...clone()                    # clone 5
self._opacity_gpu      = ...clone()                    # clone 6
```

`gpu_packed` 和 6 个 clone 副本**同时存在于 GPU**, 直到 `gpu_packed` 离开作用域被 GC.

同时 merge 阶段的中间结果 (train.py:270-275):
- `C_sorted` [K,3,H,W]
- `prefix_T` [K,1,H,W]
- `block_rank` [K,H,W]
- `colors_bg` [3,H,W]

这些 tensor 一直存活到整个 grad 循环结束, 与每个 block 的 render+backward 临时分配叠加, 形成内存高峰.

### 根因 3: Nograd + Grad 双渲染产生大量碎片

每个 iter 对每个 block 渲染**两次** (nograd phase + grad phase):
- 8 blocks × 2 passes = **16 次** allocate/free 循环
- 每次 rasterizer 内部根据 visible 点数分配不同大小的临时 buffer
- 产生大量**尺寸各异的碎片块**, 加剧 CUDA allocator 的碎片化

稳态数据:
- iter 内瞬态分配: render+backward 临时 buffer 导致 peak_alloc 比稳态 alloc (~0.36GB) 高出平均 **0.72 GB**
- 最差达 0.96 GB (iter 17084)

---

## 4. 改进方案

### 方案 A: 降低 empty_cache 阈值 [最简单, 立即可做]

```python
# config.py
GPU_CACHE_THRESHOLD_GB = 0.3  # 原值 1.0
```

- 效果: RSV 尖刺峰值从 ~2.0 GB 降至 ~1.3 GB, 锯齿幅度减半
- 代价: `empty_cache()` 调用更频繁, 可能每 iter 都触发, 预计降速 1-3%
- 改动: 1 行

### 方案 B: Grad phase clone 后立即释放 gpu_packed [简单]

```python
# gaussian_model.py: kick_h2d_and_activate(), requires_grad=True 分支末尾
if requires_grad:
    self._xyz_gpu = gpu_packed[:, s:e].clone()
    # ... 其他 5 个 clone ...
    del gpu_packed  # 新增: 立即释放, 避免 clone 期间双份驻留
```

- 效果: 减少每 block 几十 MB 的瞬态峰值
- 改动: 1 行

### 方案 C: Merge 中间结果尽早释放 [中等改动]

grad 循环中, `C_sorted`, `prefix_T` 在最后一个 block 处理完后才释放. 可以在每个 block 处理完后释放不再需要的 slice:

```python
# train.py: grad loop 结束后
del C_sorted, prefix_T, block_rank, merge_res, colors_bg
# 或者在 grad loop 内, 每个 block 用完 rank_map / prefix_T_k 后 del
```

- 效果: 减少 grad phase 常驻显存, 对大分辨率场景 (1600×1066) 约节省 50-100 MB
- 改动: 几行

### 方案 D: 统一 block 分配尺寸, 提高 CUDA allocator 复用率 [中等改动]

当前每个 block 的 `gpu_packed` 尺寸 = `n_vis × D`, `n_vis` 随相机变化. 可以将分配尺寸 round up 到固定步长:

```python
# gaussian_model.py: kick_h2d_and_activate()
ALLOC_STEP = 4096  # 以 4K 点为步长
alloc_n = ((n + ALLOC_STEP - 1) // ALLOC_STEP) * ALLOC_STEP
gpu_packed = torch.empty(alloc_n, D, device='cuda')
gpu_packed[:n].copy_(staging, non_blocking=True)
```

- 效果: 大幅减少 CUDA allocator 碎片, RSV 波动可能降低 50%+
- 代价: 每次多分配 0~4K 点的空间 (几 MB)
- 改动: ~10 行

### 方案 E: CUDA memory pool 按 block 大小分池 [较大改动]

使用 `torch.cuda.memory.CUDAPluggableAllocator` 或手动管理 memory pool, 按 block 典型大小预分配固定池:

- 效果: 根治碎片问题
- 代价: 改动量大, 需要自定义 allocator
- 改动: 较大

---

## 5. 建议优先级

1. **方案 A** (调阈值) + **方案 B** (del gpu_packed) → 改两行, 立即降低尖刺
2. **方案 D** (round up 分配) → 根治碎片的性价比最高方案
3. **方案 C** (释放 merge 结果) → 对大分辨率场景有帮助
4. **方案 E** → 除非前几个方案不够, 否则不值得投入
