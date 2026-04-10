# 流水线分析 v3 — 基于 CPU 带宽瓶颈修正

## 核心发现

Adam 是 **DRAM 带宽瓶颈**，不是计算瓶颈。

```cpp
// cpu_adam.cpp:602-619
#pragma omp parallel for num_threads(16)
for (vi = 0; vi < n_vis; ++vi) {
    row = valid_ids_ptr[vi];    // 随机行访问!
    for (d = 0; d < D; ++d) {  // D=59
        // 读: packed[row*D+d], grad[vi*D+d], m[row*D+d], v[row*D+d]
        // 写: packed[row*D+d], m[row*D+d], v[row*D+d]
    }
}
```

每行内存流量: 59 × (4 reads + 3 writes) × 4 bytes = **1,652 bytes**
500K visible rows → **826MB** 内存流量
DRAM ~40GB/s → 理论下限 **~20ms**，实测 **16ms** (已逼近极限)

16 个 OMP 线程已经把带宽打满了。后台线程、更多线程都不会更快。

**唯一的出路: 减少内存流量 = 减少行数 × 列数。**

---

## 方案 1: 列裁剪 — 只更新 hot 列（最大单项收益）

### 观察

D=59 列中:
```
xyz(3) + f_dc(3) + scaling(3) + rotation(4) + opacity(1) = 14 cols  (24%, "hot")
f_rest(45 cols)                                           = 45 cols  (76%, "cold")
```

SH 高阶系数 (f_rest) 占 76% 的内存流量，但它们：
- 梯度小（SH 高阶对 loss 贡献边际递减）
- 变化慢（学习率 2.5e-3/20 = 0.000125，远低于 xyz 的 1.6e-4 × spatial_lr_scale）
- 不影响 frustum culling 和 depth sorting

### 方案

Adam 按频率分层更新:
- **hot 列 (14 cols)**: 每 iter 都更新
- **cold 列 (45 cols)**: 每 K iter 更新一次（K=5~10）

```
当前 adam 内存流量:  n_vis × 59 × 7 × 4 = n_vis × 1,652 bytes
hot-only 流量:      n_vis × 14 × 7 × 4 = n_vis × 392 bytes  (24%)
```

### 实现

**方案 A: 拆成两个 packed buffer**

```python
# pack_to_buffer() 改为创建两个 buffer:
self._packed_hot  = torch.empty(N, 14, pin_memory=True)   # xyz, f_dc, scaling, rotation, opacity
self._packed_cold = torch.empty(N, 45, pin_memory=True)   # f_rest

# adam 分两个调用:
packed_sparse_adam(self._packed_hot, grad_hot, exp_avg_hot, ..., D=14)  # 每 iter
if iteration % K == 0:
    packed_sparse_adam(self._packed_cold, grad_cold, exp_avg_cold, ..., D=45)  # 每 K iter
```

**方案 B (更简单): 在现有 packed 上用列 mask**

不改 buffer 布局，只改 adam 内循环:

```cpp
// 新版 packed_sparse_adam_hot_only
for (int64_t vi = 0; vi < n_vis; ++vi) {
    int64_t row = valid_ids_ptr[vi];
    float* p = packed_ptr + row * D;
    float* g = grad_ptr + vi * D;
    float* m = m_ptr + row * D;
    float* vv = v_ptr + row * D;
    
    // 只处理 hot 列: [0,3) xyz, [3,6) f_dc, [51,54) scaling, [54,58) rotation, [58,59) opacity
    // 跳过 [6,51) f_rest
    for (int64_t d = 0; d < 6; ++d)       { adam_one(p,g,m,vv,d); }  // xyz + f_dc
    for (int64_t d = 51; d < D; ++d)      { adam_one(p,g,m,vv,d); }  // scaling+rotation+opacity
}
```

但方案 B 仍然读取整行的 grad（因为 grad_subset 是 [n_vis, D] 连续的），cache line 还是被污染。方案 A 更优。

### 预计收益

| 指标 | 当前 | hot-only | 倍率 |
|------|------|----------|------|
| adam 单 block (500K pts) | 16ms | 4ms | **4x** |
| adam 总 (6 blocks) | 17-35ms | 4-9ms | **4x** |
| pre_gather (500K pts) | 3ms | 0.7ms | **4x** |
| H2D (500K pts) | 6ms | 1.5ms | **4x** |

**总节省: adam 20-25ms + pre_gather 8-10ms + H2D 10-15ms ≈ 40-50ms**
**从 120ms → 70-80ms**

### 对 cold 列的处理

每 K iter 做一次 cold adam:
```
iter 1: hot adam (4ms) + full H2D (hot+cold)
iter 2: hot adam (4ms) + full H2D
...
iter K: hot adam (4ms) + cold adam (12ms) + full H2D
```

平均: 4ms + 12ms/K

K=5 时: 4 + 2.4 = 6.4ms，vs 当前 16ms。

**但 H2D 仍然需要传所有 59 列（render 需要）。** cold 列的 adam 节省不影响 H2D。

如果不做 cold cache（方案 1 先不管），pure adam 优化就已经值 20-25ms。

---

## 方案 2: 行裁剪 — 只更新高梯度行

### 观察

每个 block 的 visible_indices 中，大部分点的梯度很小（在物体内部/远处的点，已经收敛）。
只有边缘和高频区域的点有大梯度需要更新。

### 方案

在 D2H grad 后，在 CPU 上做快速梯度过滤:

```python
def packed_sparse_adam_step(self, idx, grad_subset, iteration):
    # 计算每行梯度 L2 norm（只用 hot 列做 proxy，避免读 cold）
    grad_norm = grad_subset[:, :6].norm(dim=1)  # xyz+f_dc 的梯度范数
    
    # Top-K% 或 threshold 过滤
    threshold = grad_norm.quantile(0.5)  # 只更新前 50%
    mask = grad_norm >= threshold
    
    # 只对 mask 行做 adam
    active_idx = idx[mask]
    active_grad = grad_subset[mask]
    cpu_adam.packed_sparse_adam(
        self._packed, active_grad, self._packed_exp_avg, self._packed_exp_avg_sq,
        active_idx, lr_per_col, ...)
    
    # 被跳过的行: 梯度累积到 error feedback buffer，下次补偿
    self._grad_error_buf[idx[~mask]] += grad_subset[~mask]
```

### 预计收益

如果跳过 50% 的行: adam 内存流量再减半。
与方案 1 叠加: 从 800MB → 800×0.24×0.5 = **96MB**，adam 从 16ms → **~2ms**。

### 风险

- 收敛速度可能变慢（但 error feedback 可以补偿）
- 需要验证对 PSNR 的影响
- 阈值需要调优

---

## 方案 3: Block 级稀疏 — 减少 grad 阶段 block 数

### 观察

当前: 所有通过 nograd filter 的 block 都做 grad render + backward + adam。

但很多 block 对当前视角的 loss 贡献很小。看 profiling:

```
block 0: 38K pts, contribution 可能很小
block 7: 9K pts, 几乎不影响 loss
```

### 方案

每 iter 只对 contribution 最高的 top-K 个 block 做 grad:

```python
# merge 之后已知每个 block 的 prefix_T_k
block_weights = []
for idx, submodel_id in enumerate(visible_submodel_id_list):
    # prefix_T 代表该 block 在最终图像中的权重
    weight = prefix_T[idx].mean().item()
    block_weights.append((idx, weight))

block_weights.sort(key=lambda x: x[1], reverse=True)

# 累计贡献 > 95% 就停止
cumsum = 0
grad_set = set()
for idx, w in block_weights:
    cumsum += w
    grad_set.add(idx)
    if cumsum > 0.95:
        break

# 只对 grad_set 中的 block 做 grad render
for idx in grad_order:
    if idx not in grad_set:
        continue  # 跳过，只用 nograd render 的结果
    ...
```

### 预计收益

如果 8 个可见 block 中只有 3-4 个在 grad set:
- Grad 阶段 block 数减半
- adam 总量减半: 17-35ms → 9-18ms
- 节省 H2D + render + backward: ~15ms

**但不能长期忽略低贡献 block。** 轮转策略:
```python
# 每 R iter 强制所有 block 做一次 grad
if iteration % ROTATION_INTERVAL == 0:
    grad_set = set(range(len(visible_submodel_id_list)))
```

---

## 方案 4: pre_gather 的 cold 列缓存

### 观察

`pre_gather` 做 `index_select(_packed, 0, visible_indices, out=staging)`。
每个 block 读 n_vis × 59 × 4 bytes。f_rest (45 cols) 占 76%。

如果 block 的 cold 列还没被 adam 更新（方案 1 下，cold 每 K iter 才更新一次），
那 cold 列的 pre_gather 结果和上次完全相同... **不对**，visible_indices 不同（不同相机）。

所以 pre_gather 不能跳过，但可以分两步:
```python
# 每次 pre_gather:
index_select(self._packed_hot, 0, visible_indices, out=staging_hot)   # 14 cols, 快
index_select(self._packed_cold, 0, visible_indices, out=staging_cold) # 45 cols, 慢但必需
```

pre_gather 没法省。但 H2D 分段后，可以先传 hot (GPU 马上能开始 projection)，
cold 后传（GPU 等 SH 时再到）... 这需要 rasterizer 支持分阶段输入，改动大。

### 更实际的优化: Nograd fp16 传输

Nograd 阶段只需要 render 出 rgb/depth/alpha 给 merge，精度容忍度高:

```python
def pre_gather_fp16(self):
    # CPU 端做 fp32→fp16 转换，体积减半
    staging_fp16 = self._packed_staging_fp16[:n]
    torch.index_select(self._packed, 0, idx, out=staging_fp32)
    staging_fp16.copy_(staging_fp32)  # fp32→fp16，CPU 上很快

def kick_h2d_fp16(self):
    # H2D 传 fp16，在 GPU 上 cast 回 fp32（GPU 做 cast 几乎免费）
    gpu_fp16 = staging_fp16.cuda(non_blocking=True)
    gpu_packed = gpu_fp16.float()
```

Nograd H2D 从 11.1ms → ~6ms。但 pre_gather 的内存读取量不变（仍读 fp32）。

---

## 方案 5: 跨 iter Adam 延迟更新（借鉴 GS-Scale）

### 思路

GS-Scale 的 `adam_deferred_update` + counter:
- 不是每 iter 都对每个可见点做 adam
- 维护 counter[i] = "该点跳过了几步 adam"
- 下次更新时，用 corrected step 补偿

### 在你系统中的适配

```python
# 每个 block 维护一个 adam_counter
# 如果 block 连续 M 个 iter 都在 grad set，不需要每次都做全量 adam
# 只对梯度变化大的行做 adam，其余累积

# 实现: 和方案 2 (行裁剪) 类似，但用 counter 做正确的 bias correction
```

这个和方案 2 可以合并。

---

## 综合路线图

### 阶段 1: 列裁剪（方案 1，预计 1-2 周）

- 改 `pack_to_buffer()` 拆 hot/cold
- 改 `packed_sparse_adam` C 扩展支持 D=14
- 改 `pre_gather` / `kick_h2d` 拼接 hot+cold
- 改 `kick_async_d2h` 拆分 grad

**收益: adam 从 17-35ms → 4-9ms (节省 ~20ms)**

### 阶段 2: 选择性 grad（方案 3，预计 1 周）

- 在 merge 后用 prefix_T 评估 block importance
- 只对 top-K block 做 grad
- 加轮转策略防止低贡献 block 完全停更

**收益: grad 阶段 block 数减半 (节省 ~15-20ms)**

### 阶段 3: 行裁剪（方案 2，预计 2 周）

- 梯度 norm 过滤 + error feedback
- 改 C 扩展支持 mask 输入
- 验证对 PSNR 的影响

**收益: adam 进一步 2x (累计节省 ~30-40ms)**

### 理论极限

| 组件 | 当前 ms | 优化后 ms | 节省 |
|------|---------|-----------|------|
| Nograd (8 blocks) | 31.6 | 25 (fp16 H2D) | 6.6 |
| Merge | 14.0 | 14.0 | 0 |
| Grad H2D (4 blocks) | 5.0 | 3.0 | 2.0 |
| Grad render+bwd (4 blocks) | 10.0 | 10.0 | 0 |
| Adam (4 blocks, hot, 50% rows) | 2.0 | 2.0 | 0 |
| Pipeline gaps | 10.0 | 5.0 | 5.0 |
| **总计** | **120.6** | **~60** | **~60** |

从 120ms → ~60ms ≈ **2x 加速**，不依赖后台线程，纯靠减少数据量。
