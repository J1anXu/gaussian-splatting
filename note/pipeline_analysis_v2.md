# 流水线分析 v2（基于内存约束修正）

## 核心约束

GPU 显存只够放一个 block 的数据。每个 iter 必须：

```
[逐块 nograd 渲染] → [merge 合并] → [逐块 grad 渲染 + backward + adam]
```

双渲染不可避免：merge 需要所有 block 的 rgb/depth/alpha，只有 merge 完成后才知道
每个 block 在最终图像中的 prefix_T 和 block_rank，才能计算 per-block loss。

---

## 实测数据（iter 30697, bicycle, 8 blocks, 120.6ms）

### 时间分配

| Phase | Wall ms | GPU 计算 ms | GPU idle ms | CPU 计算 ms | CPU idle ms |
|-------|---------|-------------|-------------|-------------|-------------|
| Nograd | 31.6 | 14.2 (render) | ~0 | 11.1 (h2d) | 6.3 (sync) |
| Merge | 14.0 | 12.7 | 0 | 0 | **14.0** |
| Grad | 63.5 | 20.0 (render+bwd) | **43.1** | 34.8 (adam) | 0 |
| **总计** | **120.6** | **46.9** | **43.1** | **45.9** | **20.3** |

**关键发现**: Grad 阶段 GPU 有 43ms 空闲（68% 的时间），都在等 CPU 跑 adam。

### Grad 阶段逐 block 详情

| block | n_vis | h2d ms | render ms | backward ms | **gap** ms |
|-------|-------|--------|-----------|-------------|-----------|
| 0 | 38K | 0.48 | 0.52 | 1.58 | 12.8 |
| 2 | 284K | 1.41 | 1.31 | 2.94 | 13.9 |
| 3 | 87K | 0.62 | 1.34 | 1.86 | 16.8 |
| 4 | 124K | 0.96 | 1.01 | 2.27 | 21.6 |
| 6 | 280K | 1.60 | 1.50 | 3.09 | 15.8 |
| 7 | 9K | 0.27 | 1.13 | 1.44 | 8.5 |

**gap** = 从 backward 结束到下一个 block render 开始的时间，全是 GPU 空等。

---

## 瓶颈精确定位

### 瓶颈 1: Grad 阶段 CPU adam 阻塞 GPU（43ms，占总时间 36%）

`pipeline_grad_sync.py:146-151` 的 `flush_and_prepare()`:

```python
# 步骤 5c: 阻塞等 prev D2H 完成，然后在主线程跑 adam
if self._pending_adam is not None:
    self._d2h_event.synchronize()  # 等 prev D2H（通常已完成，几乎0ms）
    self._pending_adam()           # ← 在主线程跑 adam，3-16ms，GPU 空等！
    self._pending_adam = None
```

Grad 阶段的精确时序:

```
Block i 完成后:
  GPU stream: [...backward_i | D2H_i ...完成... |  ====== IDLE ======  | H2D_{i+1} | render_{i+1}...]
  CPU main:                   [D2H_kick_i][deact][sync_prev][adam_prev!!!][record][prepare][H2D_kick_{i+1}]
                                                             ^^^^^^^^^^^^
                                                             这段时间 GPU 完全空闲
```

adam_prev 在主线程运行期间，GPU 无任何工作。这是最大的浪费。

### 瓶颈 2: Merge 期间 CPU 空闲（14ms，占 12%）

`train.py:277-279`:

```python
with torch.no_grad():
    merge_res = merge_opt_kid(rendered_list, depth_list, alpha_list)
```

GPU 做 merge 12.7ms，CPU 主线程完全空闲。此时 CPU 可以做预备工作。

### 瓶颈 3: Adam 尾部无 GPU 重叠（9-28ms，占 7-23%）

最后 1-2 个 block 的 adam 没有后续 GPU 工作可重叠：

```
...backward(last) → D2H(last) → sync → adam(prev) → adam(last) → iter 结束
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        纯 CPU 串行，GPU 空等
```

iter 30698 尾部 adam = 28.6ms（block 3 = 502K pts，adam=15.9ms）。

### 瓶颈 4: Nograd 阶段 deactivate 隐式同步（6ms，占 5%）

`train.py:253` 的 `deactivate_subset()` 本身只设 None，但紧接着的
`kick_h2d_and_activate()` 在 default stream 上排队，必须等上一个 render 完成。

实际的隐式 gap: 每 block ~2-4ms，8 blocks 累计 ~6ms。
（你已经做了 pre_gather overlap，这部分已经是优化后的状态了。）

---

## 改进方案（基于内存约束）

### 方案 1: Adam 移到后台线程（预计节省 30-38ms，最大单项收益）

**核心思路**: adam(prev) 不在主线程跑，启一个后台线程，主线程立刻去 kick 下一个 block 的 H2D。

**安全性分析**:
- adam(block_i) 写入 `block_i._packed`、`_packed_exp_avg`、`_packed_exp_avg_sq`
- 下一个 block_{i+1} 有自己独立的 buffer，无冲突
- grad 阶段的 H2D 读的是 nograd 阶段已填好的 `_db_staging`（不是 `_packed`），无冲突
- adam 只需在下一个 **iteration** 的 nograd `pre_gather()` 之前完成（读 `_packed`）
- `flush_last()` 在 iter 末尾保证所有后台 adam 完成 ✓

**改后时序**:

```
Block i:
  GPU: [H2D_i | render_i | backward_i | D2H_i]  [H2D_{i+1} | render_{i+1} | ...]
  CPU: [kick_H2D_i][densify][   GPU wait   ][D2H_kick][sync_prev][launch_bg_adam_prev][kick_H2D_{i+1}]
  BG:                                                               [adam_prev...]
```

GPU 空闲从 ~8-22ms/block 降到 ~1-2ms/block（仅 sync + thread launch 开销）。

**实现要点**:

```python
class PipelinedGradSync:
    def __init__(self, ...):
        ...
        self._adam_thread: Optional[threading.Thread] = None
    
    def flush_and_prepare(self, ...):
        state = self.kick_async_d2h(...)
        submodel.deactivate_subset()
        
        # 等上一个后台 adam 完成（通常已完成，因为 adam 与整个 block_i 的 GPU 工作并行）
        if self._adam_thread is not None:
            self._adam_thread.join()
            self._adam_thread = None
        
        # 同步 prev D2H（通常已完成）
        if self._pending_adam is not None:
            self._d2h_event.synchronize()
            # 后台线程跑 adam，主线程立刻返回
            adam_fn = self._pending_adam
            self._adam_thread = threading.Thread(target=adam_fn, daemon=True)
            self._adam_thread.start()
            self._pending_adam = None
        
        self._d2h_event.record()
        adam_fn, densify_fn = self._make_pending(...)
        self._pending_adam = adam_fn
        self._pending_densify = densify_fn
```

**预计收益**: 6 blocks × 平均 7ms gap 节省 = ~30-38ms，从 120ms → ~85ms。

---

### 方案 2: Merge 期间预备 Grad 阶段工作（预计节省 5-10ms）

Merge 的 12.7ms GPU 时间内，CPU 可以做:

```python
# 当前代码（merge 后才算）:
merge_res = merge_opt_kid(...)
gt_image = viewpoint_cam.original_image.cuda()  # ← 可以提前
grad_order = sorted(...)  # ← 可以提前

# 改为:
# 1. kick merge（非阻塞，GPU 开始工作）
merge_res_future = merge_opt_kid(...)  # 返回后 GPU 在算

# 2. CPU 趁 GPU 做 merge 时预备:
gt_image = viewpoint_cam.original_image.pin_memory().cuda(non_blocking=True)
grad_order = sorted(range(len(visible_submodel_id_list)),
    key=lambda i: submodel_list[visible_submodel_id_list[i]].visible_indices.shape[0],
    reverse=True)

# 3. 等 merge 结果（此时 CPU 预备已完成）
# merge_res 已经在变量里了，但后续操作需要 GPU 同步
```

**更重要的**: 可以在 merge 期间准备 adam 的 lr_per_col（每次都重复计算但值不变）:
```python
# 缓存 lr_per_col，不用每个 block 都重算
if not hasattr(submodel, '_cached_lr_per_col'):
    submodel._cached_lr_per_col = submodel._build_lr_per_col(iteration)
```

---

### 方案 3: 选择性 Grad 渲染（按需跳过低贡献 block 的梯度计算）

**思路**: 不是所有可见 block 都需要每 iter 都计算梯度。低贡献的 block 只做 nograd（提供 merge context），不做 grad+backward+adam。

**选择标准**（已有数据可用）:
```python
# 已有的信息:
# - valid_pixels / total_pixels: 像素覆盖率
# - submodel.visible_indices.shape[0]: 可见点数
# - prefix_T_k: 该 block 在最终图像中的透射率权重

# 方案: 每 iter 只 grad 前 top-K 个 block
# K 可以自适应: 选 contribution 累计 > 95% 的 block
block_contributions = []
for idx in range(len(visible_submodel_id_list)):
    # prefix_T_k * pixel_coverage 作为 importance
    contribution = prefix_T[idx].mean().item()  # merge 后已知
    block_contributions.append((idx, contribution))

block_contributions.sort(key=lambda x: x[1], reverse=True)
cumsum = 0
grad_set = []
for idx, contrib in block_contributions:
    cumsum += contrib
    grad_set.append(idx)
    if cumsum > 0.95:  # 贡献累计 95% 即停止
        break
```

**效果**: 如果 8 个可见 block 中只有 3-4 个贡献 >95%，grad 阶段减少一半 block，
节省 ~30ms（含 H2D + render + backward + adam）。

**代价**: 低贡献 block 参数更新变慢，但它们对最终图像影响本来就小。
可以用**轮转策略**: 每 N iter 强制 grad 一轮所有 block，防止低贡献 block 完全停更。

---

### 方案 4: 减少每个 block 的 H2D 传输量

#### 4a. Nograd 阶段 fp16 传输

Nograd 只需要 render 出 rgb/depth/alpha 用于 merge，精度要求低:

```python
def kick_h2d_and_activate_half(self):
    """Nograd-only: transfer fp16 packed buffer for lower bandwidth."""
    staging_fp16 = self._packed_staging_fp16[:n]  # 预分配的 fp16 buffer
    staging_fp16.copy_(self._db_staging[:n])  # fp32 → fp16 on CPU
    gpu_packed = torch.empty(n, D, dtype=torch.float16, device='cuda')
    gpu_packed.copy_(staging_fp16, non_blocking=True)
    # unpack 后 rasterizer 可以用 fp16? 需要 rasterizer 支持
```

如果 rasterizer 不支持 fp16 输入，可以在 GPU 上 cast:
```python
gpu_packed_f16 = staging_fp16.cuda(non_blocking=True)
gpu_packed = gpu_packed_f16.float()  # GPU 上 cast，几乎免费
```

**收益**: H2D 带宽减半。对 200K pts: 从 200K×59×4=47MB 降到 23.5MB。
Nograd H2D 从 11.1ms → ~6ms，节省 ~5ms。

#### 4b. SH rest 低频更新

SH 高阶系数（45 cols，占 76%）变化缓慢。可以:
- 每 K iter 才传一次完整 SH rest
- 中间 iter 只传 hot 部分（14 cols）
- 在 GPU 上用上次缓存的 SH rest + 本次 hot 部分拼接

```python
# 每 block 维护 GPU 端 SH cache
if block.sh_cache_valid and iteration % SH_UPDATE_INTERVAL != 0:
    # 只传 hot [n_vis, 14]
    gpu_hot = staging[:, :14].cuda(non_blocking=True)
    # GPU 上拼接 cached SH
    gpu_packed = torch.cat([gpu_hot, block.sh_cache[:n_vis]], dim=1)
else:
    # 全量传输 [n_vis, 59]
    gpu_packed = staging.cuda(non_blocking=True)
    block.sh_cache = gpu_packed[:, 14:].clone()  # 更新 cache
    block.sh_cache_valid = True
```

**问题**: SH cache 占 GPU 显存，需要评估是否放得下。
对单个 block 200K pts: 200K × 45 × 4 = 36MB，可能可以接受。

---

### 方案 5: 减少 Nograd 阶段的 block 数

#### 5a. 跨 iter 缓存不变 block 的 nograd 结果

如果 block i 在上一个 iter 不在 grad set（参数没更新），它对一个**新相机**的
render 结果仍然正确——因为参数没变，只是视角变了，必须重新 render。

**但是**: 如果同一相机在短时间内被多次采样（viewpoint_stack 是 shuffle 的），
我们可以缓存 `(block_id, camera_id) → render_result`。

这在你的场景里可能不太实用（相机随机采样，命中率低）。

#### 5b. 低分辨率 nograd render

Merge 不需要像素级精确——深度排序和透射率估计在低分辨率下也差不多:

```python
# 对低贡献 block，用 1/2 或 1/4 分辨率 nograd render
# 然后 upsample 回原始分辨率参与 merge
if block.n_vis < threshold:
    render_pkg = render(cam_half_res, block, pipe, bg, ...)
    image = F.interpolate(image, scale_factor=2, mode='bilinear')
```

**需要评估**: rasterizer 是否支持变分辨率，以及低分辨率 merge 对质量的影响。

---

### 方案 6: 借鉴 GS-Scale 的 adam_for_next（中期优化）

**核心思想**: 不是 "先 adam 更新 _packed → 再 gather 子集给下一 iter"，
而是 "adam + gather 融合为一步: 只对下一 iter 需要的子集做 adam，结果直接写 staging"。

**在你的系统中的适配**:

```python
def adam_and_gather_for_next(self, cur_idx, grad_subset, next_visible_idx, output_staging, iteration):
    """融合 adam + gather: 
    1. 对 cur_idx 的行做 adam update (写回 _packed)
    2. 对 next_visible_idx 的行 gather 到 output_staging
    
    如果 cur_idx 和 next_visible_idx 有交集，交集行的 adam 结果直接写入 staging，
    避免写入 _packed 再读出的 cache miss。
    """
    # 实现需要 C 扩展
    cpu_adam.adam_and_gather(
        self._packed, grad_subset,
        self._packed_exp_avg, self._packed_exp_avg_sq,
        cur_idx, next_visible_idx, output_staging,
        lr_per_col, self._packed_adam_step,
        0.9, 0.999, 1e-15
    )
```

**问题**: 需要提前知道下一 iter 的 visible_indices，但下一 iter 的相机还没确定。
**变通**: 可以在下一 iter 的相机确定后、nograd 阶段之前，预计算 visible_indices，
然后用 adam_for_next 准备 staging buffer。

**或者**: 在同一 iter 内，grad 阶段处理 block_i 时，block_{i+1} 的 nograd 已经做过了
（staging 已经准备好了），所以 adam_for_next 的目标不是 "下一 iter" 而是 "同 iter 内
grad 阶段的 staging buffer 复用 nograd 阶段已 gather 好的数据"。

这个在当前设计下其实已经做到了（pre_gather 复用），所以 adam_for_next 的收益
主要在跨 iter 场景，需要提前知道下一 iter 的 visible set。

---

## 推荐优先级

| 优先级 | 方案 | 预计节省 | 复杂度 | 风险 |
|--------|------|----------|--------|------|
| **P0** | 1: Adam 后台线程 | **30-38ms** | 低 | 低（数据独立，安全） |
| **P1** | 3: 选择性 grad 渲染 | **15-30ms** | 中 | 中（需验证收敛性） |
| **P1** | 2: Merge 期间 CPU 预备 | **5-10ms** | 低 | 无 |
| **P2** | 4a: Nograd fp16 H2D | **5ms** | 中 | 低 |
| **P2** | 4b: SH 低频更新 | **10-15ms** | 高 | 中（需管 GPU cache） |
| **P3** | 6: adam_for_next 融合 | **5-10ms** | 高 | 低 |

**P0 单项就可以从 120ms → ~85ms（~1.4x）。**
**P0 + P1 可以到 ~55-65ms（~2x）。**
**全部实施理论极限 ~40-50ms（~2.5-3x）。**
