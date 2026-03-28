# Fix D: round up 分配尺寸，提高 CUDA allocator 复用率

## 原理

当前每个 block 的 GPU 分配尺寸 = `n_vis × D`，`n_vis` 随相机视角变化 (vis% 10%~60%)。
CUDA caching allocator 按精确大小匹配空闲块，连续 iteration 中同一 block 的 n_vis 不同时:

```
iter 100, block 0: 分配 82000 × 59 → deactivate 释放
iter 101, block 0: 需要 79000 × 59 → 尺寸不匹配 82000 的空闲块 → 申请新 reserved
```

将 n_vis 向上对齐到固定步长，让分配落入少数几个"桶"，提高复用率。

## 修改

文件: `scene/gaussian_model.py`，`kick_h2d_and_activate()` 方法

```python
# before (line 1084-1088)
    def kick_h2d_and_activate(self, requires_grad=True):
        """H2D transfer + GPU unpack. Assumes pre_gather() was already called."""
        n = self._db_n
        staging = self._packed_staging[:n]
        gpu_packed = staging.cuda(non_blocking=True)

# after
    ALLOC_STEP = 4096  # 对齐步长

    def kick_h2d_and_activate(self, requires_grad=True):
        """H2D transfer + GPU unpack. Assumes pre_gather() was already called."""
        n = self._db_n
        staging = self._packed_staging[:n]
        # Round up 分配尺寸，让 CUDA allocator 更容易复用空闲块
        alloc_n = ((n + self.ALLOC_STEP - 1) // self.ALLOC_STEP) * self.ALLOC_STEP
        D = staging.shape[1]
        gpu_packed = torch.empty(alloc_n, D, device='cuda')
        gpu_packed[:n].copy_(staging, non_blocking=True)
        gpu_packed = gpu_packed[:n]  # slice 回实际大小，不影响后续逻辑
```

## 注意事项

- **不能直接 `staging.cuda()`**，因为那样分配的尺寸精确等于 n×D
- `torch.empty(alloc_n, D)` 分配对齐后的大块，然后 `copy_` + slice
- slice 是 view，不释放底层存储，所以 CUDA allocator 看到的仍是对齐后的大小
- `deactivate_subset()` 释放时，空闲块大小 = 对齐后大小，下次同一桶的分配可直接复用

## 预期效果

- 同一 block 在连续 iter 间 n_vis 波动 < ALLOC_STEP 时，CUDA allocator 直接复用 → RSV 不增长
- 减少 RSV 波动幅度，预计 30-50%
- 代价: 每次多分配 0~4095 点的空间，约 0~0.9 MB (4095 × 59 × 4B)

## 潜在代价

- 略微增加显存占用（每 block 最多 ~1MB）
- 改动涉及 H2D 路径，需要仔细验证正确性
- **如果已用 Taming 3DGS 的 rasterizer**，rasterizer 内部的 buffer 碎片已被处理，
  此方案只优化 `gpu_packed` 这一个 tensor 的复用，收益有限

## 适用场景

- B+C 做完后 RSV 尖刺仍然明显时再考虑
- 如果 block 数较多 (>8) 或 vis% 波动剧烈时收益更大

## 验证指标

对比 wandb 中 `rsv` 的波动标准差和 `peak_rsv` 最大值
