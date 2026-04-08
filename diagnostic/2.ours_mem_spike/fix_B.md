# Fix B: clone 后立即释放 gpu_packed

## 原理

`kick_h2d_and_activate(requires_grad=True)` 中:

```python
gpu_packed = staging.cuda(non_blocking=True)    # [n_vis, D] 整块 H2D 到 GPU
self._xyz_gpu = gpu_packed[:, s:e].clone()       # clone 副本 1
self._features_dc_gpu = ...clone()               # clone 副本 2
self._features_rest_gpu = ...clone()             # clone 副本 3
self._scaling_gpu = ...clone()                   # clone 副本 4
self._rotation_gpu = ...clone()                  # clone 副本 5
self._opacity_gpu = ...clone()                   # clone 副本 6
# gpu_packed 仍然存活! 直到函数返回或 GC 才释放
```

6 个 clone 完成后 `gpu_packed` 已无用，但它和 6 个副本**同时占用 GPU 内存**。
`gpu_packed` 的大小 = `n_vis × D`（D=59），对于 n_vis=500K 约 112MB。
这 112MB 冗余在每个 block 的 grad phase 都会出现（8 blocks = 8 次）。

## 修改

文件: `scene/gaussian_model.py:1103` (requires_grad=True 分支末尾)

```python
# before (line 1091-1103)
        if requires_grad:
            s, e, _ = slices['_xyz']
            self._xyz_gpu = gpu_packed[:, s:e].clone()
            s, e, reshape = slices['_features_dc']
            self._features_dc_gpu = gpu_packed[:, s:e].reshape(n, reshape[1], reshape[2]).clone()
            s, e, reshape = slices['_features_rest']
            self._features_rest_gpu = gpu_packed[:, s:e].reshape(n, reshape[1], reshape[2]).clone()
            s, e, _ = slices['_scaling']
            self._scaling_gpu = gpu_packed[:, s:e].clone()
            s, e, _ = slices['_rotation']
            self._rotation_gpu = gpu_packed[:, s:e].clone()
            s, e, _ = slices['_opacity']
            self._opacity_gpu = gpu_packed[:, s:e].clone()

# after
        if requires_grad:
            s, e, _ = slices['_xyz']
            self._xyz_gpu = gpu_packed[:, s:e].clone()
            s, e, reshape = slices['_features_dc']
            self._features_dc_gpu = gpu_packed[:, s:e].reshape(n, reshape[1], reshape[2]).clone()
            s, e, reshape = slices['_features_rest']
            self._features_rest_gpu = gpu_packed[:, s:e].reshape(n, reshape[1], reshape[2]).clone()
            s, e, _ = slices['_scaling']
            self._scaling_gpu = gpu_packed[:, s:e].clone()
            s, e, _ = slices['_rotation']
            self._rotation_gpu = gpu_packed[:, s:e].clone()
            s, e, _ = slices['_opacity']
            self._opacity_gpu = gpu_packed[:, s:e].clone()
            del gpu_packed  # 立即释放，避免与 6 个 clone 副本同时占用显存
```

## 预期效果

- 每个 block 的 grad phase 减少 ~n_vis×D×4 bytes 的瞬态峰值
- 对于 n_vis=500K, D=59: 约 **112MB**
- 零速度代价

## 潜在代价

无。`del` 只是提前释放引用，不产生任何计算或同步。

## 验证指标

对比 wandb 中 `peak_alloc` 的最大值
