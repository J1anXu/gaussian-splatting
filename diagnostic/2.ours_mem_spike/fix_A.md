# Fix A: 降低 empty_cache 阈值

## 原理

当前 `GPU_CACHE_THRESHOLD_GB = 1.0`，即 `reserved - allocated > 1GB` 才触发 `torch.cuda.empty_cache()`。
这意味着 CUDA allocator 可以积累最多 1GB 的碎片/空闲块才被清理，导致 RSV 尖刺高达 2.02GB。

降低阈值让清理更及时，压低 RSV 峰值。

## 修改

文件: `config.py:32`

```python
# before
GPU_CACHE_THRESHOLD_GB = 1.0

# after
GPU_CACHE_THRESHOLD_GB = 0.5
```

## 触发位置

`train.py:377-378`:
```python
if torch.cuda.memory_reserved() > torch.cuda.memory_allocated() + config.GPU_CACHE_THRESHOLD_GB * 1024**3:
    torch.cuda.empty_cache()
```

## 预期效果

- RSV 尖刺峰值从 ~2.0GB 降至 ~1.3GB
- 锯齿幅度减半

## 潜在代价

- `empty_cache()` 调用频率增加，每次调用约 0.1-0.5ms
- 预计降速 1-3%
- 如果 B+C+D 做完后尖刺已经不大，可以不做 A

## 验证指标

对比 wandb 中 `peak_rsv` 的最大值和波动幅度
