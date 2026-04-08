# Fix C: merge 中间结果尽早释放

## 原理

merge 阶段产生的中间 tensor (`train.py:270-275`):

```python
C_sorted   = merge_res["front_rgbs"]   # [K, 3, H, W]  K=8, 1600×1066
prefix_T   = merge_res["prefix_T"]     # [K, 1, H, W]
block_rank = merge_res["block_rank"]   # [K, H, W]
colors_bg  = merge_res["bg_rgb"]       # [3, H, W]
merge_res  本身还持有 "final_rgb" 等   # [3, H, W]
```

这些 tensor 在整个 grad 循环 (train.py:286-335) 中一直存活，
因为每个 block 的 grad pass 都要读 `C_sorted`, `prefix_T`, `block_rank`, `merge_res["final_rgb"]`。
**直到 grad 循环全部结束、下一个 iter 的赋值覆盖旧变量后才被 GC。**

对于 1600×1066, K=8 的情况:
- C_sorted:  8 × 3 × 1066 × 1600 × 4 bytes = **163 MB**
- prefix_T:  8 × 1 × 1066 × 1600 × 4 bytes = **54 MB**
- block_rank: 8 × 1066 × 1600 × 4 bytes     = **54 MB**
- final_rgb + bg_rgb + 其他:                   ~**20 MB**
- 总计约 **~290 MB** 常驻显存

这些 tensor 与每个 block 的 render+backward 临时分配叠加，推高了内存峰值。

## 修改

文件: `train.py`，在 grad 循环结束后 (`grad_sync.flush_last()` 之后) 添加显式释放:

```python
# before (line 337-338)
        # flush the last submodel's pending work
        grad_sync.flush_last()

# after
        # flush the last submodel's pending work
        grad_sync.flush_last()

        # 释放 merge 中间结果，避免与下一阶段的分配叠加
        del C_sorted, prefix_T, block_rank, colors_bg, merge_res
        del rendered_list, depth_list, alpha_list
        del gt_image
```

## 为什么不能在 grad 循环内释放

grad 循环中每个 block 都要读 `C_sorted`, `prefix_T`, `block_rank`, `merge_res["final_rgb"]`
来构造 `composed_img` (train.py:309-315)，所以只能在循环结束后释放。

## 预期效果

- 减少 ~290 MB 的常驻显存窗口
- 这些内存在 grad 循环结束后立即可用于下一个 iter 的 nograd phase，而不是等 Python GC
- 零速度代价

## 潜在代价

无。这些变量在 del 之后不再被任何代码读取。
下一个 iter 会重新赋值 `C_sorted`, `prefix_T` 等。

## 验证指标

对比 wandb 中 `alloc` (非 peak) 在 iter 间的基线值是否降低
