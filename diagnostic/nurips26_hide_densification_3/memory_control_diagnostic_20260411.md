# nurips26_hide_densification_3 内存控制诊断备份

日期：2026-04-11
分支：nurips26_hide_densification_3

## 背景

当前目标不是单纯追求 PyTorch `allocated` 低，而是控制服务器上能看到的真实 GPU memory 峰值。这个值更接近 PyTorch CUDA caching allocator 的 `reserved`，也就是 `nvidia-smi` 会看到的显存占用。

之前的现象是：训练速度提升了，但一些场景的 `peak_rsv` 明显升高。例如 bonsai、counter、kitchen、train、truck 等场景的 summary peak memory 变高。进一步看日志后可以区分两件事：

1. `peak_alloc` 没有同步变高，有些场景甚至下降。
2. `peak_rsv` 变高了，所以服务器上看到的显存确实会更高。

因此这不是简单的“日志打错”，也不是模型本体长期多占了很多显存，而是 CUDA allocator 的 reserved cache 被某些阶段顶高之后没有及时回落。

## 已确认的问题来源

之前一个明确问题出现在 lazy/background densify 相关路径：

1. 某些 block 正在 `_is_densifying`。
2. grad pass 会跳过这些 block。
3. 对于单块或少块场景，可能出现本轮所有可见 block 都被跳过。
4. 代码进入 `processed_grad_blocks == 0` 之后直接 `continue`。
5. 这会绕过 iteration 末尾的 `cuda_cache_check` / `empty_cache`。
6. no-grad render / merge 期间产生的临时大 allocation 虽然释放了 tensor 引用，但 allocator 仍然把 reserved cache 留着。
7. 下一轮再 `reset_peak_memory_stats()` 时，如果 stale reserved 还在，新的 `peak_rsv` 就会被污染。

这个路径已经通过 skip 前清理 cache、pre-iteration 清理、临时引用置空等方式修过。

但是即使这个问题修掉，仍然存在更大的设计问题：如果靠每轮 `empty_cache()` 控制峰值，会付出较高同步和 driver 交互成本，训练速度会受影响。

## 最新诊断里看到的主要峰值阶段

基于 `test_4090.sh` 打开的 profile sync 诊断，稳定阶段里主要推动 reserved 上升的不是 merge，而是下面几个阶段：

1. `nograd_render_dispatch`
2. `grad_render_dispatch`
3. `backward_dispatch`
4. `grad_h2d_activate`
5. `flush_and_prepare`

其中 `merge_total` 的 reserved jump 相对小，不是当前第一优先级。

另外，GPU packed cache 本身通常只有一两百 MB 级别，它不是全部问题，但它是“可选显存”。也就是说，它很适合作为主动让显存的对象。

## 为什么不应该只靠 empty_cache

`torch.cuda.empty_cache()` 的特点：

1. 它可以把 PyTorch allocator 已经空闲的 cache 还给 driver。
2. 它可以降低 `nvidia-smi` 能看到的 reserved。
3. 但它通常会引入同步和 allocator 重建成本。
4. 如果每轮都调用，会牺牲吞吐。
5. 它只能事后清理，不能避免大 render / backward 阶段先把峰值顶上去。

所以更好的方向不是“reserved 高了就清”，而是“在大分配发生前，让可选缓存主动腾位置”。

## 更好的内存控制方向

### 1. 做 stage-aware 的软预算控制

不要等一轮结束后才看 reserved，而是在危险阶段前做判断：

1. 进入 no-grad 大 render 前。
2. 进入 grad render 前。
3. 进入 backward 前。
4. 进入 grad H2D / packed activate 前。

这些阶段开始前，先估计下一步可能需要的 workspace。如果当前 `allocated + 可选 cache + 预计 workspace` 可能超过预算，就先释放 GPU packed cache 的一部分。

只有释放 cache 后仍然危险，才调用 `empty_cache()` 作为 fallback。

核心思想：

```text
先 evict 可选 GPU cache
再考虑 empty_cache
```

### 2. cache admission 用点数预算，而不是最大 block 体积

tail 小块缓存策略不应该以“最大块”为上限。最大块体积只是一个粗糙代理，不一定对应真实 render workspace。

更合理的是按 visible points 或 packed points 做预算：

```text
GPU_PACKED_CACHE_POINT_BUDGET = 1.5M
```

缓存策略：

1. 优先缓存小块 / 高频块 / 反复可见块。
2. cache 中总 points 不超过点数预算。
3. 如果下一步要处理大 visible block，则临时降低 cache budget。
4. 如果下一步是小 block，则可以保留更多 tail cache。

这样比 `SPLIT_SIZE` 或最大块体积更贴近真实内存压力。

### 3. 做按阶段的水位线，而不是每轮固定清理

建议维护几个软水位：

```text
soft_reserved_limit_gb
hard_reserved_limit_gb
cache_evict_margin_gb
empty_cache_margin_gb
```

逻辑：

```text
如果即将进入危险阶段：
    估计 next_stage_workspace
    如果 reserved + next_stage_workspace 超过 soft limit：
        evict GPU packed cache
    如果 evict 后仍然超过 hard limit：
        empty_cache()
```

这样可以减少频繁 `empty_cache()`，同时防止某一步突然把 reserved 顶高。

## 建议实现：MemoryBudgetController

可以新增一个轻量控制器，专门做内存预算和 cache 驱逐决策。

输入：

1. 当前 `torch.cuda.memory_allocated()`
2. 当前 `torch.cuda.memory_reserved()`
3. 当前 GPU packed cache 的 bytes / points
4. 下一阶段的 visible points
5. 当前 stage 名称，例如 `nograd_render`、`grad_render`、`backward`
6. block 的历史 peak workspace 估计

输出：

1. 是否允许 admission 到 GPU packed cache。
2. 需要 evict 哪些 cached blocks。
3. 是否需要 fallback 到 `empty_cache()`。

伪代码：

```python
def before_stage(stage, next_visible_points, cache_state):
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    predicted_workspace = estimate_workspace(stage, next_visible_points)

    predicted_reserved = reserved + predicted_workspace

    if predicted_reserved > soft_reserved_limit:
        evict_gpu_cache_until(
            target_reserved=soft_reserved_limit - predicted_workspace,
            prefer="low_value_or_large_cache"
        )

    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    if reserved + predicted_workspace > hard_reserved_limit:
        torch.cuda.empty_cache()
```

这里的 `estimate_workspace` 一开始可以不用复杂模型，先用日志里记录的 visible points 分桶统计：

```text
stage + visible_points_bucket -> historical peak reserved delta
```

后续再逐步做得更准。

## 推荐实验顺序

为了避免一次改太多，可以按下面顺序做 A/B：

1. 保持当前逻辑，只记录每个阶段的 visible points、reserved delta、cache points。
2. 加 point-budget cache admission，但不主动 `empty_cache()`。
3. 在危险阶段前只 evict GPU packed cache，不调用 `empty_cache()`。
4. 最后加入 fallback `empty_cache()`，只在 hard limit 可能被突破时触发。

每一步都看四个指标：

1. 平均 it/s。
2. `peak_rsv`。
3. `peak_alloc`。
4. `empty_cache()` 次数和总耗时。

## 当前判断

后续优化的主线应该是：

```text
少清理，提前避峰值
少同步，先释放可选 cache
用点数预算替代 block 数或最大块体积
只在危险阶段前做内存控制
```

这比每轮末尾固定 `empty_cache()` 更适合当前 pipeline。它既能控制服务器上看到的 GPU memory 波动，也更有机会保住训练速度。
