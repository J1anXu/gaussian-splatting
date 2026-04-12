# nurips26_hide_densification_3 内存控制三步改进计划

日期：2026-04-11
目标：在尽量少用 `empty_cache()` 的前提下，控制服务器可见 GPU memory 峰值，同时保住训练速度。

## 总原则

这轮优化不再把 `empty_cache()` 当主手段，而是把它降级成 fallback。

主线是：

```text
先预测下一阶段显存压力
再释放可选 GPU packed cache
最后才考虑 empty_cache()
```

显存预算以 points / visible points 为核心，不用 `SPLIT_SIZE` 或最大 block 体积拍脑袋。

诊断日志硬约束：

```text
所有 verbose/profile/timeline 观测只能在 test/benchmark 阶段生效。
正常训练不传 --test_diagnostics，因此不能产生日志、timeline 或 profile sync 开销。
```

## Step 1：只加预算观测，不改变训练行为

### 目的

先把“下一步为什么会涨 reserved”看清楚。这个阶段不改变 cache 策略，不改变 block 调度，不影响训练结果，只补齐后续做预算需要的数据。所有新增观测必须挂在 `--test_diagnostics` + `--profile_log` 后面。

### 要做的事

1. 在 `nograd_render_dispatch`、`grad_render_dispatch`、`backward_dispatch`、`grad_h2d_activate`、`flush_and_prepare` 前后记录：
   - 当前 `allocated`
   - 当前 `reserved`
   - stage 内 `peak_alloc`
   - stage 内 `peak_rsv`
   - visible points
   - 当前 block id / block count
   - GPU packed cache 中的 block ids
   - GPU packed cache points / bytes

2. 统计 stage 级别的历史 workspace：

   ```text
   stage + visible_points_bucket -> max_reserved_delta
   stage + visible_points_bucket -> avg_reserved_delta
   stage + visible_points_bucket -> p90_reserved_delta
   ```

3. 在 timeline / train log 里输出每轮的 compact summary：

   ```text
   iter, stage, vis_pts, cache_pts, cache_MB, alloc, rsv, d_alloc, d_rsv
   ```

4. 先不启用任何新的 evict 或 empty_cache 逻辑。

### 验收标准

1. 训练结果不变。
2. it/s 基本不变。不开 `profile_sync` 时，额外开销应该很小。
3. 能从日志里回答：
   - 哪个 stage 最容易顶 reserved。
   - visible points 到多少时开始危险。
   - GPU packed cache 在尖峰发生前占了多少 points / MB。

### 产出

1. 一个 stage workspace 表。
2. 一个 cache occupancy 表。
3. 一份 top memory spikes summary。

## Step 2：改 cache admission，用 points budget 控制 GPU packed cache

### 目的

先控制“可选显存”的上限。GPU packed cache 是性能优化，不是训练必须占着的显存，所以它应该服从 points budget。

### 要做的事

1. 新增明确的 points budget：

   ```text
   GPU_PACKED_CACHE_POINT_BUDGET
   ```

   这个值来自日志统计，不从 `SPLIT_SIZE` 或最大 block 推出来。

   实现上保留室内/室外默认值以维持旧行为，但预算变量已经和 `SPLIT_SIZE` 解耦：

   ```text
   GPU_PACKED_CACHE_POINT_BUDGET_INDOOR = 400000
   GPU_PACKED_CACHE_POINT_BUDGET_OUTDOOR = 800000
   ```

   test 阶段可用命令行覆盖：

   ```bash
   GPU_PACKED_CACHE_POINT_BUDGET=750000 bash /home/jian/gaussian-splatting/test_4090.sh
   GPU_PACKED_CACHE_POINT_BUDGET=1000000 bash /home/jian/gaussian-splatting/test_4090.sh
   GPU_PACKED_CACHE_POINT_BUDGET=1500000 bash /home/jian/gaussian-splatting/test_4090.sh
   ```

2. GPU packed cache admission 改成：

   ```text
   如果 cache_points + new_block_points <= point_budget:
       允许缓存
   否则:
       不缓存，或先 evict 低价值 cache
   ```

3. tail 小块缓存策略继续保留，但选择标准改成 points：

   ```text
   优先缓存小块 / 高频可见块 / 最近反复访问块
   总 points 不超过 budget
   ```

4. evict 策略先保持简单可解释：

   ```text
   优先 evict 低频块
   再 evict 长时间未使用块
   最后 evict points/bytes 最大但收益低的块
   ```

5. 这个阶段仍然不做 stage 前主动 `empty_cache()`。

### 验收标准

1. `peak_rsv` 不比当前版本更高。
2. `empty_cache()` 次数不增加。
3. it/s 不明显下降。
4. 日志里能看到 cache points 被稳定限制住。

### 主要风险

如果 point budget 太低，GPU packed cache 命中率下降，速度会掉。因此 Step 2 要跑至少 2-3 个 budget：

```text
0.75M points
1.0M points
1.5M points
```

最后选速度和峰值最平衡的值。

## Step 3：做 stage-aware MemoryBudgetController

### 目的

真正从 pipeline 层面避峰值：在大 render / backward 之前，先根据下一阶段压力释放可选 cache，而不是等 reserved 已经冲上去以后再清。

### 要做的事

1. 新增轻量控制器：

   ```python
   MemoryBudgetController
   ```

2. 控制器输入：

   ```text
   stage
   next_visible_points
   current_alloc
   current_reserved
   gpu_cache_points
   gpu_cache_bytes
   historical_workspace_estimate
   ```

3. 控制器输出：

   ```text
   allow_cache_admission
   evict_cache_ids
   need_empty_cache_fallback
   ```

4. 在危险 stage 前调用：

   ```text
   before nograd_render_dispatch
   before grad_render_dispatch
   before backward_dispatch
   before grad_h2d_activate
   ```

5. 决策逻辑：

   ```text
   predicted_peak = current_reserved + estimated_stage_workspace

   if predicted_peak > soft_reserved_limit:
       evict GPU packed cache until predicted_peak <= soft limit

   if still predicted_peak > hard_reserved_limit:
       empty_cache()
   ```

6. `empty_cache()` 只作为 hard fallback，不在每轮固定调用。

### 验收标准

1. 稳定阶段 `peak_rsv` 下降。
2. `empty_cache()` 次数低于当前每轮清理策略。
3. it/s 相比当前诊断版本恢复或提升。
4. 服务器上 `nvidia-smi` 峰值波动变小。

### 最终目标

在不牺牲训练正确性的前提下，把策略从：

```text
事后清理 reserved cache
```

改成：

```text
阶段前预测压力
主动让出可选 cache
只在必要时 fallback empty_cache
```

## 推荐执行顺序

1. 先做 Step 1，跑一次 `test_4090.sh` 和一组完整 scene summary，只看日志，不追求速度。
2. 再做 Step 2，针对 bicycle 先调出一个合理 point budget。
3. 最后做 Step 3，把 budget controller 接进 pipeline，再跑全场景看速度和峰值。

每一步都单独 commit，方便回滚和对比。
