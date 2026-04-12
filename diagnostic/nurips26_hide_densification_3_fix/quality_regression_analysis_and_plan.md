# nurips26_hide_densification_3 质量退化分析与修复计划

日期: 2026-04-12

目标: 在不放弃分块渲染低显存目标的前提下，先把 PSNR/SSIM/LPIPS 恢复到接近 vanilla 3DGS，再逐步把速度优化重新加回来。

## 0. 当前结论

当前分支 `nurips26_hide_densification_3` 的质量退化不是 summarize 脚本误读，也不是单纯点数变少。核心问题更像是训练语义和 vanilla 3DGS 已经不完全等价。

优先级最高的可疑点是:

1. lazy/background densify 期间 block 被 `_is_densifying` 冻结，grad pass 会跳过这些 block，严重时整轮没有任何梯度更新。
2. 当前分块 merge 的反传只用了 `prefix_T * final_rgb_grad`，没有完整传播 alpha/transmittance/depth/order 相关的跨 block 梯度。
3. 小块过滤和低贡献过滤会主动丢训练信号，尤其伤 early stage、遮挡边界、小物体和远处细节。
4. `tandt/train` 和 `tandt/truck` 当前结果无效，因为训练进程 core dumped 以后 pipeline 继续 render/metrics 了。

因此修复顺序必须是: 先恢复训练等价性，再重新引入异步/过滤/缓存优化。

## 1. 指标证据

vanilla 结果来自:

`/data/jian/output/gaussian-splatting/<dataset>/vanilla3DGS/<scene>/results.json`

当前结果来自:

`debug/nurips26_hide_densification_3/<scene>/metrics_0412_*.log`

| scene | dataset | vanilla iter | current iter | vanilla PSNR | current PSNR | delta PSNR | vanilla SSIM | current SSIM | vanilla LPIPS | current LPIPS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | mip360 | ours_30000 | ours_30000 | 25.225 | 25.035 | -0.190 | 0.7646 | 0.7527 | 0.2104 | 0.2273 |
| bonsai | mip360 | ours_30000 | ours_30000 | 31.960 | 31.750 | -0.210 | 0.9397 | 0.9384 | 0.2064 | 0.2089 |
| counter | mip360 | ours_30000 | ours_30000 | 28.985 | 28.761 | -0.224 | 0.9063 | 0.9044 | 0.2017 | 0.2053 |
| drjohnson | deepblending | ours_30000 | ours_30000 | 29.398 | 27.118 | -2.280 | 0.9028 | 0.8922 | 0.2384 | 0.2594 |
| flowers | mip360 | ours_30000 | ours_30000 | 21.488 | 21.438 | -0.050 | 0.6028 | 0.5956 | 0.3389 | 0.3443 |
| garden | mip360 | ours_30000 | ours_30000 | 27.363 | 27.292 | -0.071 | 0.8637 | 0.8625 | 0.1079 | 0.1141 |
| kitchen | mip360 | ours_30000 | ours_30000 | 31.276 | 31.081 | -0.194 | 0.9254 | 0.9248 | 0.1267 | 0.1304 |
| playroom | deepblending | ours_30000 | ours_30000 | 30.041 | 29.783 | -0.257 | 0.9029 | 0.9068 | 0.2430 | 0.2555 |
| room | mip360 | ours_30000 | ours_30000 | 31.493 | 31.010 | -0.483 | 0.9183 | 0.9159 | 0.2199 | 0.2253 |
| stump | mip360 | ours_30000 | ours_30000 | 26.662 | 26.327 | -0.336 | 0.7710 | 0.7580 | 0.2156 | 0.2321 |
| treehill | mip360 | ours_30000 | ours_30000 | 22.513 | 22.395 | -0.118 | 0.6330 | 0.6257 | 0.3265 | 0.3387 |
| truck | tandt | ours_30000 | ours_7000 | 25.395 | 23.590 | invalid | 0.8803 | 0.8439 | 0.1434 | 0.2061 |

注意:

- `truck` 当前 metrics 是 `ours_7000`，不能和 vanilla `ours_30000` 比。
- `train` 的 metrics 文件没有指标行。
- `debug/pipeline.out` 显示 `tandt/train` 和 `tandt/truck` 训练阶段 `Aborted (core dumped)`，所以 tandt 两个场景目前应标记为 invalid。

## 2. bicycle 新实验: DEFERRED_ADAM

最近的 `nurips26_hide_densification_3_DEFERRED_ADAM` bicycle full run:

训练日志:

`debug/nurips26_hide_densification_3_DEFERRED_ADAM/bicycle/train_0412_1435.log`

指标日志:

`debug/nurips26_hide_densification_3_DEFERRED_ADAM/bicycle/metrics_0412_1518.log`

结果:

| branch | iter | PSNR | SSIM | LPIPS | final points | blocks | seconds | no-gradient skips | max peak_rsv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla3DGS | 30000 | 25.225 | 0.7646 | 0.2104 | 4.867M | 1 | n/a | 0 | n/a |
| hide_densification_3 | 30000 | 25.035 | 0.7527 | 0.2273 | 4.513M | 8 | 2123.9s | 392 | 1.81GB |
| hide_densification_3_DEFERRED_ADAM | 30000 | 24.994 | 0.7581 | 0.2166 | 5.146M | 9 | 2576.8s | 363 | 1.76GB |

解读:

- DEFERRED_ADAM 让 final points 变多，LPIPS 有改善，但 PSNR 没回来，速度还更慢。
- no-gradient skips 仍然有 363 次，说明它没有解决最关键的训练语义问题。
- 因为点数已经超过 vanilla，但 PSNR 仍低，所以“点不够”不是唯一解释。

## 3. 点数证据

从 PLY header 统计 `iteration_30000` 点数:

| scene | vanilla points | current points | ratio |
| --- | ---: | ---: | ---: |
| bicycle | 4,867,163 | 4,513,100 | 0.927 |
| bonsai | 1,073,409 | 1,215,728 | 1.133 |
| counter | 1,068,745 | 1,084,840 | 1.015 |
| drjohnson | 3,073,197 | 3,074,252 | 1.000 |
| flowers | 2,875,589 | 2,820,403 | 0.981 |
| garden | 4,102,109 | 3,792,949 | 0.925 |
| kitchen | 1,577,277 | 1,608,068 | 1.020 |
| playroom | 1,850,010 | 1,688,569 | 0.913 |
| room | 1,300,181 | 1,255,009 | 0.965 |
| stump | 4,318,261 | 4,487,594 | 1.039 |
| treehill | 3,201,211 | 2,899,403 | 0.906 |

关键观察:

- `drjohnson` 当前点数几乎等于 vanilla，但 PSNR 掉了 2.28dB。
- `bonsai/stump/kitchen/counter` 点数不低于 vanilla，但仍有指标退化。
- 所以主因不是最终点数本身，而是训练路径/梯度/过滤导致的点质量、位置、opacity、SH 优化不等价。

## 4. no-gradient skip 证据

当前 full run 中大量出现:

`[iter N] no gradient block was processed; skipping optimizer/log step`

统计:

| scene | total iters | no-gradient skips | skip rate | skips <= 3000 | first split |
| --- | ---: | ---: | ---: | ---: | ---: |
| bicycle | 30000 | 392 | 1.31% | 392 | 2333 |
| drjohnson | 30000 | 357 | 1.19% | 337 | 1927 |
| flowers | 30000 | 400 | 1.33% | 400 | 2624 |
| playroom | 30000 | 350 | 1.17% | 269 | 2028 |
| stump | 30000 | 426 | 1.42% | 360 | 2136 |
| treehill | 30000 | 455 | 1.52% | 455 | 2928 |

这些 skip 大多发生在 early densification 阶段。这个阶段决定 geometry 生长，丢几百轮不是小事。

对照旧 `nurips26`:

- `debug/nurips26/drjohnson/train_0409_0600.log`: no-gradient skips = 0
- `debug/nurips26_hide_densification_3/drjohnson/train_0412_0435.log`: no-gradient skips = 357

## 5. 代码层根因分析

### 5.1 lazy/background densify 导致 visible block 被跳过

训练中，grad pass 遇到正在 densify 的 block 会跳过:

`train.py:887`

```python
if getattr(submodel, '_is_densifying', False):
    profiler.block("grad", submodel_id, skipped="densifying")
    ...
    continue
```

background densify 会设置 `_is_densifying=True`:

`pipeline_grad_sync.py:323`

```python
sm, fn = ops.pop(0)
self._freeze_for_densify(sm)
...
self._bg_densify_thread.start()
```

`_freeze_for_densify`:

`pipeline_grad_sync.py:284`

```python
sm._packed_frozen = sm._packed
...
sm._is_densifying = True
```

结果:

- no-grad pass 可以用 frozen state 旧模型渲染。
- grad pass 如果这个 block 仍在 densify，就完全跳过。
- 如果当前视角主要只看到这个 block，整轮就没有任何 optimizer step。
- 这改变了 vanilla 3DGS 的迭代语义，尤其伤 early densification。

### 5.2 分块 merge 的反传是近似，不是完整 3DGS 梯度

当前 merge 在 no-grad 下执行:

`train.py:823`

```python
with torch.no_grad():
    merge_res = merge_opt_kid(rendered_list, depth_list, alpha_list)
```

默认不是 legacy per-block loss，而是先对最终图算一次 loss grad:

`train.py:842`

```python
final_rgb_proxy = merge_res["final_rgb"].detach().requires_grad_(True)
...
image_loss.backward()
final_rgb_grad = final_rgb_proxy.grad.detach()
```

然后每个 block 用:

`train.py:993`

```python
prefix_T_k = prefix_T[:, 0].gather(dim=0, index=rank_map.unsqueeze(0)).squeeze(0)
backward_grad = prefix_T_k.unsqueeze(0) * final_rgb_grad
torch.autograd.backward(sub_img, grad_tensors=backward_grad)
```

这个近似只把最终 RGB loss 的梯度分配给 block 的 RGB 输出。它没有完整传播:

- 当前 block alpha 对后方 block 的遮挡影响。
- alphaLeft/prefix_T 对最终颜色的导数。
- depth/order 变化带来的排序边界影响。
- background transmittance 相关梯度。

所以即使点数对齐，opacity、位置、SH 的优化方向也可能偏离 vanilla。

### 5.3 小块过滤和贡献过滤会主动丢训练信号

第一处过滤:

`train.py:667`

```python
if n_vis == 0 or (config.SKIP_SMALL_BLOCK_THRESH > 0 and n_vis < max_vis * config.SKIP_SMALL_BLOCK_THRESH):
    continue
```

当前配置:

`config.py:47`

```python
SKIP_SMALL_BLOCK_THRESH = 0.05
```

第二处过滤:

`train.py:760`

```python
valid_pixels = (image > 0).any(dim=0).sum().item()
...
if valid_ratio < 0.05:
    continue
```

风险:

- 细碎小块、边缘块、远处块可能长期少更新。
- early stage 点数还不稳定，过滤更容易影响 densify 统计。
- 它是速度优化，不是质量等价优化。

### 5.4 tandt pipeline 当前会把失败训练继续拿去评估

`runall_4090.sh` 已经去掉 `set -e`，单场景训练失败后仍会继续 render/metrics。

这对自动汇总有风险:

- `train` core dumped 后没有 `training_complete`，metrics 没有指标行。
- `truck` core dumped 后 render 到 `ours_7000`，summarize 容易把它误当最终质量。

修复质量前，summarize 必须把这些结果标记成 invalid。

## 6. 评估路径检查

我检查过 vanilla render 的 GT 和当前 render 的 GT。同名图像 GT 不是逐像素完全一致，但平均 GT-vs-GT PSNR 在 47-50dB 左右。交叉计算结果显示:

以 `drjohnson` 为例:

- vanilla render vs vanilla GT: 29.398
- current render vs current GT: 27.118
- current render vs vanilla GT: 27.093
- vanilla render vs current GT: 29.430
- vanilla GT vs current GT: 50.645

所以 GT/eval 差异存在，但不是 2.28dB 退化的主因。

## 7. 修复策略

总原则:

先做 quality-first baseline，让训练语义尽量靠近 vanilla。质量回来后，再一点点恢复异步、过滤、缓存。

不要同时改多个不等价点，否则无法判断是哪一个修好了质量。

## 8. 三阶段执行计划

### Phase 1: 修复 densify/optimizer 的训练等价性

目标:

- 让 `no gradient block was processed` 在正常训练中归零。
- visible block 不允许因为 `_is_densifying` 被跳过梯度。
- 先接受一点速度损失，确认质量是否恢复。

建议实现:

1. 加一个显式开关，例如:

```python
DENSIFY_EXECUTION_MODE = "sync_quality"
```

或 CLI:

```bash
--densify_execution_mode sync|async
```

2. `sync_quality` 模式下:

- 不启动 background densify thread。
- densify_stats、adam step、densify_and_prune/reset_opacity 按确定顺序在主线程完成。
- dynamic split 仍然保留，但必须发生在 safe point。
- frustum cache 在 split/densify 后清空。

3. 如果还想保留部分异步:

- 在进入 grad pass 前，如果任何 `visible_submodel_id_list` 里的 block 正在 `_is_densifying`，必须 `join_background_densify()`，不能 continue 跳过。
- 非 visible block 可以继续异步 densify，但保存/分裂/pack_to_buffer 必须只在主线程 safe point 发生。

验证:

- 跑 bicycle 750 iter sanity，确认不崩。
- 跑 bicycle 7000 iter，检查 no-gradient skips = 0。
- 跑 drjohnson 7000 iter，检查 no-gradient skips = 0。
- 跑 bicycle 30000 和 drjohnson 30000，对比 PSNR。

通过标准:

- full train log 有 `[training_complete]`。
- metrics 是 `ours_30000`。
- no-gradient skips = 0。
- bicycle peak_rsv 最大值不超过 2GB。
- drjohnson PSNR 退化从 -2.28dB 明显收窄，目标先做到 delta > -0.5dB。

### Phase 2: 关闭/参数化训练信号过滤

目标:

确认 `SKIP_SMALL_BLOCK_THRESH=0.05` 和 `valid_ratio < 0.05` 对质量影响多大。

建议实现:

1. 把贡献过滤阈值从 hardcode 改成配置:

```python
CONTRIBUTION_PIXEL_RATIO_THRESH = 0.0
```

2. quality baseline 中:

```python
SKIP_SMALL_BLOCK_THRESH = 0.0
CONTRIBUTION_PIXEL_RATIO_THRESH = 0.0
```

3. 如果速度损失太大，再做 schedule:

- `iteration < densify_until_iter`: threshold = 0
- `iteration >= densify_until_iter`: threshold 可以恢复到 0.01 或 0.02
- 0.05 先不要作为默认质量配置

验证矩阵:

| run | densify mode | skip small | contribution filter | target |
| --- | --- | ---: | ---: | --- |
| A | sync_quality | 0.05 | 0.05 | 复现当前退化 |
| B | sync_quality | 0.00 | 0.05 | 看小块过滤影响 |
| C | sync_quality | 0.05 | 0.00 | 看贡献过滤影响 |
| D | sync_quality | 0.00 | 0.00 | quality upper bound |

优先场景:

- `drjohnson`: 质量掉得最大，点数又不缺，最能暴露训练语义问题。
- `bicycle`: 用户关注显存上限，且已有完整对照。

通过标准:

- 如果 D 明显恢复 PSNR，则过滤是主因之一。
- 如果 D 仍不恢复，则主要问题在 merge gradient 或 densify/optimizer 语义。

### Phase 3: 修复/验证分块 merge 反传

目标:

让分块训练的梯度尽可能接近 vanilla 3DGS，而不是只靠 `prefix_T * final_rgb_grad` 的 RGB 近似。

短期验证:

1. 用 `--legacy_per_block_loss` 做质量对照。

当前 legacy 路径会为每个 block 构造:

```python
composed_img = C_base + prefix_T_k * C_active
loss(composed_img, gt)
```

它仍然不是完整可微 merge，但可以判断 single-image-loss-grad 是否是质量退化源。

2. 做一个单场景小矩阵:

| run | loss mode | expected |
| --- | --- | --- |
| current single-image grad | default | 当前速度路径 |
| legacy per-block loss | `--legacy_per_block_loss` | 判断质量是否回升 |
| differentiable merge prototype | 新实现 | 长期正确方向 |

长期实现方向:

- merge 的排序 rank 可以暂时 no-grad。
- 但一旦 rank 固定，颜色/alpha/transmittance 的组合应该用可微 PyTorch 表达式重建。
- 对每个 block 的 `render_pkg["render"]` 和 alpha 输出同时保留梯度。
- 让 loss 通过 compositing 公式反传到当前 block 的 RGB 和 opacity/alpha，而不是只传 RGB。

最低限度的检查:

- forward: 分块 merge 图像和 monolithic render 的 PSNR 应非常高。
- gradient: 对一个小模型/小图，比较 monolithic loss backward 和 block loss backward 的关键参数梯度方向 cosine similarity。
- quality: drjohnson 30k PSNR 恢复到 vanilla -0.3dB 以内。

## 9. pipeline 与 summarize 必须加的保护

为了避免再次把失败结果混进平均值:

1. `summarize.sh` 必须检查 train log 是否有 `[training_complete]`。
2. metrics 必须检查 method 是 `ours_30000`，否则标记为 invalid。
3. 如果 train core dumped，render/metrics 可以继续跑，但 summarize 不应计入平均值。
4. 输出表建议增加:

```text
Status: OK | TRAIN_INCOMPLETE | METRICS_MISSING | WRONG_ITER
MetricIter: 30000 | 7000 | N/A
NoGradSkip: count
```

## 10. 当前可接受的显存边界

用户明确关注的是服务器可见真实 GPU mem 峰值，主要看最大值，不看 P90。

bicycle 目标:

- 不必压到原始 1.56GB。
- 只要最大值不超过 2GB 即可。

当前已知:

- `nurips26_hide_densification_3/bicycle/train_0412_0014.log`: max peak_rsv 约 1.81GB。
- `nurips26_hide_densification_3_DEFERRED_ADAM/bicycle/train_0412_1435.log`: max peak_rsv 约 1.76GB。

因此 Phase 1 可以先牺牲一部分速度，但要继续守住 bicycle peak_rsv < 2GB。

## 11. 建议下一步

下一步只做一件事:

实现 `sync_quality` densify 模式，目标是 no-gradient skips 归零。

完成后先跑:

```bash
bash /home/jian/gaussian-splatting/test_4090.sh
```

然后跑一个 7000iter 或 full 的 `drjohnson`。如果 drjohnson 明显恢复，说明 background densify skip 是主因。若没有恢复，再进入 Phase 2 关闭过滤。

推荐实验顺序:

1. bicycle 750iter sanity: 检查崩溃、峰值、速度。
2. drjohnson 7000iter: 检查 no-gradient skips 是否归零。
3. drjohnson 30000: 看 PSNR 是否从 27.118 回升。
4. bicycle 30000: 确认峰值 < 2GB，PSNR 不低于当前。

## 12. 成功标准

最低成功标准:

- 所有参与比较的场景 train 必须完整完成。
- metrics 必须是 `ours_30000`。
- no-gradient skips = 0。
- bicycle max peak_rsv < 2GB。
- drjohnson PSNR 从 27.118 至少回到 28.9+。

理想成功标准:

- mip360 平均 PSNR delta 在 -0.15dB 以内。
- deepblending 两个场景 PSNR delta 在 -0.3dB 以内。
- LPIPS 不比 vanilla 差超过 0.005 到 0.010。
- bicycle 速度在质量恢复后再逐步回到 14 it/s 以上。

