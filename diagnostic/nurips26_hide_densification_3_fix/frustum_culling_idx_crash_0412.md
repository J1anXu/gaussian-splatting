# frustum_culling_idx 脏索引崩溃分析与修复记录

日期: 2026-04-12 17:03 CDT

分支/实验: `nurips26_hide_densification_3_fix`

场景: `deepblending/drjohnson`

## 1. 现象

训练跑到一半失败，训练日志停在 13443 附近，wrapper 继续进入 render/metrics 阶段。

关键日志:

```text
RuntimeError: [iter 13444] submodel 1: visible_indices.max()=4946570873076711424 >= packed.shape[0]=384964, _xyz_contig=384964, cached=False
```

相关文件:

```text
debug/nurips26_hide_densification_3_fix/deepblending/drjohnson/train.log
debug/nurips26_hide_densification_3_fix/drjohnson/train_0412_1642.log
debug/nurips26_hide_densification_3_fix/drjohnson/render_0412_1653.log
```

注意: `runall_4090.sh` 当前训练失败后仍会继续 render，因此这次 render 加载的是 `iteration_7000` checkpoint，不是完整 30000 iteration 结果。

render 日志证据:

```text
Loaded 8 blocks, 2067146 total gaussians from .../iteration_7000
Rendering test set, iter=7000
```

所以这次 drjohnson 的 metrics 不能作为完整训练质量指标。

## 2. 排除项

这次不是 OOM。

日志里的显存峰值没有达到异常 OOM 水平，报错也不是 CUDA allocator 报错，而是训练代码主动校验 `visible_indices` 时发现越界。

这次也不是 FastMerge 直接导致。

运行时配置里有:

```json
"merge_fast": false
```

因此本次训练走的是 PyTorch chunked argsort merge，而不是 fused CUDA FastMerge。FastMerge K>8/K>16 的修复是另一个问题，本次崩溃发生在 merge 之前的 frustum culling index 阶段。

## 3. 直接原因

`visible_indices.max()` 出现了 `4946570873076711424` 这种天文数字。

正常情况下，`submodel 1` 的 packed 点数只有:

```text
packed.shape[0]=384964
_xyz_contig=384964
```

所以这个值不可能是正常 frustum culling 算出来的点索引，更像是未初始化内存里的脏值混进了 index tensor。

触发位置:

```text
train.py: frustum_culling_idx(...) -> model.visible_indices
train.py: validation visible_indices.max() >= packed.shape[0]
```

对应 Python 调用:

```python
model.visible_indices = frustum_culling_idx(xyz_for_cull, cpu_full_proj_transform_dict[cam_name])
```

对应 C++ 扩展:

```text
submodules/diff-gaussian-rasterization/cpu_adam.cpp
frustum_culling_idx(...)
```

## 4. 根因分析

`frustum_culling_idx` 的 C++ 实现分两步:

1. 对每个逻辑 chunk 统计 mask 中可见点数量，得到 `thread_counts`。
2. prefix sum 得到 offsets，然后把每个 chunk 的 visible index 写进 `torch::empty({total})` 输出 tensor。

旧代码的核心风险是: 第二步依赖 `omp_get_thread_num()` 对应 chunk id。

旧逻辑近似如下:

```cpp
#pragma omp parallel num_threads(NT)
{
    int tid = omp_get_thread_num();
    int64_t lo = tid * chunk;
    int64_t hi = std::min(lo + chunk, N);
    int64_t pos = offsets[tid];
    ...
}
```

这隐含了一个危险假设: OpenMP 一定会启动请求的 `NT` 个线程，并且每个 `tid=0..NT-1` 都会出现。

但 OpenMP 允许实际线程数小于 `num_threads(NT)`，尤其在 `OMP_DYNAMIC`、嵌套 OpenMP、系统线程限制或 runtime 调度策略影响下。如果第二段实际只启动了更少线程，那么后面的逻辑 chunk 根本不会执行。

由于输出 tensor 是:

```cpp
torch::empty({total}, dtype=int64)
```

没被写入的区间会保留未初始化内存。后续 Python 把这个 tensor 当作 `visible_indices` 使用，就可能出现极大的随机 int64 数字，例如:

```text
4946570873076711424
```

这和本次 crash 完全吻合。

## 5. 修复

修复文件:

```text
submodules/diff-gaussian-rasterization/cpu_adam.cpp
```

修复思路:

不再让 `omp_get_thread_num()` 决定 chunk id，而是显式用 `parallel for` 遍历逻辑 chunk id `t=0..NT-1`。

修复后代码核心:

```cpp
const int NT_REQ = 16;
const int NT = NT_REQ;
const int64_t chunk = (N + NT - 1) / NT;
std::vector<int64_t> thread_counts(NT, 0);

#pragma omp parallel for schedule(static) num_threads(NT)
for (int t = 0; t < NT; ++t) {
    int64_t lo = t * chunk;
    int64_t hi = std::min(lo + chunk, N);
    int64_t cnt = 0;
    for (int64_t i = lo; i < hi; i++) cnt += mk[i];
    thread_counts[t] = cnt;
}

...

#pragma omp parallel for schedule(static) num_threads(NT)
for (int t = 0; t < NT; ++t) {
    int64_t lo = t * chunk;
    int64_t hi = std::min(lo + chunk, N);
    int64_t pos = offsets[t];
    for (int64_t i = lo; i < hi; i++) {
        if (mk[i]) op[pos++] = i;
    }
}
```

这样即使 OpenMP runtime 实际只用了更少 worker thread，`parallel for` 也会把全部 `t=0..NT-1` 迭代分配完，不会留下没写入的 output chunk。

## 6. 编译

该修复位于 C++ 扩展，需要重新编译。

已执行:

```bash
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 /home/jian/miniconda3/envs/partgs_fix/bin/python setup.py build_ext --inplace
```

工作目录:

```text
/home/jian/gaussian-splatting/submodules/diff-gaussian-rasterization
```

编译结果: 成功。

编译过程中只有一个已有 warning:

```text
warning: unused variable 'compensation'
```

这与本次修复无关。

## 7. 验证

做了一个 CPU frustum index 压力测试，特意设置:

```bash
OMP_DYNAMIC=TRUE
```

目的是模拟 OpenMP 实际线程数可能不稳定的情况。

测试结果:

```text
{'runs': 80, 'bad': 0, 'empty': 0, 'max_idx': 199995, 'avg_visible': 46143, 'N': 200000}
```

含义:

- 连续 80 次调用 `frustum_culling_idx`
- 没有出现负 index
- 没有出现 `idx.max() >= N`
- 没有出现空结果
- `max_idx=199995 < N=200000`

这个测试不能替代完整训练，但它覆盖了本次 crash 的直接类型: C++ frustum culling 扩展返回越界/脏 index。

## 8. 后续建议

1. 重新跑 `deepblending/drjohnson`，优先确认 `iter 13444` 附近不再出现 `visible_indices.max()` 越界。
2. 修改 `runall_4090.sh` 的失败策略，训练失败后不要继续 render/metrics，否则会把 `iteration_7000` 当成当前结果混进 summarize。
3. 如果后续仍出现类似越界，下一步在 Python 层给 `frustum_culling_idx` 加防御式校验: 一旦发现 `idx.min()<0` 或 `idx.max()>=N`，立刻 fallback 到 PyTorch mask+nonzero，并记录 warning。
4. FastMerge 质量问题和本次 crash 分开跟踪。本次失败日志中 `merge_fast=false`，不能用这次 crash 证明 FastMerge 修复有效或无效。

