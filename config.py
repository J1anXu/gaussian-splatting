# config.py
# 控制加载数据集大小,减少启动时间,用于调试
LIMITED_DATASIZE = True
DATASIZE_LIMIT = 5


PARTITIONING_ENABLED = True  # 是否启用分块处理
FRUSTUM_CULLING_ENABLED = True  # 是否启用视锥剔除
FRUSTUM_CULLING_USE_GSPLAT = True  # True=gsplat CUDA kernel（考虑Gaussian范围），False=自己实现的中心点culling
FRUSTUM_CULLING_CACHE_ENABLED = True  # 是否缓存视锥剔除结果（densify_until_iter后位置不再增长，可复用）

SPLIT_SIZE = 1000000
NUM_BLOCKS = 8  # 分块数量，必须是2的幂次（如 2, 4, 8, 16）

GPU_CACHE_THRESHOLD_GB = 1.0  # reserved 超过 allocated 多少 GB 时清理 CUDA 缓存

HALF_H2D = True    # FP16 H2D transfer（CPU→GPU，SH等属性传输减半带宽）
HALF_D2H = True    # FP16 D2H transfer（GPU→CPU，梯度传输减半带宽）


TIMELINE = False    # Timeline 日志开关 (Chrome Trace JSON → Perfetto UI)

# 测速：跳过前 BENCH_WARMUP 次 iteration，测后 BENCH_ITERS 次的平均耗时
BENCHMARK = True
BENCH_WARMUP = 200   # 预热迭代数（跳过 densification、JIT 编译等不稳定阶段）
BENCH_ITERS = 500    # 计时迭代数