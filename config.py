# config.py
# 控制加载数据集大小,减少启动时间,用于调试
LIMITED_DATASIZE = False
DATASIZE_LIMIT = 5


CAL_RES_2_CPU = False  #把计算结果移动到CPU节省内存 (能节省内存但是移动非常慢)

SAVE_BLOCK_IMG = False  # 是否保存 block 渲染结果
DRAW_BOX = False # 是否在 block 渲染结果上绘制点云和block的边界框


SAVE_RGB_LAYERS = False  # 是否保存合并前的各个RGB图层
SAVE_LAYERS_CONTRIBUTION = False  # 是否保存每个block对各个图层的贡献
SAVE_DEPTH_LIST = False  # 是否保存各个block的深度图


PARTITIONING_ENABLED = True  # 是否启用分块处理
FRUSTUM_CULLING_ENABLED = True  # 是否启用视锥剔除
FRUSTUM_CULLING_CACHE_ENABLED = True  # 是否缓存视锥剔除结果（densify_until_iter后位置不再增长，可复用）

INDOOR_SCENES = {"room", "counter", "kitchen", "bonsai", "drjohnson", "playroom"}
OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill", "train", "truck"}

SPLIT_SIZE_INDOOR = 400000
SPLIT_SIZE_OUTDOOR = 800000

SPLIT_SIZE = SPLIT_SIZE_OUTDOOR  # default, overridden at runtime by scene type

GPU_PACKED_CACHE_POINT_BUDGET_INDOOR = 400000
GPU_PACKED_CACHE_POINT_BUDGET_OUTDOOR = 750000
GPU_PACKED_CACHE_POINT_BUDGET = GPU_PACKED_CACHE_POINT_BUDGET_OUTDOOR  # default, overridden at runtime by scene type

GPU_CACHE_THRESHOLD_GB = 0.5  # reserved 超过 allocated 多少 GB 时清理 CUDA 缓存，默认优先控制峰值显存
GPU_CACHE_HARD_LIMIT_GB = 1.35  # reserved 过高时无视 interval 直接清理，防止 nvidia-smi 峰值长时间堆高
GPU_CACHE_STAGE_ENTRY_LIMIT_GB = 0.0  # 进入大 render 阶段前的 reserved 水位；0 表示关闭预清理
CUDA_EMPTY_CACHE_INTERVAL = 16  # 每 N 轮允许清一次 CUDA allocator cache；skip 路径仍会强制清理

MERGE_FAST = True  # True: 全图merge(快,多用~90MB), False: 分块merge(省显存,慢)

HALF = False        # 半精度梯度拷贝加速
TIMELINE = False    # Timeline 日志开关 (Chrome Trace JSON → Perfetto UI)


KEEP_TRAINING = False  # 由命令行参数 --keep_training 控制

SKIP_SMALL_BLOCK_THRESH = 0.05  # 跳过可见点数小于最大块该比例的小块，设为0关闭

DENSIFY_GRAD_SCALE = 0.95  # densify 梯度阈值缩放，补偿 block 分割后 prefix_T 对梯度的缩放 (1.0=不补偿, 越小越激进)
