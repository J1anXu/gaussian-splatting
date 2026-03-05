# config.py
# 控制加载数据集大小,减少启动时间,用于调试
LIMITED_DATASIZE = True
DATASIZE_LIMIT = 5

# Saving and logging
CAL_RES_2_CPU = False  #把计算结果移动到CPU节省内存 (能节省内存但是移动非常慢)
SAVE_BLOCK_IMG = False  # 是否保存 block 渲染结果
DRAW_BOX = False # 是否在 block 渲染结果上绘制点云和block的边界框
SAVE_RGB_LAYERS = False  # 是否保存合并前的各个RGB图层
SAVE_LAYERS_CONTRIBUTION = False  # 是否保存每个block对各个图层的贡献
SAVE_DEPTH_LIST = False  # 是否保存各个block的深度图


PARTITIONING_ENABLED = True  # 是否启用分块处理
FRUSTUM_CULLING_ENABLED = True  # 是否启用视锥剔除

BLOCK_NUMS = 8

# ---- Gradient Accumulation ----
GA_ENABLED = False  # 是否启用梯度累积
GA_START_ITER = 5000  # 开始梯度累积的迭代次数
GA_ACCUMULATION_STEPS = 5  # 累积步数
GA_WARMUP_END_ITER = 6000  # LR warmup 结束迭代次数

# ---- Adaptive optimization scheduling (V1) ----
BLOCK_TIERED_UPDATE = True  # 是否启用分层更新频率
BLOCK_TIERED_START_ITER = 15000  # 增点结束后开始
BLOCK_TIERED_TOP_RATIO = 0.2  # top 20% 每步更新
BLOCK_TIERED_TOP_INTERVAL = 1
BLOCK_TIERED_BOT_RATIO = 0.2  # bottom 20% 每8步更新
BLOCK_TIERED_BOT_INTERVAL = 8
BLOCK_TIERED_MID_INTERVAL = 4  # 中间 60% 每4步更新

# ---- Adaptive Densification Scheduling ----
ADAPTIVE_DENS_ENABLED = False        # 是否启用自适应densification频率
ADAPTIVE_DENS_CHECK_INTERVAL = 200  # 每隔多少iter检测一次importance趋势
ADAPTIVE_DENS_HISTORY_LEN = 10      # 保留多少个快照用于趋势分析
ADAPTIVE_DENS_FLAT_SLOPE_THRESH = 0.02   # 归一化斜率阈值，低于此视为平缓
ADAPTIVE_DENS_FLAT_VAR_THRESH = 0.05     # 归一化标准差阈值，低于此视为稳定
ADAPTIVE_DENS_SCALE_FACTOR = 2.0    # 平缓时interval翻倍
ADAPTIVE_DENS_MAX_SCALE = 8         # 最大倍数（相对于base densification_interval）