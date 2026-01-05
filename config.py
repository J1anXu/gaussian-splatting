# config.py
# 控制加载数据集大小,减少启动时间,用于调试
LIMITED_DATASIZE = True
DATASIZE_LIMIT = 20
# DEBUG模式会load 30000
DEBUG_MODE = True

CAL_RES_2_CPU = False  #把计算结果移动到CPU节省内存 (能节省内存但是移动非常慢)

SAVE_BLOCK_IMG = True  # 是否保存 block 渲染结果
DRAW_BOX = False # 是否在 block 渲染结果上绘制点云和block的边界框


SAVE_RGB_LAYERS = True  # 是否保存合并前的各个RGB图层
SAVE_LAYERS_CONTRIBUTION = True  # 是否保存每个block对各个图层的贡献