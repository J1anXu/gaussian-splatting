# config.py
# 控制加载数据集大小,减少启动时间,用于调试
LIMITED_DATASIZE = False
DATASIZE_LIMIT = 20
# DEBUG模式会load 30000
DEBUG_MODE = True

CAL_RES_2_CPU = False  #把计算结果移动到CPU节省内存 (能节省内存但是移动非常慢)

DRAW_BLOCK = False  # 是否在渲染图上绘制 block 投影轮廓
BLOCK_WIRE_SAVE = True  # 是否保存 block 渲染结果