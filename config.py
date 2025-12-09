# config.py
# 控制加载数据集大小,减少启动时间,用于调试
LIMITED_DATASIZE = True
DATASIZE_LIMIT = 2
# DEBUG模式会load 30000
DEBUG_MODE = True

CAL_RES_2_CPU = False  #把计算结果移动到CPU节省内存 (能节省内存但是移动非常慢)