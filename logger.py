import os
import logging
from datetime import datetime

def get_logger(scene_name, log_path, level=logging.INFO, to_console=False):
    # 构造日志目录
    os.makedirs(log_path, exist_ok=True)

    # 构造日志文件路径
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_file = os.path.join(log_path, f"{timestamp}.log")

    # 创建 logger 对象
    logger = logging.getLogger(scene_name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复输出

    # 避免重复添加 handler
    if not logger.handlers:
        # 自定义 formatter：只显示时间和 message
        class ShortTimeFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created)
                return dt.strftime("%m%d,%H:%M")

        formatter = ShortTimeFormatter(fmt="%(asctime)s - %(message)s")

        # 文件 handler（总是写文件）
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 控制台 handler（可选）
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger
