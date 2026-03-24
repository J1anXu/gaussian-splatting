import os
import logging
from datetime import datetime


class FlushFileHandler(logging.FileHandler):
    """每条日志写完立即 flush，IDE 打开能实时看到更新。"""
    def emit(self, record):
        super().emit(record)
        self.flush()


class ShortTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%m%d,%H:%M")


_formatter = ShortTimeFormatter(fmt="%(asctime)s - %(message)s")


def get_logger(scene_name, log_path, level=logging.INFO, to_console=False):
    os.makedirs(log_path, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_file = os.path.join(log_path, f"{timestamp}.log")

    logger = logging.getLogger(scene_name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        file_handler = FlushFileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_formatter)
        logger.addHandler(file_handler)

        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(_formatter)
            logger.addHandler(console_handler)

    if not hasattr(logger, '_log_timestamp'):
        logger._log_timestamp = timestamp
    return logger


def add_output_path(logger, output_log_dir, prefix=None):
    """Add a second file handler so the same log is mirrored to the output folder."""
    os.makedirs(output_log_dir, exist_ok=True)
    timestamp = getattr(logger, '_log_timestamp', datetime.now().strftime("%m%d_%H%M"))
    fname = f"{prefix}_{timestamp}.log" if prefix else f"{timestamp}.log"
    log_file = os.path.join(output_log_dir, fname)

    fh = FlushFileHandler(log_file, encoding="utf-8")
    fh.setLevel(logger.level)
    fh.setFormatter(_formatter)
    logger.addHandler(fh)
