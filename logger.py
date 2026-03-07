import os
import logging
from datetime import datetime


def get_logger(name, log_dir, level=logging.INFO):
    """Create a file-only logger matching Reproduction-GS-Scale format.

    Writes to: {log_dir}/train_{YYYYMMDD_HHMMSS}.log
    Format:    2026-03-07 12:34:56 | message
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

    print(f"Training log: {log_file}")
    return logger
