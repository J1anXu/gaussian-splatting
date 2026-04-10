import os
import json
import logging
import platform
import socket
import subprocess
import sys
from datetime import datetime


class FlushFileHandler(logging.FileHandler):
    """每条日志写完立即 flush，IDE 打开能实时看到更新。"""
    def emit(self, record):
        super().emit(record)
        self.flush()


class ShortTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%m%d,%H:%M:%S")


_formatter = ShortTimeFormatter(fmt="%(asctime)s.%(msecs)03d - %(message)s")


def _attach_file_handler(logger, log_file, level):
    file_handler = FlushFileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
    return file_handler


def get_logger(scene_name, log_path, level=logging.INFO, to_console=False, reset_handlers=True):
    os.makedirs(log_path, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_file = os.path.join(log_path, f"{timestamp}.log")

    logger_name = f"{scene_name}:{os.path.abspath(log_path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    if not logger.handlers:
        _attach_file_handler(logger, log_file, level)
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(_formatter)
            logger.addHandler(console_handler)

    logger._log_timestamp = timestamp
    logger._primary_log_file = log_file
    logger._log_files = [log_file]
    return logger


def add_output_path(logger, output_log_dir, prefix=None):
    """Add a second file handler so the same log is mirrored to the output folder."""
    os.makedirs(output_log_dir, exist_ok=True)
    timestamp = getattr(logger, '_log_timestamp', datetime.now().strftime("%m%d_%H%M"))
    fname = f"{prefix}_{timestamp}.log" if prefix else f"{timestamp}.log"
    log_file = os.path.join(output_log_dir, fname)

    log_files = getattr(logger, '_log_files', [])
    if log_file in log_files:
        return log_file

    _attach_file_handler(logger, log_file, logger.level)
    logger._log_files = [*log_files, log_file]
    return log_file


def log_section(logger, title):
    logger.info("")
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def _json_default(obj):
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def log_kv(logger, tag, data):
    """Write one structured JSON payload with a stable, grep-friendly tag."""
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=_json_default)
    logger.info(f"[{tag}] {payload}")


def _run_cmd(cmd, cwd=None, timeout=5):
    try:
        return subprocess.check_output(
            cmd,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def collect_runtime_context(repo_dir=None):
    ctx = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "argv": sys.argv,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
    }

    repo_dir = repo_dir or os.getcwd()
    ctx["git"] = {
        "branch": _run_cmd(["git", "branch", "--show-current"], cwd=repo_dir),
        "commit": _run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir),
        "status_short": _run_cmd(["git", "status", "--short"], cwd=repo_dir),
        "submodules": _run_cmd(["git", "submodule", "status"], cwd=repo_dir),
    }

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        ctx["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": cuda_available,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        if cuda_available:
            devices = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": prop.name,
                    "total_memory_gb": round(prop.total_memory / 1024**3, 3),
                    "major": prop.major,
                    "minor": prop.minor,
                })
            ctx["torch"]["devices"] = devices
    except Exception as exc:
        ctx["torch"] = f"unavailable: {exc}"

    ctx["nvidia_smi"] = _run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ])
    return ctx


def log_runtime_context(logger, repo_dir=None, extra=None):
    ctx = collect_runtime_context(repo_dir=repo_dir)
    if extra:
        ctx["extra"] = extra
    log_kv(logger, "runtime_context", ctx)
    if hasattr(logger, "_log_files"):
        log_kv(logger, "log_files", logger._log_files)
    return ctx
