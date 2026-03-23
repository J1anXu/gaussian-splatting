"""
统一日志提取，输出标准化 CSV 到 debug/ 目录。

用法:
    python3 plot/extract.py --type capgs  logs/train/nurips26/bicycle/0323_1318.log --name bicycle
    python3 plot/extract.py --type gsscale /data/.../train_20260312_121049.log --name bicycle
"""
import argparse
import ast
import csv
import re
import sys
from pathlib import Path

# ── 统一输出列 ──
# 两种日志都归一化到这些列，缺失的填空
UNIFIED_COLS = [
    "step", "loss", "pts_M", "vis_M", "vis_pct", "blk",
    "alloc_gb", "rsv_gb", "peak_alloc_gb", "peak_rsv_gb",
    "it_s", "elapsed",
]

EVENT_COLS = ["step", "event", "detail"]

# ── CapGS 解析 ──
_CAPGS_METRIC_RE = re.compile(r"^(\d{4},\d{2}:\d{2}) - (\{.+\})$")
_CAPGS_SPLIT_RE = re.compile(
    r"^(\d{4},\d{2}:\d{2}) - \[iter (\d+)\] Split block\(s\) \[(.+?)\], now (\d+) blocks, sizes: \[(.+?)\]$"
)
_CAPGS_PARTITION_RE = re.compile(
    r"^(\d{4},\d{2}:\d{2}) - Partitioned into (\d+) blocks?, sizes: \[(.+?)\]$"
)

def _parse_capgs(log_path):
    metrics, events = [], []
    for line in open(log_path):
        line = line.strip()
        if not line:
            continue
        m = _CAPGS_METRIC_RE.match(line)
        if m:
            d = ast.literal_eval(m.group(2))
            metrics.append({
                "step":           d.get("iter", ""),
                "loss":           d.get("L", ""),
                "pts_M":          float(str(d.get("pts", "0")).rstrip("M")),
                "vis_M":          float(str(d.get("vis", "0")).rstrip("M")),
                "vis_pct":        d.get("vis%", ""),
                "blk":            d.get("blk", ""),
                "alloc_gb":       d.get("alloc", ""),
                "rsv_gb":         d.get("rsv", ""),
                "peak_alloc_gb":  d.get("peak_alloc", ""),
                "peak_rsv_gb":    d.get("peak_rsv", ""),
                "it_s":           d.get("it/s", ""),
                "elapsed":        float(str(d.get("elapsed", "0")).rstrip("s")) if d.get("elapsed") else "",
            })
            continue
        m = _CAPGS_SPLIT_RE.match(line)
        if m:
            events.append({
                "step": int(m.group(2)),
                "event": "split",
                "detail": f"blocks={m.group(4)} sizes=[{m.group(5)}]",
            })
            continue
        m = _CAPGS_PARTITION_RE.match(line)
        if m:
            events.append({
                "step": 0,
                "event": "partition",
                "detail": f"blocks={m.group(2)} sizes=[{m.group(3)}]",
            })
    return metrics, events

# ── GS-Scale 解析 ──
_GS_METRIC_RE = re.compile(
    r"step=(?P<step>\d+)/\d+ \| "
    r"loss=(?P<loss>[\d.]+) l1=[\d.]+ ssim=[\d.]+ \| "
    r"pts=(?P<pts>[\d.]+)M fru=(?P<fru>[\d.]+)M\((?P<fru_pct>[\d.]+)%\) \| "
    r"mem=[\d.]+G\((?P<alloc>[\d.]+)\+(?P<rsv>[\d.]+)\) peak=(?P<peak>[\d.]+)G \| "
    r"sh=\d+ \| "
    r"elapsed=(?P<elapsed>[\d.]+)s"
)
_GS_EVAL_RE = re.compile(
    r"\[Eval (?P<split>\w+) step=(?P<step>\d+)\] "
    r"PSNR: (?P<psnr>[\d.]+), SSIM: (?P<ssim>[\d.]+), LPIPS: (?P<lpips>[\d.]+)"
)
_GS_COMPLETE_RE = re.compile(
    r"Training complete\. Total time: (?P<time>[\d:]+) \| res: (?P<res>\S+) \| num_GS: (?P<gs>\d+)"
)

def _parse_gsscale(log_path):
    metrics, events = [], []
    for line in open(log_path):
        line = line.strip()
        if not line:
            continue
        m = _GS_METRIC_RE.search(line)
        if m:
            d = m.groupdict()
            metrics.append({
                "step":           int(d["step"]),
                "loss":           float(d["loss"]),
                "pts_M":          float(d["pts"]),
                "vis_M":          float(d["fru"]),
                "vis_pct":        float(d["fru_pct"]),
                "blk":            1,
                "alloc_gb":       float(d["alloc"]),
                "rsv_gb":         float(d["rsv"]),
                "peak_alloc_gb":  float(d["peak"]),
                "peak_rsv_gb":    "",
                "it_s":           "",
                "elapsed":        float(d["elapsed"]),
            })
            continue
        m = _GS_EVAL_RE.search(line)
        if m:
            events.append({
                "step": int(m.group("step")),
                "event": f"eval_{m.group('split')}",
                "detail": f"PSNR={m.group('psnr')} SSIM={m.group('ssim')} LPIPS={m.group('lpips')}",
            })
            continue
        m = _GS_COMPLETE_RE.search(line)
        if m:
            events.append({
                "step": -1,
                "event": "complete",
                "detail": f"time={m.group('time')} res={m.group('res')} gs={m.group('gs')}",
            })
    return metrics, events

# ── 分发 ──
PARSERS = {
    "capgs": _parse_capgs,
    "gsscale": _parse_gsscale,
}

def write_csv(rows, cols, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Extract training log to unified CSV")
    parser.add_argument("log", help="Path to .log file")
    parser.add_argument("--type", required=True, choices=PARSERS.keys(), help="Log format type")
    parser.add_argument("--name", help="Scene name (default: inferred from log stem)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"File not found: {log_path}")

    name = args.name or log_path.stem

    metrics, events = PARSERS[args.type](log_path)

    debug_dir = Path(__file__).resolve().parent.parent / "debug"
    metrics_path = debug_dir / f"{name}_metrics.csv"
    events_path = debug_dir / f"{name}_events.csv"

    write_csv(metrics, UNIFIED_COLS, metrics_path)
    # write_csv(events, EVENT_COLS, events_path)

    if metrics:
        first, last = metrics[0], metrics[-1]
        print(f"\nSummary: step {first['step']}-{last['step']}, "
              f"loss {first['loss']}->{last['loss']}, "
              f"pts {first['pts_M']}->{last['pts_M']}M, "
              f"alloc {first['alloc_gb']}->{last['alloc_gb']}G, "
              f"rsv {first['rsv_gb']}->{last['rsv_gb']}G")


if __name__ == "__main__":
    main()
