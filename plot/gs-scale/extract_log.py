"""
从 GS-Scale 训练日志中提取关键信息，输出标准化 CSV。

日志格式:
  2026-03-12 12:10:51 | step=1/30000 | loss=0.5655 l1=0.4784 ssim=0.9136 | pts=0.05M fru=0.01M(25.5%) | mem=0.30G(0.07+0.22) peak=0.21G | sh=0 | elapsed=1.1s

注意: mem=X.XXG(alloc+rsv) 中 mem 是 alloc+rsv 的和，我们提取括号内的 alloc 和 rsv。

用法:
    python3 plot/gs-scale/extract_log.py <log_path> [--name bicycle]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# 匹配训练 metric 行
METRIC_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
    r"step=(?P<step>\d+)/(?P<max_step>\d+) \| "
    r"loss=(?P<loss>[\d.]+) l1=(?P<l1>[\d.]+) ssim=(?P<ssim>[\d.]+) \| "
    r"pts=(?P<pts>[\d.]+)M fru=(?P<fru>[\d.]+)M\((?P<fru_pct>[\d.]+)%\) \| "
    r"mem=[\d.]+G\((?P<alloc>[\d.]+)\+(?P<rsv>[\d.]+)\) peak=(?P<peak>[\d.]+)G \| "
    r"sh=(?P<sh>\d+) \| "
    r"elapsed=(?P<elapsed>[\d.]+)s"
)

# 匹配 eval 行
EVAL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
    r"\[Eval (?P<split>\w+) step=(?P<step>\d+)\] "
    r"PSNR: (?P<psnr>[\d.]+), SSIM: (?P<ssim>[\d.]+), LPIPS: (?P<lpips>[\d.]+) "
    r"Time: (?P<time>[\d.]+)s/image Number of GS: (?P<num_gs>\d+)"
)

# 匹配 training complete 行
COMPLETE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
    r"Training complete\. Total time: (?P<total_time>[\d:]+) \| "
    r"res: (?P<res>\S+) \| num_GS: (?P<num_gs>\d+)"
)

METRIC_COLS = ["timestamp", "step", "loss", "l1", "ssim", "pts_M", "fru_M", "fru_pct",
               "alloc_gb", "rsv_gb", "peak_gb", "sh", "elapsed"]

EVENT_COLS = ["timestamp", "step", "event", "psnr", "ssim", "lpips", "num_gs", "detail"]


def parse_log(log_path):
    metrics = []
    events = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = METRIC_RE.match(line)
            if m:
                d = m.groupdict()
                metrics.append({
                    "timestamp": d["ts"],
                    "step": int(d["step"]),
                    "loss": float(d["loss"]),
                    "l1": float(d["l1"]),
                    "ssim": float(d["ssim"]),
                    "pts_M": float(d["pts"]),
                    "fru_M": float(d["fru"]),
                    "fru_pct": float(d["fru_pct"]),
                    "alloc_gb": float(d["alloc"]),
                    "rsv_gb": float(d["rsv"]),
                    "peak_gb": float(d["peak"]),
                    "sh": int(d["sh"]),
                    "elapsed": float(d["elapsed"]),
                })
                continue

            m = EVAL_RE.match(line)
            if m:
                d = m.groupdict()
                events.append({
                    "timestamp": d["ts"],
                    "step": int(d["step"]),
                    "event": f"eval_{d['split']}",
                    "psnr": float(d["psnr"]),
                    "ssim": float(d["ssim"]),
                    "lpips": float(d["lpips"]),
                    "num_gs": int(d["num_gs"]),
                    "detail": f"time={d['time']}s/img",
                })
                continue

            m = COMPLETE_RE.match(line)
            if m:
                d = m.groupdict()
                events.append({
                    "timestamp": d["ts"],
                    "step": -1,
                    "event": "complete",
                    "psnr": "",
                    "ssim": "",
                    "lpips": "",
                    "num_gs": int(d["num_gs"]),
                    "detail": f"time={d['total_time']} res={d['res']}",
                })
                continue

    return metrics, events


def write_csv(rows, fieldnames, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = SCRIPT_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Extract GS-Scale training log to CSV")
    parser.add_argument("log", help="Path to .log file")
    parser.add_argument("--name", help="Scene name for output files (default: inferred from log path)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"File not found: {log_path}")

    name = args.name or log_path.stem

    metrics, events = parse_log(log_path)

    metrics_path = DATA_DIR / f"{name}_metrics.csv"
    events_path = DATA_DIR / f"{name}_events.csv"

    write_csv(metrics, METRIC_COLS, metrics_path)
    write_csv(events, EVENT_COLS, events_path)

    if metrics:
        first, last = metrics[0], metrics[-1]
        print(f"\nSummary: step {first['step']}-{last['step']}, "
              f"loss {first['loss']}->{last['loss']}, "
              f"pts {first['pts_M']}->{last['pts_M']}M, "
              f"alloc {first['alloc_gb']}->{last['alloc_gb']}G, "
              f"rsv {first['rsv_gb']}->{last['rsv_gb']}G, "
              f"elapsed {last['elapsed']:.1f}s")
    if events:
        for e in events:
            if e["psnr"]:
                print(f"  {e['event']} @ step {e['step']}: PSNR={e['psnr']} SSIM={e['ssim']} LPIPS={e['lpips']}")
            else:
                print(f"  {e['event']}: {e['detail']}")


if __name__ == "__main__":
    main()
