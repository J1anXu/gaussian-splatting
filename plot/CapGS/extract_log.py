"""
从训练日志中提取关键信息，输出标准化 CSV。

用法:
    python3 plot/CapGS/extract_log.py <log_path> [--name bicycle]

示例:
    python3 plot/CapGS/extract_log.py logs/train/nurips26/bicycle/0323_1318.log
    python3 plot/CapGS/extract_log.py logs/train/nurips26/bicycle/0323_1318.log --name bicycle
"""
import argparse
import ast
import csv
import re
import sys
from pathlib import Path

# 匹配 metric 行: timestamp - {dict}
METRIC_RE = re.compile(r"^(\d{4},\d{2}:\d{2}) - (\{.+\})$")
# 匹配 split 事件行
SPLIT_RE = re.compile(r"^(\d{4},\d{2}:\d{2}) - \[iter (\d+)\] Split block\(s\) \[(.+?)\], now (\d+) blocks, sizes: \[(.+?)\]$")
# 匹配 partition 行
PARTITION_RE = re.compile(r"^(\d{4},\d{2}:\d{2}) - Partitioned into (\d+) blocks?, sizes: \[(.+?)\]$")

# metric CSV 列名 & 对应 dict key
METRIC_COLS = [
    ("iter",    "iter",    int),
    ("loss",    "L",       float),
    ("vis_M",   "vis",     lambda v: float(v.rstrip("M"))),
    ("pts_M",   "pts",     lambda v: float(v.rstrip("M"))),
    ("vis_pct", "vis%",    float),
    ("blk",     "blk",     int),
    ("alloc_gb","alloc",   float),
    ("rsv_gb",  "rsv",     float),
    ("it_s",    "it/s",    float),
    ("elapsed", "elapsed", lambda v: float(v.rstrip("s")) if v else ""),
]

# event CSV 列名
EVENT_COLS = ["iter", "event", "num_blocks", "block_sizes"]


def parse_log(log_path):
    metrics = []
    events = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 1) metric 行
            m = METRIC_RE.match(line)
            if m:
                ts, dict_str = m.group(1), m.group(2)
                d = ast.literal_eval(dict_str)
                row = {"timestamp": ts}
                for col_name, key, conv in METRIC_COLS:
                    val = d.get(key, "")
                    try:
                        row[col_name] = conv(val) if val != "" else ""
                    except (ValueError, TypeError):
                        row[col_name] = val
                metrics.append(row)
                continue

            # 2) split 事件
            m = SPLIT_RE.match(line)
            if m:
                ts = m.group(1)
                it = int(m.group(2))
                split_ids = m.group(3)
                num_blk = int(m.group(4))
                sizes = m.group(5)
                events.append({
                    "timestamp": ts,
                    "iter": it,
                    "event": f"split[{split_ids}]",
                    "num_blocks": num_blk,
                    "block_sizes": sizes,
                })
                continue

            # 3) partition 事件
            m = PARTITION_RE.match(line)
            if m:
                ts = m.group(1)
                num_blk = int(m.group(2))
                sizes = m.group(3)
                events.append({
                    "timestamp": ts,
                    "iter": 0,
                    "event": "partition",
                    "num_blocks": num_blk,
                    "block_sizes": sizes,
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

    parser = argparse.ArgumentParser(description="Extract training log to CSV")
    parser.add_argument("log", help="Path to .log file")
    parser.add_argument("--name", help="Scene name for output files (default: inferred from log path)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        sys.exit(f"File not found: {log_path}")

    # infer scene name: parent dir name or log stem
    name = args.name or log_path.stem

    metrics, events = parse_log(log_path)

    metrics_path = DATA_DIR / f"{name}_metrics.csv"
    events_path = DATA_DIR / f"{name}_events.csv"

    # write
    metric_fields = ["timestamp"] + [c[0] for c in METRIC_COLS]
    write_csv(metrics, metric_fields, metrics_path)

    event_fields = ["timestamp"] + EVENT_COLS
    write_csv(events, event_fields, events_path)

    # summary
    if metrics:
        first, last = metrics[0], metrics[-1]
        print(f"\nSummary: iter {first['iter']}-{last['iter']}, "
              f"loss {first['loss']}->{last['loss']}, "
              f"pts {first['pts_M']}->{last['pts_M']}M, "
              f"blk {first['blk']}->{last['blk']}, "
              f"it/s {first['it_s']}->{last['it_s']}")
    if events:
        print(f"Events: {len(events)} (splits: {sum(1 for e in events if 'split' in e['event'])})")


if __name__ == "__main__":
    main()
