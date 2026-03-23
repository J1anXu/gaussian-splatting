#!/usr/bin/env python3
"""Extract training log steps into CSV. Auto-detects log format."""

import ast
import csv
import re
import sys
import os

# --- Format 1: GS-Scale (pipe-delimited) ---
# step=1/30000 | loss=0.5655 | pts=0.05M fru=0.01M(25.5%) | peak_alloc=0.21G peak_rsv=0.22G | sh=0 | elapsed=1.0s
GSSCALE_RE = re.compile(
    r"step=(\d+)/(\d+)\s*\|\s*loss=([\d.]+)\s*\|"
    r"\s*pts=([\d.]+)M\s+fru=([\d.]+)M\(([\d.]+)%\)\s*\|"
    r"\s*peak_alloc=([\d.]+)G\s+peak_rsv=([\d.]+)G\s*\|"
    r"\s*sh=(\d+)\s*\|"
    r"\s*elapsed=([\d.]+)s"
)
GSSCALE_HEADER = ["step", "max_steps", "loss", "pts_M", "fru_M", "fru_pct",
                  "peak_alloc_G", "peak_rsv_G", "sh", "elapsed_s"]

# --- Format 2: dict-style ---
# 0323,14:27 - {'iter': 10, 'L': 0.2568, 'vis': '0.02M', 'pts': '0.05M', 'vis%': 35.7, ...}
DICT_RE = re.compile(r"\{.*'iter'\s*:.*\}")
DICT_HEADER = ["step", "loss", "pts_M", "vis_M", "vis_pct", "blk",
               "alloc_G", "rsv_G", "peak_alloc_G", "peak_rsv_G",
               "it_s", "elapsed_s"]


def parse_dict_line(d):
    """Convert parsed dict to row."""
    def strip_unit(v):
        if isinstance(v, str):
            return v.rstrip("Ms")
        return v
    return [
        d["iter"], d["L"],
        strip_unit(d["pts"]), strip_unit(d["vis"]), d["vis%"], d["blk"],
        d["alloc"], d["rsv"], d["peak_alloc"], d["peak_rsv"],
        d.get("it/s", ""), strip_unit(d.get("elapsed", "")),
    ]


def detect_and_extract(log_path):
    """Read first data line to detect format, then extract all."""
    with open(log_path) as f:
        for line in f:
            if GSSCALE_RE.search(line):
                return "gsscale"
            if DICT_RE.search(line):
                return "dict"
    return None


def extract(log_path, out_path):
    fmt = detect_and_extract(log_path)
    if fmt is None:
        print("Error: unrecognized log format", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(log_path) as f:
        for line in f:
            if fmt == "gsscale":
                m = GSSCALE_RE.search(line)
                if m:
                    rows.append(list(m.groups()))
            else:
                m = DICT_RE.search(line)
                if m:
                    d = ast.literal_eval(m.group())
                    rows.append(parse_dict_line(d))

    header = GSSCALE_HEADER if fmt == "gsscale" else DICT_HEADER
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[{fmt}] Wrote {len(rows)} rows -> {out_path}")


log_path = sys.argv[1]
log_name = os.path.splitext(os.path.basename(log_path))[0]
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, log_name + ".csv")
extract(log_path, out_path)
