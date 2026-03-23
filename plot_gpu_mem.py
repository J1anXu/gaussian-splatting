#!/usr/bin/env python3
"""Plot GPU memory usage (alloc / rsv) vs iteration from training log."""

import sys
import re
import matplotlib.pyplot as plt

def parse_log(log_path):
    iters, allocs, rsvs = [], [], []

    with open(log_path) as f:
        for line in f:
            # Skip lines without iter (e.g. split info lines)
            if "'iter'" not in line:
                continue

            # Extract the dict-like part after " - "
            match = re.search(r" - (\{.+\})", line)
            if not match:
                continue

            raw = match.group(1)

            # Extract iter
            iter_m = re.search(r"'iter':\s*(\d+)", raw)
            if not iter_m:
                continue
            it = int(iter_m.group(1))

            # Phase 1: gpu_mem_gb field
            mem_m = re.search(r"'gpu_mem_gb':\s*([\d.]+)", raw)
            if mem_m:
                val = float(mem_m.group(1))
                iters.append(it)
                allocs.append(val)
                rsvs.append(val)  # same value for phase 1
                continue

            # Phase 2: alloc + rsv fields
            alloc_m = re.search(r"'alloc':\s*([\d.]+)", raw)
            rsv_m = re.search(r"'rsv':\s*([\d.]+)", raw)
            if alloc_m and rsv_m:
                iters.append(it)
                allocs.append(float(alloc_m.group(1)))
                rsvs.append(float(rsv_m.group(1)))

    return iters, allocs, rsvs

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file> [output.png]")
        sys.exit(1)

    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else log_path.rsplit('.', 1)[0] + '_gpu_mem.png'

    iters, allocs, rsvs = parse_log(log_path)
    print(f"Parsed {len(iters)} data points, iter range: {iters[0]}~{iters[-1]}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(iters, allocs, label='alloc (GB)', linewidth=0.8, alpha=0.9)
    ax.plot(iters, rsvs, label='rsv (GB)', linewidth=0.8, alpha=0.9)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('GPU Memory (GB)')
    ax.set_title(f'GPU Memory Usage — {log_path.split("/")[-1]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
