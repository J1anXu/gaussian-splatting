"""
堆叠柱状图: allocated (实际使用) + fragmented (reserved - allocated, 碎片/缓存)。

用法:
    python3 plot/gs-scale/plot_mem_bar.py data/bicycle_metrics.csv
    python3 plot/gs-scale/plot_mem_bar.py data/bicycle_metrics.csv --name bicycle --bins 30
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = SCRIPT_DIR / "data"
    PIC_DIR = SCRIPT_DIR / "picture"
    PIC_DIR.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="metrics CSV")
    parser.add_argument("--name", help="Scene name (default: from csv)")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins (default: 30)")
    parser.add_argument("-o", "--output", help="output image path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = DATA_DIR / args.csv
    name = args.name or csv_path.stem.replace("_metrics", "")

    steps, alloc, rsv = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            alloc.append(float(row["alloc_gb"]))
            rsv.append(float(row["rsv_gb"]))

    steps = np.array(steps)
    alloc = np.array(alloc)
    rsv = np.array(rsv)

    # bin
    bin_edges = np.linspace(steps[0], steps[-1], args.bins + 1)
    bin_centers = []
    bin_alloc = []
    bin_frag = []
    for i in range(args.bins):
        mask = (steps >= bin_edges[i]) & (steps < bin_edges[i + 1])
        if i == args.bins - 1:  # last bin includes right edge
            mask |= (steps == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        a = alloc[mask].mean()
        r = rsv[mask].mean()
        bin_alloc.append(a)
        bin_frag.append(r - a)

    bin_centers = np.array(bin_centers)
    bin_alloc = np.array(bin_alloc)
    bin_frag = np.array(bin_frag)
    width = (bin_edges[1] - bin_edges[0]) * 0.85

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(bin_centers, bin_alloc, width=width, label="allocated", color="#2196F3")
    ax.bar(bin_centers, bin_frag, width=width, bottom=bin_alloc, label="fragmented (rsv − alloc)", color="#2196F3", alpha=0.3)

    ax.set_xlabel("Step")
    ax.set_ylabel("GPU Memory (GB)")
    ax.set_title(f"{name} — GPU Memory (GS-Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    out = Path(args.output) if args.output else PIC_DIR / f"{name}_mem_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
