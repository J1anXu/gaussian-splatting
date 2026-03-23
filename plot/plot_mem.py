"""
曲线图: allocated 填充 + reserved 填充，展示 alloc 是 rsv 的一部分。

用法:
    python3 plot/plot_mem.py debug/capgs_bicycle_metrics.csv
    python3 plot/plot_mem.py debug/capgs_bicycle_metrics.csv debug/gsscale_bicycle_metrics.csv
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_csv(csv_path):
    steps, alloc, rsv = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            a = row.get("peak_alloc_gb", "")
            if not a:
                continue
            r = row.get("peak_rsv_gb", "")
            steps.append(int(row["step"]))
            alloc.append(float(a))
            rsv.append(float(r) if r else float(a))
    return np.array(steps), np.array(alloc), np.array(rsv)


COLORS = [
    ("#2196F3", "#F57C00"),  # blue alloc, orange frag
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", help="metrics CSV(s) from extract.py")
    parser.add_argument("-o", "--output", help="output image path")
    args = parser.parse_args()

    debug_dir = Path(__file__).resolve().parent.parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    n = len(args.csv)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 5), squeeze=False)

    for idx, csv_arg in enumerate(args.csv):
        csv_path = Path(csv_arg)
        if not csv_path.exists():
            csv_path = debug_dir / csv_arg
        name = csv_path.stem.replace("_metrics", "")
        c_alloc, c_frag = COLORS[idx % len(COLORS)]

        steps, alloc, rsv = load_csv(csv_path)

        ax = axes[0][idx]
        ax.fill_between(steps, alloc, rsv, alpha=0.25, color=c_frag, label="fragmented (peak_rsv − peak_alloc)")
        ax.fill_between(steps, 0, alloc, alpha=0.5, color=c_alloc, label="peak_alloc")
        ax.plot(steps, rsv, color=c_frag, alpha=0.5, linewidth=0.5)
        ax.plot(steps, alloc, color=c_alloc, alpha=0.8, linewidth=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("GPU Memory (GB)")
        ax.set_title(name)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("GPU Peak Memory", fontsize=14, y=1.02)
    fig.tight_layout()

    if args.output:
        out = Path(args.output)
    else:
        names = [Path(c).stem.replace("_metrics", "") for c in args.csv]
        out = debug_dir / f"{'_vs_'.join(names)}_mem.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
