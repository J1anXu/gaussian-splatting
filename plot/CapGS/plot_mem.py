"""
绘制 rsv_gb-iter 和 alloc_gb-iter 曲线。

用法:
    python3 plot/CapGS/plot_mem.py data/bicycle_metrics.csv
    python3 plot/CapGS/plot_mem.py data/bicycle_metrics.csv --name bicycle
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = SCRIPT_DIR / "data"
    PIC_DIR = SCRIPT_DIR / "picture"
    PIC_DIR.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="metrics CSV from extract_log.py (relative to data/ if no path sep)")
    parser.add_argument("--name", help="Scene name for output file (default: inferred from csv)")
    parser.add_argument("-o", "--output", help="output image path (default: picture/<name>_mem.png)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = DATA_DIR / args.csv
    name = args.name or csv_path.stem.replace("_metrics", "")

    iters, alloc, rsv = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            alloc.append(float(row["alloc_gb"]))
            rsv.append(float(row["rsv_gb"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, rsv, label="reserved (rsv_gb)", alpha=0.8)
    ax.plot(iters, alloc, label="allocated (alloc_gb)", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("GPU Memory (GB)")
    ax.set_title(f"{name} — GPU Memory (CapGS)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(args.output) if args.output else PIC_DIR / f"{name}_mem.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
