#!/usr/bin/env python3
"""Plot peak_alloc_G and peak_rsv_G vs step from extracted CSV."""

import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["step"], df["peak_alloc_G"], label="peak_alloc_G", linewidth=0.5)
ax.plot(df["step"], df["peak_rsv_G"], label="peak_rsv_G", linewidth=0.5)
ax.set_xlabel("step")
ax.set_ylabel("GPU Memory (GB)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = csv_path.rsplit(".", 1)[0] + "_mem.png"
fig.savefig(out, dpi=150)
print(f"Saved -> {out}")
