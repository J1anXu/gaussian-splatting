import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Academic style setup ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# --- Data ---
scenes = ["bicycle", "flowers", "room"]
methods = {
    "Ours": "output/ours",
    "3DGS-Scale": "output/gs-scale",
}

# Muted Elegance palette
palette = ["#264653", "#2A9D8F", "#E9C46A"]
scene_colors = {s: c for s, c in zip(scenes, palette)}
scene_labels = {"bicycle": "Bicycle", "flowers": "Flowers", "room": "Room"}

# --- Figure: single-column two-panel (IEEE/NeurIPS style ~3.3in per col) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax, (method, path) in zip([ax1, ax2], methods.items()):
    for scene in scenes:
        df = pd.read_csv(f"{path}/{scene}.csv")
        raw = df["peak_rsv_G"]
        w = 200
        mean = raw.rolling(window=w, min_periods=1).mean()
        std = raw.rolling(window=w, min_periods=1).std().fillna(0)
        # mean ± std shaded band
        ax.fill_between(df["step"], mean - std, mean + std,
                        color=scene_colors[scene], alpha=0.2, linewidth=0)
        # smoothed trend on top
        ax.plot(df["step"], mean,
                color=scene_colors[scene],
                label=scene_labels[scene])

    ax.set_title(method, fontweight="semibold")
    ax.set_xlabel("Iteration")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # thousands separator for x-axis
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
    # top & right spines off
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.set_ylabel("Peak Reserved GPU Memory (GB)")
ax2.set_ylabel("Peak Reserved GPU Memory (GB)")

# unify y-axis range across both panels
y_lo = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_hi = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_lo, y_hi)
ax2.set_ylim(y_lo, y_hi)

# shared legend at bottom
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(scenes),
           frameon=False, bbox_to_anchor=(0.5, -0.02),
           handlelength=3.0, handleheight=1.5)

# thicken legend lines only
for legline in fig.legends[0].get_lines():
    legline.set_linewidth(3.0)
name = "mem_peak_comparison"
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"output/{name}.png")
plt.savefig(f"output/{name}.pdf")
plt.show()
print(f"Saved to output/{name}.{{png,pdf}}")
