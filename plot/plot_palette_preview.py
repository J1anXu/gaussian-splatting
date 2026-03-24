"""Generate a preview of 3 palette options on the actual data."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

scenes = ["bicycle", "flowers", "treehill"]
scene_labels = {"bicycle": "Bicycle", "flowers": "Flowers", "treehill": "Treehill"}
path = "output/gs-scale"  # use the more interesting one for preview

palettes = {
    "A: Tol High-Contrast": ["#004488", "#DDAA33", "#BB5566"],
    "B: Tol Vibrant":       ["#0077BB", "#EE7733", "#009988"],
    "C: Muted Elegance":    ["#264653", "#2A9D8F", "#E9C46A"],
}

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)

for ax, (pname, colors) in zip(axes, palettes.items()):
    scene_colors = {s: c for s, c in zip(scenes, colors)}
    for scene in scenes:
        df = pd.read_csv(f"{path}/{scene}.csv")
        raw = df["peak_rsv_G"]
        mean = raw.rolling(window=200, min_periods=1).mean()
        std = raw.rolling(window=200, min_periods=1).std().fillna(0)
        ax.fill_between(df["step"], mean - std, mean + std,
                        color=scene_colors[scene], alpha=0.2, linewidth=0)
        ax.plot(df["step"], mean, color=scene_colors[scene],
                label=scene_labels[scene])
    ax.set_title(pname, fontweight="semibold")
    ax.set_xlabel("Iteration")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

axes[0].set_ylabel("Peak Reserved GPU Memory (GB)")
plt.tight_layout()
plt.savefig("output/palette_preview.png")
plt.show()
print("Saved to output/palette_preview.png")
