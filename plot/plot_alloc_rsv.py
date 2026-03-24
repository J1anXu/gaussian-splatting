import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.pad": 3, "ytick.major.pad": 3,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

methods = {"Ours": "output/ours", "3DGS-Scale": "output/gs-scale"}
metrics = {"peak_alloc_G": "Allocated", "peak_rsv_G": "Reserved"}

# Muted Elegance palette
color_alloc = "#2A9D8F"
color_rsv = "#264653"

w = 200
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax, (method, path) in zip([ax1, ax2], methods.items()):
    df = pd.read_csv(f"{path}/bicycle.csv")
    for col, (metric, label) in enumerate(metrics.items()):
        color = color_rsv if "rsv" in metric else color_alloc
        ax.plot(df["step"], df[metric], color=color, label=label, linewidth=0.6)

    ax.set_title(method, fontweight="semibold")
    ax.set_xlabel("Iteration")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.set_ylabel("Peak GPU Memory (GB)")
ax2.set_ylabel("Peak GPU Memory (GB)")

# unify y-axis
y_hi = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(0, y_hi)
ax2.set_ylim(0, y_hi)

# shared legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, -0.02),
           handlelength=3.0, handleheight=1.5)
for legline in fig.legends[0].get_lines():
    legline.set_linewidth(3.0)

name = "mem_alloc_vs_rsv_bicycle"
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"output/{name}.png")
plt.savefig(f"output/{name}.pdf")
plt.show()
print(f"Saved to output/{name}.{{png,pdf}}")
