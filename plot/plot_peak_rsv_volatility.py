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

scenes = ["bicycle", "flowers", "treehill"]
methods = {"Ours": "output/ours", "3DGS-Scale": "output/gs-scale"}
palette = ["#264653", "#2A9D8F", "#E9C46A"]
scene_colors = {s: c for s, c in zip(scenes, palette)}
scene_labels = {"bicycle": "Bicycle", "flowers": "Flowers", "treehill": "Treehill"}

# 2x2: top=trend, bottom=volatility; left=Ours, right=3DGS-Scale
fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.5),
                         gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15})

w = 200
for col, (method, path) in enumerate(methods.items()):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    for scene in scenes:
        df = pd.read_csv(f"{path}/{scene}.csv")
        raw = df["peak_rsv_G"]
        mean = raw.rolling(window=w, min_periods=1).mean()
        std = raw.rolling(window=w, min_periods=1).std().fillna(0)

        # top: trend with ±std band
        ax_top.fill_between(df["step"], mean - std, mean + std,
                            color=scene_colors[scene], alpha=0.2, linewidth=0)
        ax_top.plot(df["step"], mean, color=scene_colors[scene],
                    label=scene_labels[scene])

        # bottom: volatility (rolling std)
        ax_bot.fill_between(df["step"], 0, std,
                            color=scene_colors[scene], alpha=0.3, linewidth=0)
        ax_bot.plot(df["step"], std, color=scene_colors[scene], linewidth=0.8)

    ax_top.set_title(method, fontweight="semibold")
    ax_top.set_xlim(left=0)
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(labelbottom=False)  # hide x labels on top row
    ax_bot.set_xlim(left=0)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlabel("")

    for ax in [ax_top, ax_bot]:
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

axes[0, 0].set_ylabel("Peak Reserved\nGPU Memory (GB)")
axes[1, 0].set_ylabel("Volatility (GB)")

# unify y-axis for top row
y_hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
axes[0, 0].set_ylim(0, y_hi)
axes[0, 1].set_ylim(0, y_hi)
# unify y-axis for bottom row
v_hi = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
axes[1, 0].set_ylim(0, v_hi)
axes[1, 1].set_ylim(0, v_hi)

# shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(scenes),
           frameon=False, bbox_to_anchor=(0.5, -0.01),
           handlelength=3.0, handleheight=1.5)
for legline in fig.legends[0].get_lines():
    legline.set_linewidth(3.0)

name = "mem_peak_volatility"
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(f"output/{name}.png")
plt.savefig(f"output/{name}.pdf")
plt.show()
print(f"Saved to output/{name}.{{png,pdf}}")
