#!/usr/bin/env python3
"""
Plot vis% (visible-points / total-points) curves per scene from structured
training logs at logs/train/<branch>/<scene>/*.log.

Usage:
    python analysis/plot_vis.py                         # all scenes, current branch
    python analysis/plot_vis.py bicycle counter         # specific scenes
    python analysis/plot_vis.py --branch xxx
    python analysis/plot_vis.py --grid                  # one combined figure, subplot per scene
    python analysis/plot_vis.py --overlay               # all scenes in one plot
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(REPO, "logs", "train")
OUT_DIR = os.path.join(REPO, "analysis", "out")
DICT_RE = re.compile(r"\{.*\}")

INDOOR = {"room", "counter", "kitchen", "bonsai", "drjohnson", "playroom"}


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return ""


def discover_scenes(branch: str) -> list:
    d = os.path.join(LOG_ROOT, branch)
    if not os.path.isdir(d):
        sys.exit(f"no logs dir: {d}")
    return sorted(
        name for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name))
        and glob.glob(os.path.join(d, name, "*.log"))
    )


def latest_log(scene: str, branch: str) -> str | None:
    files = sorted(glob.glob(os.path.join(LOG_ROOT, branch, scene, "*.log")))
    return files[-1] if files else None


def parse_log(path: str) -> tuple[list, list, list]:
    iters, vis_pct, rsv = [], [], []
    with open(path) as f:
        for line in f:
            m = DICT_RE.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group(0))
            except Exception:
                continue
            if "iter" not in d or "vis%" not in d:
                continue
            iters.append(d["iter"])
            vis_pct.append(float(d["vis%"]))
            rsv.append(float(d.get("rsv", 0.0)))
    return iters, vis_pct, rsv


def kind_color(scene: str) -> str:
    return "tab:red" if scene in INDOOR else "tab:blue"


def plot_per_scene(scene: str, branch: str, it, vis, rsv, out: str):
    fig, ax1 = plt.subplots(figsize=(11, 4.2))
    color_vis = kind_color(scene)
    ax1.plot(it, vis, color=color_vis, linewidth=1.0, label="vis%")
    ax1.set_ylabel("vis% (visible / total points)", color=color_vis)
    ax1.set_xlabel("iteration")
    ax1.tick_params(axis="y", labelcolor=color_vis)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(it, rsv, color="tab:gray", linewidth=0.8, alpha=0.7, label="rsv (GB)")
    ax2.set_ylabel("rsv (GB, per-iter peak)", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    kind = "indoor" if scene in INDOOR else "outdoor"
    mean_vis = sum(vis) / len(vis) if vis else 0
    max_rsv = max(rsv) if rsv else 0
    ax1.set_title(f"[{scene}] {kind} — mean vis%={mean_vis:.0f}, peak rsv={max_rsv:.2f} GB")

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_grid(runs, branch: str, out: str):
    n = len(runs)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.0), sharex=True)
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (scene, it, vis, rsv) in zip(axes, runs):
        color = kind_color(scene)
        ax.plot(it, vis, color=color, linewidth=1.0)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        kind = "indoor" if scene in INDOOR else "outdoor"
        mean_vis = sum(vis) / len(vis) if vis else 0
        peak = max(rsv) if rsv else 0
        ax.set_title(f"{scene} ({kind})\nmean vis%={mean_vis:.0f}, peak={peak:.2f}GB", fontsize=9)

    for ax in axes[len(runs):]:
        ax.axis("off")

    fig.suptitle(f"vis% per scene — branch {branch}", fontsize=12)
    fig.supxlabel("iteration")
    fig.supylabel("vis%")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_overlay(runs, branch: str, out: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for scene, it, vis, rsv in runs:
        color = kind_color(scene)
        mean_vis = sum(vis) / len(vis) if vis else 0
        peak = max(rsv) if rsv else 0
        label = f"{scene} (μ={mean_vis:.0f}, peak={peak:.1f}GB)"
        ax.plot(it, vis, color=color, linewidth=0.9, alpha=0.7, label=label)

    ax.set_xlabel("iteration")
    ax.set_ylabel("vis%")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"vis% overlay (red=indoor, blue=outdoor) — {branch}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved: {out}")


def print_summary(runs, branch: str):
    print(f"\n=== vis% vs peak rsv summary (branch {branch}) ===")
    print(f"{'scene':<12} {'type':<8} {'mean vis%':>10} {'max vis%':>9} {'peak rsv GB':>12}")
    print("-" * 55)
    rows = []
    for scene, it, vis, rsv in runs:
        kind = "indoor" if scene in INDOOR else "outdoor"
        mean_vis = sum(vis) / len(vis) if vis else 0
        max_vis = max(vis) if vis else 0
        peak = max(rsv) if rsv else 0
        rows.append((scene, kind, mean_vis, max_vis, peak))
    rows.sort(key=lambda r: -r[4])
    for scene, kind, mv, mx, pk in rows:
        print(f"{scene:<12} {kind:<8} {mv:>10.1f} {mx:>9.1f} {pk:>12.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", help="scene names (empty = all under branch)")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--grid", action="store_true", help="one combined figure (subplot per scene)")
    ap.add_argument("--overlay", action="store_true", help="all scenes in one plot")
    args = ap.parse_args()

    branch = args.branch or current_branch()
    if not branch:
        sys.exit("cannot resolve git branch; pass --branch")

    scenes = args.scenes if args.scenes else discover_scenes(branch)
    if not scenes:
        sys.exit(f"no scenes found under {branch}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"branch: {branch}")
    print(f"scenes: {', '.join(scenes)}\n")

    runs = []
    for sc in scenes:
        fp = latest_log(sc, branch)
        if not fp:
            print(f"  [{sc}] no log")
            continue
        it, vis, rsv = parse_log(fp)
        if not it:
            print(f"  [{sc}] no parseable data")
            continue
        runs.append((sc, it, vis, rsv))

    if not runs:
        sys.exit("no data")

    if args.grid:
        plot_grid(runs, branch, os.path.join(OUT_DIR, f"vis_grid_{branch}.png"))
    elif args.overlay:
        plot_overlay(runs, branch, os.path.join(OUT_DIR, f"vis_overlay_{branch}.png"))
    else:
        # default: per-scene file + grid + overlay (all three)
        for sc, it, vis, rsv in runs:
            plot_per_scene(sc, branch, it, vis, rsv,
                           os.path.join(OUT_DIR, f"vis_{branch}_{sc}.png"))
        plot_grid(runs, branch, os.path.join(OUT_DIR, f"vis_grid_{branch}.png"))
        plot_overlay(runs, branch, os.path.join(OUT_DIR, f"vis_overlay_{branch}.png"))

    print_summary(runs, branch)


if __name__ == "__main__":
    main()
