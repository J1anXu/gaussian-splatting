#!/usr/bin/env python3
"""
Usage:
    python plot_mem.py                  # 一键画当前 branch 下所有 scene（每个 scene 一张图）
    python plot_mem.py bicycle          # 只画 bicycle
    python plot_mem.py bicycle garden   # 指定几个 scene
    python plot_mem.py --branch xxx     # 指定 branch
    python plot_mem.py bicycle --all    # 叠画该 scene 的所有历史 run
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

REPO = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(REPO, "logs", "train")
DICT_RE = re.compile(r"\{.*\}")


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return ""


def find_log_files(scene: str, branch: str) -> list[str]:
    pat = os.path.join(LOG_ROOT, branch, scene, "*.log")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no log found: {pat}")
    return files


def parse_log(path: str):
    iters, allocs, rsvs, pts_list = [], [], [], []
    with open(path) as f:
        for line in f:
            m = DICT_RE.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group(0))
            except Exception:
                continue
            if "iter" not in d or "rsv" not in d:
                continue
            iters.append(d["iter"])
            allocs.append(d.get("alloc", 0.0))
            rsvs.append(d["rsv"])
            pts = d.get("pts", "0M")
            if isinstance(pts, str) and pts.endswith("M"):
                pts_list.append(float(pts[:-1]))
            else:
                pts_list.append(0.0)
    return iters, allocs, rsvs, pts_list


def plot(runs: list[tuple[str, list, list, list, list]], scene: str, out_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for label, it, alloc, rsv, pts in runs:
        ax1.plot(it, rsv, label=f"{label} rsv", linewidth=1.0)
        ax1.plot(it, alloc, label=f"{label} alloc", linewidth=0.7, alpha=0.6, linestyle="--")
        ax2.plot(it, pts, label=label, linewidth=1.0)

    ax1.set_ylabel("GPU mem (GB)  — per-iter peak")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"[{scene}] memory over training")

    ax2.set_ylabel("pts (M)")
    ax2.set_xlabel("iteration")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved: {out_path}")


def discover_scenes(branch: str) -> list:
    d = os.path.join(LOG_ROOT, branch)
    if not os.path.isdir(d):
        sys.exit(f"no logs dir: {d}")
    return sorted(
        name for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name))
        and glob.glob(os.path.join(d, name, "*.log"))
    )


def plot_one_scene(scene: str, branch: str, show_all_runs: bool, out: str | None):
    files = find_log_files(scene, branch)
    if not show_all_runs:
        files = files[-1:]

    runs = []
    for fp in files:
        label = os.path.splitext(os.path.basename(fp))[0]
        it, alloc, rsv, pts = parse_log(fp)
        if not it:
            print(f"  skip empty: {fp}")
            continue
        print(f"  [{scene}] {label}: {len(it)} samples, "
              f"rsv {min(rsv):.2f}-{max(rsv):.2f} GB, "
              f"alloc {min(alloc):.2f}-{max(alloc):.2f} GB, pts {max(pts):.2f}M")
        runs.append((label, it, alloc, rsv, pts))

    if not runs:
        print(f"  [{scene}] no parseable data, skipped")
        return

    if out is None:
        out = os.path.join(REPO, "debug", f"mem_{branch}_{scene}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot(runs, scene, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", help="scene names (empty = all under branch)")
    ap.add_argument("--branch", default=None, help="git branch name (default: current)")
    ap.add_argument("--all", action="store_true", help="叠画该 scene 的所有历史 run")
    ap.add_argument("--out", default=None, help="输出 png 路径 (仅单 scene 生效)")
    args = ap.parse_args()

    branch = args.branch or current_branch()
    if not branch:
        sys.exit("cannot resolve git branch; pass --branch")

    scenes = args.scenes if args.scenes else discover_scenes(branch)
    if not scenes:
        sys.exit(f"no scenes found under {branch}")

    print(f"branch: {branch}")
    print(f"scenes: {', '.join(scenes)}")

    for sc in scenes:
        out = args.out if (args.out and len(scenes) == 1) else None
        plot_one_scene(sc, branch, args.all, out)


if __name__ == "__main__":
    main()
