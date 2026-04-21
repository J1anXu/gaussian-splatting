#!/usr/bin/env python3
"""
Extract per-scene summary from logs/train/<branch>/<scene>/*.log and
logs/metrics/<branch>/<scene>/*.log into two CSV files:
  - train_<branch>.csv   : per-scene training stats
  - metrics_<branch>.csv : per-scene PSNR/SSIM/LPIPS

Usage:
    python analysis/extract_csv.py                 # current branch
    python analysis/extract_csv.py --branch xxx
    python analysis/extract_csv.py --out-dir path  # default: analysis/out/
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_TRAIN = os.path.join(REPO, "logs", "train")
LOG_METRICS = os.path.join(REPO, "logs", "metrics")
DEBUG_ROOT = os.path.join(REPO, "debug")
OUT_DIR_DEFAULT = os.path.join(REPO, "analysis", "out")

INDOOR = {"room", "counter", "kitchen", "bonsai", "drjohnson", "playroom"}
DICT_RE = re.compile(r"\{.*\}")
METRIC_RE = re.compile(
    r"(ours_\d+)\s*-\s*SSIM:\s*([0-9.]+),\s*PSNR:\s*([0-9.]+),\s*LPIPS:\s*([0-9.]+)"
)
TIME_RE = re.compile(r"training time cost:\s*\[([0-9.]+)\]\s*seconds", re.IGNORECASE)
BLK_RE = re.compile(r"\bblk=(\d+)")


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def kind(scene: str) -> str:
    return "indoor" if scene in INDOOR else "outdoor"


def latest_log(scene_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(scene_dir, "*.log")))
    return files[-1] if files else None


def parse_dict(s: str) -> dict | None:
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def pts_to_m(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.rstrip("M")
        try:
            return float(v)
        except ValueError:
            return None
    if isinstance(v, (int, float)):
        return float(v) / 1_000_000 if v > 1e4 else float(v)
    return None


def parse_train_log(path: str, extra_paths: list[str] | None = None) -> dict:
    """Collect per-iter dicts from structured log; pull duration / blk from
    extra stdout-redirect logs when present (they live under debug/)."""
    iters, L, pts, vis_pct, alloc, rsv, its_, blk = [], [], [], [], [], [], [], []
    with open(path) as f:
        struct_text = f.read()

    for line in struct_text.splitlines():
        m = DICT_RE.search(line)
        if not m:
            continue
        d = parse_dict(m.group(0))
        if not d or "iter" not in d:
            continue
        iters.append(int(d["iter"]))
        if "L" in d:
            L.append(float(d["L"]))
        if "pts" in d:
            p = pts_to_m(d["pts"])
            if p is not None:
                pts.append(p)
        if "vis%" in d:
            try:
                vis_pct.append(float(d["vis%"]))
            except (TypeError, ValueError):
                pass
        if "alloc" in d:
            alloc.append(float(d["alloc"]))
        if "rsv" in d:
            rsv.append(float(d["rsv"]))
        if "it/s" in d:
            its_.append(float(d["it/s"]))
        if "blk" in d:
            blk.append(int(d["blk"]))

    # Fall back to the stdout-redirected training log(s) for duration / blk
    # (these live in debug/<branch>/<scene>/train.log). They contain the
    # "Phase 2 training time cost: [X] seconds" line and tqdm postfixes with
    # blk=N that the structured LOGGER does not emit.
    extra_text = ""
    for p in extra_paths or []:
        try:
            with open(p, errors="ignore") as f:
                extra_text += "\n" + f.read()
        except OSError:
            pass

    combined = struct_text + extra_text
    duration = None
    tm = TIME_RE.search(combined)
    if tm:
        duration = float(tm.group(1))

    if not blk:
        bms = BLK_RE.findall(extra_text)
        if bms:
            blk = [int(b) for b in bms]

    return {
        "iters": iters, "L": L, "pts": pts, "vis%": vis_pct,
        "alloc": alloc, "rsv": rsv, "it/s": its_, "blk": blk,
        "duration_s": duration,
    }


def parse_metrics_log(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                rows.append({
                    "key": m.group(1),
                    "SSIM": float(m.group(2)),
                    "PSNR": float(m.group(3)),
                    "LPIPS": float(m.group(4)),
                })
    return rows


def discover_scenes(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and glob.glob(os.path.join(root, name, "*.log"))
    )


def safe(lst, fn):
    return fn(lst) if lst else None


def summarize_train(scene: str, log_path: str, branch: str) -> dict:
    extras = sorted(glob.glob(os.path.join(DEBUG_ROOT, branch, scene, "train*.log")))
    d = parse_train_log(log_path, extras)
    return {
        "scene": scene,
        "kind": kind(scene),
        "log_file": os.path.basename(log_path),
        "n_samples": len(d["iters"]),
        "max_iter": safe(d["iters"], max),
        "final_L": d["L"][-1] if d["L"] else None,
        "final_pts_M": d["pts"][-1] if d["pts"] else None,
        "final_blk": d["blk"][-1] if d["blk"] else None,
        "mean_vis%": safe(d["vis%"], lambda x: sum(x) / len(x)),
        "max_vis%": safe(d["vis%"], max),
        "min_vis%": safe(d["vis%"], min),
        "peak_alloc_gb": safe(d["alloc"], max),
        "peak_rsv_gb": safe(d["rsv"], max),
        "mean_alloc_gb": safe(d["alloc"], lambda x: sum(x) / len(x)),
        "mean_rsv_gb": safe(d["rsv"], lambda x: sum(x) / len(x)),
        "final_it_per_s": d["it/s"][-1] if d["it/s"] else None,
        "mean_it_per_s": safe(d["it/s"], lambda x: sum(x) / len(x)),
        "duration_s": d["duration_s"],
    }


def summarize_metrics(scene: str, log_path: str) -> list[dict]:
    rows = []
    for r in parse_metrics_log(log_path):
        rows.append({
            "scene": scene,
            "kind": kind(scene),
            "log_file": os.path.basename(log_path),
            "metric_key": r["key"],
            "PSNR": r["PSNR"],
            "SSIM": r["SSIM"],
            "LPIPS": r["LPIPS"],
        })
    return rows


def round_floats(d: dict, digits: int = 4) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, digits)
        else:
            out[k] = v
    return out


def write_csv(path: str, rows: list[dict], cols: list[str]):
    if not rows:
        print(f"  (no rows) {path}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"  wrote: {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    branch = args.branch or current_branch()
    if not branch:
        sys.exit("cannot resolve git branch; pass --branch")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"branch: {branch}")

    # --- training ---
    train_root = os.path.join(LOG_TRAIN, branch)
    train_scenes = discover_scenes(train_root)
    print(f"train scenes: {len(train_scenes)} -> {', '.join(train_scenes) or '(none)'}")

    train_rows = []
    for sc in train_scenes:
        lp = latest_log(os.path.join(train_root, sc))
        if not lp:
            continue
        row = summarize_train(sc, lp, branch)
        train_rows.append(round_floats(row))

    train_cols = [
        "scene", "kind", "log_file",
        "n_samples", "max_iter",
        "final_L", "final_pts_M", "final_blk",
        "mean_vis%", "max_vis%", "min_vis%",
        "peak_alloc_gb", "peak_rsv_gb",
        "mean_alloc_gb", "mean_rsv_gb",
        "final_it_per_s", "mean_it_per_s",
        "duration_s",
    ]
    write_csv(os.path.join(args.out_dir, f"train_{branch}.csv"), train_rows, train_cols)

    # --- metrics ---
    metrics_root = os.path.join(LOG_METRICS, branch)
    metrics_scenes = discover_scenes(metrics_root)
    print(f"metrics scenes: {len(metrics_scenes)} -> {', '.join(metrics_scenes) or '(none)'}")

    metrics_rows = []
    for sc in metrics_scenes:
        lp = latest_log(os.path.join(metrics_root, sc))
        if not lp:
            continue
        for r in summarize_metrics(sc, lp):
            metrics_rows.append(round_floats(r))

    metrics_cols = ["scene", "kind", "log_file", "metric_key", "PSNR", "SSIM", "LPIPS"]
    write_csv(os.path.join(args.out_dir, f"metrics_{branch}.csv"), metrics_rows, metrics_cols)


if __name__ == "__main__":
    main()
