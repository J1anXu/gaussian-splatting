#!/usr/bin/env bash
# Summarize vanilla Gaussian Splatting results.
# Usage: bash summarize_gs.sh [branch1 branch2 ...]
#   No args = summarize every branch found under /data/jian/output/gaussian-splatting
#   GS_OUTPUT_ROOT can override /data/jian/output/gaussian-splatting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 - "$SCRIPT_DIR" "$@" <<'PY'
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(sys.argv[1]).resolve()
OUTPUT_ROOT = Path(os.environ.get("GS_OUTPUT_ROOT") or "/data/jian/output/gaussian-splatting").expanduser()
BRANCH_ARGS = sys.argv[2:]

DATASETS = ("mip360", "deepblending", "tandt")
DATASET_ORDER = {name: i for i, name in enumerate(DATASETS)}

SCENE_ORDER = {
    "bicycle": 0,
    "flowers": 1,
    "garden": 2,
    "stump": 3,
    "treehill": 4,
    "room": 5,
    "counter": 6,
    "kitchen": 7,
    "bonsai": 8,
    "drjohnson": 9,
    "playroom": 10,
    "train": 11,
    "truck": 12,
}

INDOOR = {"room", "counter", "kitchen", "bonsai", "drjohnson", "playroom"}
OUTDOOR = {"bicycle", "flowers", "garden", "stump", "treehill", "train", "truck"}


@dataclass
class Summary:
    dataset: str
    kind: str
    scene: str
    status: str
    metric_iter: str
    psnr: float | None
    ssim: float | None
    lpips: float | None
    points: str
    blocks: str
    peak_mem: str
    duration_s: float | None
    source_dir: Path
    train_points_raw: int | None
    ply_points_raw: int | None


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(SCRIPT_DIR), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(errors="ignore")


def last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = next((x for x in value if x), "")
    try:
        return float(value)
    except ValueError:
        return None


def max_float(pattern: str, text: str) -> float | None:
    values: list[float] = []
    for value in re.findall(pattern, text):
        if isinstance(value, tuple):
            value = next((x for x in value if x), "")
        try:
            values.append(float(value))
        except ValueError:
            pass
    return max(values) if values else None


def last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = next((x for x in value if x), "")
    try:
        return int(value)
    except ValueError:
        return None


def format_points(raw: int | float | None) -> str:
    if raw is None:
        return "N/A"
    return f"{raw / 1_000_000:.2f}M"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def scene_kind(scene: str) -> str:
    if scene in INDOOR:
        return "indoor"
    if scene in OUTDOOR:
        return "outdoor"
    return "unknown"


def result_iteration(key: str) -> int | None:
    match = re.search(r"_(\d+)$", key)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def choose_result_entry(data: object) -> tuple[str, dict[str, object]] | None:
    if not isinstance(data, dict):
        return None
    entries = [(str(k), v) for k, v in data.items() if isinstance(v, dict)]
    if not entries:
        return None
    for key, value in entries:
        if key == "ours_30000":
            return key, value
    return max(entries, key=lambda item: (result_iteration(item[0]) or -1, item[0]))


def metric_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_results(results_path: Path | None) -> tuple[float | None, float | None, float | None, str]:
    if results_path is None or not results_path.exists():
        return None, None, None, "N/A"
    try:
        data = json.loads(read_text(results_path))
    except json.JSONDecodeError:
        return None, None, None, "N/A"

    chosen = choose_result_entry(data)
    if chosen is None:
        return None, None, None, "N/A"

    key, entry = chosen
    psnr = metric_float(entry.get("PSNR"))
    ssim = metric_float(entry.get("SSIM"))
    lpips = metric_float(entry.get("LPIPS"))
    iteration = result_iteration(key)
    metric_iter = str(iteration) if iteration is not None else "N/A"
    return psnr, ssim, lpips, metric_iter


def parse_millions(text: str) -> int | None:
    try:
        return int(round(float(text) * 1_000_000))
    except ValueError:
        return None


def parse_train(train_log: Path | None) -> tuple[str, str, float | None, int | None, bool, bool, int | None]:
    text = read_text(train_log)
    if not text:
        return "N/A", "N/A", None, None, False, False, None

    points = "N/A"
    train_points_raw = last_int(r'"final_points"\s*:\s*(\d+)', text)
    if train_points_raw is not None:
        points = format_points(train_points_raw)
    if points == "N/A":
        pts_m = re.findall(r"'pts'\s*:\s*'([0-9.]+)M'", text)
        if pts_m:
            train_points_raw = parse_millions(pts_m[-1])
            points = f"{float(pts_m[-1]):.2f}M"
    if points == "N/A":
        raw = last_int(r"'pts'\s*:\s*(\d+)", text)
        if raw is not None:
            train_points_raw = raw
            points = format_points(raw)

    peak = max_float(r"'peak_rsv'\s*:\s*([0-9.]+)", text)
    if peak is None:
        peak = max_float(r'"peak_reserved_gb"\s*:\s*([0-9.]+)', text)
    peak_mem = f"{peak:.2f}GB" if peak is not None else "N/A"

    duration = last_float(r'"seconds"\s*:\s*([0-9.]+)', text)
    if duration is None:
        duration = last_float(r"Training time cost:\s*\[([0-9.]+)\]\s*seconds", text)
    if duration is None:
        duration = last_float(r"'elapsed'\s*:\s*'([0-9.]+)s'", text)

    last_iter = last_int(r"'iter'\s*:\s*(\d+)", text)
    has_complete = "[Eval test iter=30000]" in text or "[training_complete]" in text or last_iter == 30000
    crashed = bool(re.search(r"core dumped|Traceback|RuntimeError|Aborted", text, flags=re.IGNORECASE))
    return points, peak_mem, duration, train_points_raw, has_complete, crashed, last_iter


def log_score(log_path: Path) -> tuple[int, int, float, str]:
    text = read_text(log_path)
    last_iter = last_int(r"'iter'\s*:\s*(\d+)", text) or 0
    complete = 1 if "[Eval test iter=30000]" in text or last_iter >= 30000 else 0
    return complete, last_iter, log_path.stat().st_mtime, log_path.name


def latest_train_log(scene_root: Path) -> Path | None:
    log_dir = scene_root / "logs"
    if not log_dir.is_dir():
        return None
    candidates = list(log_dir.glob("*.log"))
    if not candidates:
        return None
    return max(candidates, key=log_score)


def latest_iteration_dir(point_cloud_root: Path) -> Path | None:
    if not point_cloud_root.is_dir():
        return None
    candidates: list[tuple[int, float, Path]] = []
    for path in point_cloud_root.glob("iteration_*"):
        if not path.is_dir():
            continue
        match = re.search(r"iteration_(\d+)$", path.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        return None
    return max(candidates)[2]


def point_cloud_path(scene_root: Path, metric_iter: str) -> Path | None:
    point_cloud_root = scene_root / "point_cloud"
    if metric_iter != "N/A":
        preferred = point_cloud_root / f"iteration_{metric_iter}" / "point_cloud.ply"
        if preferred.exists():
            return preferred

    iteration_dir = latest_iteration_dir(point_cloud_root)
    if iteration_dir is None:
        return None
    candidate = iteration_dir / "point_cloud.ply"
    return candidate if candidate.exists() else None


def ply_vertex_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("ascii", errors="ignore").strip()
                match = re.match(r"element\s+vertex\s+(\d+)$", line)
                if match:
                    return int(match.group(1))
                if line == "end_header":
                    break
    except OSError:
        return None
    return None


def summarize_scene(dataset: str, scene_root: Path) -> Summary:
    scene = scene_root.name
    results_path = scene_root / "results.json"
    train_log = latest_train_log(scene_root)

    psnr, ssim, lpips, metric_iter = parse_results(results_path)
    points, peak_mem, duration, train_points_raw, train_complete, crashed, _ = parse_train(train_log)

    ply_points_raw = ply_vertex_count(point_cloud_path(scene_root, metric_iter))
    if ply_points_raw is not None:
        points = format_points(ply_points_raw)

    if crashed:
        status = "CRASH"
    elif psnr is None:
        status = "NO_METRIC"
    elif metric_iter != "N/A" and metric_iter != "30000":
        status = f"ITER_{metric_iter}"
    elif train_log is not None and not train_complete:
        status = "TRAIN?"
    else:
        status = "OK"

    return Summary(
        dataset=dataset,
        kind=scene_kind(scene),
        scene=scene,
        status=status,
        metric_iter=metric_iter,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        points=points,
        blocks="1",
        peak_mem=peak_mem,
        duration_s=duration,
        source_dir=scene_root,
        train_points_raw=train_points_raw,
        ply_points_raw=ply_points_raw,
    )


def collect_branch(branch: str) -> list[Summary]:
    rows: list[Summary] = []
    for dataset in DATASETS:
        branch_dir = OUTPUT_ROOT / dataset / branch
        if not branch_dir.is_dir():
            continue
        for scene_root in branch_dir.iterdir():
            if not scene_root.is_dir():
                continue
            if not (scene_root / "results.json").exists() and latest_train_log(scene_root) is None:
                continue
            rows.append(summarize_scene(dataset, scene_root))

    rows.sort(key=lambda r: (DATASET_ORDER.get(r.dataset, 99), SCENE_ORDER.get(r.scene, 99), r.scene))
    return rows


def discover_branches() -> list[str]:
    branches: set[str] = set()
    if not OUTPUT_ROOT.is_dir():
        return []
    for dataset in DATASETS:
        dataset_dir = OUTPUT_ROOT / dataset
        if not dataset_dir.is_dir():
            continue
        for path in dataset_dir.iterdir():
            if path.is_dir():
                branches.add(path.name)
    return sorted(branches)


def fmt_float(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_rows(rows: list[Summary]) -> None:
    sep = "-" * 112
    print(sep)
    print(
        f"{'Dataset':<13} {'Type':<8} {'Scene':<12} {'Iter':>6} "
        f"{'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Pts':>8} {'Blk':>5} "
        f"{'PeakMem':>9} {'Time':>11}"
    )
    print(sep)
    for row in rows:
        print(
            f"{row.dataset:<13} {row.kind:<8} {row.scene:<12} {row.metric_iter:>6} "
            f"{fmt_float(row.psnr):>8} {fmt_float(row.ssim):>8} {fmt_float(row.lpips):>8} "
            f"{row.points:>8} {row.blocks:>5} {row.peak_mem:>9} "
            f"{format_duration(row.duration_s):>11}"
        )
    print(sep)


def print_average_line(label: str, rows: list[Summary]) -> None:
    n = len(rows)
    psnr = sum(r.psnr for r in rows if r.psnr is not None) / n
    ssim = sum(r.ssim for r in rows if r.ssim is not None) / n
    lpips = sum(r.lpips for r in rows if r.lpips is not None) / n
    seconds = [r.duration_s for r in rows if r.duration_s is not None]
    time_text = format_duration(sum(seconds)) if seconds else "N/A"
    print(f"  {label:<13} scenes={n:<2}  PSNR={psnr:7.4f}  SSIM={ssim:7.4f}  LPIPS={lpips:7.4f}  total_time={time_text}")


def print_averages(rows: list[Summary]) -> None:
    valid = [r for r in rows if r.psnr is not None and r.ssim is not None and r.lpips is not None and r.status in {"OK", "TRAIN?"}]
    if not valid:
        return

    print("Averages:")
    for dataset in DATASETS:
        subset = [r for r in valid if r.dataset == dataset]
        if subset:
            print_average_line(dataset, subset)
    print_average_line("overall", valid)


def print_warnings(rows: list[Summary]) -> None:
    warnings: list[str] = []
    for row in rows:
        if row.train_points_raw is None or row.ply_points_raw is None:
            continue
        delta = abs(row.train_points_raw - row.ply_points_raw)
        if delta > max(25_000, row.ply_points_raw * 0.02):
            warnings.append(
                f"  {row.dataset}/{row.scene}: log_pts={format_points(row.train_points_raw)}, "
                f"ply_pts={format_points(row.ply_points_raw)}"
            )

    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print(warning)


def branches_to_print() -> list[str]:
    if BRANCH_ARGS:
        return BRANCH_ARGS
    branches = discover_branches()
    if branches:
        return branches
    print(f"No results found under {OUTPUT_ROOT}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    commit_id = run_git(["rev-parse", "HEAD"])
    hostname = os.uname().nodename

    for branch in branches_to_print():
        rows = collect_branch(branch)
        print()
        print(f"Server:  {hostname}")
        print(f"Branch:  {branch}")
        print(f"Commit:  {commit_id}")
        print(f"Source:  {OUTPUT_ROOT}")
        if not rows:
            print("No scene logs found.")
            continue
        print_rows(rows)
        print_averages(rows)
        print_warnings(rows)


if __name__ == "__main__":
    main()
PY
