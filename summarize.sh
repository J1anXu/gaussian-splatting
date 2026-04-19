#!/usr/bin/env bash
# Summarize training results from /data/jian/output/CapGS first, with
# debug/<branch>/ as a fallback mirror.
# Usage: bash summarize.sh [branch1 branch2 ...]
#   No args = use current git branch name
#   CAPGS_OUTPUT_ROOT can override /data/jian/output/CapGS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 - "$SCRIPT_DIR" "$@" <<'PY'
from __future__ import annotations

import os
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(sys.argv[1]).resolve()
DEBUG_DIR = SCRIPT_DIR / "debug"
OUTPUT_ROOT = Path(os.environ.get("CAPGS_OUTPUT_ROOT") or "/data/jian/output/CapGS").expanduser()
BRANCH_ARGS = sys.argv[2:]

DATASETS = ("mip360", "deepblending", "tandt")
DATASET_ORDER = {name: i for i, name in enumerate(DATASETS)}

SCENE_DATASET = {
    "bicycle": "mip360",
    "flowers": "mip360",
    "garden": "mip360",
    "stump": "mip360",
    "treehill": "mip360",
    "room": "mip360",
    "counter": "mip360",
    "kitchen": "mip360",
    "bonsai": "mip360",
    "drjohnson": "deepblending",
    "playroom": "deepblending",
    "train": "tandt",
    "truck": "tandt",
}

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
    no_grad: int
    source_dir: Path
    train_points_raw: int | None
    render_points_raw: int | None
    train_blocks_raw: int | None
    saved_blocks: int | None


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(SCRIPT_DIR), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def current_branch() -> str:
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"])


def latest_log(scene_dir: Path, prefix: str) -> Path | None:
    candidates = list(scene_dir.glob(f"{prefix}_*.log"))
    plain = scene_dir / f"{prefix}.log"
    if plain.exists():
        candidates.append(plain)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def latest_output_train_log(scene_root: Path) -> Path | None:
    log_dir = scene_root / "logs"
    if not log_dir.is_dir():
        return None
    candidates = list(log_dir.glob("*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def results_json_path(scene_root: Path, branch: str) -> Path | None:
    preferred = scene_root / "rendered_p" / branch / "results.json"
    if preferred.exists():
        return preferred

    rendered_root = scene_root / "rendered_p"
    if not rendered_root.is_dir():
        return None
    candidates = list(rendered_root.glob("*/results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


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


def parse_metric_iter(text: str) -> str:
    patterns = [
        r"Method:\s*ours_(\d+)",
        r"\bours_(\d+)\b",
        r"iteration[_\s-]*(\d+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1])
    return "N/A"


def parse_metrics(metrics_log: Path | None) -> tuple[float | None, float | None, float | None, str]:
    text = read_text(metrics_log)
    if not text:
        return None, None, None, "N/A"
    psnr = last_float(r"PSNR\s*:\s*([0-9.]+)", text)
    ssim = last_float(r"SSIM\s*:\s*([0-9.]+)", text)
    lpips = last_float(r"LPIPS\s*:\s*([0-9.]+)", text)
    return psnr, ssim, lpips, parse_metric_iter(text)


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


def parse_results(results_path: Path | None) -> tuple[float | None, float | None, float | None, str, str, int | None]:
    if results_path is None or not results_path.exists():
        return None, None, None, "N/A", "N/A", None
    try:
        data = json.loads(read_text(results_path))
    except json.JSONDecodeError:
        return None, None, None, "N/A", "N/A", None

    chosen = choose_result_entry(data)
    if chosen is None:
        return None, None, None, "N/A", "N/A", None

    key, entry = chosen
    psnr = metric_float(entry.get("PSNR"))
    ssim = metric_float(entry.get("SSIM"))
    lpips = metric_float(entry.get("LPIPS"))
    iteration = result_iteration(key)
    metric_iter = str(iteration) if iteration is not None else "N/A"

    raw_points_float = metric_float(entry.get("num_gaussians"))
    raw_points = int(raw_points_float) if raw_points_float is not None else None
    points = format_points(raw_points) if raw_points is not None else "N/A"
    return psnr, ssim, lpips, metric_iter, points, raw_points


def parse_train(train_log: Path | None, extra_logs: list[Path] | None = None) -> tuple[str, str, str, float | None, int, bool, bool, int | None, int | None]:
    text = read_text(train_log)
    for extra in extra_logs or []:
        text += "\n" + read_text(extra)
    if not text:
        return "N/A", "N/A", "N/A", None, 0, False, False, None, None

    final_points = last_int(r'"final_points"\s*:\s*(\d+)', text)
    train_points_raw = final_points
    points = format_points(final_points)
    if points == "N/A":
        pts_m = re.findall(r"'pts'\s*:\s*'([0-9.]+M)'", text)
        if pts_m:
            points = pts_m[-1]
    if points == "N/A":
        raw = last_int(r"'pts'\s*:\s*(\d+)", text)
        points = format_points(raw)
        train_points_raw = raw
    if points == "N/A":
        pts_m = re.findall(r"\bpts=([0-9.]+M)", text)
        if pts_m:
            points = pts_m[-1]

    final_blocks = last_int(r'"final_blocks"\s*:\s*(\d+)', text)
    if final_blocks is None:
        final_blocks = last_int(r"'blk'\s*:\s*(\d+)", text)
    if final_blocks is None:
        final_blocks = last_int(r"\bblk=(\d+)", text)
    blocks = str(final_blocks) if final_blocks is not None else "N/A"

    peak = max_float(r"'peak_rsv'\s*:\s*([0-9.]+)", text)
    if peak is None:
        peak = max_float(r'"peak_reserved_gb"\s*:\s*([0-9.]+)', text)
    if peak is None:
        peak = max_float(r"\bpeak=([0-9.]+)", text)
    if peak is None:
        # current format: per-iter peak under 'rsv'
        peak = max_float(r"'rsv'\s*:\s*([0-9.]+)", text)
    peak_mem = f"{peak:.2f}GB" if peak is not None else "N/A"

    duration = last_float(r'"seconds"\s*:\s*([0-9.]+)', text)
    if duration is None:
        # matches both "Training time cost:" and "Phase 2 training time cost:"
        duration = last_float(r"training time cost:\s*\[([0-9.]+)\]\s*seconds", text.lower())
    if duration is None:
        duration = last_float(r"'elapsed'\s*:\s*'([0-9.]+)s'", text)

    no_grad = len(re.findall(r"no gradient block was processed", text))
    has_complete = (
        "[training_complete]" in text
        or "training time cost:" in text.lower()
        or "'iter': 30000" in text
    )
    crashed = bool(re.search(r"core dumped|Traceback|RuntimeError|Aborted", text, flags=re.IGNORECASE))
    return points, blocks, peak_mem, duration, no_grad, has_complete, crashed, train_points_raw, final_blocks


def scene_kind(scene: str) -> str:
    if scene in INDOOR:
        return "indoor"
    if scene in OUTDOOR:
        return "outdoor"
    return "unknown"


def infer_dataset(scene: str, parent_dataset: str | None = None) -> str:
    if parent_dataset:
        return parent_dataset
    return SCENE_DATASET.get(scene, "unknown")


def has_any_log(scene_dir: Path) -> bool:
    return any(scene_dir.glob("train*.log")) or any(scene_dir.glob("metrics*.log")) or any(scene_dir.glob("render*.log"))


def has_real_output(scene_root: Path, branch: str) -> bool:
    return latest_output_train_log(scene_root) is not None or results_json_path(scene_root, branch) is not None


def summarize_scene(dataset: str, scene_dir: Path, branch: str) -> Summary:
    scene = scene_dir.name
    metrics_log = latest_log(scene_dir, "metrics")
    train_log = latest_log(scene_dir, "train")

    psnr, ssim, lpips, metric_iter = parse_metrics(metrics_log)
    # Also pull structured logger output at logs/train/<branch>/<scene>/*.log
    # (has 'rsv' / 'alloc' dicts written by LOGGER.info; debug/train.log
    # only has stdout/tqdm and the final time line).
    structured_logs = sorted((SCRIPT_DIR / "logs" / "train" / branch / scene).glob("*.log"))
    points, blocks, peak_mem, duration, no_grad, train_complete, crashed, train_points_raw, train_blocks_raw = parse_train(train_log, structured_logs)

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
        blocks=blocks,
        peak_mem=peak_mem,
        duration_s=duration,
        no_grad=no_grad,
        source_dir=scene_dir,
        train_points_raw=train_points_raw,
        render_points_raw=None,
        train_blocks_raw=train_blocks_raw,
        saved_blocks=None,
    )


def count_saved_blocks(scene_root: Path, branch: str, metric_iter: str) -> int | None:
    if metric_iter == "N/A":
        return None
    point_cloud_dir = scene_root / "point_cloud" / branch / f"iteration_{metric_iter}"
    if not point_cloud_dir.is_dir():
        return None
    return len(list(point_cloud_dir.glob("point_cloud_sub_*.ply")))


def summarize_output_scene(branch: str, dataset: str, scene_root: Path) -> Summary:
    scene = scene_root.name
    results_path = results_json_path(scene_root, branch)
    train_log = latest_output_train_log(scene_root)

    psnr, ssim, lpips, metric_iter, result_points, render_points_raw = parse_results(results_path)
    points, blocks, peak_mem, duration, no_grad, train_complete, crashed, train_points_raw, train_blocks_raw = parse_train(train_log)
    if points == "N/A" and result_points != "N/A":
        points = result_points
    saved_blocks = count_saved_blocks(scene_root, branch, metric_iter)

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
        blocks=blocks,
        peak_mem=peak_mem,
        duration_s=duration,
        no_grad=no_grad,
        source_dir=scene_root,
        train_points_raw=train_points_raw,
        render_points_raw=render_points_raw,
        train_blocks_raw=train_blocks_raw,
        saved_blocks=saved_blocks,
    )


def collect_debug_branch(branch_dir: Path, branch: str) -> list[Summary]:
    candidates: dict[tuple[str, str], tuple[tuple[int, float], Path, str]] = {}

    for dataset in DATASETS:
        dataset_dir = branch_dir / dataset
        if not dataset_dir.is_dir():
            continue
        for scene_dir in dataset_dir.iterdir():
            if not scene_dir.is_dir() or not has_any_log(scene_dir):
                continue
            # Canonical pipeline layout. Keep it as fallback when a richer
            # flat debug/<branch>/<scene>/ mirror exists for the same scene.
            score = (0, max((p.stat().st_mtime for p in scene_dir.glob("*.log")), default=0.0))
            candidates[(dataset, scene_dir.name)] = (score, scene_dir, dataset)

    for scene_dir in branch_dir.iterdir():
        if not scene_dir.is_dir() or scene_dir.name in DATASETS or not has_any_log(scene_dir):
            continue
        dataset = infer_dataset(scene_dir.name)
        key = (dataset, scene_dir.name)
        # Flat per-scene dirs usually contain the detailed logger output
        # (train_*.log with peak_rsv/profile fields), so prefer them over the
        # plain pipeline train.log when both are present.
        score = (1, max((p.stat().st_mtime for p in scene_dir.glob("*.log")), default=0.0))
        old = candidates.get(key)
        if old is None or score > old[0]:
            candidates[key] = (score, scene_dir, dataset)

    rows = [summarize_scene(dataset, scene_dir, branch) for _, scene_dir, dataset in candidates.values()]
    rows.sort(key=lambda r: (DATASET_ORDER.get(r.dataset, 99), SCENE_ORDER.get(r.scene, 99), r.scene))
    return rows


def collect_output_branch(branch: str) -> list[Summary]:
    if not OUTPUT_ROOT.is_dir():
        return []

    rows: list[Summary] = []
    for dataset in DATASETS:
        branch_dir = OUTPUT_ROOT / dataset / branch
        if not branch_dir.is_dir():
            continue
        for scene_root in branch_dir.iterdir():
            if not scene_root.is_dir() or not has_real_output(scene_root, branch):
                continue
            rows.append(summarize_output_scene(branch, dataset, scene_root))

    rows.sort(key=lambda r: (DATASET_ORDER.get(r.dataset, 99), SCENE_ORDER.get(r.scene, 99), r.scene))
    return rows


def collect_branch(branch: str) -> list[Summary]:
    by_scene: dict[tuple[str, str], tuple[int, Summary]] = {}

    debug_dir = DEBUG_DIR / branch
    if debug_dir.is_dir():
        for row in collect_debug_branch(debug_dir, branch):
            by_scene[(row.dataset, row.scene)] = (0, row)

    for row in collect_output_branch(branch):
        # The real output tree is authoritative. Keep debug only for scenes
        # that have not yet been copied/rendered into /data.
        by_scene[(row.dataset, row.scene)] = (1, row)

    rows = [row for _, row in by_scene.values()]
    rows.sort(key=lambda r: (DATASET_ORDER.get(r.dataset, 99), SCENE_ORDER.get(r.scene, 99), r.scene))
    return rows


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


def print_averages(rows: list[Summary]) -> None:
    valid = [r for r in rows if r.psnr is not None and r.ssim is not None and r.lpips is not None and r.status in {"OK", "TRAIN?"}]
    if not valid:
        return

    print("Averages:")
    for dataset in DATASETS:
        subset = [r for r in valid if r.dataset == dataset]
        if not subset:
            continue
        print_average_line(dataset, subset)
    print_average_line("overall", valid)


def print_warnings(rows: list[Summary]) -> None:
    warnings: list[str] = []
    for row in rows:
        label = f"{row.dataset}/{row.scene}"
        if row.train_points_raw is not None and row.render_points_raw is not None:
            delta = abs(row.train_points_raw - row.render_points_raw)
            if delta > max(10_000, row.train_points_raw * 0.005):
                warnings.append(
                    f"  {label}: train_pts={format_points(row.train_points_raw)}, "
                    f"render_pts={format_points(row.render_points_raw)}"
                )
        if row.train_blocks_raw is not None and row.saved_blocks is not None and row.train_blocks_raw != row.saved_blocks:
            warnings.append(
                f"  {label}: train_blocks={row.train_blocks_raw}, "
                f"saved_point_cloud_files={row.saved_blocks}"
            )

    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print(warning)


def print_average_line(label: str, rows: list[Summary]) -> None:
    n = len(rows)
    psnr = sum(r.psnr for r in rows if r.psnr is not None) / n
    ssim = sum(r.ssim for r in rows if r.ssim is not None) / n
    lpips = sum(r.lpips for r in rows if r.lpips is not None) / n
    seconds = [r.duration_s for r in rows if r.duration_s is not None]
    time_text = format_duration(sum(seconds)) if seconds else "N/A"
    print(f"  {label:<13} scenes={n:<2}  PSNR={psnr:7.4f}  SSIM={ssim:7.4f}  LPIPS={lpips:7.4f}  total_time={time_text}")


def branches_to_print() -> list[str]:
    if BRANCH_ARGS:
        return BRANCH_ARGS
    branch = current_branch()
    if branch and ((DEBUG_DIR / branch).is_dir() or any((OUTPUT_ROOT / dataset / branch).is_dir() for dataset in DATASETS)):
        return [branch]
    print(f"No results found for current branch '{branch}' under {OUTPUT_ROOT} or debug/", file=sys.stderr)
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
        print(f"Source:  {OUTPUT_ROOT} (debug fallback)")
        if not rows:
            print("No scene logs found.")
            continue
        print_rows(rows)
        print_averages(rows)
        print_warnings(rows)


if __name__ == "__main__":
    main()
PY
