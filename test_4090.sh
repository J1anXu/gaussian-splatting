#!/usr/bin/env bash

# ============================================================
# Benchmark (4090): load fixed 30k ply -> Phase 2 block training
# ============================================================
#
# Useful overrides:
#   GPU=1 PROFILE_SYNC=1 TRACE_FROM=650 TRACE_UNTIL=720 bash test_4090.sh
#   DISABLE_DENSIFY=0 bash test_4090.sh   # compare with densify enabled
#   SPLIT_SIZE_OVERRIDE=1200000 bash test_4090.sh
#   GPU_PACKED_CACHE_STRATEGY=largest bash test_4090.sh
#   GPU_PACKED_CACHE_STRATEGY=tail bash test_4090.sh
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="$(git -C "$SCRIPT_DIR" branch --show-current 2>/dev/null || echo no_git)"
COMMIT="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo no_commit)"
STAMP="$(date +"%Y%m%d_%H%M%S")"

# ============================================================
# Config (all can be overridden by environment variables)
# ============================================================

TRAINED_PLY_PATH="${TRAINED_PLY_PATH:-/data/jian/output/gaussian-splatting/mip360/vanilla3DGS/bicycle/point_cloud/iteration_30000/point_cloud.ply}"
DATA_PATH="${DATA_PATH:-/data/jian/data/mip360/bicycle}"
MODEL_PATH="${MODEL_PATH:-/data/jian/output}"
GPU="${GPU:-0}"
IMG_FLAG="${IMG_FLAG:-images_4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BENCH_FROM="${BENCH_FROM:-301}"
BENCH_UNTIL="${BENCH_UNTIL:-700}"
TRACE_FROM="${TRACE_FROM:-681}"
TRACE_UNTIL="${TRACE_UNTIL:-750}"
PROFILE_FROM="${PROFILE_FROM:-1}"
PROFILE_UNTIL="${PROFILE_UNTIL:-750}"
PROFILE_EVERY="${PROFILE_EVERY:-1}"
PROFILE_SYNC="${PROFILE_SYNC:-0}"
GPU_CACHE_THRESHOLD_GB="${GPU_CACHE_THRESHOLD_GB:-0.5}"
CUDA_EMPTY_CACHE_INTERVAL="${CUDA_EMPTY_CACHE_INTERVAL:-1}"
SPLIT_SIZE_OVERRIDE="${SPLIT_SIZE_OVERRIDE:-0}"
LEGACY_PER_BLOCK_LOSS="${LEGACY_PER_BLOCK_LOSS:-0}"
GPU_PACKED_CACHE_STRATEGY="${GPU_PACKED_CACHE_STRATEGY:-tail}"

# For a fixed-point-cloud Phase 2 benchmark, this should normally stay on.
DISABLE_DENSIFY="${DISABLE_DENSIFY:-1}"

LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/debug/$BRANCH/benchmark_4090}"
RUN_LOG="$LOG_ROOT/test_4090_${STAMP}.log"

mkdir -p "$LOG_ROOT" "$SCRIPT_DIR/timeline"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_and_log() {
  log "+ $*" | tee -a "$RUN_LOG"
  "$@" 2>&1 | tr -d '\r' | tee -a "$RUN_LOG"
}

{
  log "============================================================"
  log "test_4090 benchmark start"
  log "============================================================"
  log "repo=$SCRIPT_DIR"
  log "branch=$BRANCH commit=$COMMIT"
  log "host=$(hostname) user=$(whoami) pid=$$"
  log "cwd=$(pwd)"
  log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  log "TRAINED_PLY_PATH=$TRAINED_PLY_PATH"
  log "DATA_PATH=$DATA_PATH"
  log "MODEL_PATH=$MODEL_PATH"
  log "IMG_FLAG=$IMG_FLAG"
  log "PYTHON_BIN=$PYTHON_BIN"
  log "BENCH=${BENCH_FROM}-${BENCH_UNTIL} TRACE=${TRACE_FROM}-${TRACE_UNTIL}"
  log "PROFILE=${PROFILE_FROM}-${PROFILE_UNTIL}/every=${PROFILE_EVERY} sync=${PROFILE_SYNC}"
  log "GPU_CACHE_THRESHOLD_GB=$GPU_CACHE_THRESHOLD_GB CUDA_EMPTY_CACHE_INTERVAL=$CUDA_EMPTY_CACHE_INTERVAL"
  log "SPLIT_SIZE_OVERRIDE=$SPLIT_SIZE_OVERRIDE"
  log "LEGACY_PER_BLOCK_LOSS=$LEGACY_PER_BLOCK_LOSS"
  log "GPU_PACKED_CACHE_STRATEGY=$GPU_PACKED_CACHE_STRATEGY"
  log "DISABLE_DENSIFY=$DISABLE_DENSIFY"
  log "RUN_LOG=$RUN_LOG"
  log "------------------------------------------------------------"
  log "git status --short:"
  git -C "$SCRIPT_DIR" status --short || true
  log "------------------------------------------------------------"
  log "submodules:"
  git -C "$SCRIPT_DIR" submodule status || true
  log "------------------------------------------------------------"
  log "nvidia-smi:"
  nvidia-smi || true
  log "------------------------------------------------------------"
  log "python/torch:"
  "$PYTHON_BIN" - <<'PY' || true
import json
import torch
info = {
    "python": __import__("sys").version.replace("\n", " "),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
}
if torch.cuda.is_available():
    info["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_properties(i).name,
            "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 3),
        }
        for i in range(torch.cuda.device_count())
    ]
print(json.dumps(info, indent=2))
PY
  log "============================================================"
} | tee "$RUN_LOG"

TRAIN_CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/train.py"
  -s "$DATA_PATH"
  --model_path "$MODEL_PATH"
  --git_branch "$BRANCH"
  --trained_ply_path "$TRAINED_PLY_PATH"
  --keep_training
  --eval
  -i "$IMG_FLAG"
  --profile_log
  --profile_from "$PROFILE_FROM"
  --profile_until "$PROFILE_UNTIL"
  --profile_every "$PROFILE_EVERY"
  --trace_from "$TRACE_FROM"
  --trace_until "$TRACE_UNTIL"
  --bench_from "$BENCH_FROM"
  --bench_until "$BENCH_UNTIL"
  --gpu_cache_threshold_gb "$GPU_CACHE_THRESHOLD_GB"
  --cuda_empty_cache_interval "$CUDA_EMPTY_CACHE_INTERVAL"
  --split_size_override "$SPLIT_SIZE_OVERRIDE"
  --gpu_packed_cache_strategy "$GPU_PACKED_CACHE_STRATEGY"
)

if [[ "$PROFILE_SYNC" == "1" ]]; then
  TRAIN_CMD+=(--profile_sync)
fi

if [[ "$DISABLE_DENSIFY" == "1" ]]; then
  TRAIN_CMD+=(--disable_densify)
fi

if [[ "$LEGACY_PER_BLOCK_LOSS" == "1" ]]; then
  TRAIN_CMD+=(--legacy_per_block_loss)
fi

log "training command:" | tee -a "$RUN_LOG"
printf '  %q' "${TRAIN_CMD[@]}" | tee -a "$RUN_LOG"
printf '\n' | tee -a "$RUN_LOG"

SECONDS=0
set +e
run_and_log "${TRAIN_CMD[@]}"
status=$?
set -e
elapsed=$SECONDS

{
  log "============================================================"
  log "test_4090 benchmark end status=$status elapsed=${elapsed}s"
  log "latest timeline files:"
  find "$SCRIPT_DIR/timeline" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -10 || true
  log "latest train logs:"
  find "$SCRIPT_DIR/logs/train/$BRANCH" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -10 || true
  log "run log: $RUN_LOG"
  log "============================================================"
} | tee -a "$RUN_LOG"

exit "$status"
