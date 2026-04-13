#!/usr/bin/env bash

# ============================================================
# Benchmark (4090): load fixed 30k ply -> Phase 2 block training
# ============================================================
#
# Useful overrides:
#   GPU=1 bash test_4090.sh
#   MERGE_FAST=0 bash test_4090.sh
#   MERGE_FAST_VALIDATE=1 bash test_4090.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="$(git -C "$SCRIPT_DIR" branch --show-current 2>/dev/null || echo no_git)"
COMMIT="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo no_commit)"
STAMP="$(date +"%Y%m%d_%H%M%S")"

# Stable bicycle benchmark. Keep these fixed unless we intentionally create a
# new script for another scene.
TRAINED_PLY_PATH="/data/jian/output/gaussian-splatting/mip360/vanilla3DGS/bicycle/point_cloud/iteration_30000/point_cloud.ply"
DATA_PATH="/data/jian/data/mip360/bicycle"
MODEL_PATH="/data/jian/output"
IMG_FLAG="images_4"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_RASTERIZER_PATH="${LOCAL_RASTERIZER_PATH:-$SCRIPT_DIR/submodules/diff-gaussian-rasterization}"

# Fixed diagnostic window. These values make different test runs comparable.
BENCH_FROM=301
BENCH_UNTIL=700
TRACE_FROM=681
TRACE_UNTIL=750
PROFILE_FROM=1
PROFILE_UNTIL=750
PROFILE_EVERY=1

MERGE_FAST="${MERGE_FAST:-1}"
MERGE_FAST_VALIDATE="${MERGE_FAST_VALIDATE:-0}"

LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/debug/$BRANCH/benchmark_4090}"
RUN_LOG="$LOG_ROOT/test_4090_${STAMP}.log"

mkdir -p "$LOG_ROOT" "$SCRIPT_DIR/timeline"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LOCAL_RASTERIZER_PATH${PYTHONPATH:+:$PYTHONPATH}"
export MERGE_FAST
export MERGE_FAST_VALIDATE

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_and_log() {
  log "+ $*" | tee -a "$RUN_LOG"
  # Keep carriage returns so tqdm can refresh a single console line.
  "$@" 2>&1 | tee -a "$RUN_LOG"
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
  log "PROFILE=${PROFILE_FROM}-${PROFILE_UNTIL}/every=${PROFILE_EVERY} sync=1"
  log "MERGE_FAST=$MERGE_FAST"
  log "MERGE_FAST_VALIDATE=$MERGE_FAST_VALIDATE"
  log "RUN_LOG=$RUN_LOG"
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
  --test_diagnostics
  --profile_log
  --profile_sync
  --profile_from "$PROFILE_FROM"
  --profile_until "$PROFILE_UNTIL"
  --profile_every "$PROFILE_EVERY"
  --trace_from "$TRACE_FROM"
  --trace_until "$TRACE_UNTIL"
  --bench_from "$BENCH_FROM"
  --bench_until "$BENCH_UNTIL"
  --disable_densify
)

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
