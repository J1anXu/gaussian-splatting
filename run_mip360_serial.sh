#!/usr/bin/env bash

# ===== auto-daemon =====
if [[ -z "$DAEMONIZED" ]]; then
  export DAEMONIZED=1

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

  mkdir -p "$SCRIPT_DIR/debug"

  nohup bash "$SCRIPT_PATH" "$@" > "$SCRIPT_DIR/debug/pipeline.out" 2>&1 &
  echo "🚀 Pipeline started in background"
  exit 0
fi
# ======================

set -e
set -o pipefail

########################################
# 配置区
########################################
GPU=0
SCENES=(bicycle kitchen room bonsai)

DATA_ROOT=/data2/jian/data/mip360
OUT_ROOT=output/mip360
LOG_ROOT=debug

export CUDA_VISIBLE_DEVICES=$GPU

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")

mkdir -p "$LOG_ROOT/$GIT_BRANCH"

echo "🌿 Git branch: $GIT_BRANCH"
echo "🖥  GPU: $GPU"
echo ""

########################################
# 串行执行每个场景: train -> render -> metrics
########################################
for scene in "${SCENES[@]}"; do
  data_path="$DATA_ROOT/$scene"
  model_path="$OUT_ROOT/$scene"
  log_dir="$LOG_ROOT/$GIT_BRANCH/$scene"
  mkdir -p "$log_dir"

  echo "========================================"
  echo "Scene : $scene"
  echo "Time  : $(date)"
  echo "========================================"

  # 1. TRAIN
  echo "  [1/3] Training $scene ..."
  python train.py \
    -s "$data_path" \
    --model_path "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --eval \
    > "$log_dir/train.log" 2>&1

  # 2. RENDER
  echo "  [2/3] Rendering $scene ..."
  python render_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --skip_train \
    > "$log_dir/render_p.log" 2>&1

  # 3. METRICS
  echo "  [3/3] Metrics $scene ..."
  python metrics_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    > "$log_dir/metricsp.log" 2>&1

  echo "✅ Finished $scene"
  echo ""
done

echo "🎉 All scenes finished."
