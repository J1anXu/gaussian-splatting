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

# GPU 与 scene 对应关系（一个 GPU 一个队列）
GPUS=(0 1 2 3)
SCENES=(bicycle kitchen room bonsai)

DATA_ROOT=/data2/jian/data/mip360
OUT_ROOT=output/mip360
LOG_ROOT=debug

mkdir -p "$LOG_ROOT"






########################################
# 单 GPU 队列：完整 pipeline
########################################
run_pipeline() {
  local gpu=$1
  local scene=$2

  export CUDA_VISIBLE_DEVICES=$gpu

  local data_path="$DATA_ROOT/$scene"
  local model_path="$OUT_ROOT/$scene"

  ########################################
  # Git info
  ########################################
  GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")
  echo "🌿 Git branch: $GIT_BRANCH"

  local log_dir="$LOG_ROOT/$GIT_BRANCH/$scene"
  mkdir -p "$log_dir"

  echo "========================================"
  echo "GPU   : $gpu"
  echo "Scene : $scene"
  echo "Time  : $(date)"
  echo "Branch: $GIT_BRANCH"
  echo "========================================"

  # ####################
  # # 1. TRAIN
  # ####################
  python train.py \
    -s "$data_path" \
    --model_path "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --eval \
    > "$log_dir/train.log" 2>&1

  ####################
  # 2. RENDER_P
  ####################
  python render_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --skip_train \
    > "$log_dir/render_p.log" 2>&1

  ####################
  # 3. METRICS_P
  ####################
  python metrics_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    > "$log_dir/metricsp.log" 2>&1

  echo "✅ Finished $scene on GPU $gpu"
}

########################################
# 启动 4 个并发队列
########################################
echo "🚀 Launching 4 GPU queues..."
echo ""

for i in "${!GPUS[@]}"; do
  (
    run_pipeline "${GPUS[$i]}" "${SCENES[$i]}"
  ) &
done

########################################
# 等待所有队列完成
########################################
wait
echo ""
echo "🎉 All pipelines finished."
