#!/usr/bin/env bash

# ===== auto-daemon =====
if [[ -z "$DAEMONIZED" ]]; then
  export DAEMONIZED=1

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

  mkdir -p "$SCRIPT_DIR/debug"

  nohup bash "$SCRIPT_PATH" "$@" > "$SCRIPT_DIR/debug/pipeline.out" 2>&1 &
  echo "Pipeline started in background"
  exit 0
fi
# ======================

set -e
set -o pipefail

########################################
# 配置区
########################################
GPUS=(0)
NUM_GPUS=${#GPUS[@]}
DATA_BASE=/data2/jian/data
OUT_BASE=/data2/jian/output/CapGS
LOG_ROOT=debug

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")

FILTER_SCENE="${1:-}"   # 传入场景名则只跑该场景，否则跑全部
DATASETS=(mip360)

########################################
# 每个数据集的场景列表（只保留 mip360 室内场景）
########################################
get_scenes() {
  local dataset=$1
  case $dataset in
    mip360) echo "room counter kitchen bonsai" ;;
  esac
}

########################################
# mip360 室内场景用 images_2
########################################
get_img_flag() {
  local dataset=$1
  local scene=$2
  if [[ "$dataset" == "mip360" ]]; then
    echo "-i images_2"
  fi
}

########################################
# 单场景 pipeline
########################################
run_pipeline() {
  local gpu=$1 dataset=$2 scene=$3
  local img_flag
  img_flag=$(get_img_flag "$dataset" "$scene")

  export CUDA_VISIBLE_DEVICES=$gpu

  local data_path="$DATA_BASE/$dataset/$scene"
  local model_path="$OUT_BASE/$dataset/$GIT_BRANCH/$scene"
  local log_dir="$LOG_ROOT/$GIT_BRANCH/$dataset/$scene"
  mkdir -p "$log_dir"

  echo "========================================"
  echo "GPU=$gpu  Dataset=$dataset  Scene=$scene  Images=${img_flag:-default}  Branch=$GIT_BRANCH"
  echo "========================================"

  # 1. TRAIN
  echo "  [1/3] Training $scene ..."
  python train.py \
    -s "$data_path" \
    --model_path "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --eval \
    $img_flag \
    > "$log_dir/train.log" 2>&1

  # 2. RENDER
  echo "  [2/3] Rendering $scene ..."
  python render_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --skip_train \
    > "$log_dir/render.log" 2>&1

  # 3. METRICS
  echo "  [3/3] Metrics $scene ..."
  python metrics_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    > "$log_dir/metrics.log" 2>&1

  echo "Finished $scene on GPU $gpu"
}

########################################
# 收集所有 (dataset, scene) 任务，轮询分配到 GPU
########################################
ALL_TASKS=()
for dataset in "${DATASETS[@]}"; do
  for scene in $(get_scenes "$dataset"); do
    [[ -n "$FILTER_SCENE" && "$scene" != "$FILTER_SCENE" ]] && continue
    ALL_TASKS+=("$dataset/$scene")
  done
done

declare -A GPU_QUEUES
for i in "${!ALL_TASKS[@]}"; do
  gpu="${GPUS[$((i % NUM_GPUS))]}"
  GPU_QUEUES[$gpu]+="${ALL_TASKS[$i]} "
done

echo "Launching $NUM_GPUS GPU queues  (branch: $GIT_BRANCH)"
for gpu in "${GPUS[@]}"; do
  echo "  GPU $gpu: ${GPU_QUEUES[$gpu]}"
done
echo ""

for gpu in "${GPUS[@]}"; do
  (
    for task in ${GPU_QUEUES[$gpu]}; do
      dataset="${task%%/*}"
      scene="${task#*/}"
      run_pipeline "$gpu" "$dataset" "$scene"
    done
  ) &
done

wait
echo ""
echo "All pipelines finished."
