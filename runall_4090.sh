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
GPUS=(1)
NUM_GPUS=${#GPUS[@]}
DATA_BASE=/data/jian/data
OUT_BASE=/data/jian/output
LOG_ROOT=debug

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")

DATASETS=(mip360 deepblending tandt)

########################################
# 每个数据集的场景列表
########################################
get_scenes() {
  local dataset=$1
  case $dataset in
    mip360)        echo "bicycle flowers garden stump treehill room counter kitchen bonsai" ;;
    deepblending)  echo "drjohnson playroom" ;;
    tandt)         echo "train truck" ;;
  esac
}

########################################
# mip360 室内场景用 images_2，室外用 images_4
# 其他数据集用默认 images，不传 -i
########################################
get_img_flag() {
  local dataset=$1
  local scene=$2
  if [[ "$dataset" == "mip360" ]]; then
    case $scene in
      room|counter|kitchen|bonsai) echo "-i images_2" ;;
      *)                           echo "-i images_4" ;;
    esac
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
    2>&1 | tr -d '\r' > "$log_dir/train.log"

  # 2. RENDER
  echo "  [2/3] Rendering $scene ..."
  python render_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --skip_train \
    2>&1 | tr -d '\r' > "$log_dir/render.log"

  # 3. METRICS
  echo "  [3/3] Metrics $scene ..."
  python metrics_p.py \
    -m "$model_path" \
    --git_branch "$GIT_BRANCH" \
    2>&1 | tr -d '\r' > "$log_dir/metrics.log"

  echo "Finished $scene on GPU $gpu"
}

########################################
# 收集所有 (dataset, scene) 任务，轮询分配到 GPU
########################################
ALL_TASKS=()
for dataset in "${DATASETS[@]}"; do
  for scene in $(get_scenes "$dataset"); do
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
