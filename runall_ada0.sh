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
# 场景分组
########################################
OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
INDOOR_SCENES=(room counter kitchen bonsai)
ALL_SCENES=("${OUTDOOR_SCENES[@]}" "${INDOOR_SCENES[@]}")

# 室内判断表
declare -A IS_INDOOR
for s in "${INDOOR_SCENES[@]}"; do
  IS_INDOOR[$s]=1
done

########################################
# 配置区
########################################
GPUS=(0)
NUM_GPUS=${#GPUS[@]}

DATA_ROOT=/data2/jian/data/mip360
LOG_ROOT=debug

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")
OUT_ROOT=/data2/jian/output/mip360/$GIT_BRANCH

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

  # 室内 images_2，室外 images_4
  local img_flag=""
  if [[ -n "${IS_INDOOR[$scene]}" ]]; then
    img_flag="-i images_2"
  else
    img_flag="-i images_4"
  fi

  local log_dir="$LOG_ROOT/$GIT_BRANCH/$scene"
  mkdir -p "$log_dir"

  echo "========================================"
  echo "GPU   : $gpu"
  echo "Scene : $scene"
  echo "Images: $img_flag"
  echo "Time  : $(date)"
  echo "Branch: $GIT_BRANCH"
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

  echo "✅ Finished $scene on GPU $gpu"
  echo ""
}

########################################
# 轮询分配 9 个场景到 4 个 GPU 队列
# GPU 0: bicycle treehill bonsai
# GPU 1: flowers room
# GPU 2: garden counter
# GPU 3: stump kitchen
########################################
declare -A GPU_QUEUES
for i in "${!ALL_SCENES[@]}"; do
  scene="${ALL_SCENES[$i]}"
  gpu="${GPUS[$((i % NUM_GPUS))]}"
  GPU_QUEUES[$gpu]+="$scene "
done

########################################
# 启动 4 个并发队列
########################################
echo "🚀 Launching $NUM_GPUS GPU queues..."
echo "Branch: $GIT_BRANCH"
for gpu in "${GPUS[@]}"; do
  echo "  GPU $gpu: ${GPU_QUEUES[$gpu]}"
done
echo ""

for gpu in "${GPUS[@]}"; do
  (
    for scene in ${GPU_QUEUES[$gpu]}; do
      run_pipeline "$gpu" "$scene"
    done
  ) &
done

########################################
# 等待所有队列完成
########################################
wait
echo ""
echo "🎉 All pipelines finished."
