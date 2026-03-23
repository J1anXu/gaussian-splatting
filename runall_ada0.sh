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
# 数据集 & 场景定义
########################################
DATA_BASE=/data2/jian/data
OUT_BASE=/data2/jian/output

# mip360: 室内 images_2，室外 images_4
MIP360_OUTDOOR=(bicycle flowers garden stump treehill)
MIP360_INDOOR=(room counter kitchen bonsai)
MIP360_ALL=("${MIP360_OUTDOOR[@]}" "${MIP360_INDOOR[@]}")

declare -A MIP360_IS_INDOOR
for s in "${MIP360_INDOOR[@]}"; do
  MIP360_IS_INDOOR[$s]=1
done

# deepblending: 全部用 images（默认）
DB_ALL=(drjohnson playroom)

# tandt: 全部用 images（默认）
TANDT_ALL=(train truck)

########################################
# 配置区
########################################
GPUS=(0)
NUM_GPUS=${#GPUS[@]}

LOG_ROOT=debug

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")

mkdir -p "$LOG_ROOT"


########################################
# 单 GPU 队列：完整 pipeline
########################################
run_pipeline() {
  local gpu=$1
  local dataset=$2
  local scene=$3
  local img_flag=$4   # 可选，如 "-i images_2"

  export CUDA_VISIBLE_DEVICES=$gpu

  local data_path="$DATA_BASE/$dataset/$scene"
  local model_path="$OUT_BASE/$dataset/$GIT_BRANCH/$scene"

  local log_dir="$LOG_ROOT/$GIT_BRANCH/$dataset/$scene"
  mkdir -p "$log_dir"

  echo "========================================"
  echo "GPU    : $gpu"
  echo "Dataset: $dataset"
  echo "Scene  : $scene"
  echo "Images : ${img_flag:-default}"
  echo "Time   : $(date)"
  echo "Branch : $GIT_BRANCH"
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
# 构建全局场景队列（dataset scene img_flag）
########################################
ALL_TASKS=()

for scene in "${MIP360_ALL[@]}"; do
  if [[ -n "${MIP360_IS_INDOOR[$scene]}" ]]; then
    ALL_TASKS+=("mip360|$scene|-i images_2")
  else
    ALL_TASKS+=("mip360|$scene|-i images_4")
  fi
done

for scene in "${DB_ALL[@]}"; do
  ALL_TASKS+=("deepblending|$scene|")
done

for scene in "${TANDT_ALL[@]}"; do
  ALL_TASKS+=("tandt|$scene|")
done

########################################
# 轮询分配到 GPU 队列
########################################
declare -A GPU_QUEUES
for i in "${!ALL_TASKS[@]}"; do
  gpu="${GPUS[$((i % NUM_GPUS))]}"
  GPU_QUEUES[$gpu]+="${ALL_TASKS[$i]};"
done

########################################
# 启动并发队列
########################################
echo "🚀 Launching $NUM_GPUS GPU queues..."
echo "Branch: $GIT_BRANCH"
for gpu in "${GPUS[@]}"; do
  echo "  GPU $gpu:"
  IFS=';' read -ra tasks <<< "${GPU_QUEUES[$gpu]}"
  for task in "${tasks[@]}"; do
    [[ -z "$task" ]] && continue
    IFS='|' read -r ds sc _ <<< "$task"
    echo "    $ds/$sc"
  done
done
echo ""

for gpu in "${GPUS[@]}"; do
  (
    IFS=';' read -ra tasks <<< "${GPU_QUEUES[$gpu]}"
    for task in "${tasks[@]}"; do
      [[ -z "$task" ]] && continue
      IFS='|' read -r dataset scene img_flag <<< "$task"
      run_pipeline "$gpu" "$dataset" "$scene" "$img_flag"
    done
  ) &
done

########################################
# 等待所有队列完成
########################################
wait
echo ""
echo "🎉 All pipelines finished."
