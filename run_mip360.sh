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
# 场景分组 (参考 full_eval.py)
########################################
OUTDOOR_SCENES=(bicycle flowers garden stump treehill)
INDOOR_SCENES=(room counter kitchen bonsai)

########################################
# 解析命令行参数
########################################
TYPE="all"
GPU_LIST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)
      TYPE="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--type indoor|outdoor|all] [--gpus 0,1,2]"
      exit 1
      ;;
  esac
done

case "$TYPE" in
  indoor)  SCENES=("${INDOOR_SCENES[@]}") ;;
  outdoor) SCENES=("${OUTDOOR_SCENES[@]}") ;;
  all)     SCENES=("${OUTDOOR_SCENES[@]}" "${INDOOR_SCENES[@]}") ;;
  *)
    echo "Invalid --type: $TYPE (must be indoor, outdoor, or all)"
    exit 1
    ;;
esac
########################################
# 配置区
########################################
# GPU 列表：优先使用 --gpus 参数，否则自动检测全部
if [[ -n "$GPU_LIST" ]]; then
  IFS=',' read -ra GPUS <<< "$GPU_LIST"
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No GPUs detected"
    exit 1
  fi
  GPUS=($(seq 0 $((NUM_GPUS - 1))))
fi
NUM_GPUS=${#GPUS[@]}

# 室内场景集合 (用于判断 -i 参数)
declare -A IS_INDOOR
for s in "${INDOOR_SCENES[@]}"; do
  IS_INDOOR[$s]=1
done
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")
DATA_ROOT=/data2/jian/data/mip360
OUT_ROOT=/data2/jian/output/mip360/$GIT_BRANCH
LOG_ROOT=debug

mkdir -p "$LOG_ROOT"

echo "Type  : $TYPE"
echo "Scenes: ${SCENES[*]}"
echo "GPUs  : ${GPUS[*]}"
echo ""

########################################
# 单 GPU 队列：完整 pipeline
########################################
run_pipeline() {
  local gpu=$1
  shift
  local scenes=("$@")

  export CUDA_VISIBLE_DEVICES=$gpu

  for scene in "${scenes[@]}"; do
    local data_path="$DATA_ROOT/$scene"
    local model_path="$OUT_ROOT/$scene"

    # 室内 images_2, 室外 images_4
    local img_flag=""
    if [[ -n "${IS_INDOOR[$scene]}" ]]; then
      img_flag="-i images" # should be images_2
    else
      img_flag="-i images_4"
    fi

    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")

    local log_dir="$LOG_ROOT/$GIT_BRANCH/$scene"
    mkdir -p "$log_dir"

    echo "========================================"
    echo "GPU   : $gpu"
    echo "Scene : $scene"
    echo "Images: $img_flag"
    echo "Time  : $(date)"
    echo "Branch: $GIT_BRANCH"
    echo "========================================"

    ####################
    # 1. TRAIN
    ####################
    python train.py \
      -s "$data_path" \
      --model_path "$model_path" \
      --git_branch "$GIT_BRANCH" \
      --eval \
      $img_flag \
      > "$log_dir/train.log" 2>&1

    ####################
    # 2. RENDER
    ####################
    python render.py \
      -m "$model_path" \
      --git_branch "$GIT_BRANCH" \
      --skip_train \
      > "$log_dir/render.log" 2>&1

    ####################
    # 3. METRICS
    ####################
    python metrics.py \
      -m "$model_path" \
      --git_branch "$GIT_BRANCH" \
      > "$log_dir/metricsp.log" 2>&1

    echo "Finished $scene on GPU $gpu"
  done
}

########################################
# Round-robin 分配场景到 GPU
########################################
declare -a GPU_QUEUES
for i in "${!GPUS[@]}"; do
  GPU_QUEUES[$i]=""
done

for i in "${!SCENES[@]}"; do
  gpu_idx=$((i % NUM_GPUS))
  GPU_QUEUES[$gpu_idx]="${GPU_QUEUES[$gpu_idx]} ${SCENES[$i]}"
done

echo "Launching ${#GPUS[@]} GPU queues..."
echo ""

for i in "${!GPUS[@]}"; do
  queue=(${GPU_QUEUES[$i]})
  if [[ ${#queue[@]} -gt 0 ]]; then
    (
      run_pipeline "${GPUS[$i]}" "${queue[@]}"
    ) &
  fi
done

########################################
# 等待所有队列完成
########################################
wait
echo ""
echo "All pipelines finished."
