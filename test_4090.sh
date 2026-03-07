#!/usr/bin/env bash

# ===== auto-daemon =====
if [[ -z "$DAEMONIZED" ]]; then
  export DAEMONIZED=1

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

  mkdir -p "$SCRIPT_DIR/debug"

  nohup bash "$SCRIPT_PATH" "$@" > "$SCRIPT_DIR/debug/pipeline_serial.out" 2>&1 &
  echo "Pipeline started in background"
  exit 0
fi
# ======================

set -e
set -o pipefail

########################################
# 场景分组 (参考 full_eval.py)
########################################
OUTDOOR_SCENES=(bicycle)
INDOOR_SCENES=()

########################################
# 解析命令行参数
########################################
TYPE="all"
GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)
      TYPE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--type indoor|outdoor|all] [--gpu 0]"
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
export CUDA_VISIBLE_DEVICES=$GPU

# 室内场景集合 (用于判断 -i 参数)
declare -A IS_INDOOR
for s in "${INDOOR_SCENES[@]}"; do
  IS_INDOOR[$s]=1
done

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")
DATA_ROOT=/data2/jian/data/deepblending
OUT_ROOT=/data2/jian/output/deepblending/$GIT_BRANCH
LOG_ROOT=debug

mkdir -p "$LOG_ROOT"

echo "Type  : $TYPE"
echo "Scenes: ${SCENES[*]}"
echo "GPU   : $GPU"
echo "Branch: $GIT_BRANCH"
echo ""

########################################
# 串行执行: train -> render -> metrics
########################################
for scene in "${SCENES[@]}"; do
  data_path="$DATA_ROOT/$scene"
  model_path="$OUT_ROOT/$scene"

  # 室内 images_2, 室外 images_4
  img_flag=""
  if [[ -n "${IS_INDOOR[$scene]}" ]]; then
    img_flag="-i images"
  else
    img_flag="-i images"
  fi

  log_dir="$LOG_ROOT/$GIT_BRANCH/$scene"
  mkdir -p "$log_dir"

  echo "========================================"
  echo "Scene : $scene"
  echo "Images: $img_flag"
  echo "Time  : $(date)"
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
    > "$log_dir/metricsp.log" 2>&1

  echo "Finished $scene"
  echo ""
done

echo "All scenes finished."
