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
GPUS=(0 1 2 3)

DATA_ROOT=/data2/jian/data/mip360
LOG_ROOT=debug

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")
OUT_ROOT=/data/jian/output/mip360/$GIT_BRANCH

mkdir -p "$LOG_ROOT"

########################################
# 共享任务队列（临时文件 + flock 原子弹出）
########################################
QUEUE_FILE=$(mktemp)
QUEUE_LOCK=$(mktemp)
trap "rm -f '$QUEUE_FILE' '$QUEUE_LOCK'" EXIT

printf '%s\n' "${ALL_SCENES[@]}" > "$QUEUE_FILE"

pop_scene() {
  (
    flock -x 9
    local scene
    scene=$(head -1 "$QUEUE_FILE")
    if [[ -n "$scene" ]]; then
      tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
    fi
    echo "$scene"
  ) 9>"$QUEUE_LOCK"
}

########################################
# 单个场景完整 pipeline
########################################
run_pipeline() {
  local gpu=$1
  local scene=$2

  export CUDA_VISIBLE_DEVICES=$gpu

  local data_path="$DATA_ROOT/$scene"
  local model_path="$OUT_ROOT/$scene"

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
# GPU worker：不断从队列取任务直到为空
########################################
gpu_worker() {
  local gpu=$1
  while true; do
    local scene
    scene=$(pop_scene)
    [[ -z "$scene" ]] && break
    run_pipeline "$gpu" "$scene"
  done
}

########################################
# 启动 GPU worker 池
########################################
echo "🚀 Launching GPU pool (${#GPUS[@]} GPU(s))..."
echo "Branch : $GIT_BRANCH"
echo "GPUs   : ${GPUS[*]}"
echo "Scenes : ${ALL_SCENES[*]}"
echo ""

for gpu in "${GPUS[@]}"; do
  gpu_worker "$gpu" &
done

wait
echo ""
echo "🎉 All pipelines finished."
