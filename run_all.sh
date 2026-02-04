#!/usr/bin/env bash
set -e
set -o pipefail

########################################
# auto-daemon（第一件事）
########################################
if [[ -z "$DAEMONIZED" ]]; then
  export DAEMONIZED=1

  nohup bash "$0" "$@" > debug/pipeline.out 2>&1 &
  echo "🚀 Pipeline started in background"
  exit 0
fi

########################################
# 固定 Git 状态（只在启动时）
########################################
GIT_COMMIT=$(git rev-parse HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

SNAPSHOT_ROOT="/data2/jian/envsnapshot"
SNAPSHOT_DIR="$SNAPSHOT_ROOT/${GIT_BRANCH}_${GIT_COMMIT:0:8}"

echo "📌 Fixed branch : $GIT_BRANCH"
echo "📌 Fixed commit : $GIT_COMMIT"
echo "📌 Snapshot dir : $SNAPSHOT_DIR"

if [[ ! -d "$SNAPSHOT_DIR" ]]; then
  echo "📦 Creating code snapshot..."
  mkdir -p "$SNAPSHOT_DIR"
  rsync -a \
    --exclude .git \
    --exclude output \
    --exclude debug \
    ./ "$SNAPSHOT_DIR/"
fi

########################################
# 配置区
########################################
GPUS=(0 1 2 3)
SCENES=(bicycle kitchen room bonsai)

DATA_ROOT=/data2/jian/data/mip360
OUT_ROOT="$SNAPSHOT_DIR/output/mip360"
LOG_ROOT="$SNAPSHOT_DIR/debug"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

########################################
# 单 GPU pipeline（完全不碰 git）
########################################
run_pipeline() {
  local gpu=$1
  local scene=$2

  export CUDA_VISIBLE_DEVICES=$gpu

  local data_path="$DATA_ROOT/$scene"
  local model_path="$OUT_ROOT/$scene"

  echo "========================================"
  echo "GPU   : $gpu"
  echo "Scene : $scene"
  echo "Time  : $(date)"
  echo "Code  : $SNAPSHOT_DIR"
  echo "========================================"

  python "$SNAPSHOT_DIR/train.py" \
    -s "$data_path" \
    --model_path "$model_path" \
    --git_branch "$GIT_BRANCH" \
    --eval \
    > "$LOG_ROOT/train_${scene}.log" 2>&1

  python "$SNAPSHOT_DIR/render_p.py" \
    -m "$model_path" \
    --skip_train \
    > "$LOG_ROOT/render_p_${scene}.log" 2>&1

  python "$SNAPSHOT_DIR/metrics_p.py" \
    -m "$model_path" \
    > "$LOG_ROOT/metrics_p_${scene}.log" 2>&1

  echo "✅ Finished $scene on GPU $gpu"
}

########################################
# 启动并发
########################################
echo "🚀 Launching GPU pipelines..."

for i in "${!GPUS[@]}"; do
  run_pipeline "${GPUS[$i]}" "${SCENES[$i]}" &
done

wait
echo "🎉 All pipelines finished."
