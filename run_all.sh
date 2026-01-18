#!/usr/bin/env bash
set -e
set -o pipefail

########################################
# 配置区
########################################

# GPU 与 scene 对应关系（一个 GPU 一个队列）
GPUS=(4 5 6 7)
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

  echo "========================================"
  echo "GPU   : $gpu"
  echo "Scene : $scene"
  echo "Time  : $(date)"
  echo "========================================"

  ####################
  # 1. TRAIN
  ####################
  python train.py \
    -s "$data_path" \
    --model_path "$model_path" \
    --eval \
    > "$LOG_ROOT/train_${scene}.log" 2>&1

  ####################
  # 2. RENDER_P
  ####################
  python render_p.py \
    -m "$model_path" \
    --skip_train \
    > "$LOG_ROOT/render_p_${scene}.log" 2>&1

  ####################
  # 3. METRICS_P
  ####################
  python metrics_p.py \
    -m "$model_path" \
    > "$LOG_ROOT/metrics_p_${scene}.log" 2>&1

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
