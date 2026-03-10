#!/bin/bash

# ============================================================
# Benchmark: 加载固定点云 → Phase 2 分块训练 → 统计 iter 300-700
# ============================================================

TRAINED_PLY_PATH="/home/jian/Partitioned-3dgs/output/mip360/bicycle/point_cloud/iteration_30000/point_cloud.ply"
DATA_PATH="/data2/jian/data/mip360/bicycle"
MODEL_PATH="/home/jian/gaussian-splatting/output/benchmark"
GPU=0
BRANCH="$(git branch --show-current)"
IMG_FLAG="images_4"

# ============================================================

export CUDA_VISIBLE_DEVICES=$GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/train.py" \
    -s "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --git_branch "$BRANCH" \
    --trained_ply_path "$TRAINED_PLY_PATH" \
    --keep_training \
    --eval \
    -i "$IMG_FLAG"

# 清理残留进程
pkill -f "train.py.*benchmark" 2>/dev/null
