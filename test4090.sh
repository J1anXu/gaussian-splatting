#!/bin/bash

# ============================================================
# Benchmark (4090): 加载固定点云 → Phase 2 分块训练 → 统计 iter 301-700
# ============================================================

TRAINED_PLY_PATH="/home/jian/output/mip360/baseline/bicycle/point_cloud.ply"   
DATA_PATH="/home/jian/data/mip360/bicycle"         
MODEL_PATH="/home/jian/output"      
GPU=0
BRANCH="$(git branch --show-current)"
IMG_FLAG="images_4"

# ============================================================

export CUDA_VISIBLE_DEVICES=$GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/train.py" \
    -s "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --git_branch "$BRANCH" \
    --trained_ply_path "$TRAINED_PLY_PATH" \
    --keep_training \
    --eval \
    -i "$IMG_FLAG"

pkill -f "train.py.*$MODEL_PATH" 2>/dev/null
