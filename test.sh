#!/bin/bash
set -e

# ============================================================
# 继续训练配置 — 修改这里
# ============================================================

TRAINED_PLY_PATH="/home/jian/output/gaussian-splatting/mip360/baseline/bicycle/point_cloud/baseline/iteration_30000/point_cloud.ply"
DATA_PATH="/home/jian/data/mip360/bicycle"
MODEL_PATH="/home/jian/output/gaussian-splatting/mip360/keep_training/bicycle"
GPU=0
BRANCH="keep_training"
IMG_FLAG="images_4"

# ============================================================

export CUDA_VISIBLE_DEVICES=$GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Continue training from: $TRAINED_PLY_PATH"
echo "Data:       $DATA_PATH"
echo "Output:     $MODEL_PATH"
echo "GPU:        $GPU"
echo "Branch:     $BRANCH"
echo ""

python "$SCRIPT_DIR/train.py" \
    -s "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --git_branch "$BRANCH" \
    --trained_ply_path "$TRAINED_PLY_PATH" \
    --eval \
    -i "$IMG_FLAG"
