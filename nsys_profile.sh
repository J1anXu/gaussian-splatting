#!/bin/bash
# nsys 采集 + 导出分析报告

TRAINED_PLY_PATH="/home/jian/output/mip360/baseline/bicycle/point_cloud.ply"
DATA_PATH="/home/jian/data/mip360/bicycle"
MODEL_PATH="/home/jian/output"
GPU=0
BRANCH="$(git branch --show-current)"
IMG_FLAG="images_4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/timeline"
OUT_PREFIX="$OUT_DIR/nsys_nograd"

export CUDA_VISIBLE_DEVICES=$GPU

echo "=== 开始 nsys 采集 (delay=30s, duration=10s) ==="

nsys profile \
  --trace=cuda,cudnn,cublas,nvtx \
  --cuda-memory-usage=true \
  --delay=30 \
  --duration=10 \
  -o "$OUT_PREFIX" \
  --force-overwrite=true \
  python3 "$SCRIPT_DIR/train.py" \
    -s "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --git_branch "$BRANCH" \
    --trained_ply_path "$TRAINED_PLY_PATH" \
    --keep_training --eval -i "$IMG_FLAG"

echo ""
echo "=== 导出报告 ==="

# GPU kernel 耗时汇总
nsys stats "$OUT_PREFIX.nsys-rep" --report cuda_gpu_kern_sum --format csv \
  > "$OUT_DIR/nsys_kernels.csv" 2>/dev/null

# memcpy 耗时汇总 (h2d/d2h)
nsys stats "$OUT_PREFIX.nsys-rep" --report cuda_gpu_mem_time_sum --format csv \
  > "$OUT_DIR/nsys_memcpy.csv" 2>/dev/null

# memcpy 按大小分布
nsys stats "$OUT_PREFIX.nsys-rep" --report cuda_gpu_mem_size_sum --format csv \
  > "$OUT_DIR/nsys_memsize.csv" 2>/dev/null

# GPU 活动时间线概览
nsys stats "$OUT_PREFIX.nsys-rep" --report cuda_api_sum --format csv \
  > "$OUT_DIR/nsys_cuda_api.csv" 2>/dev/null

echo ""
echo "=== 报告已导出到 $OUT_DIR/ ==="
ls -lh "$OUT_DIR"/nsys_*.csv "$OUT_DIR"/nsys_nograd.nsys-rep 2>/dev/null

echo ""
echo "=== kernel 耗时 TOP 20 ==="
head -21 "$OUT_DIR/nsys_kernels.csv"

echo ""
echo "=== memcpy 汇总 ==="
cat "$OUT_DIR/nsys_memcpy.csv"
