#!/usr/bin/env bash

# ===== auto-daemon =====
if [[ -z "$DAEMONIZED" ]]; then
  export DAEMONIZED=1

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

  mkdir -p "$SCRIPT_DIR/debug"

  nohup bash "$SCRIPT_PATH" "$@" > "$SCRIPT_DIR/debug/finetune_pts.out" 2>&1 &
  echo "Pipeline started in background, log: $SCRIPT_DIR/debug/finetune_pts.out"
  exit 0
fi
# ======================

set -e
set -o pipefail

########################################
# 配置区
########################################
GPUS=(0)
NUM_GPUS=${#GPUS[@]}
DATA_BASE=/data/jian/data
OUT_BASE=/data/jian/output/vanilla3DGS
CKPT_BRANCH=vanilla3DGS_target_pts      # checkpoint 所在的 branch 名
FINETUNE_BRANCH=vanilla3DGS_keep_train # 输出 branch 名
LOG_ROOT=debug

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(mip360_outdoor) # DATASETS=(tandt deepblending mip360_indoor mip360_outdoor)

PTS_LIST=(300 400 500 600)   # 初始点云规模（万）
START_ITER=30000              # 假装起始 iter
EXTRA_ITER=10000              # 额外训练 iter
TOTAL_ITER=$((START_ITER + EXTRA_ITER))  # 40000

########################################
# 每个数据集的场景列表
########################################
get_scenes() {
  local dataset=$1
  case $dataset in
    mip360_indoor)   echo "room counter kitchen bonsai" ;;
    mip360_outdoor)  echo "bicycle flowers garden stump treehill" ;;
    deepblending)    echo "drjohnson playroom" ;;
    tandt)           echo "train truck" ;;
  esac
}

########################################
# mip360_indoor/outdoor 映射到实际磁盘路径 mip360
########################################
get_real_dataset() {
  local dataset=$1
  case $dataset in
    mip360_indoor|mip360_outdoor) echo "mip360" ;;
    *) echo "$dataset" ;;
  esac
}

########################################
# mip360_indoor 用 images_2，mip360_outdoor 用 images_4
########################################
get_img_flag() {
  local dataset=$1
  case $dataset in
    mip360_indoor)   echo "-i images_2" ;;
    mip360_outdoor)  echo "-i images_4" ;;
  esac
}

########################################
# 单场景单点云规模 finetune
########################################
run_finetune() {
  local gpu=$1 dataset=$2 scene=$3 pts=$4
  local img_flag
  img_flag=$(get_img_flag "$dataset")
  local real_dataset
  real_dataset=$(get_real_dataset "$dataset")

  export CUDA_VISIBLE_DEVICES=$gpu

  local data_path="$DATA_BASE/$real_dataset/$scene"
  local ckpt_dir="$OUT_BASE/$real_dataset/$CKPT_BRANCH/$scene"
  local ckpt_path="$ckpt_dir/chkpnt_pts_${pts}w.pth"

  # 输出路径体现 dataset 分组和初始点云 size
  local model_path="$OUT_BASE/$dataset/$FINETUNE_BRANCH/${scene}_pts${pts}w"
  local log_dir="$LOG_ROOT/$FINETUNE_BRANCH/$dataset/${scene}_pts${pts}w"
  mkdir -p "$log_dir"

  if [[ ! -f "$ckpt_path" ]]; then
    echo "[SKIP] $ckpt_path not found"
    return
  fi

  echo "========================================"
  echo "GPU=$gpu  $dataset/$scene  pts=${pts}w  iter=${START_ITER}->${TOTAL_ITER}"
  echo "  ckpt: $ckpt_path"
  echo "  out:  $model_path"
  echo "========================================"

  python train.py \
    -s "$data_path" \
    --model_path "$model_path" \
    --start_checkpoint "$ckpt_path" \
    --override_start_iter $START_ITER \
    --iterations $TOTAL_ITER \
    --save_pts \
    --no_grow_mode \
    --eval \
    --disable_viewer \
    --git_branch "$FINETUNE_BRANCH" \
    $img_flag \
    2>&1 | tr -d '\r' > "$log_dir/train.log"

  echo "[DONE] $dataset/$scene pts=${pts}w"
}

########################################
# 收集所有任务，轮询分配到 GPU
########################################
ALL_TASKS=()
for dataset in "${DATASETS[@]}"; do
  for scene in $(get_scenes "$dataset"); do
    for pts in "${PTS_LIST[@]}"; do
      ALL_TASKS+=("$dataset/$scene/$pts")
    done
  done
done

declare -A GPU_QUEUES
for i in "${!ALL_TASKS[@]}"; do
  gpu="${GPUS[$((i % NUM_GPUS))]}"
  GPU_QUEUES[$gpu]+="${ALL_TASKS[$i]} "
done

echo "Launching $NUM_GPUS GPU queues  (finetune: ${PTS_LIST[*]}万, iter ${START_ITER}->${TOTAL_ITER})"
echo "Total tasks: ${#ALL_TASKS[@]}"
for gpu in "${GPUS[@]}"; do
  echo "  GPU $gpu: $(echo "${GPU_QUEUES[$gpu]}" | wc -w) tasks"
done
echo ""

for gpu in "${GPUS[@]}"; do
  (
    for task in ${GPU_QUEUES[$gpu]}; do
      dataset="${task%%/*}"
      rest="${task#*/}"
      scene="${rest%%/*}"
      pts="${rest#*/}"
      run_finetune "$gpu" "$dataset" "$scene" "$pts"
    done
  ) &
done

wait
echo ""
echo "All finetune jobs finished."
