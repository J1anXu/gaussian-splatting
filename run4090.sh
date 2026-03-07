#!/bin/bash
set -e

# ============================================================
# Configuration
# ============================================================

DATA_ROOT="/home/jian/data"
OUTPUT_ROOT="/home/jian/output"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
GIT_BRANCH=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no_git")

# 可用 GPU 列表 —— 按需修改
GPUS=(0 1)

# Dataset -> scenes
declare -A DATASET_SCENES
DATASET_SCENES[mip360]="bicycle flowers garden stump treehill bonsai counter kitchen room"
DATASET_SCENES[tandt]="truck train"
DATASET_SCENES[deepblending]="drjohnson playroom"

# Dataset -> data subdirectory
declare -A DATASET_DATA_DIR
DATASET_DATA_DIR[mip360]="mip360"
DATASET_DATA_DIR[tandt]="tandt"
DATASET_DATA_DIR[deepblending]="deepblending"

# Per-scene downsampling
declare -A SCENE_IMAGES
for s in bicycle flowers garden stump treehill; do
    SCENE_IMAGES[$s]="images_4"
done
for s in bonsai counter kitchen room; do
    SCENE_IMAGES[$s]="images_2"
done
for s in truck train drjohnson playroom; do
    SCENE_IMAGES[$s]="images"
done

# ============================================================
# Helpers
# ============================================================

find_dataset_for_scene() {
    local scene="$1"
    for ds in "${!DATASET_SCENES[@]}"; do
        for s in ${DATASET_SCENES[$ds]}; do
            [ "$s" == "$scene" ] && echo "$ds" && return
        done
    done
    echo ""
}

# run_scene <dataset> <scene> <gpu>
# 在指定 GPU 上跑完整 train -> render -> metrics 流程
run_scene() {
    local dataset="$1"
    local scene="$2"
    local gpu="$3"

    local data_subdir="${DATASET_DATA_DIR[$dataset]}"
    local data_dir="${DATA_ROOT}/${data_subdir}/${scene}"
    local model_path="${OUTPUT_ROOT}/${PROJECT_NAME}/${data_subdir}/${GIT_BRANCH}/${scene}"
    local img_flag="${SCENE_IMAGES[$scene]:-images}"
    local log_dir="${model_path}/logs"

    if [ ! -d "$data_dir" ]; then
        echo "[SKIP] Data not found: $data_dir"
        return
    fi

    mkdir -p "$log_dir"

    echo "============================================================"
    echo "[${dataset}/${scene}] train -> render -> metrics"
    echo "  GPU:        $gpu"
    echo "  data_dir:   $data_dir"
    echo "  model_path: $model_path"
    echo "  images:     $img_flag"
    echo "  branch:     $GIT_BRANCH"
    echo "  time:       $(date)"
    echo "============================================================"

    # 1. Train
    echo "  [1/3] Training $scene on GPU $gpu ..."
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/train.py" \
        -s "$data_dir" \
        --model_path "$model_path" \
        --git_branch "$GIT_BRANCH" \
        --eval \
        -i "$img_flag" \
        > "$log_dir/train.log" 2>&1

    # 2. Render
    echo "  [2/3] Rendering $scene on GPU $gpu ..."
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/render.py" \
        -m "$model_path" \
        --git_branch "$GIT_BRANCH" \
        --skip_train \
        > "$log_dir/render.log" 2>&1

    # 3. Metrics
    echo "  [3/3] Metrics $scene on GPU $gpu ..."
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/metrics.py" \
        -m "$model_path" \
        --git_branch "$GIT_BRANCH" \
        > "$log_dir/metrics.log" 2>&1

    echo "[DONE] ${dataset}/${scene} on GPU $gpu"
    echo ""
}

# ============================================================
# Parallel scheduler: 每个 scene 独占一张卡，超出卡数则排队等待
# ============================================================

# GPU 槽位管理 (FIFO)
GPU_FIFO="/tmp/gpu_fifo_$$"
mkfifo "$GPU_FIFO"
exec 3<>"$GPU_FIFO"
rm -f "$GPU_FIFO"

# 将所有可用 GPU 写入 FIFO 作为令牌
for g in "${GPUS[@]}"; do
    echo "$g" >&3
done

# dispatch_scene <dataset> <scene>
# 从 FIFO 取一张空闲卡，后台跑完后归还令牌
dispatch_scene() {
    local dataset="$1"
    local scene="$2"

    # 阻塞等待空闲 GPU
    local gpu
    read -r gpu <&3

    (
        run_scene "$dataset" "$scene" "$gpu"
        # 归还 GPU 令牌
        echo "$gpu" >&3
    ) &
}

# ============================================================
# Parse arguments
# ============================================================

SCENE=""
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --gpus)
            # 覆盖默认卡列表, e.g. --gpus "0 1 2 3"
            IFS=' ' read -ra GPUS <<< "$2"
            shift 2
            ;;
        --branch)
            GIT_BRANCH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage:"
            echo "  ./run.sh --scene bicycle                       # Single scene"
            echo "  ./run.sh --scene bicycle/garden/room            # Multiple scenes"
            echo "  ./run.sh --dataset mip360                       # One dataset"
            echo "  ./run.sh --dataset mip360/tandt                 # Multiple datasets"
            echo "  ./run.sh                                        # All datasets"
            echo "  ./run.sh --gpus '0 1 2 3' --dataset mip360     # Specify GPUs"
            echo "  ./run.sh --branch my_branch --scene room        # Override branch"
            echo ""
            echo "Scenes auto-dispatch to available GPUs. If scenes > GPUs, extras queue."
            echo ""
            echo "Available datasets and scenes:"
            for ds in mip360 tandt deepblending; do
                echo "  ${ds}: ${DATASET_SCENES[$ds]}"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 重新填充 FIFO（如果 --gpus 覆盖了默认值）
# 先清空旧令牌，再写入新的
exec 3>&-
GPU_FIFO="/tmp/gpu_fifo_$$"
mkfifo "$GPU_FIFO"
exec 3<>"$GPU_FIFO"
rm -f "$GPU_FIFO"
for g in "${GPUS[@]}"; do
    echo "$g" >&3
done

NUM_GPUS=${#GPUS[@]}

# ============================================================
# Main
# ============================================================

cd "$SCRIPT_DIR"

echo "Project: $PROJECT_NAME"
echo "Branch:  $GIT_BRANCH"
echo "GPUs:    ${GPUS[*]} (${NUM_GPUS} available)"
echo ""

if [ -n "$SCENE" ]; then
    # --scene bicycle 或 --scene bicycle/garden/room
    IFS='/' read -ra SCENE_LIST <<< "$SCENE"
    echo "Scenes: ${SCENE_LIST[*]}"
    echo ""
    for s in "${SCENE_LIST[@]}"; do
        ds=$(find_dataset_for_scene "$s")
        if [ -z "$ds" ]; then
            echo "Error: scene '$s' not found in any dataset"
            exit 1
        fi
        dispatch_scene "$ds" "$s"
    done
    wait
    echo "All scenes completed!"

elif [ -n "$DATASET" ]; then
    # --dataset mip360 或 --dataset mip360/tandt
    IFS='/' read -ra DS_LIST <<< "$DATASET"
    echo "Datasets: ${DS_LIST[*]}"
    echo ""
    for ds in "${DS_LIST[@]}"; do
        scenes="${DATASET_SCENES[$ds]}"
        if [ -z "$scenes" ]; then
            echo "Error: dataset '$ds' not found"
            echo "Available: mip360 tandt deepblending"
            exit 1
        fi
        echo "  ${ds}: ${scenes}"
        for scene in $scenes; do
            dispatch_scene "$ds" "$scene"
        done
    done
    wait
    echo "All datasets completed!"

else
    # 全部 dataset 全部 scene
    echo "Running ALL datasets"
    echo ""
    for ds in mip360 tandt deepblending; do
        for scene in ${DATASET_SCENES[$ds]}; do
            dispatch_scene "$ds" "$scene"
        done
    done
    wait
    echo "All datasets completed!"
fi

# 关闭 FIFO
exec 3>&-
