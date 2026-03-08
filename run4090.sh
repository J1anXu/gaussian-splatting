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
# Progress display
# ============================================================

STATUS_DIR="/tmp/run4090_status_$$"
mkdir -p "$STATUS_DIR"
touch "$STATUS_DIR/.running"

# gpu_status <gpu> <scene> <phase> — 写状态文件, monitor 读取刷新
gpu_status() {
    local gpu=$1 scene=$2 phase=$3
    echo "${scene}|${phase}|$(date +%s)" > "$STATUS_DIR/gpu_$gpu"
}

# 标记一个场景完成
mark_done() {
    touch "$STATUS_DIR/done_$1"
}

# 标记一个场景失败
mark_fail() {
    touch "$STATUS_DIR/fail_$1"
}

# monitor_progress — 后台循环刷新进度条
monitor_progress() {
    local num_gpus=${#GPUS[@]}
    local total=$1
    local lines=$((num_gpus + 3))

    # 预留行
    for ((i = 0; i < lines; i++)); do printf "\n"; done

    while [ -f "$STATUS_DIR/.running" ]; do
        printf "\033[${lines}A"

        local done_count=$(ls "$STATUS_DIR"/done_* 2>/dev/null | wc -l)
        local fail_count=$(ls "$STATUS_DIR"/fail_* 2>/dev/null | wc -l)

        # 总进度条
        local pct=0
        if [ "$total" -gt 0 ]; then
            pct=$(( (done_count + fail_count) * 100 / total ))
        fi
        local bar_len=30
        local filled=$(( pct * bar_len / 100 ))
        local empty=$(( bar_len - filled ))
        local bar=$(printf '%0.s#' $(seq 1 $filled 2>/dev/null))$(printf '%0.s-' $(seq 1 $empty 2>/dev/null))

        local fail_info=""
        if [ "$fail_count" -gt 0 ]; then
            fail_info="  ${fail_count} failed"
        fi

        printf "\033[2K  \033[1mTotal: [%s] %d/%d scenes  %d%%%s\033[0m\n" "$bar" "$((done_count + fail_count))" "$total" "$pct" "$fail_info"
        printf "\033[2K\n"

        for g in "${GPUS[@]}"; do
            local st
            st=$(cat "$STATUS_DIR/gpu_$g" 2>/dev/null || echo "")
            if [ -z "$st" ]; then
                printf "\033[2K  \033[36mGPU %s\033[0m | \033[90mwaiting...\033[0m\n" "$g"
            else
                IFS='|' read -r scene phase start_ts <<< "$st"
                local now=$(date +%s)
                local elapsed=$((now - start_ts))
                local mins=$((elapsed / 60))
                local secs=$((elapsed % 60))

                local phase_icon phase_color
                case "$phase" in
                    train)   phase_icon=">>>" ; phase_color="\033[33m" ;;
                    render)  phase_icon=">>>" ; phase_color="\033[35m" ;;
                    metrics) phase_icon=">>>" ; phase_color="\033[34m" ;;
                    done)    phase_icon="OK " ; phase_color="\033[32m" ;;
                    FAIL)    phase_icon="ERR" ; phase_color="\033[31m" ;;
                    *)       phase_icon="..." ; phase_color="\033[90m" ;;
                esac

                printf "\033[2K  \033[36mGPU %s\033[0m | %-14s ${phase_color}[%s] %-8s\033[0m  %dm%02ds\n" \
                    "$g" "$scene" "$phase_icon" "$phase" "$mins" "$secs"
            fi
        done

        printf "\033[2K\n"
        sleep 1
    done
}

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
        gpu_status "$gpu" "$scene" "FAIL"
        mark_fail "$scene"
        return
    fi

    mkdir -p "$log_dir"

    # 1. Train
    gpu_status "$gpu" "$scene" "train"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/train.py" \
        -s "$data_dir" \
        --model_path "$model_path" \
        --git_branch "$GIT_BRANCH" \
        --eval \
        -i "$img_flag" \
        > "$log_dir/train.log" 2>&1

    # 2. Render
    gpu_status "$gpu" "$scene" "render"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/render.py" \
        -m "$model_path" \
        --git_branch "$GIT_BRANCH" \
        --skip_train \
        > "$log_dir/render.log" 2>&1

    # 3. Metrics
    gpu_status "$gpu" "$scene" "metrics"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/metrics.py" \
        -m "$model_path" \
        --git_branch "$GIT_BRANCH" \
        > "$log_dir/metrics.log" 2>&1

    gpu_status "$gpu" "$scene" "done"
    mark_done "$scene"
}

# ============================================================
# Parallel scheduler
# ============================================================

GPU_FIFO="/tmp/gpu_fifo_$$"
mkfifo "$GPU_FIFO"
exec 3<>"$GPU_FIFO"
rm -f "$GPU_FIFO"

for g in "${GPUS[@]}"; do
    echo "$g" >&3
done

dispatch_scene() {
    local dataset="$1"
    local scene="$2"

    local gpu
    read -r gpu <&3

    (
        trap 'echo "$gpu" >&3' EXIT
        run_scene "$dataset" "$scene" "$gpu"
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
            IFS=' ' read -ra GPUS <<< "$2"
            shift 2
            ;;
        --branch)
            GIT_BRANCH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage:"
            echo "  ./run4090.sh --scene bicycle                       # Single scene"
            echo "  ./run4090.sh --scene bicycle/garden/room            # Multiple scenes"
            echo "  ./run4090.sh --dataset mip360                       # One dataset"
            echo "  ./run4090.sh --dataset mip360/tandt                 # Multiple datasets"
            echo "  ./run4090.sh                                        # All datasets"
            echo "  ./run4090.sh --gpus '0 1 2 3' --dataset mip360     # Specify GPUs"
            echo "  ./run4090.sh --branch my_branch --scene room        # Override branch"
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

# 收集所有要跑的场景，计算总数
ALL_SCENE_LIST=()

if [ -n "$SCENE" ]; then
    IFS='/' read -ra ALL_SCENE_LIST <<< "$SCENE"
elif [ -n "$DATASET" ]; then
    IFS='/' read -ra DS_LIST <<< "$DATASET"
    for ds in "${DS_LIST[@]}"; do
        scenes="${DATASET_SCENES[$ds]}"
        if [ -z "$scenes" ]; then
            echo "Error: dataset '$ds' not found"
            exit 1
        fi
        for s in $scenes; do
            ALL_SCENE_LIST+=("$s")
        done
    done
else
    for ds in mip360 tandt deepblending; do
        for s in ${DATASET_SCENES[$ds]}; do
            ALL_SCENE_LIST+=("$s")
        done
    done
fi

TOTAL_SCENES=${#ALL_SCENE_LIST[@]}

echo "Project: $PROJECT_NAME"
echo "Branch:  $GIT_BRANCH"
echo "GPUs:    ${GPUS[*]} (${NUM_GPUS} available)"
echo "Scenes:  ${ALL_SCENE_LIST[*]} (${TOTAL_SCENES} total)"
echo ""

# 启动后台 monitor
monitor_progress "$TOTAL_SCENES" &
MONITOR_PID=$!

# cleanup on exit
cleanup() {
    rm -f "$STATUS_DIR/.running"
    sleep 1.5  # 让 monitor 最后刷新一次
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
    rm -rf "$STATUS_DIR"
    exec 3>&-
}
trap cleanup EXIT

# dispatch 所有场景
for s in "${ALL_SCENE_LIST[@]}"; do
    ds=$(find_dataset_for_scene "$s")
    if [ -z "$ds" ]; then
        echo "Error: scene '$s' not found in any dataset"
        exit 1
    fi
    dispatch_scene "$ds" "$s"
done

wait $(jobs -rp | grep -v "$MONITOR_PID")

# 停止 monitor
rm -f "$STATUS_DIR/.running"
sleep 1.5
kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null

echo ""
echo "All ${TOTAL_SCENES} scenes completed!"
