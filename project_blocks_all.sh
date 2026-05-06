#!/usr/bin/env bash
set -u

BRANCH="${BRANCH:-nurips26_fix_drjohnson_B}"
ITER="${ITER:-30000}"
NUM_VIEWS="${NUM_VIEWS:-5}"
MODE="${MODE:-per_block_render}"   # per_block_render | per_block_full_bg | gt_mono | gt_blocks | white_blocks

declare -A DATASET=(
  [bicycle]=mip360   [bonsai]=mip360   [counter]=mip360   [flowers]=mip360
  [garden]=mip360    [kitchen]=mip360  [room]=mip360      [stump]=mip360
  [treehill]=mip360
  [train]=tandt      [truck]=tandt
  [drjohnson]=deepblending  [playroom]=deepblending
)

declare -A DATA_ROOT=(
  [mip360]=/data/jian/data/mip360
  [tandt]=/data/jian/data/tandt
  [deepblending]=/data/jian/data/deepblending
)

declare -A IMAGES_FLAG=(
  [bicycle]=images_4 [flowers]=images_4 [garden]=images_4
  [stump]=images_4   [treehill]=images_4
  [bonsai]=images_2  [counter]=images_2 [kitchen]=images_2 [room]=images_2
  [train]=images     [truck]=images
  [drjohnson]=images [playroom]=images
)

SCENES=(bicycle bonsai counter flowers garden kitchen room stump treehill train truck drjohnson playroom)

case "$MODE" in
  per_block_render)  MODE_FLAGS="--per_block_render";              SUFFIX="per_block_render" ;;
  per_block_full_bg) MODE_FLAGS="--per_block_full_bg";             SUFFIX="per_block_full_bg" ;;
  gt_mono)           MODE_FLAGS="--bg gt";                         SUFFIX="gt_mono" ;;
  gt_blocks)         MODE_FLAGS="--bg gt --per_block_color";       SUFFIX="gt_blocks" ;;
  white_blocks)      MODE_FLAGS="--bg white --per_block_color";    SUFFIX="white_blocks" ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac

source /home/jian/miniconda3/etc/profile.d/conda.sh
conda activate partgs_fix
export LD_LIBRARY_PATH=/home/jian/miniconda3/envs/partgs_fix/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

LOG_DIR="/home/jian/gaussian-splatting/logs/project_blocks"
mkdir -p "$LOG_DIR"

for scene in "${SCENES[@]}"; do
    ds="${DATASET[$scene]}"
    src="${DATA_ROOT[$ds]}/$scene"
    mp="/data/jian/output/CapGS/$ds/$BRANCH/$scene"
    img_flag="${IMAGES_FLAG[$scene]}"

    if [ ! -d "$mp" ]; then
        echo "[skip] $scene: model dir not found ($mp)"
        continue
    fi
    if [ ! -d "$src" ]; then
        echo "[skip] $scene: source dir not found ($src)"
        continue
    fi

    out_dir="$mp/projected_blocks/$BRANCH/test/ours_${ITER}_${SUFFIX}"
    if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir" 2>/dev/null | wc -l)" -gt 1 ]; then
        echo "[skip] $scene: already done at $out_dir"
        continue
    fi

    echo "=============================================="
    echo "[project] $scene  ($ds, $img_flag) -> $out_dir"
    echo "=============================================="
    python project_blocks.py \
        -s "$src" \
        -m "$mp" \
        -i "$img_flag" \
        --eval \
        --git_branch "$BRANCH" \
        --iteration "$ITER" \
        --split test \
        --num_views "$NUM_VIEWS" \
        $MODE_FLAGS \
        --quiet \
        > "$LOG_DIR/${scene}.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] $scene (rc=$rc), see $LOG_DIR/${scene}.log"
    else
        echo "[done] $scene"
    fi
done

echo "All done."
