#!/usr/bin/env bash
set -u

BRANCH="nurips26_fix_drjohnson_B"
ITER=30000
MAX_VIEWS=5

# scene -> dataset subfolder under /data/jian/output/CapGS/
declare -A DATASET=(
  [bicycle]=mip360   [bonsai]=mip360   [counter]=mip360   [flowers]=mip360
  [garden]=mip360    [kitchen]=mip360  [room]=mip360      [stump]=mip360
  [treehill]=mip360
  [train]=tandt      [truck]=tandt
  [drjohnson]=deepblending  [playroom]=deepblending
)

SCENES=(bicycle bonsai counter flowers garden kitchen room stump treehill train truck drjohnson playroom)

source /home/jian/miniconda3/etc/profile.d/conda.sh
conda activate partgs_fix
export LD_LIBRARY_PATH=/home/jian/miniconda3/envs/partgs_fix/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

LOG_DIR="/home/jian/gaussian-splatting/logs/render_blocks"
mkdir -p "$LOG_DIR"

for scene in "${SCENES[@]}"; do
    ds="${DATASET[$scene]}"
    mp="/data/jian/output/CapGS/$ds/$BRANCH/$scene"
    if [ ! -d "$mp" ]; then
        echo "[skip] $scene: model dir not found ($mp)"
        continue
    fi
    out_dir="$mp/rendered_blocks/$BRANCH/test/ours_$ITER"
    if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[skip] $scene: already rendered at $out_dir"
        continue
    fi
    echo "=============================================="
    echo "[render] $scene  ($ds) -> $mp"
    echo "=============================================="
    python render_blocks.py \
        -m "$mp" \
        --git_branch "$BRANCH" \
        --iteration "$ITER" \
        --max_views "$MAX_VIEWS" \
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
