#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="${BRANCH:-$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo no_git)}"
DEFAULT_ENV_PY="/home/jian/miniconda3/envs/partgs_fix/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_ENV_PY" ]]; then
    PYTHON_BIN="$DEFAULT_ENV_PY"
  else
    PYTHON_BIN="python3"
  fi
fi
if [[ "$PYTHON_BIN" == "$DEFAULT_ENV_PY" ]]; then
  export LD_LIBRARY_PATH="/home/jian/miniconda3/envs/partgs_fix/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
fi

SCENE="bicycle"
DATASET="mip360"
SOURCE_PATH="${SOURCE_PATH:-/data/jian/data/${DATASET}/${SCENE}}"
MODEL_PATH="${MODEL_PATH:-/data/jian/output/CapGS/${DATASET}/${BRANCH}/${SCENE}}"
IMAGES_FLAG="${IMAGES_FLAG:-images_4}"
SPLIT="${SPLIT:-test}"
NUM_VIEWS="${NUM_VIEWS:-20}"
ITER_ARG="${ITER_ARG:--1}"

POINT_CLOUD_ROOT="${MODEL_PATH}/point_cloud/${BRANCH}"
LATEST_ITER=""
if [[ -d "$POINT_CLOUD_ROOT" ]]; then
  LATEST_ITER="$(find "$POINT_CLOUD_ROOT" -maxdepth 1 -type d -name 'iteration_*' | sed 's#.*iteration_##' | sort -n | tail -n 1)"
fi

echo "Scene      : ${SCENE}"
echo "Branch     : ${BRANCH}"
echo "Source     : ${SOURCE_PATH}"
echo "Model      : ${MODEL_PATH}"
echo "PointCloud : ${POINT_CLOUD_ROOT}"
if [[ -n "$LATEST_ITER" ]]; then
  echo "Latest iter: ${LATEST_ITER}"
  echo "PLY dir    : ${POINT_CLOUD_ROOT}/iteration_${LATEST_ITER}"
fi
echo "Split      : ${SPLIT}"
echo "Views      : ${NUM_VIEWS}"
echo ""

"$PYTHON_BIN" "$REPO_DIR/render_blocks.py" \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  -i "$IMAGES_FLAG" \
  --eval \
  --git_branch "$BRANCH" \
  --iteration "$ITER_ARG" \
  --split "$SPLIT" \
  --num_views "$NUM_VIEWS" \
  "$@"
