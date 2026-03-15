#!/bin/bash
set -e

MODEL=/home/jian/output/mip360/HEAD/bicycle
BRANCH=HEAD

export GIT_BRANCH_OVERRIDE=$BRANCH

echo "=== Rendering test set ==="
python render_p.py -m $MODEL --git_branch $BRANCH --skip_train

echo "=== Computing metrics ==="
python metrics_p.py -m $MODEL --git_branch $BRANCH

echo "=== Done ==="
cat $MODEL/rendered_p/$BRANCH/results.json 2>/dev/null || echo "(no results.json found)"
