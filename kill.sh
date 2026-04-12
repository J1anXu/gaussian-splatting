#!/usr/bin/env bash

echo "Killing all pipeline processes..."

# Kill the pipeline launcher first (prevents it from spawning new jobs)
pkill -9 -f "runall_4090.sh" 2>/dev/null
pkill -9 -f "runall.*\.sh" 2>/dev/null

# Then kill the workers
pkill -9 -f "train.py" 2>/dev/null
pkill -9 -f "render_p.py" 2>/dev/null
pkill -9 -f "metrics_p.py" 2>/dev/null

echo "Done."
