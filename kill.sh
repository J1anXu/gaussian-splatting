#!/usr/bin/env bash
echo "Killing all training/render/metrics processes..."
pkill -9 -f "train.py|render.py|metrics.py" 2>/dev/null || true
echo "Done."
