#!/usr/bin/env bash

echo "🔥 Killing all pipeline python processes..."

pkill -9 -f "train.py"
pkill -9 -f "render_p.py"
pkill -9 -f "metrics_p.py"

echo "🧹 Done."
