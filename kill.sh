#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔥 Killing processes running from ${PROJECT_DIR} ..."

# Kill runall daemons first (these use the full path)
pkill -9 -f "${PROJECT_DIR}/runall" 2>/dev/null

# Kill python workers whose working directory is this project
for pid in $(pgrep -f "train\.py|render_p\.py|metrics_p\.py"); do
    cwd="$(readlink /proc/$pid/cwd 2>/dev/null)"
    if [[ "$cwd" == "$PROJECT_DIR" ]]; then
        kill -9 "$pid" 2>/dev/null
    fi
done

echo "🧹 Done."
