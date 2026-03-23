#!/bin/bash
# 一键画图: 自动检测日志类型 → 提取CSV → 画图
#
# 用法:
#   plot/plot.sh <log1> [log2] ...
#
# 示例:
#   plot/plot.sh logs/train/nurips26/bicycle/0323_1427.log
#   plot/plot.sh logs/.../capgs.log /data/.../gsscale.log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEBUG_DIR="$(dirname "$SCRIPT_DIR")/debug"
mkdir -p "$DEBUG_DIR"

if [ $# -eq 0 ]; then
    echo "用法: plot/plot.sh <log1> [log2] ..."
    exit 1
fi

CSV_LIST=()

for LOG in "$@"; do
    if [ ! -f "$LOG" ]; then
        echo "Error: $LOG not found"
        exit 1
    fi

    # 自动检测类型: CapGS 日志用 "MMDD,HH:MM - {" 格式, GS-Scale 用 "| step=" 格式
    if head -20 "$LOG" | grep -q "| step="; then
        TYPE="gsscale"
    elif head -20 "$LOG" | grep -q " - {"; then
        TYPE="capgs"
    else
        echo "Error: cannot detect log type for $LOG"
        exit 1
    fi

    # 从路径推断场景名
    PARENT_DIR="$(basename "$(dirname "$LOG")")"
    GRANDPARENT_DIR="$(basename "$(dirname "$(dirname "$LOG")")")"
    if [ "$PARENT_DIR" = "logs" ]; then
        # gsscale: .../bicycle/logs/xxx.log → 取 grandparent
        SCENE="$GRANDPARENT_DIR"
    elif [ "$PARENT_DIR" = "." ]; then
        SCENE="$(basename "$LOG" .log)"
    else
        # capgs: .../bicycle/xxx.log → 取 parent
        SCENE="$PARENT_DIR"
    fi
    NAME="${TYPE}_${SCENE}"

    echo "[$NAME] extracting ($TYPE) ..."
    python3 "$SCRIPT_DIR/extract.py" --type "$TYPE" "$LOG" --name "$NAME"

    CSV_LIST+=("$DEBUG_DIR/${NAME}_metrics.csv")
done

echo ""
echo "Plotting ..."
python3 "$SCRIPT_DIR/plot_mem.py" "${CSV_LIST[@]}"
