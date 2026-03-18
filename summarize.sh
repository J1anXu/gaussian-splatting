#!/bin/bash
# Summarize training results from debug/<branch>/<scene>/ directories.
# Usage: ./summarize.sh [branch1 branch2 ...]
#   No args = use current git branch name

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEBUG_DIR="$SCRIPT_DIR/debug"

parse_seconds() {
    # Convert "DD/MM HH:MM:SS" timestamp to seconds-since-midnight (with day offset)
    local ts="$1"
    local day="${ts%%/*}"
    local rest="${ts#*/}"
    local mon="${rest%% *}"
    local time="${rest#* }"
    local h="${time%%:*}"
    local ms="${time#*:}"
    local m="${ms%%:*}"
    local s="${ms#*:}"
    # Strip leading zeros
    day=$((10#$day)); h=$((10#$h)); m=$((10#$m)); s=$((10#$s))
    echo $(( day * 86400 + h * 3600 + m * 60 + s ))
}

format_duration() {
    local total=$1
    local h=$((total / 3600))
    local m=$(( (total % 3600) / 60 ))
    local s=$((total % 60))
    if [ "$h" -gt 0 ]; then
        printf "%dh %dm %ds" "$h" "$m" "$s"
    else
        printf "%dm %ds" "$m" "$s"
    fi
}

parse_train_tail() {
    # Parse train.log for duration and Pts
    # The log has very few lines but each line can be huge (progress bar),
    # so we use tail/head -c to grab small chunks and avoid processing huge lines.
    # Sets: _duration _pts
    local train_log="$1"
    _duration="" _pts=""
    [ ! -f "$train_log" ] && return

    # Last 2KB is enough to find final Pts and end timestamp
    local tail_block
    tail_block=$(tail -c 2048 "$train_log")
    # First 512B is enough for the start timestamp
    local head_block
    head_block=$(head -c 512 "$train_log")

    # Pts: find last occurrence of Pts=X.XXXM
    _pts=$(echo "$tail_block" | grep -oP 'Pts=\K[\d.]+M' | tail -1)

    # Timestamps
    local first_ts last_ts
    first_ts=$(echo "$head_block" | grep -oP '\[\K\d+/\d+ \d+:\d+:\d+(?=\])' | head -1)
    last_ts=$(echo "$tail_block" | grep -oP '\[\K\d+/\d+ \d+:\d+:\d+(?=\])' | tail -1)
    [ -z "$first_ts" ] || [ -z "$last_ts" ] && return

    local t0 t1 diff
    t0=$(parse_seconds "$first_ts")
    t1=$(parse_seconds "$last_ts")
    diff=$((t1 - t0))
    [ "$diff" -lt 0 ] && diff=$((diff + 86400))
    _duration="$diff"
}

print_branch() {
    local branch_dir="$1"
    local branch_name
    branch_name=$(basename "$branch_dir")

    local commit_id hostname_str
    commit_id=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
    hostname_str=$(hostname)

    local sep="-----------------------------------------------------------------------"
    printf "\n Server: %s\n" "$hostname_str"
    printf " Branch: %s\n" "$branch_name"
    printf " Commit: %s\n" "$commit_id"
    echo "$sep"
    printf "%-12s %8s %8s %8s %8s %12s\n" "Scene" "PSNR" "SSIM" "LPIPS" "Pts" "Time"
    echo "$sep"

    local sum_psnr=0 sum_ssim=0 sum_lpips=0 sum_time=0 count=0

    for scene_dir in "$branch_dir"/*/; do
        [ ! -d "$scene_dir" ] && continue
        local scene
        scene=$(basename "$scene_dir")

        local psnr="N/A" ssim="N/A" lpips="N/A" time_s="N/A"
        local metrics_log="$scene_dir/metrics.log"
        local train_log="$scene_dir/train.log"

        if [ -f "$metrics_log" ]; then
            psnr=$(grep -oP 'PSNR\s*:\s*\K[\d.]+' "$metrics_log" | head -1)
            ssim=$(grep -oP 'SSIM\s*:\s*\K[\d.]+' "$metrics_log" | head -1)
            lpips=$(grep -oP 'LPIPS\s*:\s*\K[\d.]+' "$metrics_log" | head -1)
            : "${psnr:=N/A}" "${ssim:=N/A}" "${lpips:=N/A}"
        fi

        local pts="N/A"
        parse_train_tail "$train_log"
        [ -n "$_pts" ] && pts="$_pts"

        local dur="$_duration"
        if [ -n "$dur" ]; then
            time_s=$(format_duration "$dur")
            sum_time=$((sum_time + dur))
        fi

        if [ "$psnr" != "N/A" ]; then
            sum_psnr=$(awk "BEGIN{print $sum_psnr + $psnr}")
            sum_ssim=$(awk "BEGIN{print $sum_ssim + $ssim}")
            sum_lpips=$(awk "BEGIN{print $sum_lpips + $lpips}")
            count=$((count + 1))
            printf "%-12s %8.4f %8.4f %8.4f %8s %12s\n" "$scene" "$psnr" "$ssim" "$lpips" "$pts" "$time_s"
        else
            printf "%-12s %8s %8s %8s %8s %12s\n" "$scene" "$psnr" "$ssim" "$lpips" "$pts" "$time_s"
        fi
    done

    echo "$sep"
}

# Main
if [ $# -gt 0 ]; then
    branches=("$@")
else
    current_branch=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ -n "$current_branch" ] && [ -d "$DEBUG_DIR/$current_branch" ]; then
        branches=("$current_branch")
    else
        echo "No results found for current branch '$current_branch' under debug/"
        exit 1
    fi
fi

for branch in "${branches[@]}"; do
    branch_dir="$DEBUG_DIR/$branch"
    if [ ! -d "$branch_dir" ]; then
        echo "Warning: $branch_dir not found, skipping."
        continue
    fi
    print_branch "$branch_dir"
done
