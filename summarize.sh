#!/bin/sh
# Collect training metrics (Loss, pts, SSIM, PSNR, LPIPS) per scene from debug logs.
# Usage: sh summarize.sh [branch_name]
# Defaults to current git branch if not specified.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BRANCH="${1:-$(git -C "$SCRIPT_DIR" branch --show-current)}"
DEBUG_DIR="$SCRIPT_DIR/debug/$BRANCH"

echo "Branch: $BRANCH"
echo "Commit: $(git -C "$SCRIPT_DIR" log -1 --format='%h %s')"
echo

if [ ! -d "$DEBUG_DIR" ]; then
    echo "Directory not found: $DEBUG_DIR"
    exit 1
fi

W=15
scenes=""; losses=""; pts_list=""; ssims=""; psnrs=""; lpipss=""
count=0

for scene_dir in "$DEBUG_DIR"/*/; do
    scene=$(basename "$scene_dir")
    loss="N/A"; pts="N/A"; ssim="N/A"; psnr="N/A"; lpips="N/A"

    if [ -f "$scene_dir/train.log" ]; then
        last=$(grep -oP '100%.*?Loss=\K[0-9.]+' "$scene_dir/train.log" | tail -1)
        last_pts=$(grep -oP '100%.*?pts=\K[0-9]+' "$scene_dir/train.log" | tail -1)
        [ -n "$last" ] && loss="$last"
        [ -n "$last_pts" ] && pts="$last_pts"
    fi

    if [ -f "$scene_dir/metrics.log" ]; then
        ssim=$(grep -oP 'SSIM\s*:\s*\K[0-9.]+' "$scene_dir/metrics.log" | tail -1)
        psnr=$(grep -oP 'PSNR\s*:\s*\K[0-9.]+' "$scene_dir/metrics.log" | tail -1)
        lpips=$(grep -oP 'LPIPS\s*:\s*\K[0-9.]+' "$scene_dir/metrics.log" | tail -1)
        [ -z "$ssim" ] && ssim="N/A"
        [ -z "$psnr" ] && psnr="N/A"
        [ -z "$lpips" ] && lpips="N/A"
    fi

    scenes="$scenes $scene"; losses="$losses $loss"; pts_list="$pts_list $pts"
    ssims="$ssims $ssim"; psnrs="$psnrs $psnr"; lpipss="$lpipss $lpips"
    count=$((count + 1))
done

print_sep() {
    printf "+--------+"
    i=0; while [ $i -lt $count ]; do printf "%-${W}s+" "" | tr ' ' '-'; i=$((i+1)); done
    echo
}

print_row() {
    label="$1"; shift
    printf "| %-6s |" "$label"
    for val in $@; do printf " %-$((W-2))s|" "$val"; done
    echo
}

print_sep
print_row "Scene" $scenes
print_sep
print_row "Loss"  $losses
print_row "Pts"   $pts_list
print_row "SSIM"  $ssims
print_row "PSNR"  $psnrs
print_row "LPIPS" $lpipss
print_sep
