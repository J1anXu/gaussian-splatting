#!/bin/bash
# Summarize results for the current git branch

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BRANCH=$(git -C "$SCRIPT_DIR" branch --show-current 2>/dev/null || echo "unknown")
DEBUG_DIR="$SCRIPT_DIR/debug/$BRANCH"

if [ ! -d "$DEBUG_DIR" ]; then
    echo "Error: $DEBUG_DIR not found"
    exit 1
fi

echo "Branch : $BRANCH"
echo "Dir    : $DEBUG_DIR"
echo ""

printf "%-12s  %10s  %8s  %8s  %8s\n" "Scene" "Points" "PSNR" "SSIM" "LPIPS"
printf "%-12s  %10s  %8s  %8s  %8s\n" "------------" "----------" "--------" "--------" "--------"

psnr_sum=0; ssim_sum=0; lpips_sum=0; cnt=0

for scene_dir in $(ls -d "$DEBUG_DIR"/*/ 2>/dev/null | sort); do
    scene=$(basename "$scene_dir")
    train_log="$scene_dir/train.log"
    metrics_log="$scene_dir/metrics.log"

    # Points: prefer wandb summary "wandb:        pts 4878277", fallback to tqdm "pts=XXXXX"
    pts=$(grep -oP "wandb:\s+pts\s+\K[0-9]+" "$train_log" 2>/dev/null | tail -1)
    if [ -z "$pts" ]; then
        pts=$(grep -oP "pts=\K[0-9]+" "$train_log" 2>/dev/null | tail -1)
    fi
    pts="${pts:-N/A}"

    # Metrics from last few lines of metrics.log
    psnr=$(grep -oP "PSNR\s*:\s*\K[0-9.]+" "$metrics_log" 2>/dev/null | tail -1)
    ssim=$(grep -oP "SSIM\s*:\s*\K[0-9.]+" "$metrics_log" 2>/dev/null | tail -1)
    lpips=$(grep -oP "LPIPS\s*:\s*\K[0-9.]+" "$metrics_log" 2>/dev/null | tail -1)
    psnr="${psnr:-N/A}"; ssim="${ssim:-N/A}"; lpips="${lpips:-N/A}"

    printf "%-12s  %10s  %8s  %8s  %8s\n" "$scene" "$pts" "$psnr" "$ssim" "$lpips"

    if [[ "$psnr" != "N/A" && "$ssim" != "N/A" && "$lpips" != "N/A" ]]; then
        psnr_sum=$(awk "BEGIN{print $psnr_sum + $psnr}")
        ssim_sum=$(awk "BEGIN{print $ssim_sum + $ssim}")
        lpips_sum=$(awk "BEGIN{print $lpips_sum + $lpips}")
        ((cnt++))
    fi
done

if [ $cnt -gt 0 ]; then
    avg_psnr=$(awk "BEGIN{printf \"%.4f\", $psnr_sum / $cnt}")
    avg_ssim=$(awk "BEGIN{printf \"%.4f\", $ssim_sum / $cnt}")
    avg_lpips=$(awk "BEGIN{printf \"%.4f\", $lpips_sum / $cnt}")
    printf "%-12s  %10s  %8s  %8s  %8s\n" "------------" "----------" "--------" "--------" "--------"
    printf "%-12s  %10s  %8s  %8s  %8s\n" "Avg($cnt)" "-" "$avg_psnr" "$avg_ssim" "$avg_lpips"
fi
