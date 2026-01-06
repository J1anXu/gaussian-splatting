import os
import shutil
import re
fold_name = "layer_0_block_3_contribution"
src_root = "/home/jian/gaussian-splatting/debug/fcgb_pro/_DSC8680.JPG"
dst_root = f"/home/jian/gaussian-splatting/debug/fcgb_pro/{fold_name}"

os.makedirs(dst_root, exist_ok=True)

iter_pattern = re.compile(r"iter_(\d+)")

for iter in sorted(os.listdir(src_root)):
    match = iter_pattern.fullmatch(iter)
    if not match:
        continue

    iter_id = int(match.group(1))
    if iter_id < 1657:
        continue

    src_dir = os.path.join(src_root, iter, "layer_contribution")
    if not os.path.isdir(src_dir):
        print(f"[WARN] Missing block_imgs in {src_dir}")
        continue

    src_img = os.path.join(src_dir, f"{fold_name}.png")
    if not os.path.isfile(src_img):
        print(f"[WARN] Missing {fold_name}.png in {src_img}")
        continue

    dst_img = os.path.join(dst_root, f"{iter}.png")
    shutil.copy2(src_img, dst_img)
    print(f"[OK] {src_img} -> {dst_img}")
