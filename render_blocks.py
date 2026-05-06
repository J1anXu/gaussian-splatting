#
# Per-block renderer — outputs RGB + colorized depth + colorized opacity
# and residual transmitance for each PLY block, per view. Raw scalar
# exports are also kept for downstream numeric use.
#
# Output layout:
#   {model_path}/rendered_blocks/{BRANCH}/{train|test}/ours_{iter}/
#       render_manifest.json     selected views + source PLY metadata
#       {view_name}/
#           rgb/block_{N}.png
#           depth/block_{N}.png
#           depth_raw/block_{N}.png
#           depth_raw/depth_meta.json   per-view {block_id: {min, max}} for depth PNG
#           opacity/block_{N}.png
#           opacity_raw/block_{N}.png
#           residual_transmitance/block_{N}.png
#           residual_transmitance_raw/block_{N}.png
# 

import os
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from matplotlib import colormaps
from PIL import Image
import torch
import torchvision
from tqdm import tqdm

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state, get_git_branch
from utils.camera_utils import frustum_culling
from utils.system_utils import searchForMaxIteration
from arguments import ModelParams, PipelineParams, get_combined_args

try:
    from diff_gaussian_rasterization_wenqi_tam import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

DEPTH_COLORMAP = "turbo"
OPACITY_COLORMAP = "magma"
RESIDUAL_TRANSMITANCE_COLORMAP = "viridis"
OUTPUT_SUBDIRS = {
    "rgb": "rgb",
    "depth": "depth",
    "depth_raw": "depth_raw",
    "opacity": "opacity",
    "opacity_raw": "opacity_raw",
    "residual_transmitance": "residual_transmitance",
    "residual_transmitance_raw": "residual_transmitance_raw",
}


def _prepare_view_dir(view_dir):
    view_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale flat outputs from earlier versions of the script.
    for pattern in (
        "block_*_rgb.png",
        "block_*_depth.png",
        "block_*_depth_raw.png",
        "block_*_opacity.png",
        "block_*_opacity_raw.png",
        "block_*_residual_transmitance.png",
        "block_*_residual_transmitance_raw.png",
    ):
        for stale_file in view_dir.glob(pattern):
            stale_file.unlink()
    flat_depth_meta = view_dir / "depth_meta.json"
    if flat_depth_meta.exists():
        flat_depth_meta.unlink()

    out_dirs = {}
    for key, name in OUTPUT_SUBDIRS.items():
        out_dir = view_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in out_dir.glob("block_*.png"):
            stale_file.unlink()
        out_dirs[key] = out_dir

    depth_meta_path = out_dirs["depth_raw"] / "depth_meta.json"
    if depth_meta_path.exists():
        depth_meta_path.unlink()

    return out_dirs, depth_meta_path


def _depth_stats(depth_1hw):
    """Return (d_min, d_max, valid_mask) on finite, positive pixels."""
    d = depth_1hw.squeeze(0)
    valid = torch.isfinite(d) & (d > 0)
    if not valid.any():
        return None, None, valid
    return float(d[valid].min().item()), float(d[valid].max().item()), valid


def _depth_to_u16(depth_1hw, d_min, d_max, valid_mask):
    """Convert [1,H,W] depth to 16-bit normalized grayscale."""
    d = depth_1hw.squeeze(0)
    if d_min is None or (d_max - d_min) < 1e-9:
        arr = np.zeros(d.shape, dtype=np.uint16)
    else:
        norm = torch.zeros_like(d)
        norm[valid_mask] = ((d[valid_mask] - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
        arr = (norm.detach().cpu().numpy() * 65535.0 + 0.5).astype(np.uint16)
    return arr


def _save_depth_png16(depth_1hw, d_min, d_max, valid_mask, out_path):
    """Save [1,H,W] depth as 16-bit grayscale PNG, per-block min-max normalized.
    Invalid pixels (non-finite or <=0) are set to 0.
    Recover raw depth with: depth = d_min + (png/65535) * (d_max - d_min)."""
    arr = _depth_to_u16(depth_1hw, d_min, d_max, valid_mask)
    Image.fromarray(arr, mode="I;16").save(out_path, optimize=True)
    return arr


def _to_uint8_image(scalar_2d):
    return (np.clip(scalar_2d, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _colorize_scalar(norm_scalar_2d, cmap_name, invert=False, invalid_mask=None,
                     alpha_2d=None):
    norm = np.clip(norm_scalar_2d, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    rgb_f = colormaps[cmap_name](norm)[..., :3]
    if alpha_2d is not None:
        a = np.clip(alpha_2d, 0.0, 1.0)[..., None]
        rgb_f = rgb_f * a + 1.0 * (1.0 - a)
    rgb = (rgb_f * 255.0 + 0.5).astype(np.uint8)
    if invalid_mask is not None:
        rgb[~invalid_mask] = 255
    return rgb


def _sample_views(views, num_views):
    total = len(views)
    if num_views is None or num_views <= 0 or total <= num_views:
        return views, list(range(total))

    indices = np.linspace(0, total - 1, num=num_views, dtype=int).tolist()
    indices = list(dict.fromkeys(indices))
    return [views[i] for i in indices], indices


def render_set(model_path, split_name, iteration, views, model_list, pipeline,
               background, train_test_exp, separate_sh, branch, ply_files,
               num_views=None):
    out_root = Path(model_path) / "rendered_blocks" / branch / split_name / f"ours_{iteration}"
    out_root.mkdir(parents=True, exist_ok=True)

    original_view_count = len(views)
    views, selected_indices = _sample_views(views, num_views)
    if len(views) != original_view_count:
        print(
            f"  sampling {len(views)} views from {original_view_count} "
            f"(indices {selected_indices})"
        )

    manifest = {
        "model_path": str(Path(model_path).resolve()),
        "branch": branch,
        "split": split_name,
        "iteration": int(iteration),
        "depth_colormap": DEPTH_COLORMAP,
        "opacity_colormap": OPACITY_COLORMAP,
        "residual_transmitance_colormap": RESIDUAL_TRANSMITANCE_COLORMAP,
        "output_subdirs": OUTPUT_SUBDIRS,
        "num_views_requested": None if num_views is None else int(num_views),
        "num_views_selected": len(views),
        "selected_view_indices": selected_indices,
        "selected_view_names": [view.image_name for view in views],
        "block_ply_files": [
            {"block_id": block_idx, "ply_file": ply_name}
            for block_idx, ply_name in enumerate(ply_files)
        ],
    }
    with open(out_root / "render_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for idx, view in enumerate(tqdm(views, desc=f"[{split_name}] views")):
        view_dir = out_root / view.image_name
        out_dirs, depth_meta_path = _prepare_view_dir(view_dir)
        depth_meta = {}

        for block_idx, model in enumerate(model_list):
            vis_mask = frustum_culling(model._xyz, view.full_proj_transform)
            subset_indices = torch.nonzero(vis_mask, as_tuple=True)[0]
            if subset_indices.numel() == 0:
                continue

            model.visible_indices = subset_indices
            model.activate_subset()
            render_pkg = render(
                view, model, pipeline, background,
                use_trained_exp=train_test_exp, separate_sh=separate_sh,
            )
            model.deactivate_subset()

            alpha_left = render_pkg["alphaLeft"]
            if int((alpha_left < (1.0 - 1e-6)).sum().item()) == 0:
                continue

            rgb = render_pkg["render"].clamp(0, 1)
            depth = render_pkg["depth"]
            opacity = (1.0 - alpha_left).clamp(0, 1)
            residual_transmitance = alpha_left.clamp(0, 1)

            # RGB
            torchvision.utils.save_image(rgb, str(out_dirs["rgb"] / f"block_{block_idx}.png"))
            # Depth: keep a raw 16-bit export and save a paper-friendly color map.
            d_min, d_max, valid = _depth_stats(depth)
            depth_u16 = _save_depth_png16(
                depth,
                d_min,
                d_max,
                valid,
                str(out_dirs["depth_raw"] / f"block_{block_idx}.png"),
            )
            depth_norm = depth_u16.astype(np.float32) / 65535.0
            depth_rgb = _colorize_scalar(
                depth_norm,
                DEPTH_COLORMAP,
                invert=True,
                invalid_mask=valid.detach().cpu().numpy(),
            )
            Image.fromarray(depth_rgb, mode="RGB").save(
                str(out_dirs["depth"] / f"block_{block_idx}.png"), optimize=True
            )
            depth_meta[str(block_idx)] = {"min": d_min, "max": d_max}
            # Opacity: keep raw grayscale and save a colorized heatmap.
            opacity_np = opacity.squeeze(0).detach().cpu().numpy()
            opacity_u8 = _to_uint8_image(opacity_np)
            Image.fromarray(opacity_u8, mode="L").save(
                str(out_dirs["opacity_raw"] / f"block_{block_idx}.png"), optimize=True
            )
            opacity_rgb = _colorize_scalar(opacity_np, OPACITY_COLORMAP, alpha_2d=opacity_np)
            Image.fromarray(opacity_rgb, mode="RGB").save(
                str(out_dirs["opacity"] / f"block_{block_idx}.png"), optimize=True
            )
            # Residual transmitance (alphaLeft): keep raw grayscale and save a colorized heatmap.
            residual_transmitance_np = residual_transmitance.squeeze(0).detach().cpu().numpy()
            residual_transmitance_u8 = _to_uint8_image(residual_transmitance_np)
            Image.fromarray(residual_transmitance_u8, mode="L").save(
                str(out_dirs["residual_transmitance_raw"] / f"block_{block_idx}.png"), optimize=True
            )
            residual_transmitance_rgb = _colorize_scalar(
                residual_transmitance_np,
                RESIDUAL_TRANSMITANCE_COLORMAP,
                alpha_2d=residual_transmitance_np,
            )
            Image.fromarray(residual_transmitance_rgb, mode="RGB").save(
                str(out_dirs["residual_transmitance"] / f"block_{block_idx}.png"), optimize=True
            )

        if depth_meta:
            with open(depth_meta_path, "w") as f:
                json.dump(depth_meta, f, indent=2)


def render_sets(dataset, iteration, pipeline, split, separate_sh, branch, num_views=None):
    with torch.no_grad():
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud", branch))
        scene = Scene(dataset, None, load_iteration=iteration, shuffle=False, only_camera=True)
        ply_dir = Path(scene.model_path) / "point_cloud" / branch / f"iteration_{scene.loaded_iter}"
        ply_files = sorted(
            f for f in os.listdir(ply_dir)
            if f.startswith("point_cloud_sub_") and f.endswith(".ply")
        )
        if not ply_files:
            raise FileNotFoundError(f"No point_cloud_sub_*.ply found under {ply_dir}")

        model_list = []
        total_pts = 0
        for fname in ply_files:
            m = GaussianModel(dataset.sh_degree)
            m.load_ply(str(ply_dir / fname), dataset.train_test_exp)
            model_list.append(m)
            total_pts += m._xyz.shape[0]
            print(f"  loaded {fname}: {m._xyz.shape[0]} gaussians")
        print(f"[{scene.model_path}] {len(model_list)} blocks, {total_pts} total gaussians")

        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

        if split in ("train", "both"):
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), model_list, pipeline, background,
                       dataset.train_test_exp, separate_sh, branch, ply_files,
                       num_views)
        if split in ("test", "both"):
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), model_list, pipeline, background,
                       dataset.train_test_exp, separate_sh, branch, ply_files,
                       num_views)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Per-block 3DGS renderer (RGB + depth + opacity + residual transmitance)"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="test",
        help="which split to render",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--git_branch", type=str, default=None,
                        help="branch subfolder under point_cloud/; defaults to current git branch")
    parser.add_argument(
        "--num_views",
        type=int,
        default=5,
        help="number of views to render per split; use <=0 for all views",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="deprecated alias of --num_views",
    )
    args_raw = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)

    branch = args_raw.git_branch if args_raw.git_branch is not None else get_git_branch()
    num_views = args_raw.max_views if args_raw.max_views is not None else args_raw.num_views
    print(
        f"Rendering {args.model_path} "
        f"(branch={branch}, iter={args.iteration}, split={args_raw.split}, num_views={num_views})"
    )

    safe_state(args.quiet)
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args_raw.split,
        SPARSE_ADAM_AVAILABLE,
        branch,
        num_views,
    )
    print("Done.")
