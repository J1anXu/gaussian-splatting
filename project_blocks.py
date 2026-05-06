#
# Project per-block Gaussian centers onto GT images, color-coded by block.
# Produces one composite PNG per view + a color legend.
#
# Output layout:
#   {model_path}/projected_blocks/{BRANCH}/{train|test}/ours_{iter}/
#       project_manifest.json
#       legend.png
#       {view_name}.png      GT with all blocks' points overlaid
#

import os
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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


POINTS_PER_BLOCK = 100000
DOT_SIZE = 50.0          # base "core" size; glow halo extends ~2x this radius
DOT_ALPHA = 0.55          # core alpha; halo layers use lower alpha
DOT_COLOR = "#ed8936"

# Glow layers: (size_multiplier, alpha_multiplier) — drawn outermost-first
GLOW_LAYERS = [(4.0, 0.18), (2.0, 0.45), (1.0, 1.0)]


def _scatter_glow(ax, u, v, color):
    for size_mul, alpha_mul in GLOW_LAYERS:
        ax.scatter(u, v, s=DOT_SIZE * size_mul, c=[color],
                   alpha=DOT_ALPHA * alpha_mul, linewidths=0, marker="o")


def _project(xyz, M, W, H):
    """xyz: [N,3] cuda, M: [4,4] cuda. Returns (u, v) numpy arrays of in-frame points."""
    with torch.no_grad():
        clip = xyz.matmul(M[:3]) + M[3]
        x, y, z, w = clip.unbind(1)
        valid = (w > 0) & (z >= 0)
        if not valid.any():
            return np.empty(0), np.empty(0)
        x = x[valid] / w[valid]
        y = y[valid] / w[valid]
        u = (x + 1.0) * 0.5 * W
        v = (y + 1.0) * 0.5 * H
        in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[in_frame].detach().cpu().numpy()
        v = v[in_frame].detach().cpu().numpy()
        return u, v


def _subsample(xyz, k):
    n = xyz.shape[0]
    if k <= 0 or n <= k:
        return xyz
    idx = torch.randperm(n, device=xyz.device)[:k]
    return xyz[idx]


def _sample_views(views, num_views):
    total = len(views)
    if num_views is None or num_views <= 0 or total <= num_views:
        return views, list(range(total))
    indices = np.linspace(0, total - 1, num=num_views, dtype=int).tolist()
    indices = list(dict.fromkeys(indices))
    return [views[i] for i in indices], indices


def _block_colors(n):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def _render_overlay(view, model_list, W, H, out_path,
                    bg_image=None, per_block_color=False):
    M = view.full_proj_transform
    colors = _block_colors(len(model_list)) if per_block_color else None

    blocks_uv = []
    for block_idx, model in enumerate(model_list):
        xyz = _subsample(model._xyz, POINTS_PER_BLOCK)
        u, v = _project(xyz, M, W, H)
        if u.size:
            blocks_uv.append((block_idx, u, v))
    if not blocks_uv:
        return

    dpi = 100
    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    if bg_image is not None:
        ax.imshow(bg_image)
    else:
        ax.set_facecolor("white")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for block_idx, u, v in blocks_uv:
        c = colors[block_idx] if per_block_color else DOT_COLOR
        _scatter_glow(ax, u, v, c)

    fig.savefig(out_path, dpi=dpi)


def _merge_models(model_list):
    """Concatenate per-block GaussianModels into a single in-memory model for full-scene render."""
    merged = GaussianModel(model_list[0].max_sh_degree)
    merged.active_sh_degree = model_list[0].active_sh_degree
    merged._xyz = torch.cat([m._xyz for m in model_list], dim=0)
    merged._features_dc = torch.cat([m._features_dc for m in model_list], dim=0)
    merged._features_rest = torch.cat([m._features_rest for m in model_list], dim=0)
    merged._scaling = torch.cat([m._scaling for m in model_list], dim=0)
    merged._rotation = torch.cat([m._rotation for m in model_list], dim=0)
    merged._opacity = torch.cat([m._opacity for m in model_list], dim=0)
    merged.visible_indices = None
    merged.subset_mode_1 = False
    merged.subset_mode_2 = False
    return merged


def _render_full_white(view, merged_model, pipeline, train_test_exp, separate_sh):
    """Render full scene on white. Returns HWC float [H,W,3] in [0,1]."""
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    vis_mask = frustum_culling(merged_model._xyz, view.full_proj_transform)
    subset_indices = torch.nonzero(vis_mask, as_tuple=True)[0]
    if subset_indices.numel() == 0:
        H, W = view.image_height, view.image_width
        return np.ones((H, W, 3), dtype=np.float32)
    merged_model.visible_indices = subset_indices
    merged_model.activate_subset()
    pkg = render(view, merged_model, pipeline, background,
                 use_trained_exp=train_test_exp, separate_sh=separate_sh)
    merged_model.deactivate_subset()
    rgb = pkg["render"].clamp(0, 1).detach().cpu().numpy()
    return np.transpose(rgb, (1, 2, 0))


def _render_block_white(view, model, pipeline, train_test_exp, separate_sh):
    """Render a single block on a white background. Returns HWC float [H,W,3] in [0,1]."""
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    vis_mask = frustum_culling(model._xyz, view.full_proj_transform)
    subset_indices = torch.nonzero(vis_mask, as_tuple=True)[0]
    if subset_indices.numel() == 0:
        H, W = view.image_height, view.image_width
        return np.ones((H, W, 3), dtype=np.float32)
    model.visible_indices = subset_indices
    model.activate_subset()
    pkg = render(view, model, pipeline, background,
                 use_trained_exp=train_test_exp, separate_sh=separate_sh)
    model.deactivate_subset()
    rgb = pkg["render"].clamp(0, 1).detach().cpu().numpy()
    return np.transpose(rgb, (1, 2, 0))


def project_set_per_block(model_path, split_name, iteration, views, model_list,
                          pipeline, train_test_exp, separate_sh,
                          branch, ply_files, num_views=None,
                          full_bg=False):
    """If full_bg=True, background is full-scene render; else per-block render."""
    suffix = "per_block_full_bg" if full_bg else "per_block_render"
    out_root = Path(model_path) / "projected_blocks" / branch / split_name / f"ours_{iteration}_{suffix}"
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
        "mode": suffix,
        "points_per_block": POINTS_PER_BLOCK,
        "dot_size": DOT_SIZE,
        "dot_alpha": DOT_ALPHA,
        "dot_color": DOT_COLOR,
        "num_views_requested": None if num_views is None else int(num_views),
        "num_views_selected": len(views),
        "selected_view_indices": selected_indices,
        "selected_view_names": [v.image_name for v in views],
        "block_ply_files": [
            {"block_id": i, "ply_file": p}
            for i, p in enumerate(ply_files)
        ],
    }
    with open(out_root / "project_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    merged_model = _merge_models(model_list) if full_bg else None

    dpi = 100
    for view in tqdm(views, desc=f"[{split_name}] views"):
        view_dir = out_root / view.image_name
        view_dir.mkdir(parents=True, exist_ok=True)
        H, W = view.image_height, view.image_width
        M = view.full_proj_transform

        full_bg_img = None
        if full_bg:
            full_bg_img = _render_full_white(view, merged_model, pipeline,
                                             train_test_exp, separate_sh)

        for block_idx, model in enumerate(model_list):
            xyz = _subsample(model._xyz, POINTS_PER_BLOCK)
            u, v = _project(xyz, M, W, H)
            if u.size == 0:
                continue
            if full_bg:
                bg_img = full_bg_img
            else:
                bg_img = _render_block_white(view, model, pipeline,
                                             train_test_exp, separate_sh)

            fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(bg_img)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")
            _scatter_glow(ax, u, v, DOT_COLOR)
            fig.savefig(str(view_dir / f"block_{block_idx}.png"), dpi=dpi)


def project_set(model_path, split_name, iteration, views, model_list,
                branch, ply_files, num_views=None, bg="gt", per_block_color=False):
    suffix = f"{bg}_{'blocks' if per_block_color else 'mono'}"
    out_root = Path(model_path) / "projected_blocks" / branch / split_name / f"ours_{iteration}_{suffix}"
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
        "bg": bg,
        "per_block_color": per_block_color,
        "points_per_block": POINTS_PER_BLOCK,
        "dot_size": DOT_SIZE,
        "dot_alpha": DOT_ALPHA,
        "dot_color": DOT_COLOR,
        "num_views_requested": None if num_views is None else int(num_views),
        "num_views_selected": len(views),
        "selected_view_indices": selected_indices,
        "selected_view_names": [v.image_name for v in views],
        "block_ply_files": [
            {"block_id": i, "ply_file": p}
            for i, p in enumerate(ply_files)
        ],
    }
    with open(out_root / "project_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for view in tqdm(views, desc=f"[{split_name}] views"):
        if bg == "gt":
            gt = view.original_image.detach().cpu().numpy()
            bg_img = np.transpose(np.clip(gt, 0.0, 1.0), (1, 2, 0))
            H, W = bg_img.shape[:2]
        else:
            bg_img = None
            H, W = view.image_height, view.image_width
        _render_overlay(view, model_list, W, H,
                        str(out_root / f"{view.image_name}.png"),
                        bg_image=bg_img, per_block_color=per_block_color)


def project_sets(dataset, iteration, pipeline, split, branch, num_views=None,
                 bg="gt", per_block_color=False, per_block_render=False,
                 per_block_full_bg=False, separate_sh=False):
    with torch.no_grad():
        if iteration == -1:
            iteration = searchForMaxIteration(
                os.path.join(dataset.model_path, "point_cloud", branch)
            )
        scene = Scene(dataset, None, load_iteration=iteration, shuffle=False, only_camera=True)
        ply_dir = Path(scene.model_path) / "point_cloud" / branch / f"iteration_{scene.loaded_iter}"
        ply_files = sorted(
            f for f in os.listdir(ply_dir)
            if f.startswith("point_cloud_sub_") and f.endswith(".ply")
        )
        if not ply_files:
            raise FileNotFoundError(f"No point_cloud_sub_*.ply found under {ply_dir}")

        model_list = []
        for fname in ply_files:
            m = GaussianModel(dataset.sh_degree)
            m.load_ply(str(ply_dir / fname), dataset.train_test_exp)
            model_list.append(m)
            print(f"  loaded {fname}: {m._xyz.shape[0]} gaussians")
        print(f"[{scene.model_path}] {len(model_list)} blocks")

        def _do(split_name, cams):
            if per_block_render or per_block_full_bg:
                project_set_per_block(dataset.model_path, split_name, scene.loaded_iter,
                                      cams, model_list, pipeline,
                                      dataset.train_test_exp, separate_sh,
                                      branch, ply_files, num_views,
                                      full_bg=per_block_full_bg)
            else:
                project_set(dataset.model_path, split_name, scene.loaded_iter,
                            cams, model_list, branch, ply_files,
                            num_views, bg, per_block_color)

        if split in ("train", "both"):
            _do("train", scene.getTrainCameras())
        if split in ("test", "both"):
            _do("test", scene.getTestCameras())


if __name__ == "__main__":
    parser = ArgumentParser(description="Project per-block Gaussian centers onto GT images")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="test")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--git_branch", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=5,
                        help="number of views per split; <=0 for all")
    parser.add_argument("--bg", choices=["gt", "white"], default="gt",
                        help="background: GT image or pure white")
    parser.add_argument("--per_block_color", action="store_true",
                        help="color points by block (tab10/tab20) instead of single color")
    parser.add_argument("--per_block_render", action="store_true",
                        help="render each block on white bg + overlay its own gaussian centers; "
                             "outputs one PNG per (view, block). Overrides --bg/--per_block_color.")
    parser.add_argument("--per_block_full_bg", action="store_true",
                        help="background = full-scene render (all blocks, white bg); "
                             "overlay = current block's own centers. One PNG per (view, block).")
    args_raw = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)

    branch = args_raw.git_branch if args_raw.git_branch is not None else get_git_branch()
    print(
        f"Projecting {args.model_path} "
        f"(branch={branch}, iter={args.iteration}, split={args_raw.split}, "
        f"num_views={args_raw.num_views})"
    )

    safe_state(args.quiet)
    project_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args_raw.split,
        branch,
        args_raw.num_views,
        args_raw.bg,
        args_raw.per_block_color,
        args_raw.per_block_render,
        args_raw.per_block_full_bg,
        SPARSE_ADAM_AVAILABLE,
    )
    print("Done.")
