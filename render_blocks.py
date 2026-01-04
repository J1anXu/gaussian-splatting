#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render,merge_opt
import torchvision
from utils.general_utils import safe_state, get_git_branch
from utils.camera_utils import frustum_culling
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import config
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
BRANCH = "unknown_branch"

from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import random

def block_color(block_id):
    import random
    random.seed(int(block_id))
    return tuple(int(c) for c in (
        random.randint(64, 255),
        random.randint(64, 255),
        random.randint(64, 255),
    ))


AABB_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]
def projected_bbox(pts_2d, H, W):
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    xmin = int(torch.clamp(x.min(), 0, W - 1))
    xmax = int(torch.clamp(x.max(), 0, W - 1))
    ymin = int(torch.clamp(y.min(), 0, H - 1))
    ymax = int(torch.clamp(y.max(), 0, H - 1))

    if xmin >= xmax or ymin >= ymax:
        return None

    return xmin, ymin, xmax, ymax
def overlay_rect(img, rect, color, alpha=0.3):
    """
    img: [3,H,W] torch, 0~1
    rect: xmin,ymin,xmax,ymax
    color: (r,g,b) in 0~1
    """
    xmin, ymin, xmax, ymax = rect

    overlay = img.clone()
    color_t = torch.tensor( color, device=img.device, dtype=img.dtype ).view(3, 1, 1)
    overlay[:, ymin:ymax, xmin:xmax] = ( (1 - alpha) * overlay[:, ymin:ymax, xmin:xmax] + alpha * color_t )
    return overlay
import torch
import torchvision.ops as ops
def sort_points_clockwise(pts):
    """
    pts: [N,2] torch tensor
    return: [N,2] sorted clockwise
    """
    center = pts.mean(dim=0)
    angles = torch.atan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = torch.argsort(angles)
    return pts[order]

def overlay_projected_polygon(img, pts_2d, color, alpha=0.3):
    """
    img: [3,H,W] cuda tensor
    pts_2d: [8,2] projected points (float)
    """
    H, W = img.shape[1:]
    device = img.device

    # clamp to image
    pts = torch.as_tensor(
        pts_2d,
        device=img.device,
        dtype=img.dtype
    ).clone()

    pts[:, 0].clamp_(0, W - 1)
    pts[:, 1].clamp_(0, H - 1)


    # convex hull (torchvision)
    hull = sort_points_clockwise(pts)   # [8,2]


    # rasterize polygon to mask (CPU is OK for debug)
    mask = torch.zeros((H, W), device=device)

    from skimage.draw import polygon
    rr, cc = polygon(
        hull[:, 1].cpu().numpy(),
        hull[:, 0].cpu().numpy(),
        shape=(H, W)
    )
    mask[rr, cc] = 1.0

    color_t = torch.tensor(color, device=device).view(3, 1, 1)
    return img * (1 - alpha * mask) + color_t * (alpha * mask)


def draw_blocks_on_image(img_tensor, blocks_px, color=(255,0,0)):
    """
    img_tensor: [3,H,W], torch
    blocks_px: list of [8,2] pixel coords
    """
    img = TF.to_pil_image(img_tensor.cpu())
    draw = ImageDraw.Draw(img)
    idx = 0
    for pts in blocks_px:
        for i, j in AABB_EDGES:
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            draw.line((x1, y1, x2, y2), fill=color[idx], width=2)
        idx += 1

    return TF.to_tensor(img)

def block_visible(block_idx, block_indices, frustum_mask):
    idx = block_indices[block_idx]
    return frustum_mask[idx].any()

def aabb_corners(mn, mx):
    # mn, mx: Tensor[3]
    return torch.stack([
        torch.tensor([mn[0], mn[1], mn[2]]),
        torch.tensor([mx[0], mn[1], mn[2]]),
        torch.tensor([mx[0], mx[1], mn[2]]),
        torch.tensor([mn[0], mx[1], mn[2]]),
        torch.tensor([mn[0], mn[1], mx[2]]),
        torch.tensor([mx[0], mn[1], mx[2]]),
        torch.tensor([mx[0], mx[1], mx[2]]),
        torch.tensor([mn[0], mx[1], mx[2]]),
    ], dim=0)  # [8,3]
    
def project_points(xyz, full_proj, H, W):
    """
    xyz: [N,3] world coords (CPU tensor)
    full_proj: [4,4] torch tensor
    return: [N,2] pixel coords
    """
    N = xyz.shape[0]
    ones = torch.ones((N, 1), device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)  # [N,4]

    clip = (full_proj @ xyz_h.T).T          # [N,4]
    ndc = clip[:, :3] / clip[:, 3:4]        # [-1,1]

    x = (ndc[:, 0] * 0.5 + 0.5) * W
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H

    return torch.stack([x, y], dim=1)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    BRANCH = get_git_branch()

    render_path = os.path.join(model_path, BRANCH, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, BRANCH, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    debug_path = os.path.join("debug", BRANCH)
    os.makedirs(debug_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        available_mask = frustum_culling(gaussians._xyz, view.full_proj_transform)
        
        rendered_list, depth_list, alpha_list = [], [], []
        viewspace_points_list, visibility_filter_list, radii_list = [], [], []
        visible_indices_list = []
        available_num = 0
        
        
        
        for idx in range(len(gaussians.block_indices)):
            block_indice = gaussians.block_indices[idx].to("cuda")
            visible_mask_in_block = available_mask[block_indice]
            visible_indices = block_indice[visible_mask_in_block]
            if visible_indices.shape[0] == 0:
                continue
            available_num += visible_indices.shape[0]
            gaussians.set_subset(visible_indices)
            render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii, alphaLeft = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["alphaLeft"]
            
            rendered_list.append(image)
            depth_list.append(render_pkg["depth"])
            alpha_list.append(alphaLeft)
            viewspace_points_list.append(viewspace_point_tensor)
            visibility_filter_list.append(visibility_filter)
            radii_list.append(radii)
            visible_indices_list.append(visible_indices)
            gaussians.clear_subset()

        N_total = gaussians._xyz.shape[0]       
        merge_res = merge_opt(N_total, rendered_list, depth_list, alpha_list, visibility_filter_list, radii_list, visible_indices_list)   
        
        image, visibility_filter, radii = merge_res["final_rgb"], merge_res["global_visibility_filter"], merge_res["global_radii"]

        if view.alpha_mask is not None:
            alpha_mask = view.alpha_mask.cuda()
            image *= alpha_mask
        
        gt = view.original_image[0:3, :, :]
        if args.train_test_exp:
            rendering = image[..., image.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        img_name = view.image_name
        torchvision.utils.save_image(image, os.path.join(render_path, img_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, img_name + ".png"))
        
        if config.DRAW_BLOCK:
            H, W = image.shape[1:]
            frustum_mask_cpu = available_mask.detach().cpu()
            for block_id, (mn, mx) in enumerate(gaussians.block_bounds):
                
                # 跳过 frustum 外 block
                if not frustum_mask_cpu[gaussians.block_indices[block_id]].any():
                    continue

                corners = aabb_corners(mn.cpu(), mx.cpu())
                pts_2d = project_points(
                    corners,
                    view.full_proj_transform.cpu(),
                    H, W
                )

                rect = projected_bbox(pts_2d, H, W)
                if rect is None:
                    continue

                # 稳定颜色（0~1）
                random.seed(block_id)
                color = [random.random() for _ in range(3)]

                overlay = overlay_projected_polygon(
                    rendering,
                    pts_2d,        # ← 必须是 [8,2] 的投影点
                    color=color,
                    alpha=0.35
                )


                out_path = os.path.join(
                    debug_path,
                    f"{img_name}_block_{block_id}.png"
                )
                torchvision.utils.save_image(overlay, out_path)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)