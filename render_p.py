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
from utils.camera_utils import frustum_culling, draw_box, rebuild_block_bound_aabb_2d, rebuild_pointcloud_aabb_2d, extract_block_layer_contribution

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


def render_set(model_path, name, iteration, views, gaussians: GaussianModel, pipeline, background, train_test_exp, separate_sh):
    BRANCH = get_git_branch()

    render_path = os.path.join(model_path, BRANCH, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, BRANCH, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    debug_path = os.path.join("debug", BRANCH)
    os.makedirs(debug_path, exist_ok=True)
    
    # gaussians.partition()
    # gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        frustum_culling_available_mask = frustum_culling(gaussians._xyz, view.full_proj_transform)
        
        rendered_list, depth_list, alpha_list = [], [], []
        viewspace_points_list, visibility_filter_list, radii_list = [], [], []
        visible_indices_list = []
        visible_block_idxs = []
        available_num = 0
        
        
        for block_idx in range(len(gaussians.block_indices)):
            block_indice = gaussians.block_indices[block_idx].to("cuda")
            visible_mask_in_block = frustum_culling_available_mask[block_indice]
            visible_indices = block_indice[visible_mask_in_block]
            if visible_indices.shape[0] == 0:
                continue
            visible_block_idxs.append(block_idx)
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
        front_rgbs = merge_res["front_rgbs"]
        prefix_T = merge_res["prefix_T"]
        # block_rank[k, h, w] 表示： 在像素 (h, w) 处，第 k 个 block 在“按深度排序后”的层级排名（rank）
        block_rank = merge_res["block_rank"] # [K, H, W]
        
        if view.alpha_mask is not None:
            alpha_mask = view.alpha_mask.cuda()
            image *= alpha_mask
        
        gt = view.original_image[0:3, :, :]
        if args.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        img_name = view.image_name


        img_path_in_debug = os.path.join(debug_path, img_name)
        
        if config.SAVE_RGB_LAYERS:
            rgb_layers_path = os.path.join(img_path_in_debug, "rgb_layers")
            os.makedirs(rgb_layers_path, exist_ok=True)
            for idx in range(front_rgbs.shape[0]):
                layer_img = front_rgbs[idx]
                block_idx = visible_block_idxs[idx]
                torchvision.utils.save_image(layer_img, os.path.join(rgb_layers_path, f"layer_{block_idx}.png"))
            
        if config.SAVE_LAYERS_CONTRIBUTION:
            # 看看每个block在每个图层贡献了什么
            layer_contri_path = os.path.join(img_path_in_debug, "layer_contribution")
            os.makedirs(layer_contri_path, exist_ok=True)
            contribution = extract_block_layer_contribution(block_rank, front_rgbs, prefix_T, visible_block_idxs)

            for layer, blocks in contribution.items():
                for block_id, rgb in blocks.items():
                    if rgb.abs().sum() == 0:
                        continue
                    save_path = os.path.join( layer_contri_path, f"layer_{layer}_block_{block_id}_contribution.png" )
                    torchvision.utils.save_image(rgb.clamp(0, 1), save_path)
                    
        if config.SAVE_DEPTH_LIST:
            depth_list_path = os.path.join(img_path_in_debug, "depth_list")
            os.makedirs(depth_list_path, exist_ok=True)
            for idx, block_id in enumerate(visible_block_idxs):
                layer_depth = depth_list[idx]   # [1, H, W]
                depth = layer_depth.squeeze(0)       # [H, W]
                valid = torch.isfinite(depth)
                if not valid.any():
                    continue
                d_min = depth[valid].min()
                d_max = depth[valid].max()
                if (d_max - d_min) < 1e-6:
                    depth_norm = torch.zeros_like(depth)
                else:
                    depth_norm = (depth - d_min) / (d_max - d_min)
                torchvision.utils.save_image(depth_norm.unsqueeze(0), os.path.join( depth_list_path, f"block_{block_id}_depth_map.png" ))
                
        
        if config.SAVE_BLOCK_IMG:
            block_img_path = os.path.join(img_path_in_debug, "block_images")
            os.makedirs(block_img_path, exist_ok=True)
            
            for block_img, block_id in zip(rendered_list, visible_block_idxs):
                
                # bmin = (xmin, ymin, zmin); bmax = (xmax, ymax, zmax)
                # 因为一个轴对齐包围盒（AABB）在 3D 空间里， 只需要两个点：最小角 bmin 和最大角 bmax， 这两个点就唯一确定了一个长方体，而这个长方体天然有 8 个角点。
                bmin, bmax = gaussians.block_bounds[block_id]
                W, H = block_img.shape[2], block_img.shape[1]
                proj_matrix = view.full_proj_transform
                
                if config.DRAW_BOX:
                    x_min, y_min, x_max, y_max = rebuild_block_bound_aabb_2d(block_img, view, bmin, bmax, W, H)
                    block_img = draw_box(block_img, x_min, y_min, x_max, y_max, colors="red", width=4)
                    x_min, y_min, x_max, y_max = rebuild_pointcloud_aabb_2d(gaussians._xyz[gaussians.block_indices[block_id]], proj_matrix, W, H)
                    block_img = draw_box(block_img, x_min, y_min, x_max, y_max, colors="green")
                    
                torchvision.utils.save_image(block_img, os.path.join(block_img_path, f"_block_{block_id}.png"))  
            torchvision.utils.save_image(image, os.path.join(block_img_path, img_name + ".png"))            

                
        torchvision.utils.save_image(image, os.path.join(render_path, img_name + ".png"))            
        torchvision.utils.save_image(gt, os.path.join(gts_path, img_name + ".png"))



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