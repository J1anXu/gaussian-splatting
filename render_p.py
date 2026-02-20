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

import json
import sys
from pathlib import Path
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import merge_opt_kid, render,merge_opt
import torchvision
from utils.general_utils import safe_state, get_git_branch
from utils.camera_utils import frustum_culling
from utils.debug_utils import save_rgb_layers, save_layer_contribution, save_depth_list, save_block_img
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from typing import List

import config
try:
    from diff_gaussian_rasterization_jian import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
BRANCH = None
SCENE_NAME = None

def render_set(model_path, name, iteration, views, model_list: List[GaussianModel], pipeline, background, train_test_exp, separate_sh):

    render_path = os.path.join(model_path, "rendered_p", BRANCH, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, "rendered_p", BRANCH, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    debug_path = os.path.join("debug", BRANCH)
    os.makedirs(debug_path, exist_ok=True)
    
    # gaussians.partition()
    # gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # frustum_culling_available_mask = frustum_culling(gaussians._xyz, view.full_proj_transform)
        
        rendered_list, depth_list, alpha_list = [], [], []
        viewspace_points_list, radii_list = [], []
        visible_block_idxs = []
        
        N_total = 0
        for block_idx in range(len(model_list)):
            model = model_list[block_idx]
            
            visible_mask = frustum_culling(model._xyz, view.full_proj_transform)
            subset_indices = torch.nonzero(visible_mask, as_tuple=True)[0]
            model.visible_indices = subset_indices
            
            model.activate_subset()
            render_pkg = render(view, model, pipeline, background, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            model.subset_off()
            
            image, viewspace_point_tensor, visibility_filter, radii, alphaLeft = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["alphaLeft"]
            rendered_list.append(image)
            depth_list.append(render_pkg["depth"])
            alpha_list.append(alphaLeft)
            viewspace_points_list.append(viewspace_point_tensor)
            radii_list.append(radii)
            N_total += model._xyz.shape[0]
            
        merge_res = merge_opt_kid(rendered_list, depth_list, alpha_list)   
        
        image = merge_res["final_rgb"]
        front_rgbs = merge_res["front_rgbs"]
        prefix_T = merge_res["prefix_T"]
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
        
        # if config.SAVE_RGB_LAYERS:
        #     save_rgb_layers(img_path_in_debug, front_rgbs)
            
        # if config.SAVE_LAYERS_CONTRIBUTION:
        #     save_layer_contribution(img_path_in_debug, block_rank, front_rgbs, prefix_T, visible_block_idxs)
                    
        # if config.SAVE_DEPTH_LIST:
        #     save_depth_list(img_path_in_debug, depth_list, visible_block_idxs)
                

        torchvision.utils.save_image(image, os.path.join(render_path, img_name + ".png"))            
        torchvision.utils.save_image(gt, os.path.join(gts_path, img_name + ".png"))

def store_pts(res_path, pts, scene, key):
    res_path = Path(res_path) 
    data = {}
        
    data.setdefault(key, {})
    data[key].update({
        "scene": scene,
        "branch": BRANCH,
        "num_gaussians": int(pts),
    })
    with open(res_path, "w") as f:
        json.dump(data, f, indent=2)
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        model_list = []
        scene = Scene(dataset, None, load_iteration=iteration, shuffle=False, only_camera=True)
        ply_path = os.path.join(scene.model_path, "point_cloud", BRANCH, "iteration_" + str(scene.loaded_iter))
        ply_files = sorted( f for f in os.listdir(ply_path) if f.startswith("point_cloud_sub_") and f.endswith(".ply") )
        pts = 0
        for fname in ply_files:
            full_path = os.path.join(ply_path, fname)
            model = GaussianModel(dataset.sh_degree)
            model.load_ply(full_path, dataset.train_test_exp)
            model_list.append(model)
            pts+= model._xyz.shape[0]
            print("loading", full_path, "with", model._xyz.shape[0], "gaussians success", )
        
        res_path = os.path.join(scene.model_path, "rendered_p", BRANCH)
        os.makedirs(res_path, exist_ok=True)
        json_path = os.path.join(res_path, "results.json")
        store_pts(json_path, pts, scene = SCENE_NAME, key = f"ours_{scene.loaded_iter}")
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), model_list, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), model_list, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--git_branch', type=str, default=None)
    args_raw = parser.parse_args(sys.argv[1:])

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if args_raw.git_branch is not None:
        BRANCH = args_raw.git_branch
    else:
        BRANCH = get_git_branch()
    SCENE_NAME = args.model_path.strip('/').split('/')[-1]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)