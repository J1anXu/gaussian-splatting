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

import os
from typing import List
import torch
from random import randint

import torchvision
from utils.debug_utils import save_block_img, save_depth_list, save_rgb_layers, save_layer_contribution
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, merge_opt, merge_opt_kid
import sys
from scene import Scene, GaussianModel
from utils.general_utils import get_git_branch, safe_state, get_expon_lr_func, get_git_branch
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.camera_utils import frustum_culling
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import time
from logger import get_logger
import config
import diff_gaussian_rasterization

SCENE_NAME = None
BRANCH = None
DEBUG_MODE = False

WANDB = True
LOGGER = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


debug_image_name = "_DSC8680.JPG"
IMG_PATH_IN_DEBUG = None

def print_config():
    for k, v in vars(config).items():
        if not k.startswith("__"):
            print(f"{k} = {v}")

def training_phase_1(dataset, opt, pipe, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    prepare_output_and_logger(dataset)
    initial_gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, initial_gaussians, on_cpu=True)
    initial_gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        initial_gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    colors_bg = None
    
    for iteration in range(first_iter, opt.iterations + 1):
        # partition
        if config.PARTITIONING_ENABLED:
            if initial_gaussians._xyz.shape[0] > 300_000:
                print(f"Finished phase 1 training at iteration {iteration}, partitioning now...")
                LOGGER.info(f"Finished phase 1 training at iteration {iteration}, partitioning now...")
                return scene, iteration, ema_loss_for_log, ema_Ll1depth_for_log, progress_bar, colors_bg
            
        initial_gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            initial_gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # frustum culling
        if config.FRUSTUM_CULLING_ENABLED:
            visible_mask = frustum_culling(initial_gaussians._xyz, viewpoint_cam.full_proj_transform)
            initial_gaussians.visible_gaussian_indices = torch.nonzero(visible_mask, as_tuple=True)[0]
        else:
            initial_gaussians.visible_gaussian_indices = torch.arange(initial_gaussians._xyz.shape[0], device="cuda")
        

        visible_pts = 0        
        pts_total = 0

        pts_total += initial_gaussians._xyz.shape[0]
        visible_pts += initial_gaussians.visible_gaussian_indices.shape[0]
        
        initial_gaussians.active(initial_gaussians.visible_gaussian_indices)
        render_pkg = render(viewpoint_cam, initial_gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        initial_gaussians.deactive()
        
        # pixel level 
        image, alphaLeft, depth = render_pkg["render"], render_pkg["alphaLeft"], render_pkg["depth"]
        
        # gaussian points level
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        

        if viewpoint_cam.image_name == debug_image_name:
            torchvision.utils.save_image(image, os.path.join(IMG_PATH_IN_DEBUG, f"{iteration}" + ".png"))

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if colors_bg is None:
            colors_bg = torch.zeros_like(gt_image)
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth = 0
        diff_gaussian_rasterization.set_colors_bg(colors_bg)
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "pts_in_frustum": visible_pts, "pts": pts_total})
                progress_bar.update(10)
                log = {"iter": iteration, "loss": ema_loss_for_log, "pts_in_frustum": visible_pts, "pts": pts_total}
                LOGGER.info(log)
                if WANDB and not DEBUG_MODE:
                    wandb.log(log, step=iteration)
                    
            if iteration == opt.iterations:
                progress_bar.close()


            # Densification
            if iteration < opt.densify_until_iter:
                global_viewspace_points_grad = torch.zeros(initial_gaussians.get_xyz.shape[0], 3, device="cuda", requires_grad=False )
                global_viewspace_points_grad[initial_gaussians.visible_gaussian_indices] = viewspace_point_tensor.grad
                global_visibility_filter = initial_gaussians.visible_gaussian_indices[visibility_filter]
                
                initial_gaussians.max_radii2D[global_visibility_filter] = torch.max(initial_gaussians.max_radii2D[global_visibility_filter], radii[visibility_filter])
                initial_gaussians.add_densification_stats2(global_viewspace_points_grad, global_visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    initial_gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    initial_gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                initial_gaussians.optimizer.step()
                initial_gaussians.optimizer.zero_grad(set_to_none = True)



def training_phase_2(dataset, opt, pipe, saving_iterations, debug_from, res):
    scene, old_iteration, ema_loss_for_log, ema_Ll1depth_for_log, progress_bar, colors_bg = res

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    first_iter = old_iteration
    
    gaussians: GaussianModel = scene.gaussians
    
    # generate a initialized gs copy
    # gaussians = gaussians.dump_to_cpu()
    
    # partition
    gaussians.build_split_indices()
    # gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
    
    subset_list: List[GaussianModel] = gaussians.split()
        
    for subset in subset_list:
        subset.training_setup(opt)
        
    for iteration in range(first_iter, opt.iterations + 1):
        
        for subset in subset_list:
            subset.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            for subset in subset_list:
                subset.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # frustum culling
        if config.FRUSTUM_CULLING_ENABLED:
            for subset in subset_list:
                visible_mask = frustum_culling(subset._xyz, viewpoint_cam.full_proj_transform)
                subset.visible_gaussian_indices = torch.nonzero(visible_mask, as_tuple=True)[0]
        else:
            for subset in subset_list:
                subset.visible_gaussian_indices = torch.arange(subset._xyz.shape[0], device="cuda")
        
        # 无渲染全部结果 为计算Loss做准备
        rendered_list, depth_list, alpha_list = [], [], []
        visible_subset_id_list = []
        visible_pts = 0
        
        with torch.no_grad():
            for subset_id, subset in enumerate(subset_list):
                
                if subset.visible_gaussian_indices.shape[0] == 0:
                    continue
                
                visible_pts += subset.visible_gaussian_indices.shape[0]
                visible_subset_id_list.append(subset_id)
                
                active_gaussians_mask = subset.visible_gaussian_indices
                
                subset.active(active_gaussians_mask)
                render_pkg = render(viewpoint_cam, subset, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                subset.deactive()
                
                # pixel level 
                image, alphaLeft, depth = render_pkg["render"], render_pkg["alphaLeft"], render_pkg["depth"]
                
                rendered_list.append(image)
                depth_list.append(depth)
                alpha_list.append(alphaLeft)
                
        # execute merge 
        merge_res = merge_opt_kid(rendered_list, depth_list, alpha_list)
        C_sorted = merge_res["front_rgbs"] # 每个 block 的颜色贡献，已经按照正确的前后顺序排列好
        prefix_T = merge_res["prefix_T"]
        block_rank = merge_res["block_rank"]  # [K,H,W]，每个像素告诉你每个 block 的排序位置
        K, C, H, W = C_sorted.shape   
        colors_bg = merge_res["bg_rgb"]
        
        
        # 遍历所有可见block 轮流当active block
        for subset_id, rank_map in zip(visible_subset_id_list, block_rank):
            subset: GaussianModel = subset_list[subset_id]
            
            active_gaussians_mask = subset.visible_gaussian_indices
            
            subset.active(active_gaussians_mask)
            render_pkg = render(viewpoint_cam, subset, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            subset.deactive()
            
            # pixel level 
            subset_img = render_pkg["render"]
            
            # gaussian points level
            subset_viewspace_point_tensor, subset_visibility_filter, subset_radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
           # 当前subset的渲染结果在每个像素上的排序位置
            subset_rank_per_pixel = rank_map.unsqueeze(0).unsqueeze(0).expand(1, C, H, W)               # [1,3,H,W]
            
            # 3. 当前块(index = rank_map)在每个像素位置上能拿到的透射率
            prefix_T_k = prefix_T[:, 0].gather(dim=0, index = rank_map.unsqueeze(0)).squeeze(0)    # [H,W]

            # 4. 当前块(index = idx)块提供的颜色
            C_sorted_k = C_sorted.gather(dim=0, index=subset_rank_per_pixel).squeeze(0)   # [3,H,W]   
            
            # 5. 从最终结果中扣除当前块的贡献，得到背景图. 贡献由每个像素位置上提供的颜色乘以透射率得到
            C_base = merge_res["final_rgb"] - prefix_T_k * C_sorted_k                  
            
            # 6. 带梯度的渲染结果
            C_active = subset_img      # [3,H,W], has grad   
            
            # 7. 把带梯度的渲染结果拼到背景上 用于计算loss
            composed_img = C_base + prefix_T_k * C_active
            
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                composed_img *= alpha_mask
                
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(composed_img, gt_image)
            ssim_value = ssim(composed_img, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth = 0
            diff_gaussian_rasterization.set_colors_bg(colors_bg)
            loss.backward()

            with torch.no_grad():
                # Densification
                if iteration < opt.densify_until_iter:
                    global_viewspace_points_grad = torch.zeros(subset.get_xyz.shape[0], 3, device="cuda", requires_grad=False )
                    global_viewspace_points_grad[subset.visible_gaussian_indices] = subset_viewspace_point_tensor.grad
                    global_visibility_filter = subset.visible_gaussian_indices[subset_visibility_filter]
                    
                    subset.max_radii2D[global_visibility_filter] = torch.max(subset.max_radii2D[global_visibility_filter], subset_radii[subset_visibility_filter])
                    subset.add_densification_stats2(global_viewspace_points_grad, global_visibility_filter)
                    
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        subset.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        subset.reset_opacity()
                        
                # Optimizer step
                if iteration < opt.iterations:
                        subset.optimizer.step()
                        subset.optimizer.zero_grad(set_to_none = True)
                        
                        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            
            pts_total = 0
            for subset in subset_list:
                pts_total += subset._xyz.shape[0]
            
            
            if iteration % 10 == 0:
                # progress bar
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "pts_in_frustum": visible_pts, "pts": pts_total})
                progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                
                log = {"iter": iteration, "loss": ema_loss_for_log, "pts_in_frustum": visible_pts, "pts": pts_total}
                
                # logging
                LOGGER.info(log)
                
                # wandb logging
                if WANDB and not DEBUG_MODE:
                    wandb.log(log, step=iteration)
                    wandb.log({f"block/{idx}_size": gs._xyz.shape[0] for idx, gs in enumerate(subset_list)}, step=iteration)
            
        # saving Gaussians ply    
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            point_cloud_path = os.path.join(scene.model_path, f"point_cloud/{BRANCH}/iteration_{iteration}")
            for subset_id, subset in enumerate(subset_list):
                subset.save_ply(os.path.join(point_cloud_path, f"point_cloud_sub_{subset_id}.ply"), include_block=False)
                
        # save debug image
        if viewpoint_cam.image_name == debug_image_name:
            torchvision.utils.save_image(composed_img, os.path.join(IMG_PATH_IN_DEBUG, f"{iteration}" + ".png"))
                    
    # if (iteration in checkpoint_iterations):
    #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
    #     pth_path = os.path.join(args.model_path, f"point_cloud/{BRANCH}")
    #     torch.save((gaussians.capture(), iteration), pth_path + "/chkpnt" + str(iteration) + ".pth")



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--git_branch', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    SCENE_NAME = args.source_path.strip('/').split('/')[-1]
    
    if args.git_branch is not None:
        BRANCH = args.git_branch
    else:
        BRANCH = get_git_branch()
    
    LOGGER = get_logger(SCENE_NAME, os.path.join("./logs", "train", BRANCH, SCENE_NAME))
    DEBUG_MODE = sys.gettrace() is not None
    print_config()
    
    if WANDB and not DEBUG_MODE:
        wandb.login()
        run = wandb.init(
            project = "partgs", 
            name = f"{SCENE_NAME}_{BRANCH}_{time.strftime('%m%d%H%M')}", 
            config = vars(op.extract(args)) 
        )
        wandb.define_metric("iteration")  # 
        
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    os.makedirs("debug", exist_ok=True)
    IMG_PATH_IN_DEBUG = os.path.join("/data/jian/debug", BRANCH, SCENE_NAME, debug_image_name)
    os.makedirs(IMG_PATH_IN_DEBUG, exist_ok=True)
    res = training_phase_1(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.debug_from)
    training_phase_2(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.debug_from, res)

    # All done
    print("\nTraining complete.")
