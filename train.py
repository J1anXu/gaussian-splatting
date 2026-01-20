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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

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
    
    debug_image_name = "_DSC8680.JPG"
    img_path_in_debug = os.path.join("/data2/jian/debug", BRANCH, debug_image_name)
    os.makedirs(img_path_in_debug, exist_ok=True)
    
    model_list = [initial_gaussians]
    partitioned = False
    
    for iteration in range(first_iter, opt.iterations + 1):
        
        for model in model_list:
            model.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            for model in model_list:
                model.oneupSHdegree()

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
        
        if not partitioned and initial_gaussians._xyz.shape[0] > 300_000:
            initial_gaussians.partition() 
            # initial_gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
            model_list = []
            for idx in range(len(initial_gaussians.block_idx_list)):
                model = initial_gaussians.get_kid(idx, opt)
                model_list.append(model)
                LOGGER.info(f"GS {idx} size: {model._xyz.shape[0]}")  
            partitioned = True
        
        # 视锥剔除
        for model in model_list:
            visible_mask = frustum_culling(model._xyz, viewpoint_cam.full_proj_transform)
            model.visible_idx = torch.nonzero(visible_mask, as_tuple=True)[0]
        
        rendered_list, depth_list, alpha_list = [], [], []
        viewspace_points_list, visibility_filter_list, radii_list = [], [], []
        visible_pts = 0
        visible_model_idx = []
        
        pts_total = 0
        
        for idx, model in enumerate(model_list):
            pts_total += model._xyz.shape[0]
            # this model has no point in the view frustum in this view
            if model.visible_idx.shape[0] == 0:
                continue
            
            visible_model_idx.append(idx)
            visible_pts += model.visible_idx.shape[0]
            
            model.set_subset(model.visible_idx)
            render_pkg = render(viewpoint_cam, model, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            model.clear_subset()
            
            # pixel level 
            image, alphaLeft, depth = render_pkg["render"], render_pkg["alphaLeft"], render_pkg["depth"]
            
            rendered_list.append(image)
            depth_list.append(depth)
            alpha_list.append(alphaLeft)
            
            # gaussian points level
            viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            viewspace_points_list.append(viewspace_point_tensor)
            visibility_filter_list.append(visibility_filter)
            radii_list.append(radii)
            

        merged_pkg = merge_opt_kid(rendered_list, depth_list, alpha_list)
        image, colors_bg = merged_pkg["final_rgb"], merged_pkg["bg_rgb"]
    
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        
        # if viewpoint_cam.image_name == debug_image_name:
        #     iteration_path = os.path.join(img_path_in_debug, f"iter_{iteration}")
        #     # os.makedirs(iteration_path, exist_ok=True)
        #     front_rgbs = merge_res["front_rgbs"]
        #     prefix_T = merge_res["prefix_T"]
        #     # block_rank[k, h, w] 表示： 在像素 (h, w) 处，第 k 个 block 在“按深度排序后”的层级排名（rank）
        #     block_rank = merge_res["block_rank"] # [K, H, W]
        #     save_rgb_layers(iteration_path, front_rgbs)
        #     save_layer_contribution(iteration_path, block_rank, front_rgbs, prefix_T, visible_block_idxs)
        #     save_depth_list(iteration_path, depth_list, visible_block_idxs)

        #     # torchvision.utils.save_image(image, os.path.join(img_path_in_debug, f"{iteration}.png"))

        #     if gaussians.partitioned:
        #         save_block_img(iteration_path, rendered_list, visible_block_idxs, gaussians, viewpoint_cam, image, config)
        #     LOGGER.info(f"Saved debug images at iteration {iteration} for {debug_image_name}")


        # if viewpoint_cam.image_name == debug_image_name:
        #     # block_rank[k, h, w] 表示： 在像素 (h, w) 处，第 k 个 block 在“按深度排序后”的层级排名（rank）
        #     torchvision.utils.save_image(image, os.path.join(img_path_in_debug, f"{iteration}" + ".png"))
        #     LOGGER.info(f"Saved debug images at iteration {iteration} for {debug_image_name}")


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
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

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                point_cloud_path = os.path.join(scene.model_path, f"point_cloud/{BRANCH}/iteration_{iteration}")
                for idx, model in enumerate(model_list):
                    model.save_ply(os.path.join(point_cloud_path, f"point_cloud_sub_{idx}.ply"), include_block=False)

            # Densification
            if iteration < opt.densify_until_iter:
                for idx, model_idx in enumerate(visible_model_idx):
                    model = model_list[model_idx]
                    viewspace_points = viewspace_points_list[idx]
                    radii = radii_list[idx]
                    global_viewspace_points_grad = torch.zeros(model.get_xyz.shape[0], 3, device="cuda", requires_grad=False )
                    global_viewspace_points_grad[model.visible_idx] = viewspace_points.grad
                    visibility_filter = visibility_filter_list[idx]
                    global_visibility_filter = model.visible_idx[visibility_filter]
                    
                    model.max_radii2D[global_visibility_filter] = torch.max(model.max_radii2D[global_visibility_filter], radii[visibility_filter])
                    model.add_densification_stats2(global_viewspace_points_grad, global_visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    for idx, model in enumerate(model_list):
                        model.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        
                    if WANDB and initial_gaussians.partitioned and not DEBUG_MODE:
                        wandb.log(
                            {
                                f"block/{idx}_size": len(gs._xyz.shape[0])
                                for idx, gs in enumerate(model_list)
                            },
                            step=iteration
                        )

                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    for model in model_list:
                        model.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                for model in model_list:
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none = True)

                    
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
    
    if WANDB and not DEBUG_MODE:
        wandb.login()
        run = wandb.init(
            project = "3dgs_baseline", 
            name = f"{SCENE_NAME}_{BRANCH}_{time.strftime('%m%d%H%M')}", 
            job_type = "train", 
            group = SCENE_NAME,
            config = vars(op.extract(args)) 
        )
        wandb.define_metric("iteration")  # 
        
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    os.makedirs("debug", exist_ok=True)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
