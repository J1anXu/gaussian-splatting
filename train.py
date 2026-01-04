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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, merge_opt
import sys
from scene import Scene, GaussianModel
from utils.general_utils import get_git_branch, safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.camera_utils import frustum_culling
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import time
from logger import get_logger
SCENE_NAME = "unknown_scene"
BRANCH = "unknown_branch"
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
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")



    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    

    
    for iteration in range(first_iter, opt.iterations + 1):

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

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
        
        available_mask = frustum_culling(gaussians._xyz, viewpoint_cam.full_proj_transform)
        available_indices = torch.nonzero(available_mask, as_tuple=True)[0]
        
        if not gaussians.partitioned:
            if gaussians._xyz.shape[0] > 50_000:
                gaussians.partition() 
                gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
                gaussians.partitioned = True
                LOGGER.info(f"Partitioned Gaussians at iteration {iteration}, total gaussians: {gaussians._xyz.shape[0]}, num partitions: {len(gaussians.block_indices)}")
            else:
                gaussians.block_indices = [available_indices]
        
        rendered_list, depth_list, alpha_list = [], [], []
        viewspace_points_list, visibility_filter_list, radii_list = [], [], []
        visible_indices_list = []
        available_num = 0
        
        
        
        for idx in range(len(gaussians.block_indices)):
            block_indice = gaussians.block_indices[idx]
            visible_mask_in_block = available_mask[block_indice]
            visible_indices = block_indice[visible_mask_in_block]
            if visible_indices.shape[0] == 0:
                continue
            available_num += visible_indices.shape[0]
            gaussians.set_subset(visible_indices)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
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

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth = 0

        loss.backward()

        global_viewspace_points_grad = torch.zeros( N_total, 3, device="cuda", requires_grad=False )
        for viewspace_points, sub_set_mask in zip(viewspace_points_list, visible_indices_list):
            global_viewspace_points_grad[sub_set_mask] = viewspace_points.grad
        

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            total_points = gaussians.get_xyz.shape[0]
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "pts_in_frustum": available_num, "blocks": len(gaussians.block_indices), "pts": total_points})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            log = {"iter": iteration,"loss": ema_loss_for_log, "pts_in_frustum": available_num, "pts": total_points}
            
            LOGGER.info(log)
            
            if WANDB and not DEBUG_MODE:
                wandb.log(log, step=iteration)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, BRANCH)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats2(global_viewspace_points_grad, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    if gaussians.partitioned and not DEBUG_MODE:
                        wandb.log(
                            {
                                f"block/{idx}_size": len(block)
                                for idx, block in enumerate(gaussians.block_indices)
                            },
                            step=iteration
                        )

                        gaussians.repartition()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    # xyz_before = gaussians._xyz.detach().cpu().clone()
                    # gaussians.optimizer.step()
                    # xyz_after = gaussians._xyz.detach().cpu()
                    # delta = (xyz_after - xyz_before).abs().max().item()
                    # print("max |Δxyz| =", delta)
                    # gaussians.optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    SCENE_NAME = args.source_path.strip('/').split('/')[-1]
    BRANCH = get_git_branch()
    
    
    LOGGER = get_logger(SCENE_NAME, os.path.join("./logs", "train", BRANCH, SCENE_NAME))
    
    DEBUG_MODE = sys.gettrace() is not None
    
    if WANDB and not DEBUG_MODE:
        wandb.login()
        run = wandb.init( project="3dgs_baseline", name = f"{BRANCH}_{SCENE_NAME}_{time.strftime('%m%d%H%M')}", job_type="train", config=vars(op.extract(args)) )
        wandb.define_metric("iteration")  # 
        
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
