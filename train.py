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
from utils.loss_utils import l1_loss, ssim, fast_ssim
from gaussian_renderer import render, merge_opt, merge_opt_kid
import sys
from scene import Scene, GaussianModel
from utils.general_utils import get_git_branch, safe_state, get_expon_lr_func, get_git_branch
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.camera_utils import frustum_culling, frustum_culling_idx
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import time
from logger import get_logger
import config
import diff_gaussian_rasterization_wenqi_tam
from TimerManager import  TraceManager, TID_MAIN, PID_CPU
from pipeline_grad_sync import PipelinedGradSync
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
    from diff_gaussian_rasterization_wenqi_tam import SparseGaussianAdam
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
    
    if DEBUG_MODE:
    # TODO
        scene.gaussians.load_ply("output/mip360/bicycle/point_cloud/baseline/iteration_30000/point_cloud.ply")
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        opt.iterations = 1050
        return scene, 1000, 0, 0, progress_bar, None
    
    
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
            if initial_gaussians._xyz.shape[0] > config.SPLIT_SIZE:
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
        
        # frustum culling (cuda)
        if config.FRUSTUM_CULLING_ENABLED:
            initial_gaussians.visible_indices = frustum_culling_idx(initial_gaussians._xyz, viewpoint_cam.full_proj_transform)
        else:
            initial_gaussians.visible_indices = torch.arange(initial_gaussians._xyz.shape[0], device="cuda")
        

        visible_pts = 0        
        pts_total = 0

        pts_total += initial_gaussians._xyz.shape[0]
        visible_pts += initial_gaussians.visible_indices.shape[0]
        
        initial_gaussians.activate_subset()
        render_pkg = render(viewpoint_cam, initial_gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        initial_gaussians.deactivate_subset()
        
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
        ssim_value = fast_ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth = 0
        diff_gaussian_rasterization_wenqi_tam.set_colors_bg(colors_bg)
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            
            if iteration % 10 == 0:
                sh_deg = initial_gaussians.active_sh_degree
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "sh": sh_deg, "pts_in_frustum": visible_pts, "pts": pts_total})
                progress_bar.update(10)
                gpu_mem_gb = torch.cuda.memory_reserved() / 1024**3
                log = {"iter": iteration, "loss": ema_loss_for_log, "sh": sh_deg, "pts_in_frustum": visible_pts, "pts": pts_total, "gpu_mem_gb": gpu_mem_gb}
                LOGGER.info(log)
                if WANDB and not DEBUG_MODE:
                    wandb.log(log, step=iteration)

            if iteration == opt.iterations:
                progress_bar.close()


            # Densification
            if iteration < opt.densify_until_iter:
                global_viewspace_points_grad = torch.zeros(initial_gaussians.get_xyz.shape[0], 3, device="cuda", requires_grad=False )
                global_viewspace_points_grad[initial_gaussians.visible_indices] = viewspace_point_tensor.grad
                global_visibility_filter = initial_gaussians.visible_indices[visibility_filter]
                
                initial_gaussians.max_radii2D[global_visibility_filter] = torch.max(initial_gaussians.max_radii2D[global_visibility_filter], radii[visibility_filter])
                initial_gaussians.add_densification_stats2(global_viewspace_points_grad, global_visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    initial_gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    initial_gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                initial_gaussians.exposure_optimizer.step()
                initial_gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    initial_gaussians.optimizer.step(visible, radii.shape[0])
                    initial_gaussians.optimizer.zero_grad(set_to_none = True)
                else:
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
    gaussians = gaussians.dump_to_cpu()
    
    # partition
    gaussians.build_split_indices()
    # gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
    
    submodel_list: List[GaussianModel] = gaussians.split()
    
    for submodel in submodel_list:
        submodel.training_setup(opt, device = "cpu")
        # Initialize packed pinned buffer after parameters are created.
        # This packs all 6 per-Gaussian attributes into a single [N, D]
        # contiguous pinned tensor for efficient subset H2D transfer.
        submodel.pack_to_buffer()

    cpu_full_proj_transform_dict = {}
    for cam in scene.getTrainCameras():
        cpu_full_proj_transform_dict[cam.image_name] = cam.full_proj_transform.detach().cpu()

    # Cache frustum culling results per (submodel_id, camera) once positions are frozen
    # (no densification after densify_until_iter, so results are stable)
    # frustum_cache[submodel_id][image_name] = visible_indices (CPU tensor)
    frustum_cache: dict = {}

    # Pipeline: async D2H grad copies + deferred opt steps
    tracer = TraceManager(enabled=config.TIMELINE)
    grad_sync = PipelinedGradSync(submodel_list, opt, dataset, scene, tracer=tracer)

    if DEBUG_MODE:
        opt.iterations = 1050

    time_start = time.time()
    for iteration in range(first_iter, opt.iterations + 1):
        tracer.step(iteration)

        for submodel in submodel_list:
            submodel.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            for submodel in submodel_list:
                submodel.oneupSHdegree()

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
        
        # frustum culling (with cache after densify_until_iter)
        use_fc_cache = config.FRUSTUM_CULLING_CACHE_ENABLED and iteration >= opt.densify_until_iter
        with torch.no_grad():
            if config.FRUSTUM_CULLING_ENABLED:
                cam_name = viewpoint_cam.image_name
                with tracer.span("frustum_culling", tid=TID_MAIN):
                    for submodel_id, model in enumerate(submodel_list):
                        cached = use_fc_cache and submodel_id in frustum_cache and cam_name in frustum_cache[submodel_id]
                        if cached:
                            model.visible_indices = frustum_cache[submodel_id][cam_name]
                        else:
                            if model._xyz.is_cuda:
                                model.visible_indices = frustum_culling_idx(model._xyz, viewpoint_cam.full_proj_transform)
                            else:
                                # Use pre-extracted contiguous [N,3] xyz cache.
                                # Positions change ~1e-6 relative per iter and culling
                                # has inflate_ratio=0.3 margin -> stale by 1 iter is safe.
                                xyz_for_cull = model._xyz_contig if hasattr(model, '_xyz_contig') else model._xyz
                                model.visible_indices = frustum_culling_idx(xyz_for_cull, cpu_full_proj_transform_dict[cam_name])
                            if use_fc_cache:
                                if submodel_id not in frustum_cache:
                                    frustum_cache[submodel_id] = {}
                                frustum_cache[submodel_id][cam_name] = model.visible_indices
            else:
                for model in submodel_list:
                    model.visible_indices = torch.arange(model._xyz.shape[0], device="cuda")
        
        # 无渲染全部结果 为计算Loss做准备
        rendered_list, depth_list, alpha_list = [], [], []
        visible_submodel_id_list = []
        visible_pts = 0

        with torch.no_grad():
            for submodel_id, submodel in enumerate(submodel_list):

                if submodel.visible_indices.shape[0] == 0:
                    continue

                visible_pts += submodel.visible_indices.shape[0]

                with tracer.span("h2d_nograd", tid=TID_MAIN, block_id=submodel_id,
                                 n_vis=submodel.visible_indices.shape[0]):
                    submodel.move_and_activate_subset(requires_grad = False)

                with tracer.gpu_span("render_nograd", block_id=submodel_id):
                    render_pkg = render(viewpoint_cam, submodel, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

                submodel.deactivate_subset()

                # pixel level
                image, alphaLeft, depth = render_pkg["render"], render_pkg["alphaLeft"], render_pkg["depth"]

                # 能被视锥看见并不一定真的有贡献
                # image: [3, H, W]
                valid_mask = (image > 0).any(dim=0)   # [H, W] bool
                valid_pixels = valid_mask.sum().item()
                total_pixels = valid_mask.numel()
                contributed_percent = valid_pixels / total_pixels
                if contributed_percent < 0.05:
                    continue

                rendered_list.append(image)
                depth_list.append(depth)
                alpha_list.append(alphaLeft)

                visible_submodel_id_list.append(submodel_id)


        # execute merge
        with torch.no_grad():
            with tracer.gpu_span("merge_opt_kid"):
                merge_res = merge_opt_kid(rendered_list, depth_list, alpha_list)
            
        C_sorted = merge_res["front_rgbs"] # 每个 block 的颜色贡献，已经按照正确的前后顺序排列好
        prefix_T = merge_res["prefix_T"]
        block_rank = merge_res["block_rank"]  # [K,H,W]，每个像素告诉你每个 block 的排序位置
        K, C, H, W = C_sorted.shape   
        colors_bg = merge_res["bg_rgb"]
        diff_gaussian_rasterization_wenqi_tam.set_colors_bg(colors_bg)

        with torch.no_grad():
            gt_image = viewpoint_cam.original_image.cuda()

        # 遍历所有可见block 轮流当active block
        for submodel_id, rank_map in zip(visible_submodel_id_list, block_rank):
            submodel: GaussianModel = submodel_list[submodel_id]

            with tracer.span("h2d_grad", tid=TID_MAIN, block_id=submodel_id,
                             n_vis=submodel.visible_indices.shape[0]):
                submodel.move_and_activate_subset(requires_grad = True)

            with tracer.gpu_span("render_grad", block_id=submodel_id):
                render_pkg = render(viewpoint_cam, submodel, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

            # pixel level
            sub_img = render_pkg["render"]

            # gaussian points level
            sub_viewspace_point_tensor = render_pkg["viewspace_points"]

            with torch.no_grad():
                # 当前subset的渲染结果在每个像素上的排序位置
                submodel_rank_per_pixel = rank_map.unsqueeze(0).unsqueeze(0).expand(1, C, H, W)               # [1,3,H,W]
                # 3. 当前块(index = rank_map)在每个像素位置上能拿到的透射率
                prefix_T_k = prefix_T[:, 0].gather(dim=0, index = rank_map.unsqueeze(0)).squeeze(0)    # [H,W]
                # 4. 当前块(index = idx)块提供的颜色
                C_sorted_k = C_sorted.gather(dim=0, index=submodel_rank_per_pixel).squeeze(0)   # [3,H,W]
                # 5. 从最终结果中扣除当前块的贡献，得到背景图. 贡献由每个像素位置上提供的颜色乘以透射率得到
                C_base = merge_res["final_rgb"] - prefix_T_k * C_sorted_k
                # 6. 带梯度的渲染结果
                C_active = sub_img      # [3,H,W], has grad

            # 把带梯度的渲染结果拼到背景上 用于计算loss
            composed_img = C_base + prefix_T_k * C_active

            # Loss
            Ll1 = l1_loss(composed_img, gt_image)
            ssim_value = fast_ssim(composed_img, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth = 0

            with tracer.gpu_span("backward", block_id=submodel_id):
                loss.backward()

            tracer.counter("pts", {"visible": visible_pts,
                                   "total": sum(s._xyz.shape[0] for s in submodel_list)})

            grad_sync.flush_and_prepare(submodel, submodel_id, render_pkg, sub_viewspace_point_tensor, iteration)

        # flush the last submodel's pending work
        grad_sync.flush_last()
                        
                        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                pts_total = sum(submodel._xyz.shape[0] for submodel in submodel_list)
                sh_deg = submodel_list[0].active_sh_degree
                # progress bar
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "sh": sh_deg, "pts_in_frustum": visible_pts, "pts": pts_total})
                progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                gpu_mem_gb = torch.cuda.memory_reserved() / 1024**3
                log = {"iter": iteration, "loss": ema_loss_for_log, "sh": sh_deg, "pts_in_frustum": visible_pts, "pts": pts_total, "gpu_mem_gb": gpu_mem_gb}

                # logging
                LOGGER.info(log)

                # wandb logging
                if WANDB and not DEBUG_MODE:
                    wandb.log(log, step=iteration)
                    wandb.log({f"block/{idx}_size": gs._xyz.shape[0] for idx, gs in enumerate(submodel_list)}, step=iteration)
            
        # saving Gaussians ply    
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            point_cloud_path = os.path.join(scene.model_path, f"point_cloud/{BRANCH}/iteration_{iteration}")
            for submodel_id, submodel in enumerate(submodel_list):
                submodel.save_ply(os.path.join(point_cloud_path, f"point_cloud_sub_{submodel_id}.ply"), include_block=False)
                
        # # save debug image
        # if viewpoint_cam.image_name == debug_image_name:
        #     torchvision.utils.save_image(composed_img, os.path.join(IMG_PATH_IN_DEBUG, f"{iteration}" + ".png"))
    time_end = time.time()
    cost = time_end - time_start
    print(f"Phase 2 training time cost: [{cost:.2f}] seconds.")

    if config.TIMELINE:
        os.makedirs("timeline", exist_ok=True)
        tracer.export(f"timeline/trace_{BRANCH}_{SCENE_NAME}.json")
        
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
    SCENE_NAME = args.source_path.split('/')[-1]
    DATASET_NAME = args.source_path.split('/')[-2]
    
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
            project = DATASET_NAME, 
            name = f"{SCENE_NAME}_{BRANCH}", 
            group = SCENE_NAME,
            config = vars(op.extract(args)) 
        )
        wandb.define_metric("iteration")  # 
        
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    os.makedirs("debug", exist_ok=True)
    IMG_PATH_IN_DEBUG = os.path.join("/data/jian/debug", BRANCH, SCENE_NAME, debug_image_name)
    os.makedirs(IMG_PATH_IN_DEBUG, exist_ok=True)
    time_start = time.time()
    res = training_phase_1(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.debug_from)
    
    training_phase_2(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.debug_from, res)
    time_end = time.time()
    cost = time_end - time_start
    print(f"\nTraining complete. Total time: {cost}")
