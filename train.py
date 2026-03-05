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
import math
from logger import get_logger
import config
import diff_gaussian_rasterization_jian

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
    from diff_gaussian_rasterization_jian import SparseGaussianAdam
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
            initial_gaussians.visible_idx = torch.nonzero(visible_mask, as_tuple=True)[0]
        else:
            initial_gaussians.visible_idx = torch.arange(initial_gaussians._xyz.shape[0], device="cuda")
        

        visible_pts = 0        
        pts_total = 0

        pts_total += initial_gaussians._xyz.shape[0]
        visible_pts += initial_gaussians.visible_idx.shape[0]
        
        initial_gaussians.set_subset(initial_gaussians.visible_idx)
        render_pkg = render(viewpoint_cam, initial_gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        initial_gaussians.clear_subset()
        
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
        # TODO 这个不需要每个循环都做 
        diff_gaussian_rasterization_jian.set_colors_bg(colors_bg)
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "pts_in_frustum": visible_pts, "pts": pts_total})
                progress_bar.update(10)
                gpu_mem_gb = torch.cuda.memory_reserved() / 1024**3
                log = {"iter": iteration, "loss": ema_loss_for_log, "pts_in_frustum": visible_pts, "pts": pts_total, "gpu_mem_gb": gpu_mem_gb}
                LOGGER.info(log)
                if WANDB and not DEBUG_MODE:
                    wandb.log(log, step=iteration)

            if iteration == opt.iterations:
                progress_bar.close()


            # Densification
            if iteration < opt.densify_until_iter:
                global_viewspace_points_grad = torch.zeros(initial_gaussians.get_xyz.shape[0], 3, device="cuda", requires_grad=False )
                global_viewspace_points_grad[initial_gaussians.visible_idx] = viewspace_point_tensor.grad
                global_visibility_filter = initial_gaussians.visible_idx[visibility_filter]
                
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
    
    initial_gaussians: GaussianModel = scene.gaussians
    
    # partition
    initial_gaussians.partition(num_blocks=config.BLOCK_NUMS) 
    initial_gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
    
    model_list: List[GaussianModel] = []
    
    for idx in range(len(initial_gaussians.block_idx_list)):
        model = initial_gaussians.get_kid(idx, opt)
        model_list.append(model)
        print(f"GS {idx} size: {model._xyz.shape[0]}")
        LOGGER.info(f"GS {idx} size: {model._xyz.shape[0]}")  
        

    # ---- Gradient Accumulation ----
    GA_START_ITER = config.GA_START_ITER
    ACCUMULATION_STEPS = config.GA_ACCUMULATION_STEPS
    GA_WARMUP_END_ITER = config.GA_WARMUP_END_ITER
    SQRT_K = math.sqrt(ACCUMULATION_STEPS)

    original_lrs = {}
    for mid, model in enumerate(model_list):
        original_lrs[mid] = {}
        for pg in model.optimizer.param_groups:
            if pg["name"] != "xyz":
                original_lrs[mid][pg["name"]] = pg['lr']

    start = time.time()
    block_importance_ema = {}
    block_xyz_diff_ema = {}
    block_scaling_diff_ema = {}
    block_opacity_diff_ema = {}
    loss_ema = {}
    block_update_interval = {}  # 每个 block 的更新间隔

    for iteration in range(first_iter, opt.iterations + 1):

        use_ga = config.GA_ENABLED and iteration >= GA_START_ITER
        is_ga_step_iter = use_ga and ((iteration - GA_START_ITER) % ACCUMULATION_STEPS == ACCUMULATION_STEPS - 1)

        for model in model_list:
            model.update_learning_rate(iteration)

        # Apply sqrt LR scaling + warm-up for non-xyz params
        if use_ga:
            if iteration < GA_WARMUP_END_ITER:
                warmup_factor = (iteration - GA_START_ITER) / (GA_WARMUP_END_ITER - GA_START_ITER)
            else:
                warmup_factor = 1.0
            for mid, model in enumerate(model_list):
                for pg in model.optimizer.param_groups:
                    if pg["name"] != "xyz":
                        base_lr = original_lrs[mid][pg["name"]]
                        scale = 1.0 + warmup_factor * (SQRT_K - 1.0)
                        pg['lr'] = base_lr * scale
        
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
        
        # frustum culling
        if config.FRUSTUM_CULLING_ENABLED:
            for model in model_list:
                visible_mask = frustum_culling(model._xyz, viewpoint_cam.full_proj_transform)
                model.visible_idx = torch.nonzero(visible_mask, as_tuple=True)[0]
        else:
            for model in model_list:
                model.visible_idx = torch.arange(model._xyz.shape[0], device="cuda")
        
        if viewpoint_cam.image_name == debug_image_name:
            detail_path = os.path.join("/data/jian/debug", BRANCH, SCENE_NAME, debug_image_name, f"iter_{iteration}")
            os.makedirs(detail_path, exist_ok=True)
        
        # 无渲染全部结果 为计算Loss做准备
        rendered_list, depth_list, alpha_list = [], [], []
        visible_model_id_list = []
        act_contribution_list = [] # 真的有渲染结果的 block
        visible_pts = 0

        
        with torch.no_grad():
            for idx, model in enumerate(model_list):
                
                if model.visible_idx.shape[0] == 0:
                    continue
                
                visible_pts += model.visible_idx.shape[0]
                
                model.set_subset(model.visible_idx)
                render_pkg = render(viewpoint_cam, model, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                model.clear_subset()
                
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
                if viewpoint_cam.image_name == debug_image_name:
                    torchvision.utils.save_image(image, os.path.join(detail_path, f"block_{idx}_contri_{contributed_percent}" + ".png"))
                
                
                rendered_list.append(image)
                depth_list.append(depth)
                alpha_list.append(alphaLeft)
                
                    
                visible_model_id_list.append(idx)
                
        cpu_merge_result = merge_opt_kid(rendered_list, depth_list, alpha_list)
        C_sorted = cpu_merge_result["front_rgbs"] # 每个 block 的颜色贡献，已经按照正确的前后顺序排列好
        prefix_T = cpu_merge_result["prefix_T"]
        block_rank = cpu_merge_result["block_rank"]  # [K,H,W]，每个像素告诉你每个 block 的排序位置
        K, C, H, W = C_sorted.shape   
        colors_bg = cpu_merge_result["bg_rgb"]

        loss_list = {}
        
        # 分层更新：根据 importance 排名决定更新频率（增点结束后生效）
        if config.BLOCK_TIERED_UPDATE and iteration >= config.BLOCK_TIERED_START_ITER and len(block_importance_ema) > 1 and iteration % 100 == 0:
            sorted_ids = sorted(block_importance_ema.keys(), key=lambda k: block_importance_ema[k].item() if isinstance(block_importance_ema[k], torch.Tensor) else block_importance_ema[k], reverse=True)
            n_blocks = len(sorted_ids)
            top_k = max(1, int(n_blocks * config.BLOCK_TIERED_TOP_RATIO))
            bot_k = max(1, int(n_blocks * config.BLOCK_TIERED_BOT_RATIO))
            for rank, bid in enumerate(sorted_ids):
                if rank < top_k:
                    block_update_interval[bid] = config.BLOCK_TIERED_TOP_INTERVAL
                elif rank >= n_blocks - bot_k:
                    block_update_interval[bid] = config.BLOCK_TIERED_BOT_INTERVAL
                else:
                    block_update_interval[bid] = config.BLOCK_TIERED_MID_INTERVAL

        # 遍历所有可见block 轮流当active block
        for index, model_id in enumerate(visible_model_id_list):
            # 分层更新：不在更新步的 block 跳过反向传播和优化
            interval = block_update_interval.get(model_id, 1)
            should_update = (interval <= 1) or (iteration % interval == 0)

            model: GaussianModel = model_list[model_id]

            model.set_subset(model.visible_idx)
            render_pkg = render(viewpoint_cam, model, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            model.clear_subset()

            # pixel level
            image2 = render_pkg["render"]

            # gaussian points level
            viewspace_point_tensor2, visibility_filter2, radii2 = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if not should_update:
                continue

            rank_map = block_rank[index]  # [H,W]，当前block的渲染结果在每个像素上的排序位置
            idx = rank_map.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            idx = idx.expand(1, C, H, W)               # [1,3,H,W]

            # 3. 当前块(index = rank_map)在每个像素位置上能拿到的透射率
            prefix_T_k = prefix_T[:, 0].gather(dim=0, index = rank_map.unsqueeze(0)).squeeze(0)    # [H,W]

            # 4. 当前块(index = idx)块提供的颜色
            C_sorted_k = C_sorted.gather(dim=0, index=idx).squeeze(0)   # [3,H,W]

            # 5. 从最终结果中扣除当前块的贡献，得到背景图. 贡献由每个像素位置上提供的颜色乘以透射率得到
            C_base = cpu_merge_result["final_rgb"] - prefix_T_k * C_sorted_k

            # 6. 带梯度的渲染结果
            C_active = image2      # [3,H,W], has grad

            # 7. 把带梯度的渲染结果拼到背景上 用于计算loss
            image_with_block_grad = C_base + prefix_T_k * C_active

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image_with_block_grad *= alpha_mask

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image_with_block_grad, gt_image)
            ssim_value = ssim(image_with_block_grad, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth = 0
            diff_gaussian_rasterization_jian.set_colors_bg(colors_bg)
            loss.backward()
            loss_list[model_id] = loss.item()

            with torch.no_grad():

                statistic(model_id, model_list, loss, loss_ema, block_importance_ema, block_xyz_diff_ema, block_scaling_diff_ema, block_opacity_diff_ema)

                # Densification stats accumulation (每个iter都做)
                if iteration < opt.densify_until_iter:
                    global_viewspace_points_grad = torch.zeros(model.get_xyz.shape[0], 3, device="cuda", requires_grad=False )
                    global_viewspace_points_grad[model.visible_idx] = viewspace_point_tensor2.grad
                    global_visibility_filter = model.visible_idx[visibility_filter2]

                    model.max_radii2D[global_visibility_filter] = torch.max(model.max_radii2D[global_visibility_filter], radii2[visibility_filter2])
                    model.add_densification_stats2(global_viewspace_points_grad, global_visibility_filter)

                    # 判断本轮是否需要 densify / reset_opacity
                    # GA模式下只在 step 迭代触发，避免替换参数时丢失累积梯度
                    if use_ga:
                        should_densify = (is_ga_step_iter
                                          and iteration > opt.densify_from_iter
                                          and iteration % opt.densification_interval < ACCUMULATION_STEPS)
                        should_reset_opacity = (is_ga_step_iter
                                                and (iteration % opt.opacity_reset_interval < ACCUMULATION_STEPS
                                                     or (dataset.white_background and iteration == opt.densify_from_iter)))
                    else:
                        should_densify = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
                        should_reset_opacity = iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)

                    # 非GA：保持原始顺序 densify → step
                    if not use_ga:
                        if should_densify:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            model.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        if should_reset_opacity:
                            model.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    if not use_ga:
                        model.optimizer.step()
                        model.optimizer.zero_grad(set_to_none=True)
                    elif is_ga_step_iter:
                        model.optimizer.step()
                        model.optimizer.zero_grad(set_to_none=True)

                # GA：step 之后再 densify，保证累积梯度先被消费
                if use_ga and iteration < opt.densify_until_iter:
                    if should_densify:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        model.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    if should_reset_opacity:
                        model.reset_opacity()
                        
                        
        time_elapsed = time.time() - start
          
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            
            pts_total = 0
            for model in model_list:
                pts_total += model._xyz.shape[0]
            
            if iteration % 10 == 0:
                # progress bar
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "pts_in_frustum": visible_pts, "pts": pts_total})
                progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                
                gpu_mem_gb = torch.cuda.memory_reserved() / 1024**3
                log = {"iter": iteration, "loss": ema_loss_for_log, "pts_in_frustum": visible_pts, "cost": time_elapsed, "pts": pts_total, "gpu_mem_gb": gpu_mem_gb}

                # logging
                LOGGER.info(log)
                
                # wandb logging
                if WANDB and not DEBUG_MODE:
                    log_dict = {}
                    # 取所有importance
                    importance_values = torch.tensor( list(block_importance_ema.values()) )
                    imp_min = importance_values.min()
                    imp_max = importance_values.max()
                    for model_id in visible_model_id_list:
                        raw_importance = block_importance_ema[model_id]
                        # 0-1 归一化
                        norm_importance = (raw_importance - imp_min) / (imp_max - imp_min + 1e-8)
                        log_dict[f"importance/{model_id}"] = raw_importance.item()
                        log_dict[f"importance_norm/{model_id}"] = norm_importance.item()
                        log_dict[f"xyz_diff/{model_id}"] = block_xyz_diff_ema[model_id].item()
                        log_dict[f"scaling_diff/{model_id}"] = block_scaling_diff_ema[model_id].item()
                        log_dict[f"opacity_diff/{model_id}"] = block_opacity_diff_ema[model_id].item()
                        log_dict[f"block_size/{model_id}"] = model_list[model_id]._xyz.shape[0]
                        log_dict[f"block_loss/{model_id}"] = loss_list[model_id]
                    wandb.log(log_dict, step=iteration)
                        
        # saving Gaussians ply    
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            point_cloud_path = os.path.join(scene.model_path, f"point_cloud/{BRANCH}/iteration_{iteration}")
            for idx, model in enumerate(model_list):
                model.save_ply(os.path.join(point_cloud_path, f"point_cloud_sub_{idx}.ply"), include_block=False)
                
        # save debug image
        # if viewpoint_cam.image_name == debug_image_name:
        #     torchvision.utils.save_image(image_with_block_grad, os.path.join(IMG_PATH_IN_DEBUG, f"{iteration}" + ".png"))
                    
    # if (iteration in checkpoint_iterations):
    #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
    #     pth_path = os.path.join(args.model_path, f"point_cloud/{BRANCH}")
    #     torch.save((gaussians.capture(), iteration), pth_path + "/chkpnt" + str(iteration) + ".pth")


def statistic(model_id, model_list, loss, loss_ema, block_importance_ema, block_xyz_diff_ema, block_scaling_diff_ema, block_opacity_diff_ema):
    model = model_list[model_id]
    alpha = 0.95


    opacity_grad = model._opacity.grad
    scale_grad = model._scaling.grad
    xyz_grad = model._xyz.grad

    opacity_diff = opacity_grad.abs().mean() if opacity_grad is not None else torch.tensor(0.0, device="cuda")
    scale_diff = scale_grad.abs().mean() if scale_grad is not None else torch.tensor(0.0, device="cuda")
    xyz_diff = xyz_grad.abs().mean() if xyz_grad is not None else torch.tensor(0.0, device="cuda")

    importance = opacity_diff + scale_diff + xyz_diff

    if model_id not in block_importance_ema:
        block_importance_ema[model_id] = importance.detach()
        block_xyz_diff_ema[model_id] = xyz_diff.detach()
        block_scaling_diff_ema[model_id] = scale_diff.detach()
        block_opacity_diff_ema[model_id] = opacity_diff.detach()
        loss_ema[model_id] = loss.detach()
    else:
        block_importance_ema[model_id] = alpha * block_importance_ema[model_id] + (1 - alpha) * importance.detach()
        block_xyz_diff_ema[model_id] = alpha * block_xyz_diff_ema[model_id] + (1 - alpha) * xyz_diff.detach()
        block_scaling_diff_ema[model_id] = alpha * block_scaling_diff_ema[model_id] + (1 - alpha) * scale_diff.detach()
        block_opacity_diff_ema[model_id] = alpha * block_opacity_diff_ema[model_id] + (1 - alpha) * opacity_diff.detach()
        loss_ema[model_id] = alpha * loss_ema[model_id] + (1 - alpha) * loss


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
    hours = int(cost // 3600)
    minutes = int((cost % 3600) // 60)
    hhmm = f"{hours:02d}:{minutes:02d}"
    print("\nTraining complete.")

    print(f"\nTraining complete. Total time: {hhmm}")
    LOGGER.info(f"\nTraining complete. Total time: {hhmm}")
    if WANDB and not DEBUG_MODE:
        wandb.log({"time_cost": hhmm})
        run.finish()    
        
