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
from gaussian_renderer import render, merge, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import time
from partition.partition import generate_block_masks
from logger import get_logger
import torchvision
import config
WANDB = True
LOGGER = None
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def check_update(cpu_param_before, cpu_param_after, mask):
    mask = mask.to("cpu")

    # 1) 取出 mask 对应的 slice
    before_slice = cpu_param_before[mask]
    after_slice  = cpu_param_after[mask]

    diff_mask = (after_slice - before_slice).abs().sum().item()

    # 2) 取出非-mask 部分
    all_idx = torch.arange(cpu_param_before.shape[0])
    other_idx = all_idx[~torch.isin(all_idx, mask)]
    before_other = cpu_param_before[other_idx]
    after_other  = cpu_param_after[other_idx]

    diff_other = (after_other - before_other).abs().sum().item()

    print(f"[MASK  UPDATED] diff = {diff_mask:.8f}")
    print(f"[OTHERS STATIC] diff = {diff_other:.8f}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    
    gaussians.training_setup_for_part(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # ============================================================
    #   Partitioned Training Loop (recommended full structure)
    # ============================================================

    just_densified = False
    
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
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

        # --------------------------
        # split into blocks (only when densified)
        # --------------------------
        # if iteration == first_iter or just_densified:
        #     block_masks, _ = generate_block_masks(gaussians._xyz, max_size = 500_000)
        #     block_masks = [m for m in block_masks if len(m) > 0]
        #     just_densified = False

        block_masks = [torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device)]

        all_renders = []
        all_depths  = []
        all_alphas  = []
        all_viewspace_points = []
        all_visibility_filter = []
        all_radii = []

        img_name = viewpoint_cam.image_name

        # 1. 创建目录
        save_dir = os.path.join("debug", img_name)
        os.makedirs(save_dir, exist_ok=True)

        # 无渲染全部结果 为计算Loss做准备
        with torch.no_grad():
            idx = 0
            for mask in block_masks:
                if len(mask) == 0:
                    continue
                # 开启subset会导致高斯只能被访问到mask指定的部分(get()函数被mask限制) 所以渲染结果也就只包含这些高斯产生的RGB
                gaussians.start_subset(mask)
                out = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                gaussians.end_subset()
                
                if config.PRINT_EVERYTHING:
                    torchvision.utils.save_image(out["render"].detach().cpu(), os.path.join(save_dir, f"block_{idx}.png"))
                idx += 1

                # 存储所有信息(移动到CPU)
                if config.CAL_RES_2_CPU:
                    all_renders.append(out["render"].detach().cpu())
                    all_depths.append(out["depth"].detach().cpu())
                    all_alphas.append(out["alphaLeft"].detach().cpu())
                    all_viewspace_points.append(out["viewspace_points"].detach().cpu())
                    all_visibility_filter.append(out["visibility_filter"].detach().cpu())
                    all_radii.append(out["radii"].detach().cpu())
                else:
                    all_renders.append(out["render"].detach())
                    all_depths.append(out["depth"].detach())
                    all_alphas.append(out["alphaLeft"].detach())
                    all_viewspace_points.append(out["viewspace_points"].detach())
                    all_visibility_filter.append(out["visibility_filter"].detach())
                    all_radii.append(out["radii"].detach())
                
        
        # contribution_lists
        black_block_indices = []
        available_block_indices = []
        for idx, r in enumerate(all_renders):
            # r 是 CPU tensor: [3,H,W]
            if r.abs().sum().item() == 0:  # 全黑
                black_block_indices.append(idx)
            else:
                available_block_indices.append(idx)

        cpu_merge_result = merge(all_renders, all_depths, all_alphas, all_viewspace_points, all_visibility_filter, all_radii)
        
        C_sorted = cpu_merge_result["front_rgbs"] # 每个 block 的颜色贡献，已经按照正确的前后顺序排列好
        T_sorted = cpu_merge_result["front_alphas"] # 每个 block 的透明度，已经按照正确的前后顺序排列好
        prefix_T = cpu_merge_result["prefix_T"]
        sort_idx = cpu_merge_result["sort_idx"]
        block_rank = cpu_merge_result["block_rank"]  # [K,H,W]，每个像素告诉你每个 block 的排序位置
        radii_cpu = cpu_merge_result["final_radii"]
        
        
        gt_image = viewpoint_cam.original_image.cuda()


        merge_res = cpu_merge_result["final_rgb"].detach().cpu()
        
        if config.PRINT_EVERYTHING:
            torchvision.utils.save_image(merge_res, os.path.join(save_dir, f"merge_res.png"))

        # 遍历所有block 轮流当active block
        for block_id in available_block_indices:
            active_mask = block_masks[block_id]
            
            # 1. 打开subset模式 使GPU只能看到指定的高斯, 并且开启这部分高斯的梯度
            gaussians.start_subset(active_mask, requires_grad=True) 
            
            # 2. 渲染指定部分的高斯
            active_block_out = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            
            K, C, H, W = C_sorted.shape   # C应该=3
            
            rank_map = block_rank[block_id]  # [H,W]，每个像素告诉你排序位置
            
            # 3. 取出对应的 prefix （透明度前缀）
            prefix_T_k = prefix_T[:, 0].gather(dim=0, index=rank_map.unsqueeze(0)).squeeze(0)    # [H,W]

            idx = rank_map.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            idx = idx.expand(1, C, H, W)                 # [1,3,H,W]
            
            # 4. 取出对应的 C （颜色贡献）
            # C_sorted 就是 “每个 block 在每个像素上实际贡献到最终图像中的颜色项”，并且它可以直接从像素上扣除。
            C_sorted_k = C_sorted.gather(dim=0,index=idx).squeeze(0)   # [3,H,W]

            # C_base 是除了 active block 之外所有 block 的贡献的和（CPU）
            # 或者说：从全量渲染的最终图像中扣除当前 block 的旧颜色贡献，得到由其他 block 单独形成的背景图
            C_base = cpu_merge_result["final_rgb"] - prefix_T_k * C_sorted_k

            # 把 CPU 的 prefix_T_k 和 C_base 搬到 GPU（很小，成本很低）
            prefix_T_k_gpu = prefix_T_k.to("cuda")
            C_base_gpu  = C_base.to("cuda")

            # 合成 final image（GPU）
            # 因为最终图像是所有 block 按透明度前缀系数的线性加权和
            # 所以只需从全图中减去该 block 的旧贡献并加上重新渲染的新贡献，就能得到与全量渲染一致的结果
            C_active = active_block_out["render"]      # [3,H,W], has grad

            
            image = C_base_gpu + prefix_T_k_gpu * C_active
            
            if config.PRINT_EVERYTHING:
                torchvision.utils.save_image(image, os.path.join(save_dir, f"reassemble.png"))
                torchvision.utils.save_image(C_active, os.path.join(save_dir, f"C_active.png"))
                torchvision.utils.save_image(C_sorted_k, os.path.join(save_dir, f"C_sorted_k.png"))
                torchvision.utils.save_image(C_base, os.path.join(save_dir, f"C_base.png"))
                
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.to(image.device)
                image *= alpha_mask
            
            # 5. 计算 loss（依赖于 active_out → 依赖于当前 subset 的高斯）
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = final_depth # 这里需要全局depth
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0
                
            # 6. 只清 subset 的 grad
            gaussians.zero_grad_subset()
            
            loss.backward()
            
            # 7. 在 GPU 上用 subset 优化器做 Adam 更新，并把参数 & state 写回 CPU
            gaussians.adam_step_subset(active_mask)   
                         
            # 9. 关闭subset模式 清空GPU
            gaussians.end_subset() 
            
            
        
        # --- 所有 block 完成后 ---
        gaussians.adam_step += 1
        
        # 10. 更新学习率
        gaussians.update_learning_rate(iteration)
        
        iter_end.record()
        
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            total_points = gaussians.get_xyz.shape[0]

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    "Pts": total_points
                    })
                progress_bar.update(10)
                if WANDB:
                    wandb.log({
                        "iteration": iteration,
                        "loss": ema_loss_for_log,
                        "pts": total_points
                    }, step=iteration)
                LOGGER.info({
                    "iter":iteration,
                    "loss":ema_loss_for_log,
                    "pts":total_points})
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter_cpu] = torch.max(gaussians.max_radii2D[visibility_filter_cpu], radii_cpu[visibility_filter_cpu])
            #     gaussians.add_densification_stats(viewspace_point_tensor_cpu, visibility_filter_cpu)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii_cpu)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                
                #!几何参数的优化在上面的 per-block 循环里已经做完了，这里不要再 step 了
                # if use_sparse_adam:
                #     visible = radii_cpu > 0
                #     gaussians.optimizer.step(visible, radii_cpu.shape[0])
                #     gaussians.optimizer.zero_grad(set_to_none = True)
                # else:
                #     gaussians.optimizer.step()
                #     gaussians.optimizer.zero_grad(set_to_none = True)

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000]) # 这是用来保存的 不是用来load的
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    scene_name = args.source_path.strip('/').split('/')[-1]
    LOGGER = get_logger(scene_name, os.path.join("./logs", "train_p", scene_name))
    if WANDB:
        wandb.login()
        run = wandb.init(
            project="PartitionGS",
            name = f"bicycle_{time.strftime('%Y%m%d_%H%M%S')}",
            job_type="train",
            config=vars(op.extract(args))
        )
        wandb.define_metric("iteration")  # 
        
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
