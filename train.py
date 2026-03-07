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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, get_git_branch
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import diff_gaussian_rasterization

import wandb
import time
from logger import get_logger

DEBUG_MODE = False
WANDB = False
LOGGER = None
BRANCH = None

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

    # Log resolution
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    train_res = f"{train_cams[0].image_width}x{train_cams[0].image_height}" if train_cams else "N/A"
    test_res = f"{test_cams[0].image_width}x{test_cams[0].image_height}" if test_cams else "N/A"
    res_msg = f"Resolution: train={train_res} ({len(train_cams)} views), test={test_res} ({len(test_cams)} views)"
    LOGGER.info(res_msg)
    print(res_msg)

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
    debug_image_name = "_DSC8680.JPG"
    img_path_in_debug = os.path.join("debug", BRANCH, debug_image_name)
    # os.makedirs(img_path_in_debug, exist_ok=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), miniters=10, mininterval=1.0)
    first_iter += 1
    global_tic = time.time()
    step_tic = global_tic
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

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

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask


        # if viewpoint_cam.image_name == debug_image_name:
        #     # block_rank[k, h, w] 表示： 在像素 (h, w) 处，第 k 个 block 在“按深度排序后”的层级排名（rank）
        #     torchvision.utils.save_image(image, os.path.join(img_path_in_debug, f"{iteration}" + ".png"))
        #     LOGGER.info(f"Saved debug images at iteration {iteration} for {debug_image_name}")


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        
        # colors_bg = torch.zeros(image.shape, device="cuda")
        # diff_gaussian_rasterization.set_colors_bg(colors_bg)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # EMA loss
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            total_points = gaussians.get_xyz.shape[0]
            gpu_used = torch.cuda.memory_allocated() / 1024**3
            gpu_rsv = torch.cuda.memory_reserved() / 1024**3

            # wandb every iteration
            if WANDB and not DEBUG_MODE:
                wandb.log({
                    "iter": iteration,
                    "loss": ema_loss_for_log,
                    "cost": time.time() - global_tic,
                    "pts": total_points,
                    "gpu_mem_gb": gpu_rsv,
                }, step=iteration)

            # Progress bar + file log every 10 iters
            if iteration % 10 == 0:
                now = time.time()
                throughput = 10.0 / max(now - step_tic, 1e-8)
                step_tic = now
                pts_m = total_points / 1e6
                sh = gaussians.active_sh_degree
                desc = (
                    f"loss={ema_loss_for_log:.3f}| sh={sh}| "
                    f"pts={pts_m:.2f}M| "
                    f"mem={gpu_used:.2f}/{gpu_rsv:.2f}G| {throughput:.2f} it/s"
                )
                progress_bar.set_description(desc)
                progress_bar.update(10)

                elapsed = now - global_tic
                LOGGER.info(
                    f"step={iteration}/{opt.iterations} | loss={ema_loss_for_log:.4f} l1={Ll1.item():.4f} ssim={1.0 - ssim_value.item():.4f} | "
                    f"pts={pts_m:.2f}M | "
                    f"mem={gpu_used:.2f}/{gpu_rsv:.2f}G | sh={sh} | {throughput:.2f} it/s | elapsed={elapsed:.1f}s"
                )

            if iteration == opt.iterations:
                progress_bar.close()

            # Tensorboard
            if tb_writer and iteration % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                tb_writer.add_scalar("train/loss", loss.item(), iteration)
                tb_writer.add_scalar("train/l1loss", Ll1.item(), iteration)
                tb_writer.add_scalar("train/ssimloss", 1.0 - ssim_value.item(), iteration)
                tb_writer.add_scalar("train/num_GS", total_points, iteration)
                tb_writer.add_scalar("train/mem", mem, iteration)
                tb_writer.flush()

            # Eval
            if iteration in testing_iterations:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # Save
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, BRANCH)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
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
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                pth_path = os.path.join(args.model_path, f"checkpoints/{BRANCH}")
                os.makedirs(pth_path, exist_ok = True)
                torch.save((gaussians.capture(), iteration), pth_path + "/chkpnt" + str(iteration) + ".pth")

    return scene

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.model_path, "tb"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    """Run evaluation on test/train splits and log to TB + file logger."""
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                          {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    for cfg in validation_configs:
        if cfg['cameras'] and len(cfg['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(cfg['cameras']):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                if tb_writer and (idx < 5):
                    tb_writer.add_images(cfg['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(cfg['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(cfg['cameras'])
            l1_test /= len(cfg['cameras'])
            eval_msg = f"[Eval {cfg['name']} step={iteration}] L1={l1_test:.4f} PSNR={psnr_test:.3f}"
            print(f"\n{eval_msg}")
            LOGGER.info(eval_msg)
            if tb_writer:
                tb_writer.add_scalar(cfg['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(cfg['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.flush()
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
    if args.git_branch is not None:
        BRANCH = args.git_branch
    else:
        BRANCH = get_git_branch()

    # Initialize system state (RNG)
    safe_state(args.quiet)
    SCENE_NAME = args.source_path.split('/')[-1]
    DATASET_NAME = args.source_path.split('/')[-2]
    if args.git_branch is not None:
        BRANCH = args.git_branch
    else:
        BRANCH = get_git_branch()

    LOGGER = get_logger(SCENE_NAME, os.path.join(args.model_path, "logs"))
    DEBUG_MODE = sys.gettrace() is not None
    LOGGER.info(f"Scene: {SCENE_NAME} | data_dir: {args.source_path} | max_steps: {op.extract(args).iterations} | sh_degree: {lp.extract(args).sh_degree}")

    if WANDB and not DEBUG_MODE:
        wandb.login()
        run = wandb.init(
            project=DATASET_NAME,
            name=f"{SCENE_NAME}_{BRANCH}",
            group=SCENE_NAME,
            config=vars(op.extract(args)),
        )
        wandb.define_metric("iteration")
        

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    global_tic = time.time()
    scene = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    total_cost = time.time() - global_tic
    hours = int(total_cost // 3600)
    minutes = int((total_cost % 3600) // 60)
    hhmm = f"{hours:02d}:{minutes:02d}"

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    train_res = f"{train_cams[0].image_width}x{train_cams[0].image_height}" if train_cams else "N/A"
    test_res = f"{test_cams[0].image_width}x{test_cams[0].image_height}" if test_cams else "N/A"
    summary = f"Training complete. Total time: {hhmm} | Resolution: train={train_res}, test={test_res}"
    LOGGER.info(summary)
    print(f"\n{summary}")

    if WANDB and not DEBUG_MODE:
        wandb.log({"time_cost": hhmm})
        run.finish()

