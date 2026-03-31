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
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
import time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, get_git_branch
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from logger import get_logger, add_output_path
from torchvision.utils import save_image
import wandb

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
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_pts_thresholds=None, grow_mode=False):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    add_output_path(LOGGER, os.path.join(dataset.model_path, "logs"))
    add_output_path(LOGGER, os.path.join("debug", BRANCH, SCENE_NAME), prefix="train")

    # ---- densify debug log ----
    _densify_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(_densify_log_dir, exist_ok=True)
    _densify_log_path = os.path.join(_densify_log_dir, "densify_debug.log")
    _densify_log_file = open(_densify_log_path, "w")
    _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print
    def _tee_print(*args, **kwargs):
        _orig_print(*args, **kwargs)
        s = " ".join(str(a) for a in args)
        if "DENSIFY" in s or "ACCUM" in s:
            _densify_log_file.write(s + "\n")
            _densify_log_file.flush()
    import builtins
    builtins.print = _tee_print
    print(f"[INFO] densify debug log: {_densify_log_path}")
    LOGGER.info(f"Scene: {SCENE_NAME} | source: {dataset.source_path} | iterations: {opt.iterations}")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
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

    # Track which point-count thresholds have been triggered
    save_pts_triggered = set()
    if save_pts_thresholds is None:
        save_pts_thresholds = []

    time_start = time.time()
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    # benchmark stats collection
    BENCH_START, BENCH_END = 301, 700
    bench_its_list = []
    bench_alloc_list = []
    bench_rsv_list = []
    bench_loss_list = []
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        torch.cuda.reset_peak_memory_stats()

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

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

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            pts_M = gaussians.get_xyz.shape[0] / 1e6
            gpu_peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
            gpu_peak_rsv = torch.cuda.max_memory_reserved() / 1024**3
            alloc = torch.cuda.memory_allocated() / 1024**3
            rsv = torch.cuda.memory_reserved() / 1024**3
            elapsed = time.time() - time_start
            its = (iteration - first_iter) / elapsed if elapsed > 0 else 0

            # benchmark collection
            if BENCH_START <= iteration <= BENCH_END:
                bench_its_list.append(its)
                bench_alloc_list.append(alloc)
                bench_rsv_list.append(rsv)
                bench_loss_list.append(ema_loss_for_log)

            # progress bar - every iter
            progress_bar.set_postfix({"L": f"{ema_loss_for_log:.4f}", "pts": f"{pts_M:.2f}M", "alloc": f"{alloc:.2f}", "rsv": f"{rsv:.2f}", "peak": f"{gpu_peak_alloc:.2f}", "it/s": f"{its:.1f}"})
            progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            log = {"iter": iteration, "L": round(ema_loss_for_log, 4), "pts": gaussians.get_xyz.shape[0], "alloc": round(alloc, 2), "rsv": round(rsv, 2), "peak_alloc": round(gpu_peak_alloc, 2), "peak_rsv": round(gpu_peak_rsv, 2), "it/s": round(its, 1), "elapsed": round(elapsed, 1)}
            LOGGER.info(log)

            if WANDB and not DEBUG_MODE:
                wandb.log(log, step=iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if iteration % 100 == 0:
                    grad_norms = torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1)
                    n_vis = visibility_filter.sum().item() if visibility_filter.dtype == torch.bool else visibility_filter.shape[0]
                    print(f"[ACCUM-VANILLA] iter={iteration} n_total={gaussians._xyz.shape[0]} "
                          f"n_vis={n_vis} grad_mean={grad_norms.mean().item():.8f} grad_max={grad_norms.max().item():.8f}")
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if grow_mode:
                        gaussians.densify_and_prune(opt.densify_grad_threshold * 0.25, 0.005, scene.cameras_extent, size_threshold, radii,
                                                    no_prune=True, clone_times=3, split_n=6)
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if not grow_mode:
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Save by point count thresholds (unit: 万)
            if save_pts_thresholds:
                num_pts = gaussians.get_xyz.shape[0]
                num_pts_wan = num_pts / 1e4
                for threshold in save_pts_thresholds:
                    if threshold not in save_pts_triggered and num_pts_wan >= threshold:
                        save_pts_triggered.add(threshold)
                        tag = f"pts_{threshold}w"
                        print(f"\n[ITER {iteration}] Points reached {threshold}万 ({num_pts}), saving as {tag}")
                        point_cloud_path = os.path.join(scene.model_path, f"point_cloud/{tag}")
                        os.makedirs(point_cloud_path, exist_ok=True)
                        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                        torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"chkpnt_{tag}.pth"))

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
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    time_end = time.time()
    cost = time_end - time_start
    print(f"Training time cost: [{cost:.2f}] seconds.")

    # benchmark stats
    if bench_its_list:
        n = len(bench_its_list)
        p = sys.__stdout__.write
        import socket, subprocess
        hostname = socket.gethostname()
        commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        p(f"\n{'='*50}\n")
        p(f"  [{hostname}] Benchmark (iter {BENCH_START}-{BENCH_END}, {n} samples)\n")
        p(f"  Branch: {BRANCH}  Commit: {commit_id}\n")
        p(f"{'='*50}\n")
        p(f"  平均 it/s:           {sum(bench_its_list)/n:.2f}\n")
        p(f"  平均 loss:           {sum(bench_loss_list)/n:.6f}\n")
        p(f"  平均占用 mem (alloc): {sum(bench_alloc_list)/n:.2f} GB\n")
        p(f"  平均分配 mem (rsv):   {sum(bench_rsv_list)/n:.2f} GB\n")
        p(f"  总平均 mem:           {(sum(bench_alloc_list)+sum(bench_rsv_list))/(2*n):.2f} GB\n")
        p(f"{'='*50}\n")

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

                # Create output dirs for rendered images
                if BRANCH is not None:
                    renders_dir = os.path.join(scene.model_path, "rendered", BRANCH, config['name'], f"ours_{iteration}", "renders")
                    gt_dir = os.path.join(scene.model_path, "rendered", BRANCH, config['name'], f"ours_{iteration}", "gt")
                    os.makedirs(renders_dir, exist_ok=True)
                    os.makedirs(gt_dir, exist_ok=True)

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # Save rendered images to disk
                    if BRANCH is not None:
                        save_image(image, os.path.join(renders_dir, f"{viewpoint.image_name}.png"))
                        save_image(gt_image, os.path.join(gt_dir, f"{viewpoint.image_name}.png"))
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if LOGGER is not None:
                    LOGGER.info(f"[Eval {config['name']} iter={iteration}] L1: {l1_test:.4f} PSNR: {psnr_test:.4f}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                # Save results.json
                if BRANCH is not None:
                    results_dir = os.path.join(scene.model_path, "rendered", BRANCH, config['name'])
                    results_path = os.path.join(results_dir, "results.json")
                    if os.path.exists(results_path):
                        with open(results_path, "r") as f:
                            results_data = json.load(f)
                    else:
                        results_data = {}
                    results_data[f"ours_{iteration}"] = {
                        "PSNR": psnr_test.item(),
                        "L1": l1_test.item(),
                    }
                    with open(results_path, "w") as f:
                        json.dump(results_data, f, indent=2)

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
    parser.add_argument("--save_pts", nargs="+", type=int, default=[300,400,500,600,700,800,900,1000], help="Save when point count reaches these thresholds (unit: 10k)")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--grow_mode", action="store_true", default=False, help="Aggressive densify: no prune, lower threshold, 3x more points")
    parser.add_argument('--git_branch', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    SCENE_NAME = args.source_path.rstrip('/').split('/')[-1]
    DATASET_NAME = args.source_path.rstrip('/').split('/')[-2]

    if args.git_branch is not None:
        BRANCH = args.git_branch
    else:
        BRANCH = get_git_branch()

    LOGGER = get_logger(SCENE_NAME, os.path.join("./logs", "train", BRANCH, SCENE_NAME))
    DEBUG_MODE = sys.gettrace() is not None

    if WANDB and not DEBUG_MODE:
        try:
            wandb.login()
            run = wandb.init(
                project=DATASET_NAME+"_densify_test",
                name=f"{SCENE_NAME}_{BRANCH}",
                group=SCENE_NAME,
                config=vars(op.extract(args))
            )
            wandb.define_metric("iteration")
        except Exception as e:
            print(f"wandb init failed: {e}, continuing without wandb")
            WANDB = False

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    os.makedirs("debug", exist_ok=True)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.save_pts, args.grow_mode)

    # All done
    print("\nTraining complete.")
