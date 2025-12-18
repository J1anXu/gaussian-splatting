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
from scene_p import Scene_p
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_p import render, render_and_merge, render_and_merge2
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer_p import GaussianModel_p
WANDB = True
import wandb
import time

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_p")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    total_time = 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        t0 = time.perf_counter()
        rendering = render_and_merge2(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        t1 = time.perf_counter()
        dt = t1 - t0
        total_time += dt

        print(f"[View {idx}] render_and_merge time: {dt:.3f}s")
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        img_name = view.image_name
        torchvision.utils.save_image(rendering, os.path.join(render_path, img_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, img_name + ".png"))
    print(f"Total time: {total_time:.3f}s")
    print(f"Average per view: {total_time / len(views):.3f}s")
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel_p(dataset.sh_degree, max_block_size = 300000)
        scene = Scene_p(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.partition_for_rendering()

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
    scene_name = args.source_path.strip('/').split('/')[-1]

    if WANDB:
        wandb.login()
        run = wandb.init(
            project="party_rendering",
            name = f"{scene_name}_{time.strftime('%Y%m%d_%H%M%S')}",
            job_type="rendering",
        )
        wandb.define_metric("iteration")  # 
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)