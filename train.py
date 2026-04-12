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
import faulthandler; faulthandler.enable()
from typing import List
import torch
from random import randint
from contextlib import contextmanager
from collections import defaultdict

import torchvision
from utils.debug_utils import save_block_img, save_depth_list, save_rgb_layers, save_layer_contribution
from utils.loss_utils import l1_loss, ssim
from fused_ssim import fused_ssim
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
from logger import get_logger, add_output_path, log_kv, log_runtime_context, log_section
import config
import diff_gaussian_rasterization_wenqi_tam
from TimerManager import  TraceManager, TID_MAIN, PID_CPU
from pipeline_grad_sync import PipelinedGradSync
SCENE_NAME = None
BRANCH = None
DEBUG_MODE = False

WANDB = False
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


class IterationProfiler:
    """Lightweight per-iteration wall-time logger.

    By default this does not synchronize CUDA around every span, so GPU spans
    remain low overhead. Pass --profile_sync when you want slower but more
    accurate stage wall times for a short window.
    """

    def __init__(self, logger, enabled=False, start=1, end=0, every=1, sync_cuda=False):
        self.logger = logger
        self.enabled = enabled
        self.start = start
        self.end_iteration = end
        self.every = max(1, every)
        self.sync_cuda = sync_cuda
        self.active = False
        self.iteration = None
        self._t0 = None
        self.spans = {}
        self.data = {}
        self._last_peak_alloc_gb = 0.0
        self._last_peak_reserved_gb = 0.0
        self._mem_spike_threshold_gb = 0.02

    def begin(self, iteration):
        self.iteration = iteration
        in_window = iteration >= self.start and (self.end_iteration <= 0 or iteration <= self.end_iteration)
        on_stride = ((iteration - self.start) % self.every) == 0
        self.active = self.enabled and in_window and on_stride
        self._t0 = time.perf_counter()
        self.spans = {}
        self.data = {}
        self._last_peak_alloc_gb = 0.0
        self._last_peak_reserved_gb = 0.0

    def set(self, key, value):
        if self.active:
            self.data[key] = value

    def add(self, key, value):
        if self.active:
            self.data[key] = self.data.get(key, 0) + value

    def block(self, kind, block_id, **fields):
        if not self.active:
            return
        key = f"{kind}_blocks"
        blocks = self.data.setdefault(key, [])
        block_id = int(block_id)
        for item in blocks:
            if item["id"] == block_id:
                item.update(fields)
                return
        blocks.append({"id": block_id, **fields})

    @contextmanager
    def span(self, name):
        if not self.active:
            yield
            return
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.spans[name] = self.spans.get(name, 0.0) + elapsed_ms
            self._record_mem_peak(name)

    def _record_mem_peak(self, span_name):
        if not torch.cuda.is_available():
            return
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        alloc_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3

        # This cleanup intentionally runs before reset_peak_memory_stats(), so
        # the CUDA peak counters still describe the previous iteration.
        if span_name == "pre_iter_cuda_cache_cleanup":
            return

        if span_name == "reset_peak_memory_stats":
            self._last_peak_alloc_gb = peak_alloc_gb
            self._last_peak_reserved_gb = peak_reserved_gb
            return

        # reset_peak_memory_stats can lower the counters inside an active
        # profiling window; restart the local spike detector after that.
        if peak_reserved_gb + 1e-6 < self._last_peak_reserved_gb:
            self._last_peak_alloc_gb = peak_alloc_gb
            self._last_peak_reserved_gb = peak_reserved_gb
            return

        d_peak_alloc = peak_alloc_gb - self._last_peak_alloc_gb
        d_peak_reserved = peak_reserved_gb - self._last_peak_reserved_gb
        if (
            d_peak_alloc >= self._mem_spike_threshold_gb
            or d_peak_reserved >= self._mem_spike_threshold_gb
        ):
            spikes = self.data.setdefault("mem_spikes", [])
            spikes.append({
                "span": span_name,
                "d_peak_alloc_gb": round(float(max(0.0, d_peak_alloc)), 4),
                "d_peak_reserved_gb": round(float(max(0.0, d_peak_reserved)), 4),
                "peak_alloc_gb": round(float(peak_alloc_gb), 4),
                "peak_reserved_gb": round(float(peak_reserved_gb), 4),
                "alloc_gb": round(float(alloc_gb), 4),
                "reserved_gb": round(float(reserved_gb), 4),
            })
            self._last_peak_alloc_gb = max(self._last_peak_alloc_gb, peak_alloc_gb)
            self._last_peak_reserved_gb = max(self._last_peak_reserved_gb, peak_reserved_gb)

    def end(self):
        if not self.active:
            return
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        payload = {
            "iter": self.iteration,
            "total_ms": round((time.perf_counter() - self._t0) * 1000.0, 3),
            "spans_ms": {k: round(v, 3) for k, v in sorted(self.spans.items())},
            **self.data,
        }
        log_kv(self.logger, "profile_iter", payload)


def log_trace_summary(tracer: TraceManager, logger, top_n=30):
    """Summarize Chrome trace events into the normal log for quick grep."""
    events = getattr(tracer, "events", [])
    grouped = defaultdict(list)
    for ev in events:
        if ev.get("ph") == "X" and "dur" in ev:
            grouped[ev.get("name", "unknown")].append(ev["dur"] / 1000.0)
    if not grouped:
        logger.info("[trace_summary] no timed trace events")
        return

    rows = []
    for name, vals in grouped.items():
        total = sum(vals)
        rows.append({
            "name": name,
            "count": len(vals),
            "total_ms": round(total, 3),
            "avg_ms": round(total / len(vals), 3),
            "max_ms": round(max(vals), 3),
        })
    rows.sort(key=lambda x: x["total_ms"], reverse=True)
    for row in rows[:top_n]:
        log_kv(logger, "trace_summary", row)


def attach_runtime_options(opt, args):
    opt.test_diagnostics = args.test_diagnostics
    opt.profile_log_requested = args.profile_log
    opt.profile_log = args.profile_log and args.test_diagnostics
    opt.profile_from = args.profile_from
    opt.profile_until = args.profile_until
    opt.profile_every = args.profile_every
    opt.profile_sync = args.profile_sync and opt.profile_log
    opt.trace_from = args.trace_from
    opt.trace_until = args.trace_until
    opt.bench_from = args.bench_from
    opt.bench_until = args.bench_until
    opt.disable_densify = args.disable_densify
    opt.gpu_cache_threshold_gb = args.gpu_cache_threshold_gb
    opt.gpu_cache_hard_limit_gb = args.gpu_cache_hard_limit_gb
    opt.gpu_cache_stage_entry_limit_gb = args.gpu_cache_stage_entry_limit_gb
    opt.cuda_empty_cache_interval = args.cuda_empty_cache_interval
    opt.split_size_override = args.split_size_override
    opt.legacy_per_block_loss = args.legacy_per_block_loss
    opt.gpu_packed_cache_strategy = args.gpu_packed_cache_strategy
    opt.gpu_packed_cache_point_budget = (
        args.gpu_packed_cache_point_budget
        if args.gpu_packed_cache_point_budget > 0
        else config.GPU_PACKED_CACHE_POINT_BUDGET
    )
    return opt


def maybe_empty_cuda_cache(opt, iteration=None, profiler=None, reason="",
                           force=False, respect_interval=True, allow_periodic=True,
                           hard_limit_override_gb=None,
                           key_prefix="cuda"):
    """Release PyTorch CUDA allocator cache when it is mostly unused.

    `memory_reserved()` is what tools like nvidia-smi see.  A skipped iteration
    can bypass the normal end-of-iteration cleanup, so callers can force this on
    early-exit paths after dropping tensor references.
    """
    if not torch.cuda.is_available():
        return False

    alloc_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()
    cache_slack_gb = (reserved_bytes - alloc_bytes) / 1024**3
    cache_threshold_gb = getattr(opt, "gpu_cache_threshold_gb", config.GPU_CACHE_THRESHOLD_GB)
    hard_limit_gb = (
        hard_limit_override_gb
        if hard_limit_override_gb is not None
        else getattr(opt, "gpu_cache_hard_limit_gb", config.GPU_CACHE_HARD_LIMIT_GB)
    )
    cache_interval = max(1, getattr(opt, "cuda_empty_cache_interval", config.CUDA_EMPTY_CACHE_INTERVAL))
    interval_ok = (
        not respect_interval
        or iteration is None
        or iteration % cache_interval == 0
    )
    hard_limit_exceeded = hard_limit_gb > 0 and reserved_bytes / 1024**3 > hard_limit_gb
    should_empty_cache = force or (
        cache_threshold_gb > 0
        and cache_slack_gb > cache_threshold_gb
        and (
            hard_limit_exceeded
            or (allow_periodic and interval_ok)
        )
    )

    if profiler is not None:
        profiler.set(f"{key_prefix}_cache_slack_gb", round(float(cache_slack_gb), 4))
        profiler.set(f"{key_prefix}_cache_alloc_gb_before", round(float(alloc_bytes / 1024**3), 4))
        profiler.set(f"{key_prefix}_cache_reserved_gb_before", round(float(reserved_bytes / 1024**3), 4))
        profiler.set(f"{key_prefix}_cache_hard_limit_gb", round(float(hard_limit_gb), 4))
        profiler.set(f"{key_prefix}_cache_hard_limit_exceeded", bool(hard_limit_exceeded))
        profiler.set(f"{key_prefix}_cache_interval", int(cache_interval))
        profiler.set(f"{key_prefix}_cache_interval_ok", bool(interval_ok))
        profiler.set(f"{key_prefix}_cache_periodic_allowed", bool(allow_periodic))
        profiler.set(f"{key_prefix}_empty_cache", False)
        if reason:
            profiler.set(f"{key_prefix}_empty_cache_reason", reason)

    if should_empty_cache:
        torch.cuda.empty_cache()
        if profiler is not None:
            profiler.set(f"{key_prefix}_empty_cache", True)
            profiler.set(f"{key_prefix}_cache_reserved_gb_after", round(float(torch.cuda.memory_reserved() / 1024**3), 4))

    return should_empty_cache


def estimate_gpu_packed_bytes(submodel, n_vis):
    pack_d = int(getattr(submodel, "_pack_D", 0) or 0)
    alloc_step = int(getattr(submodel, "ALLOC_STEP", 4096) or 4096)
    n_vis = int(n_vis)
    alloc_n = ((n_vis + alloc_step - 1) // alloc_step) * alloc_step if n_vis > 0 else 0
    return alloc_n * pack_d * 4


def select_best_fit_gpu_packed_cache(candidates, point_budget):
    """Pick the fullest tail subset under the configured visible-point budget."""
    point_budget = int(point_budget)
    candidates = [item for item in candidates if 0 < int(item["n_vis"]) <= point_budget]
    if point_budget <= 0 or not candidates:
        return []

    dp = {0: (0, 0, ())}  # used_points -> (bytes, block_count, selected_ids)
    for item in candidates:
        weight = int(item["n_vis"])
        sid = int(item["id"])
        n_bytes = int(item["bytes"])
        for used, state in list(dp.items()):
            new_used = used + weight
            if new_used > point_budget:
                continue
            new_state = (state[0] + n_bytes, state[1] + 1, state[2] + (sid,))
            old_state = dp.get(new_used)
            if old_state is None or new_state[:2] > old_state[:2]:
                dp[new_used] = new_state

    best_used = 0
    best_state = dp[0]
    for used, state in dp.items():
        if (used, state[0], state[1]) > (best_used, best_state[0], best_state[1]):
            best_used = used
            best_state = state
    return list(best_state[2])


def plan_gpu_packed_cache(submodel_list, valid_ids, strategy, point_budget):
    strategy = strategy or "none"
    point_budget = int(point_budget or 0)
    selected_ids = []
    budget_n_vis = 0
    budget_bytes = 0
    total_n_vis = 0
    total_bytes = 0
    candidate_count = 0
    candidate_n_vis = 0
    candidate_bytes = 0
    budget_source = "none"
    selection_policy = "none"

    if valid_ids and strategy != "none":
        largest_id = valid_ids[0]
        largest_n_vis = int(submodel_list[largest_id].visible_indices.shape[0])
        largest_bytes = estimate_gpu_packed_bytes(submodel_list[largest_id], largest_n_vis)
        if strategy == "largest":
            selected_ids = [largest_id]
            budget_n_vis = largest_n_vis
            budget_bytes = largest_bytes
            total_n_vis = largest_n_vis
            total_bytes = largest_bytes
            budget_source = "largest_visible"
            selection_policy = "largest"
        elif strategy == "tail":
            budget_n_vis = point_budget
            budget_bytes = estimate_gpu_packed_bytes(submodel_list[largest_id], budget_n_vis)
            budget_source = "point_budget"
            candidates = []
            for sid in valid_ids[1:]:
                n_vis = int(submodel_list[sid].visible_indices.shape[0])
                n_bytes = estimate_gpu_packed_bytes(submodel_list[sid], n_vis)
                candidates.append({"id": int(sid), "n_vis": n_vis, "bytes": int(n_bytes)})
                candidate_n_vis += int(n_vis)
                candidate_bytes += int(n_bytes)
            candidate_count = len(candidates)
            selected_ids = select_best_fit_gpu_packed_cache(candidates, budget_n_vis)
            tail_rank = {int(sid): rank for rank, sid in enumerate(reversed(valid_ids[1:]))}
            selected_ids.sort(key=lambda sid: tail_rank.get(int(sid), 0))
            selected_set = set(selected_ids)
            for item in candidates:
                if int(item["id"]) not in selected_set:
                    continue
                total_n_vis += int(item["n_vis"])
                total_bytes += int(item["bytes"])
            selection_policy = "tail_best_fit"

    selected_ids = [int(sid) for sid in selected_ids]
    return set(selected_ids), {
        "strategy": strategy,
        "selection_policy": selection_policy,
        "budget_source": budget_source,
        "ids": selected_ids,
        "budget_n_vis": int(budget_n_vis),
        "budget_bytes_est": int(budget_bytes),
        "budget_mb_est": round(budget_bytes / 1024**2, 3),
        "candidate_count": int(candidate_count),
        "candidate_n_vis": int(candidate_n_vis),
        "candidate_bytes_est": int(candidate_bytes),
        "candidate_mb_est": round(candidate_bytes / 1024**2, 3),
        "total_n_vis": int(total_n_vis),
        "bytes_est": int(total_bytes),
        "mb_est": round(total_bytes / 1024**2, 3),
        "fill_basis": "points",
        "fill_ratio_est": round(total_n_vis / budget_n_vis, 4) if budget_n_vis > 0 else 0.0,
        "point_fill_ratio_est": round(total_n_vis / budget_n_vis, 4) if budget_n_vis > 0 else 0.0,
        "byte_fill_ratio_est": round(total_bytes / budget_bytes, 4) if budget_bytes > 0 else 0.0,
        "unused_n_vis_est": int(max(0, budget_n_vis - total_n_vis)),
        "unused_bytes_est": int(max(0, budget_bytes - total_bytes)),
        "unused_mb_est": round(max(0, budget_bytes - total_bytes) / 1024**2, 3),
    }


def clear_gpu_packed_caches(submodel_list, cache_ids):
    for sid in list(cache_ids):
        if 0 <= sid < len(submodel_list):
            clear_cache = getattr(submodel_list[sid], "clear_gpu_packed_cache", None)
            if clear_cache is not None:
                clear_cache()



def training(dataset, opt, pipe, saving_iterations, debug_from, res):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = res.get("first_iter", 1)
    prepare_output_and_logger(dataset)
    add_output_path(LOGGER, os.path.join(dataset.model_path, "logs"))
    add_output_path(LOGGER, os.path.join("debug", BRANCH, SCENE_NAME), prefix="train")
    log_section(LOGGER, "Training Setup")
    log_kv(LOGGER, "training_runtime_options", {
        "first_iter": first_iter,
        "iterations": opt.iterations,
        "test_diagnostics": getattr(opt, "test_diagnostics", False),
        "profile_log_requested": getattr(opt, "profile_log_requested", False),
        "profile_log": getattr(opt, "profile_log", False),
        "profile_from": getattr(opt, "profile_from", None),
        "profile_until": getattr(opt, "profile_until", None),
        "profile_every": getattr(opt, "profile_every", None),
        "profile_sync": getattr(opt, "profile_sync", False),
        "trace_from": getattr(opt, "trace_from", None),
        "trace_until": getattr(opt, "trace_until", None),
        "trace_enabled": bool(getattr(opt, "test_diagnostics", False)
                              and getattr(opt, "trace_from", 0) > 0
                              and getattr(opt, "trace_until", 0) >= getattr(opt, "trace_from", 0)),
        "bench_from": getattr(opt, "bench_from", None),
        "bench_until": getattr(opt, "bench_until", None),
        "disable_densify": getattr(opt, "disable_densify", False),
        "split_size": config.SPLIT_SIZE,
        "split_size_override": getattr(opt, "split_size_override", 0),
        "skip_small_block_thresh": config.SKIP_SMALL_BLOCK_THRESH,
        "merge_fast": config.MERGE_FAST,
        "frustum_culling": config.FRUSTUM_CULLING_ENABLED,
        "frustum_cache": config.FRUSTUM_CULLING_CACHE_ENABLED,
        "gpu_cache_threshold_gb": getattr(opt, "gpu_cache_threshold_gb", config.GPU_CACHE_THRESHOLD_GB),
        "gpu_cache_hard_limit_gb": getattr(opt, "gpu_cache_hard_limit_gb", config.GPU_CACHE_HARD_LIMIT_GB),
        "gpu_cache_stage_entry_limit_gb": getattr(opt, "gpu_cache_stage_entry_limit_gb", config.GPU_CACHE_STAGE_ENTRY_LIMIT_GB),
        "cuda_empty_cache_interval": getattr(opt, "cuda_empty_cache_interval", config.CUDA_EMPTY_CACHE_INTERVAL),
        "legacy_per_block_loss": getattr(opt, "legacy_per_block_loss", False),
        "gpu_packed_cache_strategy": getattr(opt, "gpu_packed_cache_strategy", "tail"),
        "gpu_packed_cache_point_budget": getattr(opt, "gpu_packed_cache_point_budget", config.GPU_PACKED_CACHE_POINT_BUDGET),
        "trained_ply_path": res.get("trained_ply_path"),
    })
    if getattr(opt, "disable_densify", False):
        LOGGER.info("[runtime] densify/prune/opacity reset disabled; dynamic block splitting remains enabled")
    if getattr(opt, "profile_log_requested", False) and not getattr(opt, "profile_log", False):
        LOGGER.info("[runtime] --profile_log ignored because --test_diagnostics was not set")

    initial_gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, initial_gaussians, on_cpu=True)

    trained_ply_path = res.get("trained_ply_path")
    if trained_ply_path:
        initial_gaussians.load_ply(trained_ply_path)

    initial_gaussians.training_setup(opt)
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    if DEBUG_MODE:
        opt.iterations = 1050

        

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    
    gaussians: GaussianModel = scene.gaussians
    gaussians = gaussians.dump_to_cpu()
    
    # Start with 1 block on CPU, dynamically split when exceeding SPLIT_SIZE
    gaussians.build_split_indices(num_blocks=1)

    ## blocks visualization
    # gaussians.visualize_blocks(save_path = f"debug/{BRANCH}_bbox")
    
    submodel_list: List[GaussianModel] = gaussians.split()

    LOGGER.info(f"Partitioned into {len(submodel_list)} blocks, sizes: {[s._xyz.shape[0] for s in submodel_list]}")
    print(f"Partitioned into {len(submodel_list)} blocks, sizes: {[s._xyz.shape[0] for s in submodel_list]}")

    for submodel in submodel_list:
        submodel.training_setup(opt, device = "cpu")
        submodel.pack_to_buffer()

    cpu_full_proj_transform_dict = {}
    for cam in scene.getTrainCameras():
        cpu_full_proj_transform_dict[cam.image_name] = cam.full_proj_transform.detach().cpu()


    frustum_cache: dict = {}

    # Pipeline: async D2H grad copies + deferred opt steps.
    # Timeline tracing is windowed to avoid CUDA event overhead across full training.
    TRACE_START = getattr(opt, "trace_from", 681)
    TRACE_END = getattr(opt, "trace_until", 750)
    TRACE_ENABLED = getattr(opt, "test_diagnostics", False) and TRACE_START > 0 and TRACE_END >= TRACE_START
    tracer = TraceManager(enabled=False)
    grad_sync = PipelinedGradSync(submodel_list, opt, dataset, scene, tracer=tracer,
                                   frustum_cache=frustum_cache)
    profiler = IterationProfiler(
        LOGGER,
        enabled=getattr(opt, "profile_log", False),
        start=getattr(opt, "profile_from", 1),
        end=getattr(opt, "profile_until", 0),
        every=getattr(opt, "profile_every", 1),
        sync_cuda=getattr(opt, "profile_sync", False),
    )

    time_start = time.time()

    # benchmark 统计收集
    BENCH_START = getattr(opt, "bench_from", 301)
    BENCH_END = getattr(opt, "bench_until", 700)
    bench_its_list = []
    bench_alloc_list = []
    bench_rsv_list = []
    bench_vis_list = []
    bench_loss_list = []

    for iteration in range(first_iter, opt.iterations + 1):
        profiler.begin(iteration)
        with profiler.span("pre_iter_cuda_cache_cleanup"):
            maybe_empty_cuda_cache(
                opt,
                iteration,
                profiler,
                reason="pre_iter_before_reset_peak",
                respect_interval=True,
                allow_periodic=False,
                key_prefix="pre_iter_cuda",
            )
        with profiler.span("reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

        if TRACE_ENABLED and iteration == TRACE_START:
            tracer.enabled = True
        tracer.step(iteration)

        # Unfreeze blocks that background densify has finished (safe between iters)
        with profiler.span("finalize_completed_densify"):
            grad_sync.finalize_completed_densify()

        with profiler.span("learning_rate_update"):
            for submodel in submodel_list:
                submodel.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            with profiler.span("oneup_sh_degree"):
                for submodel in submodel_list:
                    submodel.oneupSHdegree()

        # Pick a random Camera
        with profiler.span("pick_camera"):
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
        profiler.set("camera", viewpoint_cam.image_name)
        profiler.set("camera_index", int(vind))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # frustum culling (with cache after densify_until_iter)
        # When densify is disabled, geometry is fixed after dynamic split, so
        # cached frustum results are safe as soon as splits stop changing blocks.
        use_fc_cache = config.FRUSTUM_CULLING_CACHE_ENABLED and (
            iteration >= opt.densify_until_iter or getattr(opt, "disable_densify", False)
        )
        with profiler.span("frustum_culling_total"):
            with torch.no_grad():
                if config.FRUSTUM_CULLING_ENABLED:
                    cam_name = viewpoint_cam.image_name
                    with tracer.span("frustum_culling", tid=TID_MAIN):
                        for submodel_id, model in enumerate(submodel_list):
                            cached = use_fc_cache and submodel_id in frustum_cache and cam_name in frustum_cache[submodel_id]
                            if cached:
                                model.visible_indices = frustum_cache[submodel_id][cam_name]
                                profiler.add("frustum_cache_hits", 1)
                            else:
                                profiler.add("frustum_cache_misses", 1)
                                if model._xyz.is_cuda:
                                    model.visible_indices = frustum_culling_idx(model._xyz, viewpoint_cam.full_proj_transform)
                                else:
                                    # Use frozen xyz when block is being densified in background
                                    if hasattr(model, '_xyz_contig_frozen'):
                                        xyz_for_cull = model._xyz_contig_frozen
                                    elif hasattr(model, '_xyz_contig'):
                                        xyz_for_cull = model._xyz_contig
                                    else:
                                        xyz_for_cull = model._xyz
                                    model.visible_indices = frustum_culling_idx(xyz_for_cull, cpu_full_proj_transform_dict[cam_name])
                                if use_fc_cache:
                                    if submodel_id not in frustum_cache:
                                        frustum_cache[submodel_id] = {}
                                    frustum_cache[submodel_id][cam_name] = model.visible_indices
                else:
                    for model in submodel_list:
                        model.visible_indices = torch.arange(model._xyz.shape[0], device="cuda")
                # DEBUG: validate visible_indices vs _packed
                for sid, m in enumerate(submodel_list):
                    vi = m.visible_indices
                    packed_ref = getattr(m, '_packed_frozen', m._packed) if hasattr(m, '_packed') else None
                    if vi is not None and len(vi) > 0 and packed_ref is not None:
                        mx = vi.max().item()
                        if mx >= packed_ref.shape[0]:
                            raise RuntimeError(
                                f"[iter {iteration}] submodel {sid}: visible_indices.max()={mx} >= "
                                f"packed.shape[0]={packed_ref.shape[0]}, _xyz_contig={m._xyz_contig.shape[0] if hasattr(m,'_xyz_contig') else 'N/A'}, "
                                f"cached={use_fc_cache and sid in frustum_cache and cam_name in frustum_cache.get(sid,{})}"
                            )

        # 无渲染全部结果 为计算Loss做准备
        all_rendered, all_depth, all_alpha, all_submodel_ids = [], [], [], []
        rendered_list, depth_list, alpha_list = [], [], []
        visible_submodel_id_list = []
        visible_pts = 0

        with profiler.span("nograd_pass_total"):
          with torch.no_grad():
            # Phase 1: 连续发射所有 block 的 render，不做任何 CPU 同步
            # 按点数从多到少排序，让大 block 先上 GPU
            sorted_submodel_ids = sorted( range(len(submodel_list)), key=lambda i: submodel_list[i].visible_indices.shape[0], reverse=True )
            max_vis = submodel_list[sorted_submodel_ids[0]].visible_indices.shape[0] if sorted_submodel_ids else 0

            # 过滤出有效 block
            valid_ids = []
            for sid in sorted_submodel_ids:
                n_vis = submodel_list[sid].visible_indices.shape[0]
                if n_vis == 0 or (config.SKIP_SMALL_BLOCK_THRESH > 0 and n_vis < max_vis * config.SKIP_SMALL_BLOCK_THRESH):
                    continue
                valid_ids.append(sid)
            profiler.set("valid_ids", [int(x) for x in valid_ids])
            profiler.set("valid_n_vis", {str(sid): int(submodel_list[sid].visible_indices.shape[0]) for sid in valid_ids})
            gpu_packed_cache_ids, gpu_packed_cache_info = plan_gpu_packed_cache(
                submodel_list,
                valid_ids,
                getattr(opt, "gpu_packed_cache_strategy", "tail"),
                getattr(opt, "gpu_packed_cache_point_budget", config.GPU_PACKED_CACHE_POINT_BUDGET),
            )
            profiler.set("gpu_packed_cache", gpu_packed_cache_info)
            gpu_packed_cache_n_vis_actual = 0
            gpu_packed_cache_bytes_actual = 0

            stage_entry_limit_gb = getattr(
                opt,
                "gpu_cache_stage_entry_limit_gb",
                config.GPU_CACHE_STAGE_ENTRY_LIMIT_GB,
            )
            if stage_entry_limit_gb > 0:
                # Max-memory guard: render can request a large workspace on top
                # of stale reserved cache, so clear slack before entering it.
                with profiler.span("stage_entry_cuda_cache_cleanup"):
                    maybe_empty_cuda_cache(
                        opt,
                        iteration,
                        profiler,
                        reason="before_nograd_render_pass",
                        respect_interval=True,
                        allow_periodic=False,
                        hard_limit_override_gb=stage_entry_limit_gb,
                        key_prefix="stage_entry_cuda",
                    )

            # 流水线: 提前 gather 下一个 block，与当前 block 的 h2d+render 重叠
            # 每个 submodel 有自己的 _packed_staging，天然双缓冲
            if valid_ids:
                # 预热: gather 第一个 block
                with tracer.span("gather_nograd", block_id=valid_ids[0], n_vis=submodel_list[valid_ids[0]].visible_indices.shape[0]):
                    with profiler.span("nograd_pre_gather"):
                        submodel_list[valid_ids[0]].pre_gather()
                    profiler.block("nograd", valid_ids[0], n_vis=int(submodel_list[valid_ids[0]].visible_indices.shape[0]))

            for i, submodel_id in enumerate(valid_ids):
                submodel = submodel_list[submodel_id]
                visible_pts += submodel.visible_indices.shape[0]
                cache_gpu_packed = submodel_id in gpu_packed_cache_ids
                if cache_gpu_packed:
                    profiler.block("nograd", submodel_id, gpu_packed_cache_planned=True)

                with tracer.transfer_span("h2d_nograd", block_id=submodel_id):
                    with profiler.span("nograd_h2d_activate"):
                        submodel.kick_h2d_and_activate(requires_grad=False, cache_gpu_packed=cache_gpu_packed)
                if cache_gpu_packed:
                    cache_bytes = submodel.cached_gpu_packed_bytes()
                    gpu_packed_cache_n_vis_actual += int(submodel.visible_indices.shape[0])
                    gpu_packed_cache_bytes_actual += int(cache_bytes)
                    profiler.add("gpu_packed_cache_bytes_actual", cache_bytes)
                    profiler.block("nograd", submodel_id, gpu_packed_cache_bytes=cache_bytes)

                # 趁 h2d (non_blocking) + render 占 GPU 时，CPU 提前 gather 下一个 block
                if i + 1 < len(valid_ids):
                    next_id = valid_ids[i + 1]
                    with tracer.span("gather_nograd", block_id=next_id, n_vis=submodel_list[next_id].visible_indices.shape[0]):
                        with profiler.span("nograd_pre_gather"):
                            submodel_list[next_id].pre_gather()
                        profiler.block("nograd", next_id, n_vis=int(submodel_list[next_id].visible_indices.shape[0]))

                with tracer.gpu_span("render_nograd", block_id=submodel_id):
                    with profiler.span("nograd_render_dispatch"):
                        render_pkg = render(
                            viewpoint_cam,
                            submodel,
                            pipe,
                            bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            retain_viewspace_grad=False,
                        )

                with profiler.span("nograd_deactivate"):
                    submodel.deactivate_subset()

                image, alphaLeft, depth = render_pkg["render"], render_pkg["alphaLeft"], render_pkg["depth"]
                all_rendered.append(image)
                all_depth.append(depth)
                all_alpha.append(alphaLeft)
                all_submodel_ids.append(submodel_id)
                render_pkg = None
                profiler.block("nograd", submodel_id, rendered=True)

            # Phase 2: 所有 render 完成后，批量过滤低贡献 block（此时 .item() 不会阻塞 render pipeline）
            with profiler.span("filter_contribution"):
                for image, depth, alphaLeft, submodel_id in zip(all_rendered, all_depth, all_alpha, all_submodel_ids):
                    valid_pixels = (image > 0).any(dim=0).sum().item()
                    total_pixels = image.shape[1] * image.shape[2]
                    valid_ratio = valid_pixels / total_pixels
                    profiler.block("nograd", submodel_id, valid_pixels=int(valid_pixels), valid_ratio=round(valid_ratio, 6))
                    if valid_ratio < 0.05:
                        continue

                    rendered_list.append(image)
                    depth_list.append(depth)
                    alpha_list.append(alphaLeft)
                    visible_submodel_id_list.append(submodel_id)
            if gpu_packed_cache_ids:
                visible_cache_ids = [sid for sid in sorted(gpu_packed_cache_ids) if sid in visible_submodel_id_list]
                dropped_cache_ids = [sid for sid in sorted(gpu_packed_cache_ids) if sid not in visible_submodel_id_list]
                clear_gpu_packed_caches(submodel_list, dropped_cache_ids)
                profiler.set("gpu_packed_cache_visible_ids", [int(sid) for sid in visible_cache_ids])
                profiler.set("gpu_packed_cache_dropped_ids", [int(sid) for sid in dropped_cache_ids])
                budget_bytes = int(gpu_packed_cache_info.get("budget_bytes_est", 0))
                budget_n_vis = int(gpu_packed_cache_info.get("budget_n_vis", 0))
                visible_cache_n_vis = sum(
                    int(submodel_list[sid].visible_indices.shape[0])
                    for sid in visible_cache_ids
                )
                visible_cache_bytes = sum(
                    estimate_gpu_packed_bytes(
                        submodel_list[sid],
                        int(submodel_list[sid].visible_indices.shape[0]),
                    )
                    for sid in visible_cache_ids
                )
                profiler.set("gpu_packed_cache_fill_actual", round(gpu_packed_cache_n_vis_actual / budget_n_vis, 4) if budget_n_vis > 0 else 0.0)
                profiler.set("gpu_packed_cache_point_fill_actual", round(gpu_packed_cache_n_vis_actual / budget_n_vis, 4) if budget_n_vis > 0 else 0.0)
                profiler.set("gpu_packed_cache_byte_fill_actual", round(gpu_packed_cache_bytes_actual / budget_bytes, 4) if budget_bytes > 0 else 0.0)
                profiler.set("gpu_packed_cache_visible_n_vis", int(visible_cache_n_vis))
                profiler.set("gpu_packed_cache_visible_bytes_est", int(visible_cache_bytes))
                profiler.set("gpu_packed_cache_visible_fill_est", round(visible_cache_n_vis / budget_n_vis, 4) if budget_n_vis > 0 else 0.0)
                profiler.set("gpu_packed_cache_visible_point_fill_est", round(visible_cache_n_vis / budget_n_vis, 4) if budget_n_vis > 0 else 0.0)
                profiler.set("gpu_packed_cache_visible_byte_fill_est", round(visible_cache_bytes / budget_bytes, 4) if budget_bytes > 0 else 0.0)
            profiler.set("visible_submodel_ids", [int(x) for x in visible_submodel_id_list])
            profiler.set("visible_pts", int(visible_pts))

        if len(rendered_list) == 0:
            print(f"Iteration {iteration}: No visible blocks after filtering, skipping.")
            profiler.set("skip_reason", "no_visible_blocks_after_filter")
            clear_gpu_packed_caches(submodel_list, gpu_packed_cache_ids)
            all_rendered, all_depth, all_alpha, all_submodel_ids = None, None, None, None
            rendered_list, depth_list, alpha_list = None, None, None
            image, alphaLeft, depth = None, None, None
            render_pkg = None
            with profiler.span("skip_cuda_cache_cleanup"):
                maybe_empty_cuda_cache(
                    opt,
                    iteration,
                    profiler,
                    reason="no_visible_blocks_after_filter",
                    force=True,
                    respect_interval=False,
                    key_prefix="skip_cuda",
                )
            profiler.end()
            continue
        # execute merge
        with profiler.span("merge_total"):
            with torch.no_grad():
                with tracer.gpu_span("merge_opt_kid"):
                    merge_res = merge_opt_kid(rendered_list, depth_list, alpha_list)
            
        C_sorted = merge_res["front_rgbs"] # 每个 block 的颜色贡献，已经按照正确的前后顺序排列好
        prefix_T = merge_res["prefix_T"]
        block_rank = merge_res["block_rank"]  # [K,H,W]，每个像素告诉你每个 block 的排序位置
        K, C, H, W = C_sorted.shape   
        colors_bg = merge_res["bg_rgb"]
        diff_gaussian_rasterization_wenqi_tam.set_colors_bg(colors_bg)
        profiler.set("merge_k", int(K))
        profiler.set("image_hw", [int(H), int(W)])

        with profiler.span("gt_image_to_gpu"):
            with torch.no_grad():
                gt_image = viewpoint_cam.original_image.cuda()

        final_rgb_grad = None
        if not getattr(opt, "legacy_per_block_loss", False):
            with profiler.span("loss_grad_once"):
                final_rgb_proxy = merge_res["final_rgb"].detach().requires_grad_(True)
                Ll1 = l1_loss(final_rgb_proxy, gt_image)
                ssim_value = fused_ssim(final_rgb_proxy.unsqueeze(0), gt_image.unsqueeze(0))
                image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
                image_loss.backward()
                final_rgb_grad = final_rgb_proxy.grad.detach()
                loss = image_loss.detach()
                del final_rgb_proxy, image_loss
            profiler.set("single_image_loss_grad", True)

        # 遍历所有可见block 轮流当active block，按点数从多到少排序
        grad_order = sorted(
            range(len(visible_submodel_id_list)),
            key=lambda i: submodel_list[visible_submodel_id_list[i]].visible_indices.shape[0],
            reverse=True
        )
        if getattr(opt, "gpu_packed_cache_strategy", "tail") == "tail" and gpu_packed_cache_ids:
            grad_order = sorted(
                range(len(visible_submodel_id_list)),
                key=lambda i: (
                    visible_submodel_id_list[i] not in gpu_packed_cache_ids,
                    int(submodel_list[visible_submodel_id_list[i]].visible_indices.shape[0])
                    if visible_submodel_id_list[i] in gpu_packed_cache_ids
                    else -int(submodel_list[visible_submodel_id_list[i]].visible_indices.shape[0]),
                ),
            )
            profiler.set("grad_order_policy", "tail_cache_first")
        else:
            profiler.set("grad_order_policy", "largest_visible_first")
        profiler.set("grad_order_ids", [int(visible_submodel_id_list[idx]) for idx in grad_order])
        if getattr(opt, "legacy_per_block_loss", False):
            loss = None
        processed_grad_blocks = 0
        Ll1depth = 0
        with profiler.span("grad_pass_total"):
            for idx in grad_order:
                submodel_id = visible_submodel_id_list[idx]
                rank_map = block_rank[idx]
                submodel: GaussianModel = submodel_list[submodel_id]
                n_vis_grad = int(submodel.visible_indices.shape[0])
                profiler.block("grad", submodel_id, n_vis=n_vis_grad)

                # Skip blocks being densified in background — use stale nograd render only
                if getattr(submodel, '_is_densifying', False):
                    profiler.block("grad", submodel_id, skipped="densifying")
                    if submodel_id in gpu_packed_cache_ids:
                        submodel.clear_gpu_packed_cache()
                    continue

                # pre_gather() 已在 nograd 阶段完成，staging buffer 仍有效，无需重复 gather
                use_gpu_packed_cache = submodel_id in gpu_packed_cache_ids and submodel.has_cached_gpu_packed()
                profiler.block("grad", submodel_id, gpu_packed_cache_hit=use_gpu_packed_cache)
                if use_gpu_packed_cache:
                    profiler.add("gpu_packed_cache_grad_hits", 1)
                    profiler.add("gpu_packed_cache_grad_bytes", submodel.cached_gpu_packed_bytes())
                    with tracer.gpu_span("gpu_packed_cache_grad_activate", block_id=submodel_id, n_vis=n_vis_grad):
                        with profiler.span("grad_gpu_cache_activate"):
                            submodel.kick_h2d_and_activate(
                                requires_grad=True,
                                use_cached_gpu_packed=True,
                            )
                    # The grad activation clones the cached packed tensor into
                    # trainable per-attribute tensors, so the packed source can
                    # be released before render/backward request their workspace.
                    released_bytes = submodel.cached_gpu_packed_bytes()
                    submodel.clear_gpu_packed_cache()
                    profiler.add("gpu_packed_cache_grad_released_after_activate_bytes", released_bytes)
                    profiler.block(
                        "grad",
                        submodel_id,
                        gpu_packed_cache_released_after_activate=True,
                        gpu_packed_cache_released_bytes=int(released_bytes),
                    )
                else:
                    if submodel_id in gpu_packed_cache_ids:
                        profiler.add("gpu_packed_cache_grad_misses", 1)
                    with tracer.transfer_span("h2d_grad", block_id=submodel_id):
                        with profiler.span("grad_h2d_activate"):
                            submodel.kick_h2d_and_activate(
                                requires_grad=True,
                            )

                # GPU 正在做 H2D，CPU 趁机跑上一个 block 的 densify_stats
                with profiler.span("run_deferred_densify"):
                    grad_sync.run_deferred_densify()

                with tracer.gpu_span("render_grad", block_id=submodel_id):
                    with profiler.span("grad_render_dispatch"):
                        retain_viewspace_grad = (
                            not getattr(opt, "disable_densify", False)
                            and iteration < opt.densify_until_iter
                        )
                        profiler.block("grad", submodel_id, retain_viewspace_grad=retain_viewspace_grad)
                        render_pkg = render(
                            viewpoint_cam,
                            submodel,
                            pipe,
                            bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            retain_viewspace_grad=retain_viewspace_grad,
                        )

                # pixel level
                sub_img = render_pkg["render"]

                # gaussian points level
                sub_viewspace_point_tensor = render_pkg["viewspace_points"]
                prefix_T_k = None
                backward_target = None
                backward_grad = None
                composed_img = None
                C_base = None
                C_sorted_k = None
                C_active = None
                submodel_rank_per_pixel = None
                Ll1 = None
                ssim_value = None

                if getattr(opt, "legacy_per_block_loss", False):
                    with profiler.span("compose_loss"):
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
                        ssim_value = fused_ssim(composed_img.unsqueeze(0), gt_image.unsqueeze(0))
                        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

                        # Depth regularization
                        Ll1depth = 0
                    backward_target = loss
                    backward_grad = None
                else:
                    with profiler.span("compose_loss"):
                        with torch.no_grad():
                            prefix_T_k = prefix_T[:, 0].gather(dim=0, index=rank_map.unsqueeze(0)).squeeze(0)
                            backward_grad = prefix_T_k.unsqueeze(0) * final_rgb_grad
                    backward_target = sub_img

                with tracer.gpu_span("backward", block_id=submodel_id):
                    with profiler.span("backward_dispatch"):
                        torch.autograd.backward(backward_target, grad_tensors=backward_grad)

                tracer.counter("pts", {"visible": visible_pts, "total": sum(s._xyz.shape[0] for s in submodel_list)})

                with profiler.span("flush_and_prepare"):
                    grad_sync.flush_and_prepare(submodel, submodel_id, render_pkg, sub_viewspace_point_tensor, iteration)
                if submodel_id in gpu_packed_cache_ids:
                    submodel.clear_gpu_packed_cache()
                with profiler.span("grad_temp_cleanup"):
                    render_pkg = None
                    sub_img = None
                    sub_viewspace_point_tensor = None
                    prefix_T_k = None
                    backward_target = None
                    backward_grad = None
                    composed_img = None
                    C_base = None
                    C_sorted_k = None
                    C_active = None
                    submodel_rank_per_pixel = None
                    Ll1 = None
                    ssim_value = None
                profiler.block("grad", submodel_id, backward=True)
                processed_grad_blocks += 1

        # flush the last submodel's pending work
        with profiler.span("flush_last"):
            grad_sync.flush_last()
        clear_gpu_packed_caches(submodel_list, gpu_packed_cache_ids)

        if processed_grad_blocks == 0 or loss is None:
            LOGGER.warning(f"[iter {iteration}] no gradient block was processed; skipping optimizer/log step")
            profiler.set("skip_reason", "no_gradient_block_processed")
            del C_sorted, prefix_T, block_rank, colors_bg, merge_res
            del rendered_list, depth_list, alpha_list
            all_rendered, all_depth, all_alpha, all_submodel_ids = None, None, None, None
            del gt_image
            final_rgb_grad = None
            image = None
            alphaLeft = None
            depth = None
            rank_map = None
            sub_img = None
            sub_viewspace_point_tensor = None
            render_pkg = None
            backward_target = None
            backward_grad = None
            prefix_T_k = None
            with profiler.span("skip_cuda_cache_cleanup"):
                maybe_empty_cuda_cache(
                    opt,
                    iteration,
                    profiler,
                    reason="no_gradient_block_processed",
                    force=True,
                    respect_interval=False,
                    key_prefix="skip_cuda",
                )
            profiler.end()
            continue

        del C_sorted, prefix_T, block_rank, colors_bg, merge_res
        del rendered_list, depth_list, alpha_list
        all_rendered, all_depth, all_alpha, all_submodel_ids = None, None, None, None
        del gt_image
        final_rgb_grad = None
        image = None
        alphaLeft = None
        depth = None
        rank_map = None
        sub_img = None
        sub_viewspace_point_tensor = None
        render_pkg = None
        backward_target = None
        backward_grad = None
        prefix_T_k = None

        # Queue non-visible blocks for densify/reset (visible blocks already queued in adam)
        with profiler.span("queue_densify_nonvisible"):
            if iteration < opt.densify_until_iter and not getattr(opt, "disable_densify", False):
                processed_ids = set(visible_submodel_id_list)
                for sm_id, sm in enumerate(submodel_list):
                    if sm_id in processed_ids:
                        continue
                    grad_sync.queue_densify_prune(sm, sm_id, iteration)
            profiler.set("densify_queue_len", len(getattr(grad_sync, "_deferred_densify_prune", [])))

        # Run all queued densify ops in parallel threads
        with profiler.span("flush_all_densify"):
            grad_sync.flush_all_densify()
        profiler.set("bg_densify_alive", bool(getattr(grad_sync, "_bg_densify_thread", None) is not None and grad_sync._bg_densify_thread.is_alive()))

        # Dynamic block splitting: any block exceeding SPLIT_SIZE gets binary split
        # Skip blocks still being densified in background
        allow_dynamic_split = iteration < opt.densify_until_iter or getattr(opt, "disable_densify", False)
        if allow_dynamic_split:
            blocks_to_split = [i for i, sm in enumerate(submodel_list)
                               if sm._xyz.shape[0] > config.SPLIT_SIZE and not getattr(sm, '_is_densifying', False)]
            if blocks_to_split:
                with profiler.span("dynamic_split"):
                    for sm_id in sorted(blocks_to_split, reverse=True):
                        sm = submodel_list[sm_id]
                        left, right = sm.split_in_half(opt)
                        grad_sync.reallocate_pinned_buffers(left)
                        grad_sync.reallocate_pinned_buffers(right)
                        submodel_list[sm_id] = left
                        submodel_list.insert(sm_id + 1, right)
                frustum_cache.clear()
                LOGGER.info(f"[iter {iteration}] Split block(s) {blocks_to_split}, now {len(submodel_list)} blocks, sizes: {[s._xyz.shape[0] for s in submodel_list]}")
                print(f"[iter {iteration}] Split block(s) {blocks_to_split}, now {len(submodel_list)} blocks, sizes: {[s._xyz.shape[0] for s in submodel_list]}")
                profiler.set("blocks_split", [int(x) for x in blocks_to_split])

        # reserved 超过 allocated 太多时才清缓存，避免频繁清导致性能下降
        with profiler.span("cuda_cache_check"):
            maybe_empty_cuda_cache(
                opt,
                iteration,
                profiler,
                reason="end_of_iteration",
                respect_interval=True,
                key_prefix="cuda",
            )

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            pts_total = sum(submodel._xyz.shape[0] for submodel in submodel_list)
            vis_M = visible_pts / 1e6
            pts_M = pts_total / 1e6
            vis_pct = visible_pts / pts_total * 100 if pts_total > 0 else 0
            gpu_peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
            gpu_peak_rsv = torch.cuda.max_memory_reserved() / 1024**3
            alloc = torch.cuda.memory_allocated() / 1024**3
            rsv = torch.cuda.memory_reserved() / 1024**3
            elapsed = time.time() - time_start
            its = (iteration - first_iter) / elapsed if elapsed > 0 else 0
            profiler.set("pts_total", int(pts_total))
            profiler.set("block_count", int(len(submodel_list)))
            profiler.set("block_sizes", [int(submodel._xyz.shape[0]) for submodel in submodel_list])
            profiler.set("mem_gb", {
                "alloc": round(alloc, 4),
                "reserved": round(rsv, 4),
                "peak_alloc": round(gpu_peak_alloc, 4),
                "peak_reserved": round(gpu_peak_rsv, 4),
            })
            profiler.set("loss", round(float(ema_loss_for_log), 6))
            profiler.set("it_per_s_running", round(float(its), 4))
            # benchmark 收集
            if BENCH_START <= iteration <= BENCH_END:
                bench_its_list.append(its)
                bench_alloc_list.append(alloc)
                bench_rsv_list.append(rsv)
                bench_vis_list.append(visible_pts)
                bench_loss_list.append(ema_loss_for_log)

            # progress bar - every iter
            progress_bar.set_postfix({"L": f"{ema_loss_for_log:.4f}", "vis": f"{vis_M:.2f}M", "pts": f"{pts_M:.2f}M", "vis%": f"{vis_pct:.0f}", "blk": len(submodel_list), "alloc": f"{alloc:.2f}", "rsv": f"{rsv:.2f}", "peak": f"{gpu_peak_alloc:.2f}", "it/s": f"{its:.1f}"})
            progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 1 == 0:
                log = {"iter": iteration, "L": round(ema_loss_for_log, 4), "vis": f"{vis_M:.2f}M", "pts": f"{pts_M:.2f}M", "vis%": round(vis_pct, 1), "blk": len(submodel_list), "alloc": round(alloc, 2), "rsv": round(rsv, 2), "peak_alloc": round(gpu_peak_alloc, 2), "peak_rsv": round(gpu_peak_rsv, 2), "it/s": round(its, 1), "elapsed": f"{elapsed:.1f}s"}
                wandb_log = {"iter": iteration, "L": round(ema_loss_for_log, 4), "vis": visible_pts, "pts": pts_total, "vis%": round(vis_pct, 1), "blk": len(submodel_list), "alloc": round(alloc, 2), "rsv": round(rsv, 2), "peak_alloc": round(gpu_peak_alloc, 2), "peak_rsv": round(gpu_peak_rsv, 2), "it/s": round(its, 1), "elapsed": round(elapsed, 1)}

                # logging
                LOGGER.info(log)

                # wandb logging
                if WANDB and not DEBUG_MODE:
                    wandb.log(wandb_log, step=iteration)
                    wandb.log({f"block/{idx}_size": gs._xyz.shape[0] for idx, gs in enumerate(submodel_list)}, step=iteration)
            profiler.end()
            
        # saving Gaussians ply
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            # Wait for background densify to finish so save_ply sees consistent shapes.
            # Without this, densify thread may resize _xyz/_opacity/etc between reads,
            # causing "dimension mismatch" in np.concatenate.
            grad_sync.join_background_densify()
            point_cloud_path = os.path.join(scene.model_path, f"point_cloud/{BRANCH}/iteration_{iteration}")
            for submodel_id, submodel in enumerate(submodel_list):
                submodel.save_ply(os.path.join(point_cloud_path, f"point_cloud_sub_{submodel_id}.ply"), include_block=False)
                

        if TRACE_ENABLED and iteration == TRACE_END:
             tracer.enabled = False
             
    # Ensure all background densify threads are done before reporting/exporting
    grad_sync.join_background_densify()

    time_end = time.time()
    cost = time_end - time_start
    print(f"Training time cost: [{cost:.2f}] seconds.")
    log_kv(LOGGER, "training_complete", {
        "seconds": round(cost, 3),
        "iterations": opt.iterations,
        "final_blocks": len(submodel_list),
        "final_points": int(sum(s._xyz.shape[0] for s in submodel_list)),
    })

    # 打印 benchmark 统计（用 sys.__stdout__ 避免时间戳）
    if bench_its_list:
        n = len(bench_its_list)
        import socket, subprocess
        hostname = socket.gethostname()
        commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        bench_summary = {
            "hostname": hostname,
            "branch": BRANCH,
            "commit": commit_id,
            "samples": n,
            "iter_from": BENCH_START,
            "iter_until": BENCH_END,
            "avg_it_s": sum(bench_its_list) / n,
            "avg_loss": sum(bench_loss_list) / n,
            "avg_visible_pts": sum(bench_vis_list) / n,
            "avg_alloc_gb": sum(bench_alloc_list) / n,
            "avg_reserved_gb": sum(bench_rsv_list) / n,
        }
        log_kv(LOGGER, "benchmark_summary", bench_summary)
        p = sys.__stdout__.write
        p(f"\n{'='*50}\n")
        p(f"  [{hostname}] Benchmark (iter {BENCH_START}-{BENCH_END}, {n} samples)\n")
        p(f"  Branch: {BRANCH}  Commit: {commit_id}\n")
        p(f"{'='*50}\n")
        p(f"  平均 it/s:           {sum(bench_its_list)/n:.2f}\n")
        p(f"  平均 loss:           {sum(bench_loss_list)/n:.6f}\n")
        p(f"  平均 visible pts:    {sum(bench_vis_list)/n/1e6:.2f}M\n")
        p(f"  平均占用 mem (alloc): {sum(bench_alloc_list)/n:.2f} GB\n")
        p(f"  平均分配 mem (rsv):   {sum(bench_rsv_list)/n:.2f} GB\n")
        p(f"  总平均 mem:           {(sum(bench_alloc_list)+sum(bench_rsv_list))/(2*n):.2f} GB\n")
        p(f"{'='*50}\n")

    if getattr(opt, "test_diagnostics", False):
        os.makedirs("timeline", exist_ok=True)
        trace_stamp = time.strftime("%Y%m%d_%H%M%S")
        trace_path = f"timeline/trace_{BRANCH}_{SCENE_NAME}_{trace_stamp}.json"
        tracer.export(trace_path)
        log_kv(LOGGER, "timeline_export", {"path": trace_path, "stamp": trace_stamp})
        log_trace_summary(tracer, LOGGER)
        
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
    parser.add_argument("--trained_ply_path", type=str, default=None)
    parser.add_argument('--git_branch', type=str, default=None)
    parser.add_argument('--keep_training', action='store_true', default=False)
    parser.add_argument('--disable_densify', action='store_true', default=False,
                        help='Skip densify/prune/opacity reset while keeping dynamic block splitting.')
    parser.add_argument('--profile_log', action='store_true', default=False,
                        help='Request detailed per-iteration timing/profile payloads; only active with --test_diagnostics.')
    parser.add_argument('--test_diagnostics', action='store_true', default=False,
                        help='Enable test/benchmark-only diagnostics. Normal training should not pass this.')
    parser.add_argument('--profile_from', type=int, default=1)
    parser.add_argument('--profile_until', type=int, default=0,
                        help='Last iteration for profile logs; 0 means no upper bound.')
    parser.add_argument('--profile_every', type=int, default=1)
    parser.add_argument('--profile_sync', action='store_true', default=False,
                        help='Synchronize CUDA around profile spans for more accurate but slower timings.')
    parser.add_argument('--trace_from', type=int, default=681)
    parser.add_argument('--trace_until', type=int, default=750)
    parser.add_argument('--bench_from', type=int, default=301)
    parser.add_argument('--bench_until', type=int, default=700)
    parser.add_argument('--gpu_cache_threshold_gb', type=float, default=config.GPU_CACHE_THRESHOLD_GB,
                        help='Call torch.cuda.empty_cache when reserved-allocated exceeds this many GB; default is mem-first.')
    parser.add_argument('--gpu_cache_hard_limit_gb', type=float, default=config.GPU_CACHE_HARD_LIMIT_GB,
                        help='Force empty_cache when reserved memory exceeds this GB limit; 0 disables the hard limit.')
    parser.add_argument('--gpu_cache_stage_entry_limit_gb', type=float, default=config.GPU_CACHE_STAGE_ENTRY_LIMIT_GB,
                        help='Force empty_cache before no-grad render when reserved exceeds this GB limit; 0 disables the stage-entry guard.')
    parser.add_argument('--cuda_empty_cache_interval', type=int, default=config.CUDA_EMPTY_CACHE_INTERVAL,
                        help='Only allow periodic empty_cache every N iterations; hard-limit and force paths can still clean sooner.')
    parser.add_argument('--split_size_override', type=int, default=0,
                        help='Override config.SPLIT_SIZE after indoor/outdoor scene default is selected; 0 keeps the default.')
    parser.add_argument('--legacy_per_block_loss', action='store_true', default=False,
                        help='Use the older per-block composed-image loss/backward path instead of one image-loss gradient per iteration.')
    parser.add_argument('--gpu_packed_cache_strategy', type=str, default="tail", choices=["none", "largest", "tail"],
                        help='Reuse no-grad GPU packed buffers in grad pass: none, largest block, or best-fit tail blocks within the configured visible-point budget.')
    parser.add_argument('--gpu_packed_cache_point_budget', type=int, default=0,
                        help='Visible-point budget for tail GPU packed cache admission; 0 uses the scene default.')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    SCENE_NAME = args.source_path.split('/')[-1]
    DATASET_NAME = args.source_path.split('/')[-2]

    # Set SPLIT_SIZE based on indoor/outdoor scene
    if SCENE_NAME in config.INDOOR_SCENES:
        config.SPLIT_SIZE = config.SPLIT_SIZE_INDOOR
        config.GPU_PACKED_CACHE_POINT_BUDGET = config.GPU_PACKED_CACHE_POINT_BUDGET_INDOOR
    elif SCENE_NAME in config.OUTDOOR_SCENES:
        config.SPLIT_SIZE = config.SPLIT_SIZE_OUTDOOR
        config.GPU_PACKED_CACHE_POINT_BUDGET = config.GPU_PACKED_CACHE_POINT_BUDGET_OUTDOOR
    if args.split_size_override > 0:
        config.SPLIT_SIZE = args.split_size_override
    if args.gpu_packed_cache_point_budget > 0:
        config.GPU_PACKED_CACHE_POINT_BUDGET = args.gpu_packed_cache_point_budget
    print(
        f"Scene: {SCENE_NAME} ({'indoor' if SCENE_NAME in config.INDOOR_SCENES else 'outdoor'}), "
        f"SPLIT_SIZE={config.SPLIT_SIZE}, "
        f"GPU_PACKED_CACHE_POINT_BUDGET={config.GPU_PACKED_CACHE_POINT_BUDGET}"
    )

    if args.git_branch is not None:
        BRANCH = args.git_branch
    else:
        BRANCH = get_git_branch()
    
    LOGGER = get_logger(SCENE_NAME, os.path.join("./logs", "train", BRANCH, SCENE_NAME))
    log_section(LOGGER, "Run Context")
    log_runtime_context(LOGGER, repo_dir=os.path.dirname(os.path.abspath(__file__)), extra={
        "scene": SCENE_NAME,
        "dataset": DATASET_NAME,
        "branch": BRANCH,
    })
    log_kv(LOGGER, "args", vars(args))
    log_kv(LOGGER, "config", {
        k: v for k, v in vars(config).items()
        if k.isupper() and isinstance(v, (bool, int, float, str, list, tuple, set, dict))
    })
    DEBUG_MODE = sys.gettrace() is not None
    
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

    trained_ply_path = args.trained_ply_path

    opt = attach_runtime_options(op.extract(args), args)
    if args.keep_training:
        assert trained_ply_path is not None, "--keep_training requires --trained_ply_path"
        print("KEEP_TRAINING MODEL, LOADING FROM CHECKPOINT: ", trained_ply_path)
        res = {
            "first_iter": 1,
            "trained_ply_path": trained_ply_path,
        }
        opt.iterations = 750
    else:
        res = {"first_iter": 1}
    log_kv(LOGGER, "optimization_params", vars(opt))

    training(lp.extract(args), opt, pp.extract(args), args.save_iterations, args.debug_from, res)
