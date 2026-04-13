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
import math
from diff_gaussian_rasterization_wenqi_tam import GaussianRasterizationSettings, GaussianRasterizer
import diff_gaussian_rasterization_wenqi_tam._C as _merge_C
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

_MERGE_FAST_VALIDATE_CALLS = 0
_MERGE_FAST_VALIDATE_BAD = 0
_MERGE_FAST_FALLBACK_CALLS = 0
_MERGE_FAST_FALLBACK_WARNED_KS = set()


def _tensor_diff_stats(a, b):
    diff = (a - b).abs()
    if diff.numel() == 0:
        return 0.0, 0.0
    return float(diff.max().item()), float(diff.mean().item())


def _stable_bg_from_sorted(front_rgbs, front_alphas):
    if front_rgbs.shape[0] <= 1:
        return torch.zeros_like(front_rgbs[0])
    tail_rgbs = front_rgbs[1:]
    tail_alphas = front_alphas[1:]
    tail_cumT = torch.cumprod(tail_alphas, dim=0)
    tail_prefix_T = torch.cat([torch.ones_like(tail_cumT[:1]), tail_cumT[:-1]], dim=0)
    return (tail_prefix_T * tail_rgbs).sum(dim=0)


def _format_merge_pixel_debug(render_list, depth_list, alphaLeft_list, fast, ref, y, x):
    depths = []
    alphas = []
    for d in depth_list:
        d0 = d if d.dim() == 3 else d.unsqueeze(0)
        depths.append(float(d0[0, y, x].item()))
    for a in alphaLeft_list:
        a0 = a if a.dim() == 3 else a.unsqueeze(0)
        alphas.append(float(a0[0, y, x].item()))

    fast_rank = fast["block_rank"][:, y, x]
    ref_rank = ref["block_rank"][:, y, x]
    fast_order = torch.argsort(fast_rank).detach().cpu().tolist()
    ref_order = torch.argsort(ref_rank).detach().cpu().tolist()
    fast_prefix = fast["prefix_T"][:, 0, y, x].detach().cpu().tolist()
    ref_prefix = ref["prefix_T"][:, 0, y, x].detach().cpu().tolist()

    return {
        "pixel": [int(y), int(x)],
        "depths": [round(v, 8) for v in depths],
        "alphas": [round(v, 8) for v in alphas],
        "fast_order": [int(v) for v in fast_order],
        "ref_order": [int(v) for v in ref_order],
        "fast_rank": fast_rank.detach().cpu().tolist(),
        "ref_rank": ref_rank.detach().cpu().tolist(),
        "fast_prefix_T": [round(float(v), 8) for v in fast_prefix],
        "ref_prefix_T": [round(float(v), 8) for v in ref_prefix],
    }


def _validate_merge_fast(render_list, depth_list, alphaLeft_list, fast, eps, chunk_size):
    import config
    global _MERGE_FAST_VALIDATE_CALLS, _MERGE_FAST_VALIDATE_BAD

    _MERGE_FAST_VALIDATE_CALLS += 1
    call_id = _MERGE_FAST_VALIDATE_CALLS
    every = max(1, int(getattr(config, "MERGE_FAST_VALIDATE_EVERY", 1)))
    until = int(getattr(config, "MERGE_FAST_VALIDATE_UNTIL", 750))
    if call_id > until or call_id % every != 0:
        return

    ref = _merge_opt_kid_chunked(render_list, depth_list, alphaLeft_list, eps, chunk_size)
    atol = float(getattr(config, "MERGE_FAST_VALIDATE_ATOL", 1e-5))
    bg_atol = float(getattr(config, "MERGE_FAST_VALIDATE_BG_ATOL", atol))
    ref_bg_stable = _stable_bg_from_sorted(ref["front_rgbs"], ref["front_alphas"])

    final_max, final_mean = _tensor_diff_stats(fast["final_rgb"], ref["final_rgb"])
    bg_max, bg_mean = _tensor_diff_stats(fast["bg_rgb"], ref_bg_stable)
    bg_legacy_max, bg_legacy_mean = _tensor_diff_stats(fast["bg_rgb"], ref["bg_rgb"])
    front_max, front_mean = _tensor_diff_stats(fast["front_rgbs"], ref["front_rgbs"])
    prefix_max, prefix_mean = _tensor_diff_stats(fast["prefix_T"], ref["prefix_T"])
    rank_mismatch = int((fast["block_rank"] != ref["block_rank"]).sum().item())

    bad = (
        final_max > atol
        or bg_max > bg_atol
        or front_max > atol
        or prefix_max > atol
        or rank_mismatch > 0
    )
    if not bad and not (call_id <= 3 or call_id % max(1, every * 50) == 0):
        return

    status = "BAD" if bad else "OK"
    details = {
        "call": int(call_id),
        "status": status,
        "K": int(len(render_list)),
        "final_rgb_max": round(final_max, 8),
        "final_rgb_mean": round(final_mean, 10),
        "bg_rgb_max": round(bg_max, 8),
        "bg_rgb_mean": round(bg_mean, 10),
        "bg_rgb_legacy_max": round(bg_legacy_max, 8),
        "bg_rgb_legacy_mean": round(bg_legacy_mean, 10),
        "front_rgbs_max": round(front_max, 8),
        "front_rgbs_mean": round(front_mean, 10),
        "prefix_T_max": round(prefix_max, 8),
        "prefix_T_mean": round(prefix_mean, 10),
        "rank_mismatch": rank_mismatch,
        "atol": atol,
        "bg_atol": bg_atol,
    }

    if bad:
        _MERGE_FAST_VALIDATE_BAD += 1
        max_print = int(getattr(config, "MERGE_FAST_VALIDATE_MAX_PRINT", 20))
        if _MERGE_FAST_VALIDATE_BAD <= max_print:
            H, W = fast["final_rgb"].shape[-2:]
            if rank_mismatch > 0:
                details["bad_metric"] = "block_rank"
                bad_pos = (fast["block_rank"] != ref["block_rank"]).nonzero(as_tuple=False)[0]
                y = int(bad_pos[1].item())
                x = int(bad_pos[2].item())
            else:
                metric_tensors = [
                    ("final_rgb", (fast["final_rgb"] - ref["final_rgb"]).abs()),
                    ("bg_rgb", (fast["bg_rgb"] - ref_bg_stable).abs()),
                    ("front_rgbs", (fast["front_rgbs"] - ref["front_rgbs"]).abs()),
                    ("prefix_T", (fast["prefix_T"] - ref["prefix_T"]).abs()),
                ]
                metric_name, metric_diff = max(
                    metric_tensors,
                    key=lambda item: float(item[1].max().item()) if item[1].numel() else 0.0,
                )
                details["bad_metric"] = metric_name
                flat = metric_diff.reshape(-1).argmax()
                pixel = int(flat.item()) % (H * W)
                y = pixel // W
                x = pixel % W
            details["debug"] = _format_merge_pixel_debug(
                render_list, depth_list, alphaLeft_list, fast, ref, y, x
            )
    print(f"[merge_fast_validate] {details}", flush=True)

    if bad and bool(getattr(config, "MERGE_FAST_VALIDATE_FATAL", False)):
        raise RuntimeError(f"FastMerge validation failed: {details}")


def _warn_merge_fast_fallback(K, max_k, render_list):
    global _MERGE_FAST_FALLBACK_CALLS

    _MERGE_FAST_FALLBACK_CALLS += 1
    first_for_k = K not in _MERGE_FAST_FALLBACK_WARNED_KS
    periodic = _MERGE_FAST_FALLBACK_CALLS % 100 == 0
    if not first_for_k and not periodic:
        return

    _MERGE_FAST_FALLBACK_WARNED_KS.add(K)
    _, H, W = render_list[0].shape
    details = {
        "status": "WARNING",
        "reason": "merge_k_exceeds_fast_limit",
        "K": int(K),
        "fast_max_k": int(max_k),
        "fallback": "pytorch_chunked_argsort",
        "fallback_calls": int(_MERGE_FAST_FALLBACK_CALLS),
        "image_hw": [int(H), int(W)],
    }
    print(f"[merge_fast_fallback_warning] {details}", flush=True)


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, separate_sh = False, override_color = None,
           use_trained_exp=False, retain_viewspace_grad=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    means3D = pc.get_xyz
    if retain_viewspace_grad is None:
        retain_viewspace_grad = torch.is_grad_enabled()

    # Only densification needs the retained 2D/screen-space gradient.
    screenspace_points = torch.zeros_like(
        means3D,
        dtype=means3D.dtype,
        requires_grad=retain_viewspace_grad,
        device="cuda",
    )
    if retain_viewspace_grad:
        screenspace_points = screenspace_points + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity
    
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        _,_,rendered_image, radii, depth_image, alphaLeft = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        _,_,rendered_image, radii, depth_image, alphaLeft = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "alphaLeft" : alphaLeft
        }
    
    return out


def merge_opt( N_total, render_list, depth_list, alphaLeft_list, vis_filter_list=None, radii_list=None, visible_indices_list=None, eps=1e-10, chunk_size=32 ):
    """
    Memory-efficient Multi-block compositing using Spatial Chunking.
    
    Logic is mathematically identical to the original implementation but 
    processes the image in horizontal strips to minimize peak VRAM usage.
    """
    
    K = len(render_list)
    assert K > 0, "render_list is empty"

    # 获取基本信息
    device = render_list[0].device
    dtype  = render_list[0].dtype
    _, H, W = render_list[0].shape  # assuming [3, H, W]

    # ==========================================
    # Part 1: 非图像数据的合并 (Global Merge)
    # 这部分数据量小，直接全局合并即可，无需分块
    # ==========================================

    # 1.1 radii
    global_radii = torch.zeros(N_total, device=device)
    for radii, visible_indice in zip(radii_list, visible_indices_list):
        radii_trans = radii.view(-1).to(global_radii.dtype)
        global_radii.scatter_reduce_(dim=0, index=visible_indice, src=radii_trans, reduce="amax", include_self=True)

    # 1.3 visibility_filter
    # 相当于再过一遍细筛, vis_filter表示当前送去的高斯哪些是有效的(实际参与了渲染的)
    global_visibility_filter = torch.unique(
        torch.cat(
            [
                visible_indices[vis_filter]
                for visible_indices, vis_filter
                in zip(visible_indices_list, vis_filter_list)
            ]
        ),
        sorted=True,
    )

    
    # ==========================================
    # Part 2: 图像数据的分块合并 (Chunked Merge)
    # 核心优化：避免创建 [K, C, H, W] 的全图张量
    # ==========================================
    
    # 用于收集分块结果的容器
    out_final_rgb = []
    out_bg_rgb = []
    out_final_depth = []
    out_sort_idx = []
    out_front_rgbs = []
    out_front_alphas = []
    out_prefix_T = []
    out_block_rank = []

    # 按行进行分块循环
    for i in range(0, H, chunk_size):
        # 确定当前块的结束行
        end = min(i + chunk_size, H)
        h_chunk = end - i
        
        # -------------------------------------------------
        # 2.1 局部 Stack (仅针对当前 chunk 的区域)
        # -------------------------------------------------
        # Render: [K, 3, h_chunk, W]
        renders_chunk = torch.stack([r[:, i:end, :] for r in render_list], dim=0)
        
        # Depth: [K, 1, h_chunk, W]
        depth_tensors = []
        for d in depth_list:
            # 处理 slice
            d_slice = d[:, i:end, :] if d.dim() == 3 else d[i:end, :]
            if d_slice.dim() == 2: d_slice = d_slice.unsqueeze(0)
            depth_tensors.append(d_slice)
        depths_chunk = torch.stack(depth_tensors, dim=0)

        # Alpha: [K, 1, h_chunk, W]
        alpha_tensors = []
        for a in alphaLeft_list:
            # 处理 slice
            a_slice = a[:, i:end, :] if a.dim() == 3 else a[i:end, :]
            if a_slice.dim() == 2: a_slice = a_slice.unsqueeze(0)
            alpha_tensors.append(a_slice)
        alphas_chunk = torch.stack(alpha_tensors, dim=0)
        
        # -------------------------------------------------
        # 2.2 局部 Sort & Gather (原版逻辑)
        # -------------------------------------------------
        # Sort along depth
        sort_idx_chunk = torch.argsort(depths_chunk.squeeze(1), dim=0, descending=True, stable=True) # [K, h, W]
        
        # Gather RGB
        idx_rgb = sort_idx_chunk.unsqueeze(1).expand(-1, 3, -1, -1)
        front_rgbs_chunk = torch.gather(renders_chunk, 0, idx_rgb)
        
        # Gather Alpha
        idx_alpha = sort_idx_chunk.unsqueeze(1)
        front_alphas_chunk = torch.gather(alphas_chunk, 0, idx_alpha)
        
        # Gather Depth
        front_depths_chunk = torch.gather(depths_chunk, 0, idx_alpha)

        # -------------------------------------------------
        # 2.3 Forward Compositing (原版逻辑)
        # -------------------------------------------------
        cumT = torch.cumprod(front_alphas_chunk, dim=0)
        prefix_T_chunk = torch.cat([torch.ones_like(cumT[:1]), cumT[:-1]], dim=0)

        final_rgb_chunk = (prefix_T_chunk * front_rgbs_chunk).sum(dim=0).clamp(0, 1) # [3, h, W]
        final_depth_chunk = (prefix_T_chunk * front_depths_chunk).sum(dim=0)         # [1, h, W]

        # -------------------------------------------------
        # 2.4 Background Color (原版逻辑)
        # -------------------------------------------------
        log_front_Ts = torch.log(front_alphas_chunk.clamp(min=eps))
        log_post_prod_inc = torch.cumsum(log_front_Ts.flip(0), dim=0).flip(0)
        log_post_prod_shift = torch.cat(
            [log_post_prod_inc[1:], torch.zeros_like(log_post_prod_inc[:1])], dim=0
        )
        
        inv_scale = torch.exp(-log_post_prod_inc).clamp(max=1e6)
        C_scaled = front_rgbs_chunk * inv_scale
        suffix_sum_C = torch.cumsum(C_scaled.flip(0), dim=0).flip(0) - C_scaled
        scale = torch.exp(log_post_prod_shift)
        suffix_color = scale * suffix_sum_C
        bg_rgb_chunk = suffix_color[0] # [3, h, W]

        # -------------------------------------------------
        # 2.5 Block Rank Calculation (局部计算)
        # -------------------------------------------------
        # sort_idx_chunk: [K, h, W]
        flat = sort_idx_chunk.reshape(K, -1)
        br_chunk = torch.empty_like(flat)
        cols = torch.arange(flat.shape[1], device=device)
        
        for k in range(K):
            br_chunk[flat[k], cols] = k
        
        block_rank_chunk = br_chunk.reshape(K, h_chunk, W)

        # -------------------------------------------------
        # 2.6 收集结果
        # -------------------------------------------------
        out_final_rgb.append(final_rgb_chunk)
        out_bg_rgb.append(bg_rgb_chunk)
        out_final_depth.append(final_depth_chunk)
        
        # 如果下游任务需要由于backward，这些中间变量也需要保存
        # 虽然保存了，但避免了在创建时峰值过高
        out_sort_idx.append(sort_idx_chunk)
        out_front_rgbs.append(front_rgbs_chunk)
        out_front_alphas.append(front_alphas_chunk)
        out_prefix_T.append(prefix_T_chunk)
        out_block_rank.append(block_rank_chunk)

    # ==========================================
    # Part 3: 拼接最终结果
    # ==========================================
    return {
        "final_rgb": torch.cat(out_final_rgb, dim=1),            # [3, H, W]
        "bg_rgb": torch.cat(out_bg_rgb, dim=1),                  # [3, H, W]
        "final_depth": torch.cat(out_final_depth, dim=1),        # [1, H, W]
        
        "global_visibility_filter": global_visibility_filter,
        "global_radii": global_radii,
        
        "sort_idx": torch.cat(out_sort_idx, dim=1),              # [K, H, W]
        "front_rgbs": torch.cat(out_front_rgbs, dim=2),          # [K, 3, H, W] 注意dim=2因为是H轴
        "front_alphas": torch.cat(out_front_alphas, dim=2),      # [K, 1, H, W]
        "prefix_T": torch.cat(out_prefix_T, dim=2),              # [K, 1, H, W]
        "block_rank": torch.cat(out_block_rank, dim=1)           # [K, H, W]
    }


def merge_opt_kid(render_list, depth_list, alphaLeft_list, eps=1e-10, chunk_size=32):
    import config
    K = len(render_list)
    max_k = int(getattr(config, "MERGE_FAST_MAX_K", 16))
    if config.MERGE_FAST:
        if K <= max_k:
            fast = _merge_opt_kid_fast(render_list, depth_list, alphaLeft_list, eps)
            if getattr(config, "MERGE_FAST_VALIDATE", False):
                _validate_merge_fast(render_list, depth_list, alphaLeft_list, fast, eps, chunk_size)
            return fast
        _warn_merge_fast_fallback(K, max_k, render_list)
    return _merge_opt_kid_chunked(render_list, depth_list, alphaLeft_list, eps, chunk_size)


def _merge_opt_kid_fast(render_list, depth_list, alphaLeft_list, eps=1e-10):
    """
    Fused single-kernel merge. One CUDA kernel handles sort + composite
    + prefix_T + bg_rgb + block_rank per pixel in registers.
    """
    K = len(render_list)
    assert K > 0, "render_list is empty"

    renders = torch.stack(render_list, dim=0).contiguous()              # [K, 3, H, W]
    depths = torch.stack([d if d.dim() == 3 else d.unsqueeze(0)
                          for d in depth_list], dim=0).contiguous()     # [K, 1, H, W]
    alphas = torch.stack([a if a.dim() == 3 else a.unsqueeze(0)
                          for a in alphaLeft_list], dim=0).contiguous() # [K, 1, H, W]

    out = _merge_C.merge_blocks(renders, depths, alphas, eps)
    # out = [final_rgb, bg_rgb, front_rgbs, prefix_T(K,H,W), block_rank(K,H,W int32)]

    return {
        "final_rgb":  out[0],                        # [3, H, W]
        "bg_rgb":     out[1],                        # [3, H, W]
        "front_rgbs": out[2],                        # [K, 3, H, W]
        "prefix_T":   out[3].unsqueeze(1),           # [K, 1, H, W]
        "block_rank": out[4].long(),                 # [K, H, W] int64
    }


def _merge_opt_kid_fast_py(render_list, depth_list, alphaLeft_list, eps=1e-10):
    """
    Original Python reference implementation (kept for validation).
    """
    K = len(render_list)
    assert K > 0, "render_list is empty"
    device = render_list[0].device

    renders = torch.stack(render_list, dim=0)
    depths = torch.stack([d if d.dim() == 3 else d.unsqueeze(0)
                          for d in depth_list], dim=0)
    alphas = torch.stack([a if a.dim() == 3 else a.unsqueeze(0)
                          for a in alphaLeft_list], dim=0)

    sort_idx = torch.argsort(depths.squeeze(1), dim=0, descending=True, stable=True)
    idx_rgb = sort_idx.unsqueeze(1).expand(-1, 3, -1, -1)
    idx_1ch = sort_idx.unsqueeze(1)

    front_rgbs = torch.gather(renders, 0, idx_rgb)
    front_alphas = torch.gather(alphas, 0, idx_1ch)

    cumT = torch.cumprod(front_alphas, dim=0)
    prefix_T = torch.cat([torch.ones_like(cumT[:1]), cumT[:-1]], dim=0)
    final_rgb = (prefix_T * front_rgbs).sum(dim=0).clamp(0, 1)

    log_front_Ts = torch.log(front_alphas.clamp(min=eps))
    log_post_prod_inc = torch.cumsum(log_front_Ts.flip(0), dim=0).flip(0)
    log_post_prod_shift = torch.cat(
        [log_post_prod_inc[1:], torch.zeros_like(log_post_prod_inc[:1])], dim=0
    )
    inv_scale = torch.exp(-log_post_prod_inc).clamp(max=1e6)
    C_scaled = front_rgbs * inv_scale
    suffix_sum_C = torch.cumsum(C_scaled.flip(0), dim=0).flip(0) - C_scaled
    bg_rgb = (torch.exp(log_post_prod_shift) * suffix_sum_C)[0]

    ranks = torch.arange(K, device=device).view(K, 1, 1).expand_as(sort_idx)
    block_rank = torch.zeros_like(sort_idx)
    block_rank.scatter_(0, sort_idx, ranks)

    return {
        "final_rgb": final_rgb,
        "bg_rgb": bg_rgb,
        "front_rgbs": front_rgbs,
        "prefix_T": prefix_T,
        "block_rank": block_rank,
    }


def _merge_opt_kid_chunked(render_list, depth_list, alphaLeft_list, eps=1e-10, chunk_size=32):
    """
    Memory-efficient Multi-block compositing using Spatial Chunking.

    Logic is mathematically identical to the original implementation but
    processes the image in horizontal strips to minimize peak VRAM usage.
    """

    K = len(render_list)
    assert K > 0, "render_list is empty"

    # 获取基本信息
    device = render_list[0].device
    dtype  = render_list[0].dtype
    _, H, W = render_list[0].shape  # assuming [3, H, W]

    # ==========================================
    # Part 1: 非图像数据的合并 (Global Merge)
    # 这部分数据量小，直接全局合并即可，无需分块
    # ==========================================


    # ==========================================
    # Part 2: 图像数据的分块合并 (Chunked Merge)
    # 核心优化：避免创建 [K, C, H, W] 的全图张量
    # ==========================================
    
    # 用于收集分块结果的容器
    out_final_rgb = []
    out_bg_rgb = []
    out_final_depth = []
    out_sort_idx = []
    out_front_rgbs = []
    out_front_alphas = []
    out_prefix_T = []
    out_block_rank = []

    # 按行进行分块循环
    for i in range(0, H, chunk_size):
        # 确定当前块的结束行
        end = min(i + chunk_size, H)
        h_chunk = end - i
        
        # -------------------------------------------------
        # 2.1 局部 Stack (仅针对当前 chunk 的区域)
        # -------------------------------------------------
        # Render: [K, 3, h_chunk, W]
        renders_chunk = torch.stack([r[:, i:end, :] for r in render_list], dim=0)
        
        # Depth: [K, 1, h_chunk, W]
        depth_tensors = []
        for d in depth_list:
            # 处理 slice
            d_slice = d[:, i:end, :] if d.dim() == 3 else d[i:end, :]
            if d_slice.dim() == 2: d_slice = d_slice.unsqueeze(0)
            depth_tensors.append(d_slice)
        depths_chunk = torch.stack(depth_tensors, dim=0)

        # Alpha: [K, 1, h_chunk, W]
        alpha_tensors = []
        for a in alphaLeft_list:
            # 处理 slice
            a_slice = a[:, i:end, :] if a.dim() == 3 else a[i:end, :]
            if a_slice.dim() == 2: a_slice = a_slice.unsqueeze(0)
            alpha_tensors.append(a_slice)
        alphas_chunk = torch.stack(alpha_tensors, dim=0)
        
        # -------------------------------------------------
        # 2.2 局部 Sort & Gather (原版逻辑)
        # -------------------------------------------------
        # Sort along depth
        sort_idx_chunk = torch.argsort(depths_chunk.squeeze(1), dim=0, descending=True, stable=True) # [K, h, W]
        
        # Gather RGB
        idx_rgb = sort_idx_chunk.unsqueeze(1).expand(-1, 3, -1, -1)
        front_rgbs_chunk = torch.gather(renders_chunk, 0, idx_rgb)
        
        # Gather Alpha
        idx_alpha = sort_idx_chunk.unsqueeze(1)
        front_alphas_chunk = torch.gather(alphas_chunk, 0, idx_alpha)
        
        # Gather Depth
        front_depths_chunk = torch.gather(depths_chunk, 0, idx_alpha)

        # -------------------------------------------------
        # 2.3 Forward Compositing (原版逻辑)
        # -------------------------------------------------
        cumT = torch.cumprod(front_alphas_chunk, dim=0)
        prefix_T_chunk = torch.cat([torch.ones_like(cumT[:1]), cumT[:-1]], dim=0)

        final_rgb_chunk = (prefix_T_chunk * front_rgbs_chunk).sum(dim=0).clamp(0, 1) # [3, h, W]
        final_depth_chunk = (prefix_T_chunk * front_depths_chunk).sum(dim=0)         # [1, h, W]

        # -------------------------------------------------
        # 2.4 Background Color (原版逻辑)
        # -------------------------------------------------
        log_front_Ts = torch.log(front_alphas_chunk.clamp(min=eps))
        log_post_prod_inc = torch.cumsum(log_front_Ts.flip(0), dim=0).flip(0)
        log_post_prod_shift = torch.cat(
            [log_post_prod_inc[1:], torch.zeros_like(log_post_prod_inc[:1])], dim=0
        )
        
        inv_scale = torch.exp(-log_post_prod_inc).clamp(max=1e6)
        C_scaled = front_rgbs_chunk * inv_scale
        suffix_sum_C = torch.cumsum(C_scaled.flip(0), dim=0).flip(0) - C_scaled
        scale = torch.exp(log_post_prod_shift)
        suffix_color = scale * suffix_sum_C
        bg_rgb_chunk = suffix_color[0] # [3, h, W]

        # -------------------------------------------------
        # 2.5 Block Rank Calculation (局部计算)
        # -------------------------------------------------
        # sort_idx_chunk: [K, h, W]
        flat = sort_idx_chunk.reshape(K, -1)
        br_chunk = torch.empty_like(flat)
        cols = torch.arange(flat.shape[1], device=device)
        
        for k in range(K):
            br_chunk[flat[k], cols] = k
        
        block_rank_chunk = br_chunk.reshape(K, h_chunk, W)

        # -------------------------------------------------
        # 2.6 收集结果
        # -------------------------------------------------
        out_final_rgb.append(final_rgb_chunk)
        out_bg_rgb.append(bg_rgb_chunk)
        out_final_depth.append(final_depth_chunk)
        
        # 如果下游任务需要由于backward，这些中间变量也需要保存
        # 虽然保存了，但避免了在创建时峰值过高
        out_sort_idx.append(sort_idx_chunk)
        out_front_rgbs.append(front_rgbs_chunk)
        out_front_alphas.append(front_alphas_chunk)
        out_prefix_T.append(prefix_T_chunk)
        out_block_rank.append(block_rank_chunk)

    # ==========================================
    # Part 3: 拼接最终结果
    # ==========================================
    return {
        "final_rgb": torch.cat(out_final_rgb, dim=1),            # [3, H, W]
        "bg_rgb": torch.cat(out_bg_rgb, dim=1),                  # [3, H, W]
        "final_depth": torch.cat(out_final_depth, dim=1),        # [1, H, W]
        

        "sort_idx": torch.cat(out_sort_idx, dim=1),              # [K, H, W]
        "front_rgbs": torch.cat(out_front_rgbs, dim=2),          # [K, 3, H, W] 注意dim=2因为是H轴
        "front_alphas": torch.cat(out_front_alphas, dim=2),      # [K, 1, H, W]
        "prefix_T": torch.cat(out_prefix_T, dim=2),              # [K, 1, H, W]
        "block_rank": torch.cat(out_block_rank, dim=1)           # [K, H, W]
    }
