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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
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

    means3D = pc.get_xyz
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
        rendered_image, radii, depth_image, alphaLeft = rasterizer(
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
        rendered_image, radii, depth_image, alphaLeft = rasterizer(
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
        "alphaLeft": alphaLeft
        }
    
    return out

def render_subset(subset_mask, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    pc.start_subset(subset_mask)
    out = render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp)   # 自动只渲染 subset
    pc.end_subset()
    return out



def merge(
    render_list,
    depth_list,
    alphaLeft_list,
    cache_viewspace_points_list=None,
    cache_visibility_filters_list=None,
    cache_radii_list=None,
    eps=1e-10
):
    """
    Multi-block compositing for partitioned Gaussian rendering.

    Inputs per block k:
        render_list[k]  : [3, H, W] RGB -> 这个 block 自己贡献给这个像素的总颜色
        depth_list[k]   : [1, H, W] depth
        alphaLeft_list[k]: [H, W] or [1, H, W] -> 光线穿过这个 block 之后剩下的透射率（还没被前面挡掉的比例）

        cache_viewspace_points_list[k]: [N_k, 3]
        cache_visibility_filters_list[k]: [M_k, 1] or [M_k]
        cache_radii_list[k]: [N_k]

    Returns:
        final_rgb      : [3, H, W]
        bg_rgb         : [3, H, W]
        final_depth    : [1, H, W]
        merged_vps     : [sum_k N_k, 3] or None
        merged_vis_idx : [sum_k M_k, 1] or None (indices into merged_radii)
        merged_radii   : [sum_k N_k] or None
    """



    K = len(render_list)
    assert K > 0, "render_list is empty"

    device = render_list[0].device
    dtype  = render_list[0].dtype

    # ------------------------
    # 0. stack tensors
    # ------------------------
    # RGB: [K, 3, H, W]
    renders = torch.stack(render_list, dim=0)

    # depth: 保证 [K, 1, H, W]
    depth_tensors = []
    for d in depth_list:
        if d.dim() == 2:
            depth_tensors.append(d.unsqueeze(0))
        elif d.dim() == 3:
            depth_tensors.append(d)
        else:
            raise ValueError(f"depth dim must be 2 or 3, got {d.dim()}")
    depths = torch.stack(depth_tensors, dim=0)  # [K,1,H,W]

    # alphaLeft: 保证 [K, 1, H, W]
    alpha_tensors = []
    for a in alphaLeft_list:
        if a.dim() == 2:
            alpha_tensors.append(a.unsqueeze(0))
        elif a.dim() == 3:
            alpha_tensors.append(a)
        else:
            raise ValueError(f"alphaLeft dim must be 2 or 3, got {a.dim()}")
    alphas = torch.stack(alpha_tensors, dim=0)  # [K,1,H,W]

    _, C, H, W = renders.shape
    assert C == 3, f"render channel should be 3, got {C}"

    # ------------------------
    # 1. sort pixels along depth
    # ------------------------
    # 如果你的 depth 是 "越大越近" 或 "inverse depth"，这里可以改成 descending=True/False
    sort_idx = torch.argsort(depths.squeeze(1), dim=0, descending=True)  # [K,H,W]

    # RGB 排序
    idx_rgb = sort_idx.unsqueeze(1).expand(-1, C, -1, -1)        # [K,3,H,W]
    front_rgbs = torch.gather(renders, 0, idx_rgb)               # [K,3,H,W]

    # alpha 排序
    idx_alpha = sort_idx.unsqueeze(1)                            # [K,1,H,W]
    front_alphas = torch.gather(alphas, 0, idx_alpha)            # [K,1,H,W]

    # depth 排序（用于 final_depth）
    front_depths = torch.gather(depths, 0, idx_alpha)            # [K,1,H,W]

    # ------------------------
    # 2. forward compositing
    # ------------------------
    # cumT[k] = prod_{i<=k} alpha_i
    cumT = torch.cumprod(front_alphas, dim=0)                    # [K,1,H,W]
    # prefix_T[k] = prod_{i<k} alpha_i
    prefix_T = torch.cat([torch.ones_like(cumT[:1]), cumT[:-1]], dim=0)  # [K,1,H,W]

    # color & depth 合成
    final_rgb   = (prefix_T * front_rgbs).sum(dim=0)       # [3,H,W]
    final_rgb = final_rgb.clamp(0, 1)

    
    

    final_depth = (prefix_T * front_depths).sum(dim=0)           # [1,H,W]

    # ------------------------
    # 3. background color
    # ------------------------
    log_front_Ts = torch.log(front_alphas.clamp(min=eps))        # [K,1,H,W]
    log_post_prod_inc   = torch.cumsum(log_front_Ts.flip(0), dim=0).flip(0)
    log_post_prod_shift = torch.cat(
        [log_post_prod_inc[1:], torch.zeros_like(log_post_prod_inc[:1])],
        dim=0
    )

    inv_scale    = torch.exp(-log_post_prod_inc).clamp(max=1e6)
    C_scaled     = front_rgbs * inv_scale
    suffix_sum_C = torch.cumsum(C_scaled.flip(0), dim=0).flip(0) - C_scaled
    scale        = torch.exp(log_post_prod_shift)
    suffix_color = scale * suffix_sum_C
    bg_rgb       = suffix_color[0]                               # [3,H,W]

    # ------------------------
    # 4. merge caches (viewspace_points, radii, visibility_filter)
    # ------------------------
    final_viewspace_points   = None
    final_radii = None
    final_visibility_filter   = None

    # 4.1 viewspace_points: 直接 cat
    if cache_viewspace_points_list is not None and len(cache_viewspace_points_list) > 0:
        final_viewspace_points = torch.cat(cache_viewspace_points_list, dim=0)

    # 4.2 radii: 直接 cat
    if cache_radii_list is not None and len(cache_radii_list) > 0:
        final_radii = torch.cat(cache_radii_list, dim=0)

    # 4.3 visibility_filter: 需要做 index offset 后 cat
    if (
        cache_visibility_filters_list is not None
        and cache_radii_list is not None
        and len(cache_visibility_filters_list) == len(cache_radii_list)
    ):
        vis_list = []
        offset = 0
        for vf, radii in zip(cache_visibility_filters_list, cache_radii_list):
            num_r = radii.shape[0]

            if vf is None or vf.numel() == 0:
                offset += num_r
                continue

            # vf 可能是 [M,1] 或 [M]
            if vf.dim() == 2 and vf.size(1) == 1:
                local_idx = vf[:, 0]
            else:
                local_idx = vf.view(-1)

            global_idx = local_idx + offset
            vis_list.append(global_idx)

            offset += num_r

        if len(vis_list) > 0:
            final_visibility_filter = torch.cat(vis_list, dim=0).unsqueeze(1)  # [M_total,1]
        else:
            final_visibility_filter = torch.empty(
                (0, 1), dtype=torch.long, device=device
            )

    return final_rgb, bg_rgb, final_depth, final_viewspace_points, final_visibility_filter, final_radii





def merge2(
    render_list,
    depth_list,
    alphaLeft_list,

    eps=1e-10
):
    """
    Multi-block compositing for partitioned Gaussian rendering.

    Inputs per block k:
        render_list[k]  : [3, H, W] RGB
        depth_list[k]   : [1, H, W] depth
        alphaLeft_list[k]: [H, W] or [1, H, W]  (transmittance / alpha-like)

        cache_viewspace_points_list[k]: [N_k, 3]
        cache_visibility_filters_list[k]: [M_k, 1] or [M_k]
        cache_radii_list[k]: [N_k]

    Returns:
        final_rgb      : [3, H, W]
        bg_rgb         : [3, H, W]
        final_depth    : [1, H, W]
        merged_vps     : [sum_k N_k, 3] or None
        merged_vis_idx : [sum_k M_k, 1] or None (indices into merged_radii)
        merged_radii   : [sum_k N_k] or None
    """

    K = len(render_list)
    assert K > 0, "render_list is empty"

    device = render_list[0].device
    dtype  = render_list[0].dtype

    # ------------------------
    # 0. stack tensors
    # ------------------------
    # RGB: [K, 3, H, W]
    renders = torch.stack(render_list, dim=0)

    # depth: 保证 [K, 1, H, W]
    depth_tensors = []
    for d in depth_list:
        if d.dim() == 2:
            depth_tensors.append(d.unsqueeze(0))
        elif d.dim() == 3:
            depth_tensors.append(d)
        else:
            raise ValueError(f"depth dim must be 2 or 3, got {d.dim()}")
    depths = torch.stack(depth_tensors, dim=0)  # [K,1,H,W]

    # alphaLeft: 保证 [K, 1, H, W]
    alpha_tensors = []
    for a in alphaLeft_list:
        if a.dim() == 2:
            alpha_tensors.append(a.unsqueeze(0))
        elif a.dim() == 3:
            alpha_tensors.append(a)
        else:
            raise ValueError(f"alphaLeft dim must be 2 or 3, got {a.dim()}")
    alphas = torch.stack(alpha_tensors, dim=0)  # [K,1,H,W]

    _, C, H, W = renders.shape
    assert C == 3, f"render channel should be 3, got {C}"

    # ------------------------
    # 1. sort pixels along depth
    # ------------------------
    # 如果你的 depth 是 "越大越近" 或 "inverse depth"，这里可以改成 descending=True/False
    sort_idx = torch.argsort(depths.squeeze(1), dim=0, descending=True)  # [K,H,W]

    # RGB 排序
    idx_rgb = sort_idx.unsqueeze(1).expand(-1, C, -1, -1)        # [K,3,H,W]
    front_rgbs = torch.gather(renders, 0, idx_rgb)               # [K,3,H,W]

    # alpha 排序
    idx_alpha = sort_idx.unsqueeze(1)                            # [K,1,H,W]
    front_alphas = torch.gather(alphas, 0, idx_alpha)            # [K,1,H,W]

    # depth 排序（用于 final_depth）
    front_depths = torch.gather(depths, 0, idx_alpha)            # [K,1,H,W]

    # ------------------------
    # 2. forward compositing
    # ------------------------
    # cumT[k] = prod_{i<=k} alpha_i
    cumT = torch.cumprod(front_alphas, dim=0)                    # [K,1,H,W]
    # prefix_T[k] = prod_{i<k} alpha_i
    prefix_T = torch.cat([torch.ones_like(cumT[:1]), cumT[:-1]], dim=0)  # [K,1,H,W]

    # color & depth 合成
    final_rgb   = (prefix_T * front_rgbs).sum(dim=0)       # [3,H,W]
    final_rgb = final_rgb.clamp(0, 1)

    
    

    final_depth = (prefix_T * front_depths).sum(dim=0)           # [1,H,W]

    # ------------------------
    # 3. background color
    # ------------------------
    log_front_Ts = torch.log(front_alphas.clamp(min=eps))        # [K,1,H,W]
    log_post_prod_inc   = torch.cumsum(log_front_Ts.flip(0), dim=0).flip(0)
    log_post_prod_shift = torch.cat(
        [log_post_prod_inc[1:], torch.zeros_like(log_post_prod_inc[:1])],
        dim=0
    )

    inv_scale    = torch.exp(-log_post_prod_inc).clamp(max=1e6)
    C_scaled     = front_rgbs * inv_scale
    suffix_sum_C = torch.cumsum(C_scaled.flip(0), dim=0).flip(0) - C_scaled
    scale        = torch.exp(log_post_prod_shift)
    suffix_color = scale * suffix_sum_C
    bg_rgb       = suffix_color[0]                               # [3,H,W]




    return final_rgb, bg_rgb, final_depth
