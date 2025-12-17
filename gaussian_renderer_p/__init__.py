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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torchvision
from scene_p.gaussian_model import GaussianModel_p
from utils_p.sh_utils import eval_sh
from torchvision.utils import draw_bounding_boxes

import config
def render(viewpoint_camera, pc : GaussianModel_p, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
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

    # ------- 新增：计算 block_rank -------
    # sort_idx[k,h,w] = block_id
    # block_rank[block_id,h,w] = k     (反向映射）
    K, H, W = sort_idx.shape
    flat = sort_idx.reshape(K, -1)        # [K,HW]
    br = torch.empty_like(flat)
    cols = torch.arange(flat.shape[1])

    for k in range(K):
        br[ flat[k], cols ] = k

    block_rank = br.reshape(K, H, W)

    
    return {
        "final_rgb": final_rgb,
        "bg_rgb": bg_rgb,
        "final_depth": final_depth,
        "final_viewspace_points": final_viewspace_points,
        "final_visibility_filter": final_visibility_filter,
        "final_radii": final_radii,
        "sort_idx": sort_idx,
        "front_rgbs": front_rgbs,
        "front_alphas": front_alphas,
        "prefix_T": prefix_T,
        "block_rank": block_rank
    }


def get_frustum_planes(view_proj_matrix):
    """
    view_proj_matrix: viewpoint_cam.full_proj_transform (已经转置好的 4x4)
    """
    # 如果 matrix 是 [4,4], 我们直接按行提取
    # 3DGS 的 full_proj_transform 通常已经是转置后的逻辑
    matrix = view_proj_matrix.T 
    planes = torch.stack([
        matrix[3] + matrix[0], # Left
        matrix[3] - matrix[0], # Right
        matrix[3] + matrix[1], # Bottom
        matrix[3] - matrix[1], # Top
        matrix[3] + matrix[2], # Near
        matrix[3] - matrix[2]  # Far
    ])
    # 归一化平面系数，方便后续计算
    norms = torch.norm(planes[:, :3], dim=1, keepdim=True)
    return planes / norms

def is_block_visible(block, planes):
    """
    判断 Block 对象的 AABB 是否在视锥体内
    """
    # 拿到 Block 里的 mins 和 maxs
    # 确保它们在 GPU 上且是 torch 类型
    b_min = torch.as_tensor(block.mins, device=planes.device)
    b_max = torch.as_tensor(block.maxs, device=planes.device)
    
    for i in range(6):
        # 寻找 P-vertex (法线方向上最远的点)
        p_vertex = torch.where(planes[i, :3] > 0, b_max, b_min)
        # 如果最远点都在平面后面，则盒子完全不可见
        if torch.dot(planes[i, :3], p_vertex) + planes[i, 3] < 0:
            return False
    return True

def is_block_within_dist(block, cam_center, max_dist):
    """
    block: 你的 Block 对象，包含 mins, maxs
    cam_center: 相机在世界坐标系的位置 (viewpoint_cam.camera_center)
    max_dist: 你设定的可见距离阈值
    """
    # 计算块的中心点
    b_min = torch.as_tensor(block.mins, device=cam_center.device)
    b_max = torch.as_tensor(block.maxs, device=cam_center.device)
    center = (b_min + b_max) * 0.5
    
    # 计算欧式距离的平方（避免开方计算，更快）
    dist_sq = torch.sum((center - cam_center) ** 2)
    return dist_sq < (max_dist ** 2)

def is_block_on_screen(block, viewpoint_cam):
    """
    更精细的判断：检查块投影到 2D 屏幕后的范围
    """
    # 1. 拿到 AABB 的 8 个顶点
    mins, maxs = block.mins, block.maxs
    corners = torch.tensor([
        [mins[0], mins[1], mins[2]], [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]], [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]], [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]], [maxs[0], maxs[1], maxs[2]],
    ], device='cuda', dtype=torch.float32)

    # 2. 投影到屏幕空间 (使用相机的全局矩阵)
    # world_to_clip: p_clip = p_world * full_proj_transform
    # 注意维度匹配
    p_homo = torch.cat([corners, torch.ones((8, 1), device='cuda')], dim=-1)
    p_clip = p_homo @ viewpoint_cam.full_proj_transform
    
    # 归一化设备坐标 (NDC)
    w = p_clip[:, 3:4]
    # 如果所有点的 w 都是负数，说明在相机后面
    if (w < 0.001).all():
        return False
        
    ndc = p_clip[:, :3] / (w + 1e-7)
    
    # 3. 检查 NDC 是否在有效范围内 [-1, 1]
    # 如果 8 个顶点的投影全在屏幕外（比如全在左边），则剔除
    if (ndc[:, 0] < -1.1).all() or (ndc[:, 0] > 1.1).all() or \
       (ndc[:, 1] < -1.1).all() or (ndc[:, 1] > 1.1).all():
        return False

    return True

def get_visible_mask_in_block(block_mask, xyz, planes):
    """
    block_mask: 该块对应的原始索引 [N_block]
    xyz: 全局点云坐标 [N_total, 3]
    planes: 视锥体 6 个平面 [6, 4]
    """
    # 1. 提取该块内的点
    p_xyz = xyz[block_mask] # [N_block, 3]
    device = p_xyz.device
    
    # 2. 确保 planes 在同一设备
    planes = planes.to(device)

    # 3. 无齐次坐标计算点到平面的距离
    # 距离公式: Dist = A*x + B*y + C*z + D
    # planes[:, :3] 是 (A, B, C), planes[:, 3] 是 D
    # [6, 3] @ [3, N_block] + [6, 1] -> [6, N_block]
    distances = planes[:, :3] @ p_xyz.T + planes[:, 3:4]
    
    # 4. 只有在所有 6 个平面“内侧”（距离 >= 0）的点才是可见的
    visible_bool = torch.all(distances >= 0, dim=0) # [N_block]
    
    return block_mask[visible_bool]

def get_block_screen_bbox(xyz, full_proj_transform, W, H):
    """
    xyz: [N, 3] 高斯点坐标 (GPU)
    full_proj_transform: viewpoint_cam.full_proj_transform
    W, H: 图像宽高
    """
    device = xyz.device
    # 1. 确保矩阵在 GPU 上且类型匹配
    full_proj_transform = full_proj_transform.to(device=device, dtype=xyz.dtype)

    # 2. 构造齐次坐标
    p_homo = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=device, dtype=xyz.dtype)], dim=-1)
    
    # 3. 投影变换
    p_clip = p_homo @ full_proj_transform
    
    # 4. 关键：剔除 w <= 0.05 的点（相机背后的点）
    w = p_clip[:, 3:4]
    mask = (w > 0.05).squeeze() 
    
    # 容错：如果没有点在相机前方
    if not mask.any():
        return 0, 0, 0, 0
    
    # 5. 提取有效点并进行透视除法
    valid_p_clip = p_clip[mask]
    valid_w = w[mask]
    
    # 这里的关键：即使只有一个点，也通过 reshape 确保 ndc 是 [N, 2]
    # 避免 [2] 这种一维向量导致 ndc[:, 1] 报错
    ndc = (valid_p_clip[:, :2] / valid_w).reshape(-1, 2)
    
    # 6. 映射到像素空间 (保持你验证正确的 +1.0 逻辑)
    screen_x = (ndc[:, 0] + 1.0) * W / 2.0
    screen_y = (ndc[:, 1] + 1.0) * H / 2.0
    
    # 7. 取得边界并转为整数，增加简单的边界溢出保护
    x_min = max(0, int(screen_x.min().item()))
    y_min = max(0, int(screen_y.min().item()))
    x_max = min(W, int(screen_x.max().item()))
    y_max = min(H, int(screen_y.max().item()))
    
    return x_min, y_min, x_max, y_max



def print_box_on_image(image, x_min, y_min, x_max, y_max):
    """
    在渲染图上绘制 BBox 并返回结果 Tensor
    :param image: [3, H, W] 的 torch.Tensor, 范围 [0, 1]
    :param x_min, y_min, x_max, y_max: 像素坐标 (int)
    :return: [3, H, W] 的 torch.Tensor, 范围 [0, 1]
    """
    with torch.no_grad():
        # 1. 转换到 uint8 格式 (draw_bounding_boxes 的标准要求)
        # clamp 保证 [0, 1] 范围，避免溢出
        img_uint8 = (image.detach().clamp(0, 1) * 255).to(torch.uint8).cpu()
        
        # 2. 坐标合法性裁剪，防止投影计算出屏导致的报错
        H, W = img_uint8.shape[1], img_uint8.shape[2]
        x1, y1 = max(0, int(x_min)), max(0, int(y_min))
        x2, y2 = min(W, int(x_max)), min(H, int(y_max))
        
        # 3. 如果有效区域太小或非法，直接返回原图
        if x2 <= x1 or y2 <= y1:
            return image
        
        # 4. 构造 boxes [N, 4] 格式为 [xmin, ymin, xmax, ymax]
        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
        
        # 5. 绘制框：颜色红色，宽度可调
        # 注意：这里返回的也是 uint8 Tensor
        res_uint8 = draw_bounding_boxes(img_uint8, boxes, colors="red", width=3)
        
        # 6. 转回 float32 且范围回到 [0, 1] 并送回原始设备
        return res_uint8.to(image.device).float() / 255.0
    


def render_and_merge(viewpoint_cam, gaussians : GaussianModel_p, pipe, bg : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    all_renders = []
    all_depths  = []
    all_alphas  = []
    all_viewspace_points = []
    all_visibility_filter = []
    all_radii = []
    img_name = viewpoint_cam.image_name
    save_dir = os.path.join("debug", f"{img_name}")
    os.makedirs(save_dir, exist_ok=True)
    visible_indices = []
    # 1. 提取当前相机的视锥平面

    # 设定一个合理的全局可见距离，根据你的场景大小调整（比如 50.0 或 100.0）
    MAX_RENDER_DIST = 70.0
    cam_center = viewpoint_cam.camera_center # 获取相机位置
    proj_matrix = viewpoint_cam.full_proj_transform
    planes = get_frustum_planes(proj_matrix)
    
    view_matrix=viewpoint_cam.world_view_transform
    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    for block_idx in range(len(gaussians.block_masks)):
        mask = gaussians.block_masks[block_idx]
        blk = gaussians.blocks[block_idx]
        
        
        # --- 第一层：距离粗筛 ---
        # # 过滤掉那些在视锥内但离得太远、投影后几乎没像素的块
        # if not is_block_within_dist(blk, cam_center, MAX_RENDER_DIST):
        #     print(f"In {viewpoint_cam.image_name} Block {block_idx} skipped due to distance.")
        #     continue
        
        # 2. 视锥剔除：如果块不在视野内，直接跳过
        if not is_block_visible(blk, planes):
            continue
        

        # --- 第二步：点级精筛 ---
        # 此时只处理那些“部分可见”的块，剔除掉该块中在视野外的冗余点
        original_count = len(mask)
        fine_mask = get_visible_mask_in_block(mask, gaussians.get_xyz, planes)
        culled_count = original_count - len(fine_mask)
        
        # 打印剔除情况
        if culled_count > 0:
            percent = (culled_count / original_count) * 100
            print(f"  [Block {block_idx:2d}] Fine Culling: {original_count:7d} -> {len(fine_mask):7d} points (-{percent:.1f}%)")

        if len(fine_mask) == 0:
            print(f"  [Block {block_idx:2d}] Fully culled by point-level check.")
            continue

        visible_indices.append(block_idx)

        xyz=gaussians.get_xyz[fine_mask]
        x_min, y_min, x_max, y_max = get_block_screen_bbox(xyz, proj_matrix, W, H)


        # 3. 渲染可见块
        gaussians.start_subset(fine_mask)
        out = render(viewpoint_cam, gaussians, pipe, bg,  scaling_modifier=scaling_modifier, separate_sh=separate_sh, override_color=override_color, use_trained_exp=use_trained_exp)
        gaussians.end_subset()
        
        if config.PRINT_EVERYTHING:
            block_img_with_box = print_box_on_image(out["render"], x_min, y_min, x_max, y_max)
            torchvision.utils.save_image(block_img_with_box, os.path.join(save_dir, f"{block_idx}.png"))

        all_renders.append(out["render"].detach())
        all_depths.append(out["depth"].detach())
        all_alphas.append(out["alphaLeft"].detach())
        all_viewspace_points.append(out["viewspace_points"].detach())
        all_visibility_filter.append(out["visibility_filter"].detach())
        all_radii.append(out["radii"].detach())
        
    print(f"Rendered {len(visible_indices)} / {len(gaussians.block_masks)} blocks for view {img_name}")
    
    cpu_merge_result = merge(all_renders, all_depths, all_alphas, all_viewspace_points, all_visibility_filter, all_radii)
    
    rendered_image = cpu_merge_result["final_rgb"]
    screenspace_points = cpu_merge_result["final_viewspace_points"]
    radii = cpu_merge_result["final_radii"]
    depth_image = cpu_merge_result["final_depth"]
    alphaLeft = cpu_merge_result["front_alphas"][-1]

    out = {
    "render": rendered_image,
    "viewspace_points": screenspace_points,
    "visibility_filter" : (radii > 0).nonzero(),
    "radii": radii,
    "depth" : depth_image,
    "alphaLeft": alphaLeft
    }
    
    return out


