

import os
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes




def draw_box(image, x_min, y_min, x_max, y_max, colors, width=2):
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
        res_uint8 = draw_bounding_boxes(img_uint8, boxes, colors=colors, width=width)
        
        # 6. 转回 float32 且范围回到 [0, 1] 并送回原始设备
        return res_uint8.to(image.device).float() / 255.0
    
def aabb_8_corners(bmin: torch.Tensor, bmax: torch.Tensor):
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax

    corners = torch.stack([
        torch.tensor([x0, y0, z0], device=bmin.device, dtype=bmin.dtype),
        torch.tensor([x1, y0, z0], device=bmin.device, dtype=bmin.dtype),
        torch.tensor([x0, y1, z0], device=bmin.device, dtype=bmin.dtype),
        torch.tensor([x1, y1, z0], device=bmin.device, dtype=bmin.dtype),
        torch.tensor([x0, y0, z1], device=bmin.device, dtype=bmin.dtype),
        torch.tensor([x1, y0, z1], device=bmin.device, dtype=bmin.dtype),
        torch.tensor([x0, y1, z1], device=bmin.device, dtype=bmin.dtype),
        torch.tensor([x1, y1, z1], device=bmin.device, dtype=bmin.dtype),
    ], dim=0)

    return corners

def rebuild_block_bound_aabb_2d(rendered, view, bmin, bmax, W, H):
    """
    rendered: [3, H, W] 渲染结果（只用于 device / dtype）
    view: camera view (contains full_proj_transform)
    bmin, bmax: block AABB in world space
    W, H: image width / height
    """
    device = rendered.device
    dtype = rendered.dtype

    # 1. AABB 的 8 个角点（world space）
    corners = aabb_8_corners(bmin, bmax).to(device=device, dtype=dtype)  # [8,3]

    # 2. 齐次坐标（行向量语义，和点云版本一致）
    corners_h = torch.cat(
        [corners, torch.ones((8, 1), device=device, dtype=dtype)],
        dim=1
    )  # [8,4]

    # 3. 投影（关键：p @ M）
    full_proj_transform = view.full_proj_transform.to(device=device, dtype=dtype)
    clip = corners_h @ full_proj_transform  # [8,4]

    # 4. 只保留在相机前方的角点（和点云一致）
    w = clip[:, 3]
    mask = w > 0.05

    if not mask.any():
        # block 完全在相机后 / 不可见
        return 0, 0, 0, 0

    valid_clip = clip[mask]
    valid_w = w[mask]

    # 5. 透视除法 → NDC（只用 x,y，和点云一致）
    ndc = (valid_clip[:, :2] / valid_w.unsqueeze(1)).reshape(-1, 2)

    # 6. NDC → 像素坐标（和点云版本完全一致）
    screen_x = (ndc[:, 0] + 1.0) * W / 2.0
    screen_y = (ndc[:, 1] + 1.0) * H / 2.0

    # 7. screen-space AABB（与点云版本一致的 clamp 逻辑）
    x_min = max(0, int(screen_x.min().item()))
    y_min = max(0, int(screen_y.min().item()))
    x_max = min(W, int(screen_x.max().item()))
    y_max = min(H, int(screen_y.max().item()))

    return x_min, y_min, x_max, y_max


def rebuild_pointcloud_aabb_2d(xyz, full_proj_transform, W, H):
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


def extract_block_layer_contribution(
    block_rank,          # [K, H, W]
    front_rgbs,          # [K, 3, H, W]
    prefix_T,            # [K, 1, H, W]
    visible_block_idxs,  # list of block_id, length K
):
    """
    Returns:
        contrib[layer][block_id] -> RGB tensor [3, H, W]
    """
    K, H, W = block_rank.shape
    device = front_rgbs.device
    dtype = front_rgbs.dtype

    # 1. 真实的 layer-wise RGB 贡献
    layer_rgb = prefix_T * front_rgbs        # [K, 3, H, W]

    # 2. 初始化结果容器
    contrib = {
        layer: {
            block_id: torch.zeros((3, H, W), device=device, dtype=dtype)
            for block_id in visible_block_idxs
        }
        for layer in range(K)
    }

    # 3. 按 layer + block 拆分
    for layer in range(K):
        for block_pos, block_id in enumerate(visible_block_idxs):
            mask = (block_rank[block_pos] == layer)   # [H, W]
            if mask.any():
                contrib[layer][block_id][:, mask] = layer_rgb[layer][:, mask]

    return contrib


def save_rgb_layers(img_path_in_debug, front_rgbs):
    rgb_layers_path = os.path.join(img_path_in_debug, "rgb_layers")
    os.makedirs(rgb_layers_path, exist_ok=True)
    for idx in range(front_rgbs.shape[0]):
        layer_img = front_rgbs[idx]
        torchvision.utils.save_image(layer_img, os.path.join(rgb_layers_path, f"layer_{idx}.png"))
        
def save_layer_contribution(img_path_in_debug, block_rank, front_rgbs, prefix_T, visible_block_idxs):
    # 看看每个block在每个图层贡献了什么
    layer_contri_path = os.path.join(img_path_in_debug, "layer_contribution")
    os.makedirs(layer_contri_path, exist_ok=True)
    contribution = extract_block_layer_contribution(block_rank, front_rgbs, prefix_T, visible_block_idxs)

    for layer, blocks in contribution.items():
        for block_id, rgb in blocks.items():
            if rgb.abs().sum() == 0:
                continue
            save_path = os.path.join( layer_contri_path, f"layer_{layer}_block_{block_id}_contribution.png" )
            torchvision.utils.save_image(rgb.clamp(0, 1), save_path)

def save_depth_list(img_path_in_debug, depth_list, visible_block_idxs):
    depth_list_path = os.path.join(img_path_in_debug, "depth_list")
    os.makedirs(depth_list_path, exist_ok=True)
    for idx, block_id in enumerate(visible_block_idxs):
        layer_depth = depth_list[idx]   # [1, H, W]
        depth = layer_depth.squeeze(0)       # [H, W]
        valid = torch.isfinite(depth)
        if not valid.any():
            continue
        d_min = depth[valid].min()
        d_max = depth[valid].max()
        if (d_max - d_min) < 1e-6:
            depth_norm = torch.zeros_like(depth)
        else:
            depth_norm = (depth - d_min) / (d_max - d_min)
        torchvision.utils.save_image(depth_norm.unsqueeze(0), os.path.join( depth_list_path, f"block_{block_id}_depth_map.png" ))
        
def save_block_img(img_path_in_debug, rendered_list, visible_block_idxs, gaussians, view, image, config):
    block_img_path = os.path.join(img_path_in_debug, "block_images")
    os.makedirs(block_img_path, exist_ok=True)
    
    for block_img, block_id in zip(rendered_list, visible_block_idxs):
        
        # bmin = (xmin, ymin, zmin); bmax = (xmax, ymax, zmax)
        # 因为一个轴对齐包围盒（AABB）在 3D 空间里， 只需要两个点：最小角 bmin 和最大角 bmax， 这两个点就唯一确定了一个长方体，而这个长方体天然有 8 个角点。
        bmin, bmax = gaussians.block_bounds[block_id]
        W, H = block_img.shape[2], block_img.shape[1]
        proj_matrix = view.full_proj_transform
        
        if config.DRAW_BOX:
            x_min, y_min, x_max, y_max = rebuild_block_bound_aabb_2d(block_img, view, bmin, bmax, W, H)
            block_img = draw_box(block_img, x_min, y_min, x_max, y_max, colors="red", width=4)
            x_min, y_min, x_max, y_max = rebuild_pointcloud_aabb_2d(gaussians._xyz[gaussians.block_indices[block_id]], proj_matrix, W, H)
            block_img = draw_box(block_img, x_min, y_min, x_max, y_max, colors="green")
            
        torchvision.utils.save_image(block_img, os.path.join(block_img_path, f"_block_{block_id}.png"))  
    torchvision.utils.save_image(image, os.path.join(block_img_path, view.image_name + ".png"))    

def save_iteration_render(img_path_in_debug, image, iter):
    render_path = os.path.join(img_path_in_debug, "iteration_render")
    os.makedirs(render_path, exist_ok=True)
    torchvision.utils.save_image(image, os.path.join(render_path, f"{iter}.png"))