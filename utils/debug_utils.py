

import os
import torch
import torchvision

from utils.camera_utils import frustum_culling, draw_box, rebuild_block_bound_aabb_2d, rebuild_pointcloud_aabb_2d, extract_block_layer_contribution


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