import torch
import numpy as np
from tqdm import tqdm


# ============================================
#               KD-tree Block Node
# ============================================
class Block:
    def __init__(self, indices, mins, maxs):
        self.indices = indices                  # numpy array of int
        self.mins    = mins.clone()             # torch [3]
        self.maxs    = maxs.clone()             # torch [3]


# ============================================
#                Split Block
# ============================================
def split_block(block, xyz, max_size):
    pts = xyz[block.indices]  # [M,3] torch

    # 1. 选择最长维度（可以保持）
    extent = block.maxs - block.mins
    axis = torch.argmax(extent).item()

    # 2. 按 axis 排序
    order = torch.argsort(pts[:, axis]).detach().cpu().numpy()
    sorted_idx = block.indices[order]
    
    # 3. 数量对半切（balanced）
    mid = len(sorted_idx) // 2
    left_idx  = sorted_idx[:mid]
    right_idx = sorted_idx[mid:]

    # 4. 重新计算 bounding box（基于真实点）
    left_xyz  = xyz[left_idx]
    right_xyz = xyz[right_idx]

    left_block = Block(
        left_idx,
        left_xyz.min(dim=0).values,
        left_xyz.max(dim=0).values
    )
    right_block = Block(
        right_idx,
        right_xyz.min(dim=0).values,
        right_xyz.max(dim=0).values
    )

    return left_block, right_block




def generate_octant_blocks( xyz: torch.Tensor, inflate_ratio: float = 0.05 ):
    device = xyz.device
    dtype  = xyz.dtype
    N = xyz.shape[0]

    # ---------- 1. 计算整体 AABB ----------
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values

    center = (mins + maxs) * 0.5
    extent = (maxs - mins) * 0.5

    # ---------- 2. inflate 一点（防止边界点被漏掉） ----------
    extent = extent * (1.0 + inflate_ratio)

    mins = center - extent
    maxs = center + extent

    # ---------- 3. 8 个子块的边界 ----------
    # 每个维度：low / high
    x_lo, x_hi = mins[0], maxs[0]
    y_lo, y_hi = mins[1], maxs[1]
    z_lo, z_hi = mins[2], maxs[2]

    cx, cy, cz = center

    # (xmin, xmax, ymin, ymax, zmin, zmax)
    octant_bounds = [
        # lower z
        (x_lo, cx, y_lo, cy, z_lo, cz),
        (cx, x_hi, y_lo, cy, z_lo, cz),
        (x_lo, cx, cy, y_hi, z_lo, cz),
        (cx, x_hi, cy, y_hi, z_lo, cz),
        # upper z
        (x_lo, cx, y_lo, cy, cz, z_hi),
        (cx, x_hi, y_lo, cy, cz, z_hi),
        (x_lo, cx, cy, y_hi, cz, z_hi),
        (cx, x_hi, cy, y_hi, cz, z_hi),
    ]

    block_bounds = []
    block_indices = []

    all_idx = torch.arange(N, device=device)

    # ---------- 4. 给每个子块分配点 ----------
    for i, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(octant_bounds):
        mask = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] < xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] < ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] < zmax)
        )

        idx = all_idx[mask]

        block_bounds.append((
            torch.tensor([xmin, ymin, zmin], device=device, dtype=dtype),
            torch.tensor([xmax, ymax, zmax], device=device, dtype=dtype),
        ))
        block_indices.append(idx)

        print(f"Block {i:2d}: {idx.numel():7d} points")

    return block_bounds, block_indices

