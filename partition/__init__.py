import math
import torch
import numpy as np
from tqdm import tqdm
import config


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


def generate_octant_blocks_kdtree(xyz: torch.Tensor, inflate_ratio: float = 0.05):
    """
    KD-tree style balanced partitioning into exactly 8 blocks.

    Semantics aligned with generate_octant_blocks:
    - block_indices: index tensors (no change)
    - block_bounds: NEW tensors (no grad), safe for visualization
    """

    device = xyz.device
    dtype  = xyz.dtype
    N = xyz.shape[0]

    all_idx = torch.arange(N, device=device)

    block_bounds = []
    block_indices = []

    def recurse(idx: torch.Tensor, depth: int):
        # 3 levels -> 2^3 = 8 blocks
        if depth == 3:
            pts = xyz[idx]

            # ---- compute AABB (may require grad) ----
            mins = pts.min(dim=0).values
            maxs = pts.max(dim=0).values

            # ---- inflate (pure geometry) ----
            center = (mins + maxs) * 0.5
            extent = (maxs - mins) * 0.5
            extent = extent * (1.0 + inflate_ratio)

            mins = center - extent
            maxs = center + extent

            # ---- IMPORTANT PART (octant-style detach) ----
            # Re-create tensors so they DO NOT require grad
            min_xyz = torch.tensor(
                [mins[0], mins[1], mins[2]],
                device=device,
                dtype=dtype
            )
            max_xyz = torch.tensor(
                [maxs[0], maxs[1], maxs[2]],
                device=device,
                dtype=dtype
            )

            block_bounds.append((min_xyz, max_xyz))
            block_indices.append(idx)
            return

        # ---- KD split axis ----
        axis = depth % 3

        # ---- sort by axis ----
        coords = xyz[idx, axis]
        _, order = torch.sort(coords)
        sorted_idx = idx[order]

        mid = sorted_idx.numel() // 2
        left_idx  = sorted_idx[:mid]
        right_idx = sorted_idx[mid:]

        recurse(left_idx,  depth + 1)
        recurse(right_idx, depth + 1)

    recurse(all_idx, depth=0)

    # ---- debug stats ----
    for i, idx in enumerate(block_indices):
        print(f"Block {i:2d}: {idx.numel():7d} points")

    return block_bounds, block_indices




def generate_space_kdtree_blocks(xyz: torch.Tensor, inflate_ratio: float = 0.05):
    device = xyz.device
    dtype  = xyz.dtype
    N = xyz.shape[0]

    num_blocks = config.NUM_BLOCKS
    assert num_blocks > 0 and (num_blocks & (num_blocks - 1)) == 0, \
        f"NUM_BLOCKS must be a power of 2, got {num_blocks}"
    max_depth = int(math.log2(num_blocks))

    all_idx = torch.arange(N, device=device)

    # ---------- 1) 根 AABB（可 inflate 防漏） ----------
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    center = (mins + maxs) * 0.5
    extent = (maxs - mins) * 0.5
    extent = extent * (1.0 + inflate_ratio)
    root_min = center - extent
    root_max = center + extent

    block_bounds  = []
    block_indices = []

    # ---------- 2) KD-tree recursion: max_depth levels => num_blocks leaves ----------
    def recurse(idx: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor, depth: int):
        if depth == max_depth:
            # 叶子：bounds 必须是 “由 split 平面传下来的” bmin/bmax，保证不重叠
            # 为了和你原 octant 一样“不会进 autograd / 可视化安全”，这里用 torch.tensor 重建
            min_xyz = torch.tensor([bmin[0], bmin[1], bmin[2]], device=device, dtype=dtype)
            max_xyz = torch.tensor([bmax[0], bmax[1], bmax[2]], device=device, dtype=dtype)
            block_bounds.append((min_xyz, max_xyz))
            block_indices.append(idx)
            return

        axis = depth % 3

        # 按该轴排序，按点数对半分（严格均分靠这里）
        coords = xyz[idx, axis]
        _, order = torch.sort(coords)
        sorted_idx = idx[order]
        mid = sorted_idx.numel() // 2

        left_idx  = sorted_idx[:mid]
        right_idx = sorted_idx[mid:]

        # split plane 取“切分值”
        # 用排序后的中位坐标作为分割面（左: < split, 右: >= split 的空间意义）
        split_val = xyz[sorted_idx[mid], axis] if sorted_idx.numel() > 0 else (bmin[axis] + bmax[axis]) * 0.5

        # 构造非重叠 bounds（核心：由 split 平面更新父 bounds）
        left_min = bmin
        left_max = bmax.clone()
        left_max[axis] = split_val

        right_min = bmin.clone()
        right_min[axis] = split_val
        right_max = bmax

        recurse(left_idx,  left_min,  left_max,  depth + 1)
        recurse(right_idx, right_min, right_max, depth + 1)

    recurse(all_idx, root_min, root_max, depth=0)

    # ---------- 3) 打印统计 ----------
    for i, idx in enumerate(block_indices):
        print(f"Block {i:2d}: {idx.numel():7d} points")

    return block_bounds, block_indices




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

