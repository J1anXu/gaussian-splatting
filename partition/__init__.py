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
    sorted_idx = block.indices[pts[:, axis].argsort()]  # numpy index
    
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



# ============================================
#               主 KD-tree 构建
# ============================================
def generate_block_masks(xyz, max_size=100000):
    """
    xyz: torch [N,3] (GPU or CPU 都行)
    """
    N = xyz.shape[0]

    mins = xyz.min(dim=0).values   # torch [3]
    maxs = xyz.max(dim=0).values   # torch [3]

    blocks = [Block(np.arange(N), mins, maxs)]
    expected = max(1, N // max_size)

    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if len(blk.indices) > max_size:
            left, right = split_block(blk, xyz, max_size)
            blocks.pop(i)
            blocks.append(left)
            blocks.append(right)
        else:
            i += 1


    block_masks = [blk.indices for blk in blocks]
    for i, mask in enumerate(block_masks):
        print(f"  Block {i:3d}: {len(mask):7d} points")

    return block_masks, blocks
