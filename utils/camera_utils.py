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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def downsample_points(xyz: torch.Tensor, factor=100):
    return xyz[::factor]

def get_frustum_corners_world(full_proj_transform: torch.Tensor):
    """
    full_proj_transform: [4,4], world -> clip
    return: [8,3] frustum corners in world space
    """
    device = full_proj_transform.device
    inv = torch.inverse(full_proj_transform)

    # NDC space corners
    corners_ndc = torch.tensor([
        [-1, -1, -1, 1],
        [ 1, -1, -1, 1],
        [ 1,  1, -1, 1],
        [-1,  1, -1, 1],
        [-1, -1,  1, 1],
        [ 1, -1,  1, 1],
        [ 1,  1,  1, 1],
        [-1,  1,  1, 1],
    ], dtype=torch.float32, device=device)

    corners_world = (inv @ corners_ndc.T).T
    corners_world = corners_world[:, :3] / corners_world[:, 3:4]

    return corners_world.cpu()




def _compute_corners_from_inv(invM: torch.Tensor, z_mode: str, device=None):
    # NDC corners: x,y in {-1,1}; z in {-1,1} (OpenGL) or {0,1} (D3D/OpenCV-like)
    if z_mode == "neg1_pos1":
        zs = [-1.0, 1.0]
    elif z_mode == "0_1":
        zs = [0.0, 1.0]
    else:
        raise ValueError("z_mode must be 'neg1_pos1' or '0_1'")

    corners_ndc = torch.tensor([
        [-1, -1, zs[0], 1],
        [ 1, -1, zs[0], 1],
        [ 1,  1, zs[0], 1],
        [-1,  1, zs[0], 1],
        [-1, -1, zs[1], 1],
        [ 1, -1, zs[1], 1],
        [ 1,  1, zs[1], 1],
        [-1,  1, zs[1], 1],
    ], dtype=torch.float32, device=invM.device)

    w = (invM @ corners_ndc.T).T
    xyz = w[:, :3] / w[:, 3:4]
    return xyz  # [8,3]


def _score_candidate(corners: torch.Tensor, cam_center: torch.Tensor, xyz_bbox: torch.Tensor):
    # 选“合理”的解：角点必须 finite，且尺度和点云 bbox 同量级，且 near quad 离相机更近
    if not torch.isfinite(corners).all():
        return -1e30

    # bbox 尺度
    min_xyz = xyz_bbox.min(dim=0).values
    max_xyz = xyz_bbox.max(dim=0).values
    bbox_diag = torch.norm(max_xyz - min_xyz) + 1e-9

    # frustum 尺度
    fru_diag = torch.norm(corners.max(dim=0).values - corners.min(dim=0).values) + 1e-9

    # near/far 平均距离（假设前4个是 near，后4个是 far）
    d_near = torch.norm(corners[:4].mean(dim=0) - cam_center)
    d_far  = torch.norm(corners[4:].mean(dim=0) - cam_center)

    # 评分：尺度比接近、且 far > near
    scale_ratio = fru_diag / bbox_diag
    scale_score = -abs(torch.log(scale_ratio))  # 越接近 1 越好
    depth_score = 1.0 if d_far > d_near else -1.0

    # 额外：相机不应离 frustum 极端远
    cam_to_fru = torch.norm(corners.mean(dim=0) - cam_center)
    cam_score = - (cam_to_fru / bbox_diag) * 0.2

    return scale_score + depth_score + cam_score



def visualize_frustum(
    xyz_any: torch.Tensor,                 # [N,3]
    full_proj_transform_any: torch.Tensor, # [4,4]
    camera_center_any: torch.Tensor,       # [3]
    downsample_factor=100,
    save_path="frustum_debug.png"
):
    # -------- 数据准备 --------
    xyz = xyz_any.detach().cpu()[::downsample_factor]
    M   = full_proj_transform_any.detach().cpu()
    cam = camera_center_any.detach().cpu()

    # -------- 自动判断矩阵约定 --------
    candidates = []
    for use_T in [False, True]:
        A = M.T if use_T else M
        invA = torch.inverse(A)
        for z_mode in ["neg1_pos1", "0_1"]:
            corners = _compute_corners_from_inv(invA, z_mode)
            score = _score_candidate(corners, cam, xyz)
            candidates.append((score, use_T, z_mode, corners))

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, use_T, z_mode, corners = candidates[0]

    A = M.T if use_T else M

    # -------- clip-space 判定 inside / outside --------
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    clip = (A @ xyz_h.T).T
    x, y, z, w = clip[:,0], clip[:,1], clip[:,2], clip[:,3]

    if z_mode == "neg1_pos1":
        inside = (x>=-w)&(x<=w)&(y>=-w)&(y<=w)&(z>=-w)&(z<=w)
    else:
        inside = (x>=-w)&(x<=w)&(y>=-w)&(y<=w)&(z>=0)&(z<=w)

    xin  = xyz[inside]
    xout = xyz[~inside]

    # -------- 画图 --------
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 视锥外（非常透明）
    if xout.numel() > 0:
        ax.scatter(
            xout[:,0], xout[:,1], xout[:,2],
            c="gray", s=1, alpha=0.05,
            label="Outside frustum"
        )

    # 视锥内（醒目）
    if xin.numel() > 0:
        ax.scatter(
            xin[:,0], xin[:,1], xin[:,2],
            c="red", s=3, alpha=0.9,
            label="Inside frustum"
        )

    # 相机（大圆）
    ax.scatter(
        cam[0], cam[1], cam[2],
        s=100, c="blue", marker="o",
        edgecolors="k", linewidths=2,
        label="Camera"
    )

    # 视锥线框（椎体）
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i,j in edges:
        ax.plot(
            [corners[i,0], corners[j,0]],
            [corners[i,1], corners[j,1]],
            [corners[i,2], corners[j,2]],
            linewidth=2.5, color="black"
        )

    # 坐标比例
    all_pts = torch.cat([xyz, corners, cam[None]], dim=0)
    mn = all_pts.min(0).values
    mx = all_pts.max(0).values
    ctr = (mn + mx) / 2
    rad = (mx - mn).max() / 2

    ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[OK] saved to {save_path} | use_T={use_T}, z_mode={z_mode}")
    
    

def frustum_culling(
    xyz: torch.Tensor,                  # [N,3]
    full_proj_transform: torch.Tensor,   # [4,4]
    assume_opengl: bool = False    # None = 自动；True = z∈[-1,1]; False = z∈[0,1]
) -> torch.BoolTensor:
    """
    Returns:
        mask: BoolTensor [N], True means inside frustum
    """

    # -------- 安全处理 --------
    xyz_ = xyz.detach()
    M = full_proj_transform.detach()

    device = xyz_.device
    dtype = xyz_.dtype

    # -------- 齐次坐标 --------
    ones = torch.ones((xyz_.shape[0], 1), device=device, dtype=dtype)
    xyz_h = torch.cat([xyz_, ones], dim=1)  # [N,4]

    # -------- 尝试两种矩阵乘法约定 --------
    # 1) clip = M @ x
    clip1 = (M @ xyz_h.T).T
    # 2) clip = M.T @ x
    clip2 = (M.T @ xyz_h.T).T

    def inside_clip(clip, opengl: bool, inflate_ratio=0.3):
        x, y, z, w = clip.unbind(dim=1)

        valid_w = w > 0
        inflate = inflate_ratio * w

        if opengl:
            inside = (
                (x >= -w - inflate) & (x <= w + inflate) &
                (y >= -w - inflate) & (y <= w + inflate) &
                (z >= -w - inflate) & (z <= w + inflate)
            )
        else:
            inside = (
                (x >= -w - inflate) & (x <= w + inflate) &
                (y >= -w - inflate) & (y <= w + inflate) &
                (z >= 0) & (z <= w + inflate)   # ❗只放 far
            )

        return inside & valid_w


    # -------- 自动 / 手动 z 约定 --------
    if assume_opengl is None:
        # 自动：哪个结果“合理”（inside 点更多）就用哪个
        mask1_gl = inside_clip(clip1, True)
        mask1_dx = inside_clip(clip1, False)
        mask2_gl = inside_clip(clip2, True)
        mask2_dx = inside_clip(clip2, False)

        candidates = [
            mask1_gl, mask1_dx,
            mask2_gl, mask2_dx
        ]
        mask = max(candidates, key=lambda m: int(m.sum()))
    else:
        if assume_opengl:
            mask = inside_clip(clip1, True) | inside_clip(clip2, True)
        else:
            mask = inside_clip(clip1, False) | inside_clip(clip2, False)

    return mask


def overlay_block_aabb_edges(img, block_idx, block_bounds, view, alpha=0.35):
    device = img.device
    H, W = img.shape[1:]

    mn, mx = block_bounds[block_idx]

    # corners: [8,3]
    corners = torch.stack([
        torch.stack([mn[0], mn[1], mn[2]]),
        torch.stack([mx[0], mn[1], mn[2]]),
        torch.stack([mx[0], mx[1], mn[2]]),
        torch.stack([mn[0], mx[1], mn[2]]),
        torch.stack([mn[0], mn[1], mx[2]]),
        torch.stack([mx[0], mn[1], mx[2]]),
        torch.stack([mx[0], mx[1], mx[2]]),
        torch.stack([mn[0], mx[1], mx[2]]),
    ], dim=0).to(device=device, dtype=img.dtype)

    ones = torch.ones((8, 1), device=device, dtype=img.dtype)
    corners_h = torch.cat([corners, ones], dim=1)  # [8,4]

    M = view.full_proj_transform.to(device)

    clip_a = (M @ corners_h.T).T      # option A
    clip_b = corners_h @ M            # option B

    def ndc_from_clip(clip):
        w = clip[:, 3:4]
        good = w.abs() > 1e-6
        ndc = torch.zeros_like(clip[:, :3])
        m = good.squeeze(1)
        ndc[m] = clip[m, :3] / w[m]
        return ndc, m

    ndc_a, good_a = ndc_from_clip(clip_a)
    ndc_b, good_b = ndc_from_clip(clip_b)

    score_a = ((ndc_a[:, 0].abs() <= 1) & (ndc_a[:, 1].abs() <= 1) & good_a).sum().item()
    score_b = ((ndc_b[:, 0].abs() <= 1) & (ndc_b[:, 1].abs() <= 1) & good_b).sum().item()

    if score_b >= score_a:
        ndc, good = ndc_b, good_b
    else:
        ndc, good = ndc_a, good_a

    # 只用有效点（避免 w<=0 / inf 把 min/max 搞炸）
    valid = good & torch.isfinite(ndc).all(dim=1)
    ndc = ndc[valid]
    if ndc.shape[0] < 2:
        return img

    pts_2d = torch.empty((ndc.shape[0], 2), device=device, dtype=img.dtype)
    pts_2d[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * W
    pts_2d[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H

    xmin = int(torch.clamp(pts_2d[:, 0].min(), 0, W - 1))
    xmax = int(torch.clamp(pts_2d[:, 0].max(), 0, W - 1))
    ymin = int(torch.clamp(pts_2d[:, 1].min(), 0, H - 1))
    ymax = int(torch.clamp(pts_2d[:, 1].max(), 0, H - 1))

    if xmin >= xmax or ymin >= ymax:
        return img

    overlay = img.clone()

    # 稳定颜色（别用 manual_seed 影响全局随机：用 Generator 更干净）
    g = torch.Generator(device=device)
    g.manual_seed(int(block_idx))
    color = torch.rand((3, 1, 1), device=device, dtype=img.dtype, generator=g)

    overlay[:, ymin:ymax, xmin:xmax] = (1 - alpha) * overlay[:, ymin:ymax, xmin:xmax] + alpha * color
    return overlay
