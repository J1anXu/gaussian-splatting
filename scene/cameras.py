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

import threading
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image_path, invdepthmap_path,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        # Added by jian
        self.image_path = image_path
        self.invdepthmap_path = invdepthmap_path
        self.resolution = resolution  # (w,h)
        self._prefetch_thread = None
        self._prefetched_image = None
        self._prefetch_lock = threading.Lock()
        
        
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # resized_image_rgb = PILtoTorch(image, resolution)
        # gt_image = resized_image_rgb[:3, ...]
        # self.alpha_mask = None
        # if resized_image_rgb.shape[0] == 4:
        #     self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        # else: 
        #     self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        # self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        # self.image_width = self.original_image.shape[2]
        # self.image_height = self.original_image.shape[1]
        self.image_width  = resolution[0]
        self.image_height = resolution[1]

        self.invdepthmap = None
        self.depth_reliable = False
        # if invdepthmap is not None:
        #     self.depth_mask = torch.ones_like(self.alpha_mask)
        #     self.invdepthmap = cv2.resize(invdepthmap, resolution)
        #     self.invdepthmap[self.invdepthmap < 0] = 0
        #     self.depth_reliable = True

        #     if depth_params is not None:
        #         if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
        #             self.depth_reliable = False
        #             self.depth_mask *= 0
                
        #         if depth_params["scale"] > 0:
        #             self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

        #     if self.invdepthmap.ndim != 2:
        #         self.invdepthmap = self.invdepthmap[..., 0]
        #     self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def load_original_image(self, device="cuda"):
        """
        Return GT image tensor in [0,1], shape [3,H,W], resized to self.resolution.
        Does NOT cache -> disk is the source of truth.
        """
        W, H = self.resolution  # 你这里 resolution 是 (w,h)

        with Image.open(self.image_path) as img:
            # 原版 3DGS 常用 RGB；如果有 alpha，也先读出来
            img = img.convert("RGBA")
            if (img.size[0], img.size[1]) != (W, H):
                img = img.resize((W, H), resample=Image.BILINEAR)

            arr = np.array(img).astype(np.float32) / 255.0  # [H,W,4]
            rgb = arr[..., :3]                               # [H,W,3]
            alpha = arr[..., 3:4]                            # [H,W,1]

            # 如果你希望像原版那样把 alpha 合成背景：
            # 这里用黑背景（0）做合成；如果你要白背景，把 bg 改成 1
            bg = 0.0
            rgb = rgb * alpha + bg * (1.0 - alpha)

        gt = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # [3,H,W]
        return gt.to(device, non_blocking=True)

    def load_alpha_mask(self, device="cuda"):
        """
        If image has alpha channel, return alpha mask tensor [1,H,W] on device.
        Otherwise return None.
        """
        W, H = self.resolution
        with Image.open(self.image_path) as img:
            if img.mode != "RGBA":
                # 没 alpha
                return None
            if (img.size[0], img.size[1]) != (W, H):
                img = img.resize((W, H), resample=Image.BILINEAR)
            alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0  # [H,W]

        m = torch.from_numpy(alpha)[None, ...].contiguous()  # [1,H,W]
        return m.to(device, non_blocking=True)

    def load_invdepth(self):
        """
        Load invdepthmap on CPU float32, scaled properly.
        No cache.
        """
        if not self.depth_path:
            return None
        depth = cv2.imread(self.depth_path, -1)
        if depth is None:
            raise FileNotFoundError(f"Depth file not found or unreadable: {self.depth_path}")
        depth = depth.astype(np.float32)
        depth = depth / (512.0 if self.is_nerf_synthetic else float(2**16))
        return depth  # CPU numpy
    
    def load_image_cpu(self):
        """
        Load and preprocess GT image on CPU only.
        Safe to be called in background threads.
        """
        image = Image.open(self.image_path).convert("RGB")

        # Resize + to torch tensor (CPU)
        resized_image_rgb = PILtoTorch(image, self.resolution)[:3, ...]
        gt_image = resized_image_rgb.clamp(0.0, 1.0)

        return gt_image
    
    def load_image(self, device="cuda"):
        # 1. 拿到 PIL.Image（来自 prefetch 或磁盘）
        if self._prefetched_image is not None:
            img = self._prefetched_image
            self._prefetched_image = None
        elif self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
            img = self._prefetched_image
            self._prefetched_image = None
        else:
            with Image.open(self.image_path) as im:
                img = im.copy()

        # 2. PIL → Tensor
        if img.mode != "RGB":
            img = img.convert("RGB")

        W, H = self.resolution
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)

        arr = np.asarray(img).astype(np.float32) / 255.0   # [H,W,3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        return tensor.to(device, non_blocking=True)


    def prefetch_image(self):
        # 已经有数据 or 正在加载 → 不重复开线程
        if self._prefetched_image is not None or self._prefetch_thread is not None:
            return

        def _load():
            try:
                with Image.open(self.image_path) as im:
                    img = im.copy()
                with self._prefetch_lock:
                    self._prefetched_image = img   # ✅ 真正写入
            finally:
                self._prefetch_thread = None

        t = threading.Thread(target=_load, daemon=True)
        self._prefetch_thread = t
        t.start()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]



