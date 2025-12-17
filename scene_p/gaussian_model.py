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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from partition import generate_block_masks
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel_p:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default", max_block_size = 300_000):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        
        # -------- CPU master parameters --------
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        # -------- optimizer buffers (CPU) --------
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        # ----- subset mode -----
        self.subset_mode = False
        self.subset_indices = None

        # GPU subset buffers
        self._xyz_gpu = None
        self._opacity_gpu = None
        self._scaling_gpu = None
        self._rotation_gpu = None
        self._features_dc_gpu = None
        self._features_rest_gpu = None

        # 一个block最多包含点数,超过这个数量就要重新分
        self.max_block_size = max_block_size
        self.block_masks = None
        self.blocks = None
        self.just_densified = False

        # 标记 optimier state 是否已经为 partition 训练初始化
        self.partition_training_initialized = False

        self.setup_functions()
        
    def partition(self):
        # 刚刚增加过点数 & 总数超过限制
        if self.just_densified and self._xyz.shape[0]>=self.max_block_size:
            block_masks , blocks = generate_block_masks(self._xyz, max_size = self.max_block_size)
            block_masks = [m for m in block_masks if len(m) > 0]
            self.blocks = blocks
            self.block_masks = block_masks
        else:
            self.block_masks = [torch.arange(self._xyz.shape[0])]
        self.just_densified = False
            

    def partition_for_rendering(self):
        block_masks , blocks = generate_block_masks(self._xyz, max_size = self.max_block_size)
        block_masks = [m for m in block_masks if len(m) > 0]
        self.blocks = blocks
        self.block_masks = block_masks

        
        
    def _to_cpu_index(self, idx):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long)
        return idx.to("cpu")

        
    # =============================
    #  Subset control API
    # =============================
    def start_subset(self, subset_indices, requires_grad=False):
        self.subset_mode = True
        idx = self._to_cpu_index(subset_indices)
        self.subset_indices = idx

        def send_subset_to_gpu(tensor):
            subset = tensor[idx].cuda(non_blocking=True)
            subset = subset.clone().detach()
            if requires_grad:
                subset.requires_grad_(True)
            return subset
        
        # --------------- 参数子集（带梯度） ---------------
        self._xyz_gpu           = send_subset_to_gpu(self._xyz)
        self._opacity_gpu       = send_subset_to_gpu(self._opacity)
        self._scaling_gpu       = send_subset_to_gpu(self._scaling)
        self._rotation_gpu      = send_subset_to_gpu(self._rotation)
        self._features_dc_gpu   = send_subset_to_gpu(self._features_dc)
        self._features_rest_gpu = send_subset_to_gpu(self._features_rest)

        # --------------- 对应的 Adam 状态子集（不需要 grad） ---------------
        if self.partition_training_initialized:
            self._m_xyz_gpu,      self._v_xyz_gpu      = send_subset_to_gpu(self.m_xyz),      send_subset_to_gpu(self.v_xyz)
            self._m_f_dc_gpu,     self._v_f_dc_gpu     = send_subset_to_gpu(self.m_f_dc),     send_subset_to_gpu(self.v_f_dc)
            self._m_f_rest_gpu,   self._v_f_rest_gpu   = send_subset_to_gpu(self.m_f_rest),   send_subset_to_gpu(self.v_f_rest)
            self._m_opacity_gpu,  self._v_opacity_gpu  = send_subset_to_gpu(self.m_opacity),  send_subset_to_gpu(self.v_opacity)
            self._m_scaling_gpu,  self._v_scaling_gpu  = send_subset_to_gpu(self.m_scaling),  send_subset_to_gpu(self.v_scaling)
            self._m_rotation_gpu, self._v_rotation_gpu = send_subset_to_gpu(self.m_rotation), send_subset_to_gpu(self.v_rotation)


    # 注意不需要拷贝回去，直接清空就好了，因为梯度已经在训练过程中通过 copy_grad_to_cpu 写回 CPU master 参数了
    def end_subset(self):
        self.subset_mode = False
        self.subset_indices = None

        # 删除 GPU 子集参数以释放显存
        del self._xyz_gpu
        del self._opacity_gpu
        del self._scaling_gpu
        del self._rotation_gpu
        del self._features_dc_gpu
        del self._features_rest_gpu
        
        self._xyz_gpu = None
        self._opacity_gpu = None
        self._scaling_gpu = None
        self._rotation_gpu = None
        self._features_dc_gpu = None
        self._features_rest_gpu = None
        
        if self.partition_training_initialized:
            del self._m_xyz_gpu
            del self._v_xyz_gpu
            del self._m_f_dc_gpu
            del self._v_f_dc_gpu
            del self._m_f_rest_gpu
            del self._v_f_rest_gpu
            del self._m_opacity_gpu
            del self._v_opacity_gpu
            del self._m_scaling_gpu
            del self._v_scaling_gpu
            del self._m_rotation_gpu
            del self._v_rotation_gpu
        
            self._m_xyz_gpu = None
            self._v_xyz_gpu = None
            self._m_f_dc_gpu = None
            self._v_f_dc_gpu = None
            self._m_f_rest_gpu = None
            self._v_f_rest_gpu = None
            self._m_opacity_gpu = None
            self._v_opacity_gpu = None
            self._m_scaling_gpu = None
            self._v_scaling_gpu = None
            self._m_rotation_gpu = None
            self._v_rotation_gpu = None
            

        
        # 强制释放 GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    def zero_grad_subset(self):
        for t in [
            getattr(self, "_xyz_gpu", None),
            getattr(self, "_opacity_gpu", None),
            getattr(self, "_scaling_gpu", None),
            getattr(self, "_rotation_gpu", None),
            getattr(self, "_features_dc_gpu", None),
            getattr(self, "_features_rest_gpu", None),
        ]:
            if t is not None and t.grad is not None:
                t.grad.zero_()  
  
    def adam_step_subset(self):
        if not hasattr(self, "subset_indices"):
            return

        idx_cpu = self.subset_indices        # CPU long tensor
        
        grads = [
            ("xyz", self._xyz_gpu),
            ("opacity", self._opacity_gpu),
            ("scaling", self._scaling_gpu),
            ("rotation", self._rotation_gpu),
            ("f_dc", self._features_dc_gpu),
            ("f_rest", self._features_rest_gpu),
        ]

        for name, tensor in grads:
            if tensor is not None and tensor.grad is None:
                # 直接创建一个0梯度（和 copy_grad_to_cpu 完全一致的策略）
                tensor.grad = torch.zeros_like(tensor)


        def adam_update_block(param_gpu, grad_gpu, m_gpu, v_gpu, lr):
            b1, b2, eps = self.beta1, self.beta2, self.eps
            step = self.adam_step

            # ---- m / v 更新（无 autograd） ----
            with torch.no_grad():
                m_gpu.mul_(b1).add_(grad_gpu, alpha=1 - b1)
                v_gpu.mul_(b2).addcmul_(grad_gpu, grad_gpu, value=1 - b2)

                m_hat = m_gpu / (1 - b1 ** step)
                v_hat = v_gpu / (1 - b2 ** step)

                # ---- 参数更新（必须禁用 autograd） ----
                param_gpu.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)



        # ======================================================
        # xyz
        # ======================================================
        if hasattr(self, "_xyz_gpu") and self._xyz_gpu.grad is not None:
            adam_update_block(
                self._xyz_gpu,
                self._xyz_gpu.grad,
                self._m_xyz_gpu,
                self._v_xyz_gpu,
                lr=self.adam_lrs["xyz"],
            )
            with torch.no_grad():
                self._xyz[idx_cpu] = self._xyz_gpu.detach().to(self._xyz.device)
                self.m_xyz[idx_cpu] = self._m_xyz_gpu.detach().to(self.m_xyz.device)
                self.v_xyz[idx_cpu] = self._v_xyz_gpu.detach().to(self.v_xyz.device)

        # ======================================================
        # features_dc
        # ======================================================
        if hasattr(self, "_features_dc_gpu") and self._features_dc_gpu.grad is not None:
            adam_update_block(
                self._features_dc_gpu,
                self._features_dc_gpu.grad,
                self._m_f_dc_gpu,
                self._v_f_dc_gpu,
                lr=self.adam_lrs["f_dc"],
            )
            with torch.no_grad():
                self._features_dc[idx_cpu] = self._features_dc_gpu.detach().to(self._features_dc.device)
                self.m_f_dc[idx_cpu] = self._m_f_dc_gpu.detach().to(self.m_f_dc.device)
                self.v_f_dc[idx_cpu] = self._v_f_dc_gpu.detach().to(self.v_f_dc.device)

        # ======================================================
        # features_rest
        # ======================================================
        if hasattr(self, "_features_rest_gpu") and self._features_rest_gpu.grad is not None:
            adam_update_block(
                self._features_rest_gpu,
                self._features_rest_gpu.grad,
                self._m_f_rest_gpu,
                self._v_f_rest_gpu,
                lr=self.adam_lrs["f_rest"],
            )
            with torch.no_grad():
                self._features_rest[idx_cpu] = self._features_rest_gpu.detach().to(self._features_rest.device)
                self.m_f_rest[idx_cpu] = self._m_f_rest_gpu.detach().to(self.m_f_rest.device)
                self.v_f_rest[idx_cpu] = self._v_f_rest_gpu.detach().to(self.v_f_rest.device)

        # ======================================================
        # opacity
        # ======================================================
        if hasattr(self, "_opacity_gpu") and self._opacity_gpu.grad is not None:
            adam_update_block(
                self._opacity_gpu,
                self._opacity_gpu.grad,
                self._m_opacity_gpu,
                self._v_opacity_gpu,
                lr=self.adam_lrs["opacity"],
            )
            with torch.no_grad():
                self._opacity[idx_cpu] = self._opacity_gpu.detach().to(self._opacity.device)
                self.m_opacity[idx_cpu] = self._m_opacity_gpu.detach().to(self.m_opacity.device)
                self.v_opacity[idx_cpu] = self._v_opacity_gpu.detach().to(self.v_opacity.device)

        # ======================================================
        # scaling
        # ======================================================
        if hasattr(self, "_scaling_gpu") and self._scaling_gpu.grad is not None:
            adam_update_block(
                self._scaling_gpu,
                self._scaling_gpu.grad,
                self._m_scaling_gpu,
                self._v_scaling_gpu,
                lr=self.adam_lrs["scaling"],
            )
            with torch.no_grad():
                self._scaling[idx_cpu] = self._scaling_gpu.detach().to(self._scaling.device)
                self.m_scaling[idx_cpu] = self._m_scaling_gpu.detach().to(self.m_scaling.device)
                self.v_scaling[idx_cpu] = self._v_scaling_gpu.detach().to(self.v_scaling.device)

        # ======================================================
        # rotation
        # ======================================================
        if hasattr(self, "_rotation_gpu") and self._rotation_gpu.grad is not None:
            adam_update_block(
                self._rotation_gpu,
                self._rotation_gpu.grad,
                self._m_rotation_gpu,
                self._v_rotation_gpu,
                lr=self.adam_lrs["rotation"],
            )
            with torch.no_grad():
                self._rotation[idx_cpu] = self._rotation_gpu.detach().to(self._rotation.device)
                self.m_rotation[idx_cpu] = self._m_rotation_gpu.detach().to(self.m_rotation.device)
                self.v_rotation[idx_cpu] = self._v_rotation_gpu.detach().to(self.v_rotation.device)
  
  
    def copy_grad_to_cpu(self, mask):
        """
        将 GPU subset 的梯度写回 CPU master 参数。
        mask: 当前block的CPU索引 (Tensor[long])
        """

        # 确保 CPU master 参数有 grad buffer
        if self._xyz.grad is None:
            self._xyz.grad = torch.zeros_like(self._xyz)
        if self._opacity.grad is None:
            self._opacity.grad = torch.zeros_like(self._opacity)
        if self._scaling.grad is None:
            self._scaling.grad = torch.zeros_like(self._scaling)
        if self._rotation.grad is None:
            self._rotation.grad = torch.zeros_like(self._rotation)
        if self._features_dc.grad is None:
            self._features_dc.grad = torch.zeros_like(self._features_dc)
        if self._features_rest.grad is None:
            self._features_rest.grad = torch.zeros_like(self._features_rest)

        # ------- 将 GPU block 的梯度拷回 CPU master 参数 -------
        self._xyz.grad[mask] = self._xyz_gpu.grad.detach().cpu()
        self._opacity.grad[mask] = self._opacity_gpu.grad.detach().cpu()
        self._scaling.grad[mask] = self._scaling_gpu.grad.detach().cpu()
        self._rotation.grad[mask] = self._rotation_gpu.grad.detach().cpu()
        self._features_dc.grad[mask] = self._features_dc_gpu.grad.detach().cpu()
        self._features_rest.grad[mask] = self._features_rest_gpu.grad.detach().cpu()
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup_for_part(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)



    @property
    def get_scaling(self):
        if self.subset_mode:
            return self.scaling_activation(self._scaling_gpu)
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self.subset_mode:
            return self.rotation_activation(self._rotation_gpu)
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self.subset_mode:
            return self._xyz_gpu
        return self._xyz
    
    @property
    def get_features(self):
        if self.subset_mode:
            features_dc = self._features_dc_gpu
            features_rest = self._features_rest_gpu
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        if self.subset_mode:
            return self._features_dc_gpu
        return self._features_dc

    
    @property
    def get_features_rest(self):
        if self.subset_mode:
            return self._features_rest_gpu
        return self._features_rest
    
    @property
    def get_opacity(self):
        if self.subset_mode:
            return self.opacity_activation(self._opacity_gpu)
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        if self.subset_mode:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation_gpu)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        # self.pretrained_exposures = None
        # exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        # self._exposure = nn.Parameter(exposure.requires_grad_(True))
        
        device = torch.device("cpu")
        self._xyz = fused_point_cloud.to(device).clone().requires_grad_(False)
        self._features_dc = torch.tensor(features[:, :, 0:1], dtype=torch.float, device=device).transpose(1, 2).contiguous().clone().requires_grad_(False)
        self._features_rest = torch.tensor(features[:, :, 1:], dtype=torch.float, device=device).transpose(1, 2).contiguous().clone().requires_grad_(False)
        self._scaling = torch.tensor(scales, dtype=torch.float, device=device).clone().requires_grad_(False)
        self._rotation = torch.tensor(rots, dtype=torch.float, device=device).clone().requires_grad_(False)
        self._opacity = torch.tensor(opacities, dtype=torch.float, device=device).clone().requires_grad_(False)
        self.max_radii2D = torch.zeros(self._xyz.shape[0], device=device)
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        self._exposure = torch.eye(3, 4, device=device)[None].repeat(len(cam_infos), 1, 1).clone().requires_grad_(False)


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def training_setup_for_part(self, training_args):
        self.percent_dense = training_args.percent_dense
        # 每个高斯的累积梯度 用于判断这个地方是否重要 重要的话需要分裂更多的高斯
        # 注意这个地方扩增以后需要清空
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1)) 
        # 每个 Gaussian 在 densification 统计中，被“看到并产生梯度”的次数（计数器）
        self.denom = torch.zeros((self.get_xyz.shape[0], 1))

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        
        # ============================================================
        #     Block-wise Adam State Initialization (CPU master)
        # ============================================================

        device_cpu = torch.device("cpu")

        # 1. 获取每个参数的学习率
        xyz_lr      = training_args.position_lr_init * self.spatial_lr_scale
        f_dc_lr     = training_args.feature_lr
        f_rest_lr   = training_args.feature_lr / 20.0
        opacity_lr  = training_args.opacity_lr
        scaling_lr  = training_args.scaling_lr
        rotation_lr = training_args.rotation_lr

        self.adam_lrs = {
            "xyz": xyz_lr,
            "f_dc": f_dc_lr,
            "f_rest": f_rest_lr,
            "opacity": opacity_lr,
            "scaling": scaling_lr,
            "rotation": rotation_lr
        }

        # 2. Adam 超参
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps    = 1e-8
        self.adam_step = 1

        # 3. CPU master m/v 为所有可学习参数创建
        def make_m_v_like(p):
            return torch.zeros_like(p, device=device_cpu), torch.zeros_like(p, device=device_cpu)

        self.m_xyz,      self.v_xyz      = make_m_v_like(self._xyz.cpu())
        self.m_f_dc,     self.v_f_dc     = make_m_v_like(self._features_dc.cpu())
        self.m_f_rest,   self.v_f_rest   = make_m_v_like(self._features_rest.cpu())
        self.m_opacity,  self.v_opacity  = make_m_v_like(self._opacity.cpu())
        self.m_scaling,  self.v_scaling  = make_m_v_like(self._scaling.cpu())
        self.m_rotation, self.v_rotation = make_m_v_like(self._rotation.cpu())
        
        self.partition_training_initialized = True

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 把所有 Gaussian 的 opacity 强行压到一个很小的上限（≈0.01），
    # 然后把它当成“新参数”重新塞回优化器里，从头再学 opacity。
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_party(self):
        # 1. 计算新的 opacity（数值语义不变）
        with torch.no_grad():
            opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))

        # 2. 直接替换 CPU master 参数
        self._opacity = nn.Parameter(opacities_new.requires_grad_(True))

        # 3. 重置你自己维护的 Adam 状态
        self.m_opacity.zero_()
        self.v_opacity.zero_()



    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        device = torch.device("cpu")
        self._xyz = torch.tensor(xyz, dtype=torch.float, device=device).clone().requires_grad_(False)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().clone().requires_grad_(False)
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().clone().requires_grad_(False)
        self._opacity = torch.tensor(opacities, dtype=torch.float, device=device).clone().requires_grad_(False)
        self._scaling = torch.tensor(scales, dtype=torch.float, device=device).clone().requires_grad_(False)
        self._rotation = torch.tensor(rots, dtype=torch.float, device=device).clone().requires_grad_(False)

        self.active_sh_degree = self.max_sh_degree
        num_points = xyz.shape[0]
        
        print(f"[load_ply] Loaded {num_points} Gaussian points from {path}")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        
        # ===== 同步 prune 你自己维护的 Adam 状态（必须） =====
        self.m_xyz      = self.m_xyz[valid_points_mask]
        self.v_xyz      = self.v_xyz[valid_points_mask]

        self.m_opacity  = self.m_opacity[valid_points_mask]
        self.v_opacity  = self.v_opacity[valid_points_mask]

        self.m_scaling  = self.m_scaling[valid_points_mask]
        self.v_scaling  = self.v_scaling[valid_points_mask]

        self.m_rotation = self.m_rotation[valid_points_mask]
        self.v_rotation = self.v_rotation[valid_points_mask]

        self.m_f_dc     = self.m_f_dc[valid_points_mask]
        self.v_f_dc     = self.v_f_dc[valid_points_mask]

        self.m_f_rest   = self.m_f_rest[valid_points_mask]
        self.v_f_rest   = self.v_f_rest[valid_points_mask]
        # ======================================================


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors




    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 复制一份高梯度的高斯, 就直接cat上去
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        #ADD - 维护新添加点的adam中间状态
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1))
        self.denom = torch.zeros((self.get_xyz.shape[0], 1))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]))
        
        
        # 你自己维护的 Adam 状态（关键）
        self.m_xyz      = torch.cat([self.m_xyz,      torch.zeros_like(new_xyz)], dim=0)
        self.v_xyz      = torch.cat([self.v_xyz,      torch.zeros_like(new_xyz)], dim=0)

        self.m_opacity  = torch.cat([self.m_opacity,  torch.zeros_like(new_opacities)], dim=0)
        self.v_opacity  = torch.cat([self.v_opacity,  torch.zeros_like(new_opacities)], dim=0)

        self.m_scaling  = torch.cat([self.m_scaling,  torch.zeros_like(new_scaling)], dim=0)
        self.v_scaling  = torch.cat([self.v_scaling,  torch.zeros_like(new_scaling)], dim=0)

        self.m_rotation = torch.cat([self.m_rotation, torch.zeros_like(new_rotation)], dim=0)
        self.v_rotation = torch.cat([self.v_rotation, torch.zeros_like(new_rotation)], dim=0)

        self.m_f_dc     = torch.cat([self.m_f_dc,     torch.zeros_like(new_features_dc)], dim=0)
        self.v_f_dc     = torch.cat([self.v_f_dc,     torch.zeros_like(new_features_dc)], dim=0)

        self.m_f_rest   = torch.cat([self.m_f_rest,   torch.zeros_like(new_features_rest)], dim=0)
        self.v_f_rest   = torch.cat([self.v_f_rest,   torch.zeros_like(new_features_rest)], dim=0)

        

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points))
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        # 筛选： “尺寸已经很大、但梯度仍然很大的 Gaussian”
        size = torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        selected_pts_mask = torch.logical_and(selected_pts_mask, size)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3))
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), dtype=bool)))
        
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        
        # 需要克隆的点 (克隆操作没什么优化的 判断梯度然后复制就是了)
        self.densify_and_clone(grads, max_grad, extent)
        
        # 需要分裂的点
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
        self.just_densified = True

    def add_densification_stats(self, full_viewspace_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(full_viewspace_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1 # denom = 每个 Gaussian 被“观察到/产生梯度”的次数（visibility count） 平均梯度 = 累积梯度 / 出现次数(denom)



