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

import config
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
from partition import generate_octant_blocks, generate_octant_blocks_kdtree, generate_space_kdtree_blocks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy


try:
    from diff_gaussian_rasterization_wenqi_tam import SparseGaussianAdam
except:
    pass

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None

try:
    from diff_gaussian_rasterization_wenqi_tam import _C as cpu_adam
except ImportError:
    cpu_adam = None

class GaussianModel:

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


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # visible_indices should keep None unless set by set_subset
        self.visible_indices = None # 
        self.block_bounds = []
        self.block_idx_list = []
        self.partitioned = False
        
        self.subset_mode_1 = False # for render first time
        self.subset_mode_2 = False # for render second time
        
        self.setup_functions()
        
    def dump_to_cpu(self, free_gpu=True):
        """
        Create a frozen CPU snapshot and optionally
        free GPU tensors from the original object.
        """
        new = copy.copy(self)

        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                # CPU snapshot
                setattr(new, k, v.detach().cpu())

                if free_gpu:
                    # 🔥 关键：切断原对象对 GPU tensor 的引用
                    setattr(self, k, None)

            elif k == "optimizer":
                setattr(new, k, None)
                if free_gpu:
                    self.optimizer = None

        if free_gpu:
            torch.cuda.empty_cache()

        return new

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
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # TODO 这个_gpu好像不太必要 可以直接tocuda
    @property
    def get_scaling(self):
        if self.subset_mode_2:
            return self.scaling_activation(self._scaling_gpu)
        elif self.subset_mode_1:
            return self.scaling_activation(self._scaling[self.visible_indices])
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.subset_mode_2:
            return self.rotation_activation(self._rotation_gpu)
        elif self.subset_mode_1:
            return self.rotation_activation(self._rotation[self.visible_indices])
        else:
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self.subset_mode_2:
            return self._xyz_gpu
        elif self.subset_mode_1:
            return self._xyz[self.visible_indices]
        else:
            return self._xyz
    
    @property
    def get_features(self):
        if self.subset_mode_2:
            features_dc = self._features_dc_gpu
            features_rest = self._features_rest_gpu
        elif self.subset_mode_1:
            features_dc = self._features_dc[self.visible_indices]
            features_rest = self._features_rest[self.visible_indices]
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        if self.subset_mode_2:
            return self._features_dc_gpu
        elif self.subset_mode_1:
            return self._features_dc[self.visible_indices]
        else:
            return self._features_dc
    
    @property
    def get_features_rest(self):
        if self.subset_mode_2:
            return self._features_rest_gpu
        elif self.subset_mode_1:
            return self._features_rest[self.visible_indices]
        else:
            return self._features_rest
    
    @property
    def get_opacity(self):
        if self.subset_mode_2:
            return self.opacity_activation(self._opacity_gpu)
        elif self.subset_mode_1:
            return self.opacity_activation(self._opacity[self.visible_indices])
        else:
            return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    # def get_exposure_from_name(self, image_name):
    #     if self.pretrained_exposures is None:
    #         return self._exposure[self.exposure_mapping[image_name]]
    #     else:
    #         return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        if self.subset_mode_2:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation_gpu)
        elif self.subset_mode_1:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation[self.visible_indices])
        else:
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

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        # self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def create_from_pcd_cpu(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
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

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        # self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args, device = "cuda"):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, foreach=True)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        # ''' Learning rate scheduling per step '''
        # if self.pretrained_exposures is None:
        #     for param_group in self.exposure_optimizer.param_groups:
        #         param_group['lr'] = self.exposure_scheduler_args(iteration)

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

    def build_block_id(self):
        N = self._xyz.shape[0]
        block_id = np.full(N, -1, dtype=np.int32)
        for b, idx in enumerate(self.block_idx_list):
            idx = idx.detach().cpu().numpy()
            block_id[idx] = b
        return block_id

    def build_block_elements(self):
        B = len(self.block_bounds)
        dtype = [ ("xmin", "f4"), ("ymin", "f4"), ("zmin", "f4"), ("xmax", "f4"), ("ymax", "f4"), ("zmax", "f4"), ]
        blocks = np.empty(B, dtype=dtype)
        for i, (mn, mx) in enumerate(self.block_bounds):
            mn = mn.detach().cpu().numpy()
            mx = mx.detach().cpu().numpy()
            blocks[i] = (*mn, *mx)
        return PlyElement.describe(blocks, "block")

    def save_ply(self, path, include_block=True):
        mkdir_p(os.path.dirname(path))
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        if include_block:
            block_id = self.build_block_id()[:, None]  # [N,1]
            dtype_full = ( [(attribute, 'f4') for attribute in self.construct_list_of_attributes()] + [('block_id', 'i4')] )
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate( (xyz, normals, f_dc, f_rest, opacities, scale, rotation, block_id), axis=1 )
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            el_block = self.build_block_elements()
            PlyData([el, el_block]).write(path)
        else:        
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
        
            

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        # if use_train_test_exp:
        #     exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
        #     if os.path.exists(exposure_file):
        #         with open(exposure_file, "r") as f:
        #             exposures = json.load(f)
        #         self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
        #         print(f"Pretrained exposures loaded.")
        #     else:
        #         print(f"No exposure to be loaded at {exposure_file}")
        #         self.pretrained_exposures = None

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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        self.block_bounds = None
        self.block_idx_list = None
        
        if "block" in plydata:
            block_elem = plydata["block"].data  # structured array, shape (B,)
            block_bounds = []
            for b in block_elem:
                mn = torch.tensor( [b["xmin"], b["ymin"], b["zmin"]], dtype=torch.float32 )
                mx = torch.tensor( [b["xmax"], b["ymax"], b["zmax"]], dtype=torch.float32 )
                block_bounds.append((mn, mx))
            self.block_bounds = block_bounds

        if self.block_bounds is not None:
            xyz_cpu = self._xyz.detach().cpu()  # [N,3]
            block_indices = []
            for mn, mx in self.block_bounds:
                mn = mn.cpu()
                mx = mx.cpu()
                inside = (
                    (xyz_cpu[:, 0] >= mn[0]) & (xyz_cpu[:, 0] <= mx[0]) &
                    (xyz_cpu[:, 1] >= mn[1]) & (xyz_cpu[:, 1] <= mx[1]) &
                    (xyz_cpu[:, 2] >= mn[2]) & (xyz_cpu[:, 2] <= mx[2])
                )
                idx = torch.nonzero(inside, as_tuple=False).squeeze(1)
                block_indices.append(idx)
                self.block_idx_list = block_indices
            

    def get_subset_by_id(self, idx):
        """
        Create an independent GaussianModel for block idx.
        This kid has its OWN parameters and OWN optimizer,
        and is optimized from scratch.
        """
        assert self.block_idx_list is not None, "block_indices is None"
        assert idx < len(self.block_idx_list), f"block idx {idx} out of range"

        indices = self.block_idx_list[idx].to(self._xyz.device)

        kid = GaussianModel( sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type )

        # ====== 核心参数：深拷贝 & 断梯度 ======
        kid._xyz = nn.Parameter(self._xyz[indices].clone().detach())
        kid._features_dc = nn.Parameter(self._features_dc[indices].clone().detach())
        kid._features_rest = nn.Parameter(self._features_rest[indices].clone().detach())
        kid._scaling = nn.Parameter(self._scaling[indices].clone().detach())
        kid._rotation = nn.Parameter(self._rotation[indices].clone().detach())
        kid._opacity = nn.Parameter(self._opacity[indices].clone().detach())

        # exposure：通常不 block 化，但如果你希望完全独立，也 clone
        kid._exposure = nn.Parameter(self._exposure.clone().detach())

        # ====== 其他状态 ======
        kid.active_sh_degree = self.active_sh_degree
        kid.spatial_lr_scale = self.spatial_lr_scale
        kid.max_radii2D = torch.zeros( kid._xyz.shape[0], device=kid._xyz.device )

        kid.block_bounds = None
        kid.block_idx_list = None
        kid.partitioned = False
        kid.visible_indices = None
        
        return kid

    def split(self):
        subsets = []
        for idx in range(len(self.block_idx_list)):
            subset = self.get_subset_by_id(idx)
            subsets.append(subset)
        return subsets


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, device="cuda"):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, device = "cuda"):
        # 梯度大 + 尺度已经很大的 Gaussian → 不该再 clone，而是必须 split（拆分）: 沿 Gaussian 自身尺度与朝向，在空间上强制生成 N 个彼此分离的子 Gaussian
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        rots = rots.to(device) # add by jian
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, device=device)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, device="cuda"):
        # Extract points that satisfy the gradient condition
        # 只克隆那些"梯度大、但尺度还不算大"的高斯点 → 克隆, 让它们变得更密集 (原地复制)
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, device=device)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, device="cuda"):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, device=device)
        self.densify_and_split(grads, max_grad, extent, device=device)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, global_visibility_filter, frustum_visibility_filter):
        self.xyz_gradient_accum[global_visibility_filter] += torch.norm(viewspace_point_tensor.grad[frustum_visibility_filter,:2], dim=-1, keepdim=True)
        self.denom[global_visibility_filter] += 1

    def add_densification_stats2(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _compute_pack_layout(self):
        """
        Compute the column layout for the packed parameter buffer.

        All per-Gaussian attributes are concatenated along the feature dimension
        into a single row of width D. The layout is determined by the spherical
        harmonics degree and follows a fixed ordering:

            [xyz(3) | f_dc(3) | f_rest((L+1)^2-1)*3 | scaling(3) | rotation(4) | opacity(1)]

        where L = max_sh_degree. For L=3, D = 3+3+45+3+4+1 = 59.

        Returns:
            slices: dict mapping attribute name -> (col_start, col_end, reshape_dims)
                    reshape_dims is None for 2D attributes, or a tuple for those
                    requiring reshape (e.g., features_dc: [N,1,3]).
            D:      total number of columns in the packed buffer.
        """
        sh_rest_cols = ((self.max_sh_degree + 1) ** 2 - 1) * 3
        slices = {}
        c = 0
        slices['_xyz'] = (c, c + 3, None); c += 3
        slices['_features_dc'] = (c, c + 3, (-1, 1, 3)); c += 3
        slices['_features_rest'] = (c, c + sh_rest_cols, (-1, sh_rest_cols // 3, 3)); c += sh_rest_cols
        slices['_scaling'] = (c, c + 3, None); c += 3
        slices['_rotation'] = (c, c + 4, None); c += 4
        slices['_opacity'] = (c, c + 1, None); c += 1
        return slices, c

    _GROUP_TO_ATTR = {
        "xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest",
        "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation",
    }

    def pack_to_buffer(self):
        """
        Pack six per-Gaussian CPU nn.Parameter tensors into a single contiguous
        pinned-memory buffer of shape [N, D], then replace each param with a
        view into the packed buffer so that optimizer.step() writes directly
        into _packed (eliminating the need for sync_packed_from_params).
        """
        slices, D = self._compute_pack_layout()
        N = self._xyz.shape[0]
        packed = torch.empty(N, D, dtype=torch.float32, pin_memory=True)
        for name, (s, e, _) in slices.items():
            param = getattr(self, name)
            packed[:, s:e] = param.data.detach().reshape(N, e - s)

        old_params = {name: getattr(self, name) for name in slices}

        self._packed = packed
        self._pack_slices = slices
        self._pack_D = D

        # Create view-params: Adam writes directly into _packed
        for name, (s, e, reshape) in slices.items():
            view = packed[:, s:e]
            if reshape is not None:
                view = view.view(N, *reshape[1:])
            setattr(self, name, nn.Parameter(view, requires_grad=True))

        self._packed_staging = torch.empty(N, D, dtype=torch.float32, pin_memory=True)

        if self.optimizer is not None:
            self._migrate_optimizer_state(old_params)

        self._init_packed_adam_state()

    def _migrate_optimizer_state(self, old_params):
        """Migrate optimizer state from old params to new view-params after pack_to_buffer."""
        for group in self.optimizer.param_groups:
            attr = self._GROUP_TO_ATTR.get(group["name"])
            if attr is None:
                continue
            old_p = old_params[attr]
            new_p = getattr(self, attr)
            stored = self.optimizer.state.pop(old_p, None)
            if stored is not None:
                for key in ["exp_avg", "exp_avg_sq"]:
                    if key in stored and stored[key].shape == new_p.shape:
                        pass  # shape unchanged, reuse as-is
                    else:
                        stored[key] = torch.zeros_like(new_p)
                self.optimizer.state[new_p] = stored
            group["params"][0] = new_p

    def _re_view_opacity(self):
        """After reset_opacity, write new values back to packed and rebuild the view."""
        s, e, _ = self._pack_slices['_opacity']
        N = self._packed.shape[0]
        self._packed[:, s:e] = self._opacity.data.detach().reshape(N, e - s)
        new_p = nn.Parameter(self._packed[:, s:e], requires_grad=True)
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                stored = self.optimizer.state.pop(group["params"][0], None)
                if stored is not None:
                    for key in ["exp_avg", "exp_avg_sq"]:
                        stored[key] = torch.zeros_like(new_p)
                    self.optimizer.state[new_p] = stored
                group["params"][0] = new_p
        self._opacity = new_p
        # Zero out packed adam state for opacity columns
        if hasattr(self, '_packed_exp_avg'):
            self._packed_exp_avg[:, s:e] = 0
            self._packed_exp_avg_sq[:, s:e] = 0

    def _init_packed_adam_state(self):
        """Initialize or resize [N, D] exp_avg / exp_avg_sq for packed_sparse_adam.

        On first call: allocate zero tensors and set step=0.
        On subsequent calls (after densify/prune): resize to new N, preserving
        step counter. Momentum is reset to zero (new points need zero init anyway,
        and densify only happens in early training).
        """
        N, D = self._packed.shape
        if not hasattr(self, '_packed_adam_step'):
            self._packed_adam_step = 0
        # Always reallocate to match current N (densify/prune changes N)
        self._packed_exp_avg = torch.zeros(N, D, dtype=torch.float32, pin_memory=True)
        self._packed_exp_avg_sq = torch.zeros(N, D, dtype=torch.float32, pin_memory=True)

    def _build_lr_per_col(self, iteration):
        """Build [D] tensor with per-column learning rate from optimizer param_groups."""
        D = self._pack_D
        lr_vec = torch.zeros(D, dtype=torch.float32)
        for group in self.optimizer.param_groups:
            attr = self._GROUP_TO_ATTR.get(group["name"])
            if attr is None:
                continue
            s, e, _ = self._pack_slices[attr]
            lr_vec[s:e] = group["lr"]
        return lr_vec

    def _assemble_grad_subset(self, gpu_grads):
        """Concatenate 6 per-attr grad pinned bufs [n_vis, cols_i] into [n_vis, D]."""
        n_vis = gpu_grads[0].shape[0]
        staging = self._packed_staging[:n_vis]
        attr_order = ['_xyz', '_features_dc', '_features_rest',
                      '_scaling', '_rotation', '_opacity']
        for gi, name in enumerate(attr_order):
            s, e, _ = self._pack_slices[name]
            staging[:, s:e] = gpu_grads[gi].float().reshape(n_vis, e - s)
        return staging

    def packed_sparse_adam_step(self, idx, grad_subset, iteration):
        """
        idx: [n_vis] int64, visible row indices
        grad_subset: [n_vis, D] float32, assembled gradient
        """
        self._packed_adam_step += 1
        lr_per_col = self._build_lr_per_col(iteration)
        cpu_adam.packed_sparse_adam(
            self._packed, grad_subset,
            self._packed_exp_avg, self._packed_exp_avg_sq,
            idx, lr_per_col,
            self._packed_adam_step,
            0.9, 0.999, 1e-15
        )

    def sync_packed_from_params(self):
        """
        No-op when params are views into _packed (optimizer writes directly).
        Falls back to full repack only if N changed (densify/prune).
        """
        N = self._xyz.shape[0]
        if not hasattr(self, '_packed') or self._packed.shape[0] != N:
            self.pack_to_buffer()
            return
        # params are views of _packed — nothing to sync

    def activate_subset(self):
        self.subset_mode_1 = True
        
    def deactivate_subset(self):
        self.subset_mode_1 = False    
        self.subset_mode_2 = False  

    def move_and_activate_subset(self, requires_grad=True):
        """
        Gather visible Gaussian attributes from CPU and transfer to GPU.

        This method replaces the naive per-attribute implementation that performed
        six independent fancy-indexing + H2D transfers. The optimized pipeline:

        1. **Single gather**: torch.index_select on the packed [N, D] pinned buffer
           writes directly into a pre-allocated pinned staging buffer, avoiding
           memory allocation and ensuring the result resides in page-locked memory.

        2. **Single H2D transfer**: The pinned staging buffer is transferred to GPU
           via .cuda(non_blocking=True). Because the source is pinned, PyTorch
           dispatches a true cudaMemcpyAsync on the current stream, enabling
           overlap with concurrent CPU work or GPU computation on other streams.

        3. **GPU-side unpack**: The packed GPU tensor is sliced and cloned into
           individual attribute tensors. clone() is mandatory — views would share
           the same storage, causing gradient accumulation conflicts during
           backward (scatter_grad expects independent .grad tensors per attribute).
           GPU-internal memcpy (clone) is negligible (~0.1ms) compared to the
           PCIe transfer savings.

        Complexity reduction:
            - CPU indexing: 6 random-access passes -> 1 pass (better cache locality)
            - PCIe transactions: 6 -> 1 (reduced launch overhead)
            - Memory allocation: 6 temporary tensors -> 0 (pre-allocated staging)
        """
        idx = self.visible_indices
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long)
        idx = idx.to("cpu")
        n = idx.shape[0]

        # Gather into pre-allocated pinned staging buffer.
        # Using out= parameter ensures the result is written directly into
        # page-locked memory without intermediate allocation.
        staging = self._packed_staging[:n]
        torch.index_select(self._packed, 0, idx, out=staging)

        # Single DMA transfer: pinned -> GPU. non_blocking=True is effective
        # only when the source tensor is in pinned (page-locked) memory.
        if config.HALF_H2D:
            gpu_packed = staging.half().cuda(non_blocking=True)
        else:
            gpu_packed = staging.cuda(non_blocking=True)

        # Unpack on GPU. Each clone() creates an independent tensor with its
        # own storage, which is required for correct per-attribute gradient
        # computation in the subsequent backward pass.
        # .float() ensures FP32 for rasterizer regardless of HALF_H2D setting.
        slices = self._pack_slices
        s, e, _ = slices['_xyz']
        self._xyz_gpu = gpu_packed[:, s:e].float().clone()
        s, e, reshape = slices['_features_dc']
        self._features_dc_gpu = gpu_packed[:, s:e].reshape(n, reshape[1], reshape[2]).float().clone()
        s, e, reshape = slices['_features_rest']
        self._features_rest_gpu = gpu_packed[:, s:e].reshape(n, reshape[1], reshape[2]).float().clone()
        s, e, _ = slices['_scaling']
        self._scaling_gpu = gpu_packed[:, s:e].float().clone()
        s, e, _ = slices['_rotation']
        self._rotation_gpu = gpu_packed[:, s:e].float().clone()
        s, e, _ = slices['_opacity']
        self._opacity_gpu = gpu_packed[:, s:e].float().clone()

        if requires_grad:
            self._xyz_gpu.requires_grad_(True)
            self._features_dc_gpu.requires_grad_(True)
            self._features_rest_gpu.requires_grad_(True)
            self._scaling_gpu.requires_grad_(True)
            self._rotation_gpu.requires_grad_(True)
            self._opacity_gpu.requires_grad_(True)

        self.subset_mode_2 = True


        
    def build_split_indices(self):
        block_bounds, block_indices = generate_space_kdtree_blocks(self._xyz)
        self.block_bounds = block_bounds
        self.block_idx_list = block_indices
    
    def visualize_blocks(self, point_alpha=0.02, box_alpha=0.15, save_path="boxxes.png"):
        xyz = self._xyz.detach().cpu()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        def random_subsample(xyz, max_points=100_000):
            N = xyz.shape[0]
            if N <= max_points:
                return xyz
            idx = torch.randperm(N, device=xyz.device)[:max_points]
            return xyz[idx]
        # ---------- 1. 画点云 ----------
        xyz_vis = random_subsample(xyz, max_points=50_000)

        ax.scatter(
            xyz_vis[:, 0].cpu(),
            xyz_vis[:, 1].cpu(),
            xyz_vis[:, 2].cpu(),
            s=1,
            c="gray",
            alpha=point_alpha,
        )


        # ---------- 2. 给 block 上不同颜色 ----------
        colors = plt.cm.tab10.colors

        # ---------- 3. 画每个 block 的盒子 ----------
        for i, (mn, mx) in enumerate(self.block_bounds):
            mn = mn.cpu()
            mx = mx.cpu()

            # 8 个顶点
            v = [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ]

            # 6 个面（每个面 4 个点）
            faces = [
                [v[0], v[1], v[2], v[3]],  # bottom
                [v[4], v[5], v[6], v[7]],  # top
                [v[0], v[1], v[5], v[4]],
                [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]],
                [v[0], v[3], v[7], v[4]],
            ]

            box = Poly3DCollection(
                faces,
                alpha=box_alpha,
                facecolor=colors[i % len(colors)],
                edgecolor="k",
                linewidths=1.0
            )
            ax.add_collection3d(box)

            # ---------- 4. 可选：标注 block ----------
            if self.block_idx_list is not None:
                idx = self.block_idx_list[i]
                if idx.numel() > 0:
                    center = (mn + mx) * 0.5
                    ax.text(
                        center[0], center[1], center[2],
                        f"B{i}\n{idx.numel()}",
                        color="black",
                        fontsize=10,
                        ha="center"
                    )

        # ---------- 5. 坐标等比例 ----------
        all_pts = xyz
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
        ax.set_title("Blocks Visualization")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
            print(f"[OK] saved to {save_path}")
        plt.show()
        
    def repartition(self):
        """
        根据固定的 block_bounds，
        用当前 self._xyz 重新生成 block_indices
        """

        assert hasattr(self, "block_bounds"), "block_bounds not initialized. Call partition() first."

        xyz = self._xyz
        device = xyz.device
        dtype = xyz.dtype
        N = xyz.shape[0]

        all_idx = torch.arange(N, device=device)

        new_block_indices = []
        covered_mask = torch.zeros(N, device=device, dtype=torch.bool)  # 记录是否被某个 block 覆盖

        # ---------- 遍历每个 block ----------
        for i, (mn, mx) in enumerate(self.block_bounds):
            # 确保在同一 device
            mn = mn.to(device)
            mx = mx.to(device)

            mask = (
                (xyz[:, 0] >= mn[0]) & (xyz[:, 0] < mx[0]) &
                (xyz[:, 1] >= mn[1]) & (xyz[:, 1] < mx[1]) &
                (xyz[:, 2] >= mn[2]) & (xyz[:, 2] < mx[2])
            )

            idx = all_idx[mask]
            new_block_indices.append(idx)
            covered_mask[idx] = True  # 标记被覆盖

            # debug 用
            # print(f"[repartition] Block {i:2d}: {idx.numel():7d} points")
        uncovered_idx = all_idx[~covered_mask]
        self.block_idx_list = new_block_indices
        print(f"[repartition] Uncovered points ({uncovered_idx.numel()}): {uncovered_idx.tolist()}")
