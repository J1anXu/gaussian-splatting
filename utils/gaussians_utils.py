from scene.gaussian_model import GaussianModel
import torch
import torch.nn as nn




def _leaf_param(x: torch.Tensor) -> nn.Parameter:
    # 变成 leaf + 可训练 Parameter
    return nn.Parameter(x.detach().clone(), requires_grad=True)

def create_sub_gaussians(parent, block_indices):
    kids = []
    for inds in block_indices:
        kid = GaussianModel(parent.max_sh_degree, parent.optimizer_type)

        kid._xyz          = _leaf_param(parent._xyz[inds])
        kid._features_dc  = _leaf_param(parent._features_dc[inds])
        kid._features_rest= _leaf_param(parent._features_rest[inds])
        kid._scaling      = _leaf_param(parent._scaling[inds])
        kid._rotation     = _leaf_param(parent._rotation[inds])
        kid._opacity      = _leaf_param(parent._opacity[inds])
        kid._exposure     = _leaf_param(parent._exposure)
        kid.pretrained_exposures = None
        kid.max_radii2D = torch.zeros((kid._xyz.shape[0]), device="cuda")


        kids.append(kid)
    return kids