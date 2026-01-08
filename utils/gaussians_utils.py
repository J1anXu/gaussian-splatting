from scene.gaussian_model import GaussianModel
import torch


def create_sub_gaussians(parent: GaussianModel, indices_list):
    kids = []
    for indices in indices_list:
        kids.append(create_sub_gaussian(parent, indices))
    return kids

def create_sub_gaussian(parent: GaussianModel, indices: torch.Tensor):
    """
    Create a sub GaussianModel by indexing parent Gaussian parameters
    """
    sub = GaussianModel(
        sh_degree=parent.max_sh_degree,
        optimizer_type=parent.optimizer_type,
    )

    # ⚠️ 一定要 .clone()，否则会共享 storage
    sub._xyz = parent._xyz[indices].clone()
    sub._features_dc = parent._features_dc[indices].clone()
    sub._features_rest = parent._features_rest[indices].clone()
    sub._opacity = parent._opacity[indices].clone()
    sub._scaling = parent._scaling[indices].clone()
    sub._rotation = parent._rotation[indices].clone()

    # 如果你有其他属性，也一并 copy
    # sub.some_attr = parent.some_attr[indices].clone()

    return sub