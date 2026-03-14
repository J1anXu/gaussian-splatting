"""
Debug script: 对比 split 版和 vanilla 版的 densify 梯度统计。

用法:
  1. 先正常训练 split 版到 iter 3000+ (会自动输出日志到 densify_debug_split.log)
  2. 再正常训练 vanilla 版到 iter 3000+ (会自动输出日志到 densify_debug_vanilla.log)
  3. 对比两个 log 文件中的 grad_mean, above_thresh, cloned, split 等数值

集成方式: 在 train.py 顶部加一行:
    import debug_densify
即可自动 monkey-patch densify_and_prune，无需改动原始代码。
"""

import torch
import os
import sys

# 检测是 split 版还是 vanilla 版
_is_split = os.path.exists(os.path.join(os.path.dirname(__file__), "pipeline_grad_sync.py"))
_log_name = "densify_debug_split.log" if _is_split else "densify_debug_vanilla.log"
_log_path = os.path.join(os.path.dirname(__file__), _log_name)
_log_file = open(_log_path, "w")
_call_count = 0
_current_iter = [0]  # mutable for closure

def _log(msg):
    print(msg)
    _log_file.write(msg + "\n")
    _log_file.flush()


def set_iteration(iteration):
    """在 train.py 的 densify 调用前设置当前 iteration."""
    _current_iter[0] = iteration


def patch_densify():
    """Monkey-patch GaussianModel.densify_and_prune to add debug logging."""
    from scene.gaussian_model import GaussianModel

    _orig_densify = GaussianModel.densify_and_prune

    def _patched_densify(self, max_grad, min_opacity, extent, max_screen_size, *args, **kwargs):
        global _call_count
        _call_count += 1

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        n_before = self.get_xyz.shape[0]
        valid_mask = grads > 0
        valid_grads = grads[valid_mask]
        n_valid = valid_grads.numel()

        grad_norms = torch.norm(grads, dim=-1)
        above = (grad_norms >= max_grad).sum().item()
        denom_nz = (self.denom > 0).sum().item()

        if n_valid > 0:
            gmean = valid_grads.mean().item()
            gmax = valid_grads.max().item()
            # 分位数
            sorted_g = valid_grads.flatten().sort().values
            p50 = sorted_g[len(sorted_g)//2].item()
            p90 = sorted_g[int(len(sorted_g)*0.9)].item()
            p99 = sorted_g[int(len(sorted_g)*0.99)].item()
        else:
            gmean = gmax = p50 = p90 = p99 = 0.0

        _log(f"\n=== DENSIFY #{_call_count} iter={_current_iter[0]} thresh={max_grad} size_thresh={max_screen_size} ===")
        _log(f"  pts={n_before}  denom>0={denom_nz}/{n_before} ({100*denom_nz/max(n_before,1):.1f}%)")
        _log(f"  grad: mean={gmean:.7f} p50={p50:.7f} p90={p90:.7f} p99={p99:.7f} max={gmax:.7f}")
        _log(f"  above_thresh(>={max_grad}): {above} ({100*above/max(n_before,1):.2f}%)")

        # 调用原始方法
        _orig_densify(self, max_grad, min_opacity, extent, max_screen_size, *args, **kwargs)

        n_after = self.get_xyz.shape[0]
        delta = n_after - n_before
        _log(f"  result: {n_before} -> {n_after} (delta={delta:+d})")

    GaussianModel.densify_and_prune = _patched_densify
    _log(f"[debug_densify] patched! log -> {_log_path}")


# 自动 patch
patch_densify()
