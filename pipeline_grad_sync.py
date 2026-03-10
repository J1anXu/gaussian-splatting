import torch
from typing import List, Optional, Callable
from scene import GaussianModel
from TimerManager import TraceManager, TID_PIPELINE, TID_ADAM, PID_CPU, TID_D2H


ATTR_NAMES = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']


class PipelinedGradSync:
    """Manages async D2H gradient copies and deferred optimizer steps for submodel pipeline."""

    def __init__(self, submodel_list: List[GaussianModel], opt, dataset, scene,
                 tracer: Optional[TraceManager] = None):
        self.submodel_list = submodel_list
        self.opt = opt
        self.dataset = dataset
        self.scene = scene
        self.tracer = tracer or TraceManager(enabled=False)

        self._pending_opt: Optional[Callable] = None
        # CUDA Event 用于精确同步 D2H，替代 cuda.synchronize()
        self._d2h_event = torch.cuda.Event()

        # Pre-allocate pinned buffers for each submodel
        for submodel in submodel_list:
            self._allocate_pinned_buffers(submodel)

    @staticmethod
    def _allocate_pinned_buffers(submodel: GaussianModel):
        n_vis = submodel._xyz.shape[0]  # worst case: all visible
        submodel._pinned_grad_bufs = {
            '_xyz': torch.empty(n_vis, 3, dtype=torch.float32, pin_memory=True),
            '_features_dc': torch.empty_like(submodel._features_dc, pin_memory=True),
            '_features_rest': torch.empty_like(submodel._features_rest, pin_memory=True),
            '_scaling': torch.empty(n_vis, 3, dtype=torch.float32, pin_memory=True),
            '_rotation': torch.empty(n_vis, 4, dtype=torch.float32, pin_memory=True),
            '_opacity': torch.empty(n_vis, 1, dtype=torch.float32, pin_memory=True),
        }
        submodel._pinned_vf_buf = torch.empty(n_vis, 1, dtype=torch.long, pin_memory=True)
        submodel._pinned_radii_buf = torch.empty(n_vis, dtype=torch.int32, pin_memory=True)
        submodel._pinned_vpt_grad_buf = torch.empty(n_vis, 3, dtype=torch.float32, pin_memory=True)

    def reallocate_pinned_buffers(self, submodel: GaussianModel):
        self._allocate_pinned_buffers(submodel)

    def kick_async_d2h(self, submodel: GaussianModel, submodel_id: int,
                        render_pkg: dict, sub_viewspace_point_tensor):
        """Kick off non-blocking D2H copies of grads into pre-allocated pinned buffers.

        Returns captured state tuple for deferred work.
        """
        tm = self.tracer
        with tm.transfer_span("d2h", block_id=submodel_id, tid=TID_D2H):
            cur_idx = submodel.visible_indices.to("cpu")
            n_vis = cur_idx.shape[0]
            cur_gpu_grads = []
            gpu_attrs = [submodel._xyz_gpu, submodel._features_dc_gpu, submodel._features_rest_gpu,
                            submodel._scaling_gpu, submodel._rotation_gpu, submodel._opacity_gpu]
            for name, gpu_p in zip(ATTR_NAMES, gpu_attrs):
                g = gpu_p.grad
                if g is None:
                    continue
                pinned = submodel._pinned_grad_bufs[name][:n_vis]
                pinned.copy_(g, non_blocking=True)
                cur_gpu_grads.append(pinned)

            sub_visibility_filter = render_pkg["visibility_filter"]
            sub_radii = render_pkg["radii"]

            pin_sub_vf = submodel._pinned_vf_buf[:sub_visibility_filter.shape[0]]
            pin_sub_vf.copy_(sub_visibility_filter, non_blocking=True)
            pin_sub_radii = submodel._pinned_radii_buf[:sub_radii.shape[0]]
            pin_sub_radii.copy_(sub_radii, non_blocking=True)
            pin_vpt_grad = submodel._pinned_vpt_grad_buf[:n_vis]
            pin_vpt_grad.copy_(sub_viewspace_point_tensor.grad, non_blocking=True)

        return cur_idx, cur_gpu_grads, pin_sub_vf, pin_sub_radii, pin_vpt_grad

    def _make_pending(self, sm, sm_id, idx, gpu_grads, sub_vf, sub_radii, vpt_grad, iteration):
        """Build a closure that performs densify + packed_sparse_adam for one submodel."""
        opt, dataset, scene = self.opt, self.dataset, self.scene
        tm = self.tracer

        def _do():
            with torch.no_grad():
                # Adam step first — before densify/prune which may change N
                if iteration < opt.iterations:
                    with tm.span("assemble_grad", tid=TID_ADAM, block_id=sm_id):
                        grad_subset = sm._assemble_grad_subset(gpu_grads)
                    with tm.span("packed_sparse_adam", tid=TID_ADAM, block_id=sm_id, n_vis=idx.shape[0]):
                        sm.packed_sparse_adam_step(idx, grad_subset, iteration)

                if iteration < opt.densify_until_iter:
                    with tm.span("densify_stats", tid=TID_PIPELINE, block_id=sm_id):
                        gvpg = torch.zeros(sm.get_xyz.shape[0], 3, device="cpu", requires_grad=False)
                        gvpg[sm.visible_indices] = vpt_grad
                        gvf = sm.visible_indices[sub_vf]
                        sm.max_radii2D[gvf] = torch.max(sm.max_radii2D[gvf], sub_radii[sub_vf])
                        sm.add_densification_stats2(gvpg, gvf)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        with tm.span("densify_and_prune", tid=TID_PIPELINE, block_id=sm_id):
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            sm.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, device="cpu")
                            sm.pack_to_buffer()
                            self.reallocate_pinned_buffers(sm)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        with tm.span("reset_opacity", tid=TID_PIPELINE, block_id=sm_id):
                            sm.reset_opacity()
                            sm._re_view_opacity()

        return _do

    def after_backward(self, submodel: GaussianModel, submodel_id: int,
                       render_pkg: dict, sub_viewspace_point_tensor, iteration: int):
        """backward 后立即调用：发起 D2H，释放 GPU，准备 pending。"""
        tm = self.tracer

        # 1. D2H（non_blocking）
        state = self.kick_async_d2h(submodel, submodel_id, render_pkg, sub_viewspace_point_tensor)
        cur_idx, cur_gpu_grads, pin_sub_vf, pin_sub_radii, pin_vpt_grad = state

        # 2. 释放 GPU subset
        submodel.deactivate_subset()

        # 3. 记录 D2H event（下一次 flush_pending 会等这个）
        self._d2h_event.record()

        # 4. 准备 deferred adam
        self._pending_opt = self._make_pending(submodel, submodel_id, cur_idx, cur_gpu_grads,
                                                pin_sub_vf, pin_sub_radii, pin_vpt_grad, iteration)

    def flush_pending(self):
        """h2d 前调用：等上一个 block 的 D2H 完成，立即执行 adam。"""
        if self._pending_opt is not None:
            tm = self.tracer
            with tm.span("d2h_event_sync", tid=TID_PIPELINE):
                self._d2h_event.synchronize()
            self._pending_opt()
            self._pending_opt = None

    def flush_last(self):
        """最后一个 block 的 pending work。"""
        self.flush_pending()
        # 全部 block 处理完后，同步所有 GPU 操作，一次性 resolve 所有 pending trace events
        torch.cuda.synchronize()
        self.tracer.flush_gpu_events()
