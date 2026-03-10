import torch
from typing import List, Optional, Callable
from scene import GaussianModel
from TimerManager import TraceManager, TID_PIPELINE, TID_ADAM, PID_CPU, TID_D2H, TID_OPTIMIZE


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
        # Grad D2H now goes directly into _packed_staging (GPU-side cat + single DMA),
        # so per-attr _pinned_grad_bufs are no longer needed.
        submodel._pinned_vf_buf = torch.empty(n_vis, 1, dtype=torch.long, pin_memory=True)
        submodel._pinned_radii_buf = torch.empty(n_vis, dtype=torch.int32, pin_memory=True)
        submodel._pinned_vpt_grad_buf = torch.empty(n_vis, 3, dtype=torch.float32, pin_memory=True)

    def reallocate_pinned_buffers(self, submodel: GaussianModel):
        self._allocate_pinned_buffers(submodel)

    def kick_async_d2h(self, submodel: GaussianModel, submodel_id: int,
                        render_pkg: dict, sub_viewspace_point_tensor):
        """Kick off non-blocking D2H copies of grads into pre-allocated pinned buffers.

        Packs grads on GPU with torch.cat, then single D2H into _packed_staging.
        This eliminates the CPU-side _assemble_grad_subset (column-scatter into pinned memory).

        Returns captured state tuple for deferred work.
        """
        tm = self.tracer
        with tm.transfer_span("d2h_kick", block_id=submodel_id):
            cur_idx = submodel.visible_indices.to("cpu")
            n_vis = cur_idx.shape[0]

            # Pack 6 grad tensors into [n_vis, D] on GPU, then single D2H
            gpu_attrs = [submodel._xyz_gpu, submodel._features_dc_gpu, submodel._features_rest_gpu,
                            submodel._scaling_gpu, submodel._rotation_gpu, submodel._opacity_gpu]
            grads_flat = []
            for gpu_p in gpu_attrs:
                g = gpu_p.grad
                grads_flat.append(g.reshape(n_vis, -1))
            gpu_grad_packed = torch.cat(grads_flat, dim=1)  # [n_vis, D], contiguous

            # Single contiguous D2H into pinned staging (H2D already consumed it)
            staging = submodel._packed_staging[:n_vis]
            staging.copy_(gpu_grad_packed, non_blocking=True)

            sub_visibility_filter = render_pkg["visibility_filter"]
            sub_radii = render_pkg["radii"]

            pin_sub_vf = submodel._pinned_vf_buf[:sub_visibility_filter.shape[0]]
            pin_sub_vf.copy_(sub_visibility_filter, non_blocking=True)
            pin_sub_radii = submodel._pinned_radii_buf[:sub_radii.shape[0]]
            pin_sub_radii.copy_(sub_radii, non_blocking=True)
            pin_vpt_grad = submodel._pinned_vpt_grad_buf[:n_vis]
            pin_vpt_grad.copy_(sub_viewspace_point_tensor.grad, non_blocking=True)

        return cur_idx, n_vis, pin_sub_vf, pin_sub_radii, pin_vpt_grad

    def _make_pending(self, sm, sm_id, idx, n_vis, sub_vf, sub_radii, vpt_grad, iteration):
        """Build a closure that performs densify + packed_sparse_adam for one submodel."""
        opt, dataset, scene = self.opt, self.dataset, self.scene
        tm = self.tracer

        def _do():
            with torch.no_grad():
                # Adam step first — before densify/prune which may change N
                # Grads already packed in _packed_staging by GPU-side cat + single D2H
                if iteration < opt.iterations:
                    grad_subset = sm._packed_staging[:n_vis]
                    with tm.span("packed_sparse_adam", tid=TID_OPTIMIZE, block_id=sm_id, n_vis=idx.shape[0]):
                        sm.packed_sparse_adam_step(idx, grad_subset, iteration)

                if iteration < opt.densify_until_iter:
                    with tm.span("densify_stats", tid=TID_OPTIMIZE, block_id=sm_id):
                        gvf = sm.visible_indices[sub_vf]
                        sm.max_radii2D[gvf] = torch.max(sm.max_radii2D[gvf], sub_radii[sub_vf])
                        # vpt_grad[sub_vf] == gvpg[gvf]: skip the [N,3] alloc + scatter + re-gather
                        sm.xyz_gradient_accum[gvf] += torch.norm(vpt_grad[sub_vf, :2], dim=-1, keepdim=True)
                        sm.denom[gvf] += 1

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        with tm.span("densify_and_prune", tid=TID_OPTIMIZE, block_id=sm_id):
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            sm.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, device="cpu")
                            sm.pack_to_buffer()
                            self.reallocate_pinned_buffers(sm)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        with tm.span("reset_opacity", tid=TID_OPTIMIZE, block_id=sm_id):
                            sm.reset_opacity()
                            sm._re_view_opacity()

        return _do

    def flush_and_prepare(self, submodel: GaussianModel, submodel_id: int, render_pkg: dict, sub_viewspace_point_tensor, iteration: int):
        """Kick async D2H, flush previous pending, sync, and prepare new pending.

        Key optimization: use Event.synchronize() instead of cuda.synchronize().
        Event only waits for the PREVIOUS block's D2H, not current block's.
        This lets GPU continue processing current D2H + next block's H2D/render
        while CPU runs adam.
        """
        tm = self.tracer

        # 1. kick off async D2H for current submodel
        state = self.kick_async_d2h(submodel, submodel_id, render_pkg, sub_viewspace_point_tensor)
        cur_idx, cur_n_vis, pin_sub_vf, pin_sub_radii, pin_vpt_grad = state

        # 2. deactivate current submodel's GPU subset
        submodel.deactivate_subset()

        # 3. flush PREVIOUS submodel's opt step:
        #    Event.sync waits only for PREV D2H (not current block's GPU work!)
        #    → adam runs while GPU continues processing current block's D2H
        if self._pending_opt is not None:
            with tm.span("d2h_event_sync", tid=TID_OPTIMIZE):
                self._d2h_event.synchronize()
            tm.flush_gpu_events()
            self._pending_opt()
            self._pending_opt = None

        # 4. record Event AFTER current D2H is queued (next call will sync on this)
        self._d2h_event.record()

        # 5. prepare deferred work for current submodel
        self._pending_opt = self._make_pending(submodel, submodel_id, cur_idx, cur_n_vis, pin_sub_vf, pin_sub_radii, pin_vpt_grad, iteration)

    def flush_last(self):
        """Flush the last submodel's pending work after the loop ends."""
        if self._pending_opt is not None:
            self._d2h_event.synchronize()
            self.tracer.flush_gpu_events()
            self._pending_opt()
            self._pending_opt = None
