import torch
from typing import List, Optional, Callable
from scene import GaussianModel
from TimerManager import TraceManager, TID_PIPELINE, TID_ADAM, PID_CPU, TID_D2H, TID_OPTIMIZE


ATTR_NAMES = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']


class PipelinedGradSync:
    """Pipelined D2H gradient sync with one-block stagger.

    Usage in main loop:
        for block in blocks:
            ... render + backward ...
            grad_sync.kick_d2h(block)        # 1. async D2H current
            grad_sync.step_previous()        # 2. wait prev D2H + adam prev
            grad_sync.commit(block, iter)    # 3. record event + defer current
        grad_sync.step_previous()            # flush last block
    """

    def __init__(self, submodel_list: List[GaussianModel], opt, dataset, scene,
                 tracer: Optional[TraceManager] = None):
        self.submodel_list = submodel_list
        self.opt = opt
        self.dataset = dataset
        self.scene = scene
        self.tracer = tracer or TraceManager(enabled=False)

        self._pending_opt: Optional[Callable] = None
        self._d2h_event = torch.cuda.Event()

        # Captured state from kick_d2h, consumed by commit
        self._current_state = None

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

    # ------------------------------------------------------------------
    # Step 1: kick off async D2H for current block
    # ------------------------------------------------------------------
    def kick_d2h(self, submodel: GaussianModel, submodel_id: int,
                 render_pkg: dict, sub_viewspace_point_tensor):
        """Start non-blocking D2H copies of gradients, then free GPU subset."""
        tm = self.tracer
        with tm.transfer_span("d2h_kick", block_id=submodel_id):
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

        submodel.deactivate_subset()

        self._current_state = (submodel, submodel_id, cur_idx, cur_gpu_grads,
                               pin_sub_vf, pin_sub_radii, pin_vpt_grad)

    # ------------------------------------------------------------------
    # Step 2: wait for previous block's D2H and run its optimizer
    # ------------------------------------------------------------------
    def step_previous(self):
        """Sync previous block's D2H event and execute its deferred adam/densify."""
        if self._pending_opt is None:
            return
        tm = self.tracer
        with tm.span("d2h_event_sync", tid=TID_OPTIMIZE):
            self._d2h_event.synchronize()
        tm.flush_gpu_events()
        self._pending_opt()
        self._pending_opt = None

    # ------------------------------------------------------------------
    # Step 3: record D2H event and defer current block's optimizer
    # ------------------------------------------------------------------
    def commit(self, iteration: int):
        """Record CUDA event for current D2H and prepare deferred optimizer closure."""
        self._d2h_event.record()

        sm, sm_id, idx, gpu_grads, sub_vf, sub_radii, vpt_grad = self._current_state
        self._current_state = None

        self._pending_opt = self._make_pending(
            sm, sm_id, idx, gpu_grads, sub_vf, sub_radii, vpt_grad, iteration)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _make_pending(self, sm, sm_id, idx, gpu_grads, sub_vf, sub_radii, vpt_grad, iteration):
        """Build a closure that performs adam + densify for one submodel."""
        opt, dataset, scene = self.opt, self.dataset, self.scene
        tm = self.tracer

        def _do():
            with torch.no_grad():
                if iteration < opt.iterations:
                    with tm.span("assemble_grad", tid=TID_OPTIMIZE, block_id=sm_id):
                        grad_subset = sm._assemble_grad_subset(gpu_grads)
                    with tm.span("packed_sparse_adam", tid=TID_OPTIMIZE, block_id=sm_id, n_vis=idx.shape[0]):
                        sm.packed_sparse_adam_step(idx, grad_subset, iteration)

                if iteration < opt.densify_until_iter:
                    with tm.span("densify_stats", tid=TID_OPTIMIZE, block_id=sm_id):
                        gvpg = torch.zeros(sm.get_xyz.shape[0], 3, device="cpu", requires_grad=False)
                        gvpg[sm.visible_indices] = vpt_grad
                        gvf = sm.visible_indices[sub_vf]
                        sm.max_radii2D[gvf] = torch.max(sm.max_radii2D[gvf], sub_radii[sub_vf])
                        sm.add_densification_stats2(gvpg, gvf)

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
