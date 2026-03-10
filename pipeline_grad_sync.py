import torch
from typing import List, Optional
from scene import GaussianModel
from TimerManager import TraceManager, TID_OPTIMIZE


ATTR_NAMES = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']


class PipelinedGradSync:
    """Pipelined adam — overlap CPU adam(N) with GPU render(N+1).

    GS-Scale pattern: GPU work runs on subthread, CPU adam runs on main thread.
    Main thread does ZERO CUDA calls during overlap period.

    Usage in main loop (threaded=True):
        for block in blocks:
            pending = grad_sync.pop_pending()
            if pending:
                launch GPU work for current block on subthread
                pending()          # main thread: adam for prev block
                join subthread
            else:
                do GPU work on main thread
            grad_sync.kick_d2h(block)
            grad_sync.submit(iteration)   # sync D2H, save closure
        grad_sync.drain()                 # run last adam
    """

    def __init__(self, submodel_list: List[GaussianModel], opt, dataset, scene,
                 tracer: Optional[TraceManager] = None, threaded: bool = False):
        self.submodel_list = submodel_list
        self.opt = opt
        self.dataset = dataset
        self.scene = scene
        self.tracer = tracer or TraceManager(enabled=False)
        self.threaded = threaded

        self._current_state = None
        self._d2h_event = torch.cuda.Event()
        self._pending_closure = None  # adam closure waiting to run on main thread

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
    # Main thread API
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

        # Record event AFTER all D2H copies are queued
        self._d2h_event.record()

        self._current_state = (submodel, submodel_id, cur_idx, cur_gpu_grads,
                               pin_sub_vf, pin_sub_radii, pin_vpt_grad)

    def submit(self, iteration: int):
        """Sync D2H on main thread, build closure. Threaded mode saves it for later."""
        tm = self.tracer

        with tm.span("d2h_sync", tid=TID_OPTIMIZE):
            self._d2h_event.synchronize()
        tm.flush_gpu_events()

        sm, sm_id, idx, gpu_grads, sub_vf, sub_radii, vpt_grad = self._current_state
        self._current_state = None

        closure = self._make_closure(sm, sm_id, idx, gpu_grads, sub_vf, sub_radii, vpt_grad, iteration)

        if self.threaded:
            self._pending_closure = closure  # save for main thread to run during overlap
        else:
            closure()

    def pop_pending(self):
        """Retrieve and clear the pending adam closure. Returns None if none."""
        c = self._pending_closure
        self._pending_closure = None
        return c

    def drain(self):
        """Run last pending adam closure. Call at end of each iteration."""
        if self._pending_closure is not None:
            self._pending_closure()
            self._pending_closure = None

    def shutdown(self):
        """No-op (no persistent worker thread)."""
        pass

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _make_closure(self, sm, sm_id, idx, gpu_grads, sub_vf, sub_radii, vpt_grad, iteration):
        """Build a closure that performs assemble_grad + adam + densify (pure CPU)."""
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
