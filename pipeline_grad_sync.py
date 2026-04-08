import config
import torch
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from scene import GaussianModel
from TimerManager import TraceManager, TID_PIPELINE, TID_ADAM, PID_CPU, TID_D2H, TID_OPTIMIZE
from diff_gaussian_rasterization_wenqi_tam import _C as cpu_adam


ATTR_NAMES = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']


class PipelinedGradSync:
    """Manages async D2H gradient copies and deferred optimizer steps for submodel pipeline.

    Key design (fix_2): CPU Adam runs in a background thread with GIL released
    (via py::call_guard<py::gil_scoped_release> on packed_sparse_adam).
    D2H uses a dedicated comm_stream so it overlaps with default-stream GPU work.
    Each block gets its own CUDA Event; the worker thread waits on it independently.
    Main thread never blocks on Adam — only an iteration barrier in flush_last().
    """

    # Block-partitioned rendering produces slightly lower gradient magnitudes
    # than vanilla due to block-level transmittance approximation and per-block
    # loss computation. Scale down the densification threshold to compensate.
    DENSIFY_GRAD_SCALE = config.DENSIFY_GRAD_SCALE

    def __init__(self, submodel_list: List[GaussianModel], opt, dataset, scene,
                 tracer: Optional[TraceManager] = None, frustum_cache: Optional[dict] = None):
        self.submodel_list = submodel_list
        self.opt = opt
        self.dataset = dataset
        self.scene = scene
        self.tracer = tracer or TraceManager(enabled=False)
        self.frustum_cache = frustum_cache

        # Dedicated communication stream for D2H copies (high priority)
        self._comm_stream = torch.cuda.Stream(priority=-1)

        # Background thread for CPU Adam (GIL released in C++ packed_sparse_adam)
        self._adam_executor = ThreadPoolExecutor(max_workers=1)
        self._adam_futures: List[Future] = []

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

        D2H runs on comm_stream so it overlaps with default-stream GPU work (next
        block's H2D + render). Returns captured state tuple for deferred work.
        gpu_grad_packed is returned to keep the GPU tensor alive until D2H completes.
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

            sub_visibility_filter = render_pkg["visibility_filter"]
            sub_radii = render_pkg["radii"]

            # comm_stream must wait for default stream's backward + cat to finish
            sync_event = torch.cuda.Event()
            sync_event.record()  # record on default (current) stream

            # D2H on comm_stream — overlaps with default stream's next H2D/render
            with torch.cuda.stream(self._comm_stream):
                self._comm_stream.wait_event(sync_event)

                # Single contiguous D2H into pinned staging
                staging = submodel._packed_staging[:n_vis]
                staging.copy_(gpu_grad_packed, non_blocking=True)

                pin_sub_vf = submodel._pinned_vf_buf[:sub_visibility_filter.shape[0]]
                pin_sub_vf.copy_(sub_visibility_filter, non_blocking=True)
                pin_sub_radii = submodel._pinned_radii_buf[:sub_radii.shape[0]]
                pin_sub_radii.copy_(sub_radii, non_blocking=True)
                pin_vpt_grad = submodel._pinned_vpt_grad_buf[:n_vis]
                pin_vpt_grad.copy_(sub_viewspace_point_tensor.grad, non_blocking=True)

        # Return gpu_grad_packed to keep it alive — caller must hold the reference
        # until D2H on comm_stream completes (via event.synchronize in worker)
        return cur_idx, n_vis, pin_sub_vf, pin_sub_radii, pin_vpt_grad, gpu_grad_packed

    def _make_adam_fn(self, sm, sm_id, idx, n_vis, lr_per_col, iteration):
        """Build adam closure. lr_per_col is pre-computed on main thread to avoid
        GIL contention in the worker."""
        opt, dataset, scene = self.opt, self.dataset, self.scene
        tm = self.tracer

        def _adam():
            with torch.no_grad():
                if iteration < opt.iterations:
                    grad_subset = sm._packed_staging[:n_vis]
                    with tm.span("packed_sparse_adam", tid=TID_OPTIMIZE, block_id=sm_id, n_vis=idx.shape[0]):
                        # This C++ call releases the GIL — main thread runs freely
                        sm._packed_adam_step += 1
                        cpu_adam.packed_sparse_adam(
                            sm._packed, grad_subset,
                            sm._packed_exp_avg, sm._packed_exp_avg_sq,
                            idx, lr_per_col,
                            sm._packed_adam_step,
                            0.9, 0.999, 1e-15
                        )

                if iteration < opt.densify_until_iter:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        with tm.span("densify_and_prune", tid=TID_OPTIMIZE, block_id=sm_id):
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            sm.densify_and_prune(opt.densify_grad_threshold * self.DENSIFY_GRAD_SCALE, 0.005, scene.cameras_extent, size_threshold, device="cpu")
                            sm.pack_to_buffer()
                            self.reallocate_pinned_buffers(sm)
                            if self.frustum_cache and sm_id in self.frustum_cache:
                                del self.frustum_cache[sm_id]

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        with tm.span("reset_opacity", tid=TID_OPTIMIZE, block_id=sm_id):
                            sm.reset_opacity()
                            sm._re_view_opacity()

        return _adam

    def _make_densify_stats_fn(self, sm, sm_id, sub_vf, sub_radii, vpt_grad, iteration):
        """Build densify_stats closure."""
        opt = self.opt
        tm = self.tracer

        def _densify_stats():
            with torch.no_grad():
                if iteration < opt.densify_until_iter:
                    with tm.span("densify_stats", tid=TID_OPTIMIZE, block_id=sm_id):
                        gvf = sm.visible_indices[sub_vf]
                        sm.max_radii2D[gvf] = torch.max(sm.max_radii2D[gvf], sub_radii[sub_vf])
                        sm.xyz_gradient_accum[gvf] += torch.norm(vpt_grad[sub_vf, :2], dim=-1, keepdim=True)
                        sm.denom[gvf] += 1

        return _densify_stats

    def flush_and_prepare(self, submodel: GaussianModel, submodel_id: int, render_pkg: dict, sub_viewspace_point_tensor, iteration: int):
        """Kick async D2H on comm_stream, then submit adam+densify to worker thread.

        Main thread does NOT block on adam — it returns immediately to enqueue
        the next block's H2D + render on the default stream.
        """
        tm = self.tracer

        # 1. kick off async D2H on comm_stream
        state = self.kick_async_d2h(submodel, submodel_id, render_pkg, sub_viewspace_point_tensor)
        cur_idx, cur_n_vis, pin_sub_vf, pin_sub_radii, pin_vpt_grad, gpu_grad_ref = state

        # 2. deactivate current submodel's GPU subset
        submodel.deactivate_subset()

        # 3. pre-compute lr_per_col on main thread (avoids GIL in worker)
        lr_per_col = submodel._build_lr_per_col(iteration)

        # 4. record event on comm_stream for this block's D2H
        event = torch.cuda.Event()
        event.record(self._comm_stream)

        # 5. build closures
        adam_fn = self._make_adam_fn(submodel, submodel_id, cur_idx, cur_n_vis, lr_per_col, iteration)
        densify_fn = self._make_densify_stats_fn(submodel, submodel_id, pin_sub_vf, pin_sub_radii, pin_vpt_grad, iteration)

        # 6. submit to worker thread: wait D2H event → densify_stats → adam
        #    densify_stats must run before adam (adam's densify_and_prune may
        #    call pack_to_buffer which invalidates visible_indices)
        #    gpu_grad_ref is captured to keep the GPU tensor alive until D2H completes
        def _worker(gpu_ref=gpu_grad_ref):
            event.synchronize()  # wait for this block's D2H to complete
            del gpu_ref           # safe to release GPU memory now
            densify_fn()
            adam_fn()

        self._adam_futures.append(self._adam_executor.submit(_worker))

    def run_deferred_densify(self):
        """No-op in async mode. Densify_stats runs in worker thread."""
        pass

    def flush_last(self):
        """Iteration barrier: wait for all async adam tasks to complete.

        Must be called before next iteration's pre_gather, which reads _packed.
        """
        for fut in self._adam_futures:
            fut.result()
        self._adam_futures.clear()
        self.tracer.flush_gpu_events()
