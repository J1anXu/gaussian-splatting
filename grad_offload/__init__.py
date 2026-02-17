import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class PendingCopy:
    name: str
    idx: torch.Tensor               # CPU LongTensor indices
    cpu_param: torch.Tensor         # full CPU param
    grad_buf: torch.Tensor          # pinned CPU buffer (shape = cpu_param[idx].shape, dtype=float32)
    ready_event: torch.cuda.Event   # recorded on copy stream when copy enqueued
    half_buf: Optional[torch.Tensor] = None  # optional pinned fp16 buffer if using HALF


class AsyncGradScatter:
    """
    Pipeline:
      - enqueue(gpu_param.grad -> pinned grad buffer) in copy_stream (async)
      - later apply() will wait on event (per copy) then do cpu_param.grad[idx] += grad_buf (CPU)
    """

    def __init__(self, use_half: bool = False, device: str = "cuda"):
        self.use_half = use_half
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)
        # (name, idx_len, slice_shape, dtype) -> pinned buffers reused
        self._buf_cache: Dict[Tuple[str, int, Tuple[int, ...], torch.dtype], Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        self._pending: list[PendingCopy] = []

    @staticmethod
    def _to_cpu_long(idx) -> torch.Tensor:
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long)
        idx = idx.to("cpu", non_blocking=False)
        if idx.dtype != torch.long:
            idx = idx.long()
        return idx

    def _get_pinned_buffers(self, name: str, cpu_slice: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        cpu_slice: cpu_param[idx] view/copy result shape (no grad)
        Returns:
          grad_buf_fp32 pinned
          optionally half_buf_fp16 pinned if use_half
        """
        idx_len = cpu_slice.shape[0]
        slice_shape = tuple(cpu_slice.shape)
        key = (name, idx_len, slice_shape, torch.float32)

        if key in self._buf_cache:
            return self._buf_cache[key]

        # Always allocate fp32 pinned buffer for applying to cpu_param.grad
        grad_buf = torch.empty(slice_shape, dtype=torch.float32, device="cpu", pin_memory=True)

        half_buf = None
        if self.use_half:
            half_buf = torch.empty(slice_shape, dtype=torch.float16, device="cpu", pin_memory=True)

        self._buf_cache[key] = (grad_buf, half_buf)
        return grad_buf, half_buf

    @torch.no_grad()
    def enqueue(self, name: str, cpu_param: torch.Tensor, gpu_param: torch.Tensor, idx,
                timeline=None, block_id: int = -1) -> None:
        """
        Enqueue async D2H copy of gpu_param.grad for a subset indexed by idx (CPU long tensor indices).
        Does NOT touch cpu_param.grad yet.
        """
        grad_gpu = gpu_param.grad
        if grad_gpu is None:
            return

        idx = self._to_cpu_long(idx)

        # Ensure cpu_param.grad exists
        if cpu_param.grad is None:
            cpu_param.grad = torch.zeros_like(cpu_param, device="cpu")

        # Slice shape for buffer allocation
        cpu_slice = cpu_param[idx]  # CPU tensor view/copy of subset (shape reference)
        grad_buf_fp32, half_buf_fp16 = self._get_pinned_buffers(name, cpu_slice)

        # Record event after enqueue so CPU can later wait efficiently
        evt = torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)

        # Put async ops on a dedicated copy stream
        with torch.cuda.stream(self.copy_stream):
            # Important: wait for grad_gpu to be produced on current stream (usually default)
            # This ties copy_stream to the stream that computed grad_gpu without a full device sync.
            self.copy_stream.wait_stream(torch.cuda.current_stream(device=self.device))

            if self.use_half:
                # 1) cast grad on GPU (kernel) then 2) copy to pinned fp16 host buffer async
                grad_half_gpu = grad_gpu.detach().to(torch.float16)
                # Copy to pinned fp16 host buffer
                half_buf_fp16.copy_(grad_half_gpu, non_blocking=True)
            else:
                # Direct copy to pinned fp32 host buffer (dtype must match)
                # If grad_gpu isn't fp32, cast on GPU first to fp32 to keep copy dtype consistent.
                if grad_gpu.dtype != torch.float32:
                    grad_fp32_gpu = grad_gpu.detach().to(torch.float32)
                    grad_buf_fp32.copy_(grad_fp32_gpu, non_blocking=True)
                else:
                    grad_buf_fp32.copy_(grad_gpu.detach(), non_blocking=True)

            # Mark: all queued on copy_stream up to here
            evt.record(self.copy_stream)

        self._pending.append(PendingCopy(
            name=name,
            idx=idx,
            cpu_param=cpu_param,
            grad_buf=grad_buf_fp32,
            half_buf=half_buf_fp16,
            ready_event=evt
        ))

        if timeline is not None:
            async_id = f"copy_b{block_id}_{name}"
            timeline.async_begin(f"copy_grad.{name}", async_id=async_id,
                                 block_id=block_id, param=name)

    @torch.no_grad()
    def apply_ready(self, max_items: Optional[int] = None, timeline=None, block_id: int = -1) -> int:
        """
        Apply any copies that are already done (non-blocking where possible).
        Returns number applied.
        """
        applied = 0
        remaining = []
        for item in self._pending:
            if item.ready_event.query():  # does not block
                self._apply_one(item, timeline=timeline, block_id=block_id)
                applied += 1
                if max_items is not None and applied >= max_items:
                    # keep rest
                    remaining.extend(self._pending[self._pending.index(item)+1:])
                    break
            else:
                remaining.append(item)
        self._pending = remaining
        return applied

    @torch.no_grad()
    def flush(self, timeline=None, block_id: int = -1) -> None:
        """
        Block until all pending copies are done, then apply all.
        """
        for item in self._pending:
            item.ready_event.synchronize()  # blocks until this copy finished
            self._apply_one(item, timeline=timeline, block_id=block_id)
        self._pending.clear()

    @torch.no_grad()
    def flush_profiled(self, timeline=None, block_id: int = -1):
        """
        Like flush(), but returns per-attribute timing breakdown and emits
        fine-grained timeline events.
        """
        import time
        results = []
        for item in self._pending:
            was_ready = item.ready_event.query()

            if timeline:
                timeline._record(f"d2h_wait.{item.name}", "d2h", "B", "CPU_profile",
                                 block_id=block_id, was_ready=was_ready)
            t0 = time.perf_counter()
            if not was_ready:
                item.ready_event.synchronize()
            t1 = time.perf_counter()
            if timeline:
                timeline._record(f"d2h_wait.{item.name}", "d2h", "E", "CPU_profile",
                                 block_id=block_id)

            if timeline:
                timeline._record(f"d2h_apply.{item.name}", "d2h", "B", "CPU_profile",
                                 block_id=block_id)
            self._apply_one(item, timeline=timeline, block_id=block_id)
            t2 = time.perf_counter()
            if timeline:
                timeline._record(f"d2h_apply.{item.name}", "d2h", "E", "CPU_profile",
                                 block_id=block_id)

            results.append({
                "name": item.name,
                "wait_us": (t1 - t0) * 1e6,
                "apply_us": (t2 - t1) * 1e6,
                "was_ready": was_ready,
            })
        self._pending.clear()
        return results

    @torch.no_grad()
    def _apply_one(self, item: PendingCopy, timeline=None, block_id: int = -1) -> None:
        """
        CPU-side scatter: directly assign grad (no accumulation needed,
        each submodel gets exactly one backward per iteration).
        """
        if self.use_half:
            grad_cpu_fp32 = item.half_buf.to(torch.float32)
            item.cpu_param.grad[item.idx] = grad_cpu_fp32
        else:
            item.cpu_param.grad[item.idx] = item.grad_buf

        if timeline is not None:
            async_id = f"copy_b{block_id}_{item.name}"
            timeline.async_end(f"copy_grad.{item.name}", async_id=async_id,
                               block_id=block_id, param=item.name)
