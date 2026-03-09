import time
import threading
import json
import os
from contextlib import contextmanager

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── Process / thread IDs for chrome://tracing lanes ──────────────────────────
PID_GPU      = 1
PID_TRANSFER = 2
PID_CPU      = 3

TID_MAIN     = 1
TID_PIPELINE = 2
TID_ADAM     = 3


class TraceManager:
    """
    Chrome Trace Event Format logger (chrome://tracing / Perfetto UI).

    Three process lanes:
      PID_CPU      – CPU work (span)
      PID_GPU      – GPU kernels (gpu_span), positioned using CUDA event relative timing
      PID_TRANSFER – H2D / D2H data transfers (transfer_span)

    GPU and transfer events are placed on the timeline using a per-iteration
    reference CUDA event so their absolute position reflects real GPU timing,
    not the CPU dispatch timestamp.
    """

    def __init__(self, enabled: bool = True):
        self.enabled      = enabled
        self._iter        = 0
        self._active      = enabled
        self._lock        = threading.Lock()
        self.events       = []
        self._gpu_pending = []   # (e_start, e_end, meta)
        # per-iteration reference point for GPU absolute timing
        self._ref_event   = None
        self._ref_cpu_ts  = None  # CPU timestamp (us) when ref was recorded

    # ── iteration gate ────────────────────────────────────────────────────────

    def step(self, iteration: int):
        self._iter   = iteration
        self._active = self.enabled
        if self._active and _TORCH_AVAILABLE:
            self._ref_event = torch.cuda.Event(enable_timing=True)
            self._ref_event.record()
            self._ref_cpu_ts = time.time_ns() // 1000

    @property
    def active(self) -> bool:
        return self._active

    # ── CPU span ──────────────────────────────────────────────────────────────

    @contextmanager
    def span(self, name: str, pid: int = PID_CPU, tid: int = TID_MAIN,
             block_id=None, **kwargs):
        if not self._active:
            yield
            return
        ts = time.time_ns() // 1000
        yield
        dur = time.time_ns() // 1000 - ts
        ev = {
            "name": name, "ph": "X",
            "ts": ts, "dur": max(dur, 1),
            "pid": pid, "tid": tid,
            "args": {"block": block_id, "iter": self._iter, **kwargs},
        }
        with self._lock:
            self.events.append(ev)

    # ── GPU span (accurate device-side timing) ────────────────────────────────

    @contextmanager
    def gpu_span(self, name: str, block_id=None, tid: int = 1, **kwargs):
        """
        Wraps CUDA work with torch.cuda.Event markers.
        GPU start time and duration are resolved in flush_gpu_events() using
        the per-iteration reference event for accurate positioning.
        """
        if not self._active or not _TORCH_AVAILABLE:
            yield
            return
        e_start = torch.cuda.Event(enable_timing=True)
        e_end   = torch.cuda.Event(enable_timing=True)
        e_start.record()
        yield
        e_end.record()
        meta = {"name": name, "pid": PID_GPU, "tid": tid,
                "block_id": block_id, "iter": self._iter,
                "ref_event": self._ref_event, "ref_cpu_ts": self._ref_cpu_ts,
                **kwargs}
        with self._lock:
            self._gpu_pending.append((e_start, e_end, meta))

    # ── Transfer span (H2D / D2H, accurate device-side timing) ───────────────

    @contextmanager
    def transfer_span(self, name: str, block_id=None, tid: int = 1, **kwargs):
        """
        Like gpu_span but placed on the PID_TRANSFER lane.
        Use for H2D / D2H data transfers.
        """
        if not self._active or not _TORCH_AVAILABLE:
            yield
            return
        e_start = torch.cuda.Event(enable_timing=True)
        e_end   = torch.cuda.Event(enable_timing=True)
        e_start.record()
        yield
        e_end.record()
        meta = {"name": name, "pid": PID_TRANSFER, "tid": tid,
                "block_id": block_id, "iter": self._iter,
                "ref_event": self._ref_event, "ref_cpu_ts": self._ref_cpu_ts,
                **kwargs}
        with self._lock:
            self._gpu_pending.append((e_start, e_end, meta))

    # ── instant marker ────────────────────────────────────────────────────────

    def mark(self, name: str, pid: int = PID_CPU, tid: int = TID_MAIN,
             block_id=None, **kwargs):
        if not self._active:
            return
        ev = {
            "name": name, "ph": "i", "s": "g",
            "ts": time.time_ns() // 1000,
            "pid": pid, "tid": tid,
            "args": {"block": block_id, "iter": self._iter, **kwargs},
        }
        with self._lock:
            self.events.append(ev)

    # ── counter (shows as graph) ──────────────────────────────────────────────

    def counter(self, name: str, values: dict,
                pid: int = PID_CPU, tid: int = TID_MAIN):
        if not self._active:
            return
        ev = {
            "name": name, "ph": "C",
            "ts": time.time_ns() // 1000,
            "pid": pid, "tid": tid,
            "args": values,
        }
        with self._lock:
            self.events.append(ev)

    # ── GPU event resolution ──────────────────────────────────────────────────

    def flush_gpu_events(self):
        """
        Resolve pending GPU/transfer events. Call after torch.cuda.synchronize().
        Uses the per-iteration reference CUDA event to compute absolute GPU
        timestamps: ts = ref_cpu_ts + ref.elapsed_time(e_start).
        """
        if not self._gpu_pending:
            return
        resolved = []
        for e_start, e_end, meta in self._gpu_pending:
            try:
                gpu_dur_us = int(e_start.elapsed_time(e_end) * 1000)
                ref_event = meta.pop("ref_event", None)
                ref_cpu_ts = meta.pop("ref_cpu_ts", None)
                if ref_event is not None and ref_cpu_ts is not None:
                    # offset from iteration reference → absolute GPU start
                    offset_us = int(ref_event.elapsed_time(e_start) * 1000)
                    gpu_ts = ref_cpu_ts + offset_us
                else:
                    # fallback: use CPU dispatch time (less accurate)
                    gpu_ts = meta.pop("cpu_ts", time.time_ns() // 1000)

                pid = meta.pop("pid", PID_GPU)
                tid = meta.pop("tid", 1)
                name = meta.pop("name")
                ev = {
                    "name": name, "ph": "X",
                    "ts":  gpu_ts,
                    "dur": max(gpu_dur_us, 1),
                    "pid": pid, "tid": tid,
                    "args": {k: v for k, v in meta.items()
                             if k not in ("cpu_ts",)},
                }
                resolved.append(ev)
            except Exception:
                pass
        with self._lock:
            self.events.extend(resolved)
            self._gpu_pending.clear()

    # ── export ────────────────────────────────────────────────────────────────

    def export(self, path: str = "timeline/trace.json"):
        if not self.events and not self._gpu_pending:
            print("[TraceManager] no events recorded, skipping export")
            return
        if _TORCH_AVAILABLE:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.flush_gpu_events()

        meta_events = [
            {"name": "process_name", "ph": "M", "pid": PID_CPU, "tid": 0,
             "args": {"name": "CPU"}},
            {"name": "process_name", "ph": "M", "pid": PID_GPU, "tid": 0,
             "args": {"name": "GPU"}},
            {"name": "process_name", "ph": "M", "pid": PID_TRANSFER, "tid": 0,
             "args": {"name": "Transfer (H2D/D2H)"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_CPU, "tid": TID_MAIN,
             "args": {"name": "main_loop"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_CPU, "tid": TID_PIPELINE,
             "args": {"name": "pipeline_grad_sync"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_CPU, "tid": TID_ADAM,
             "args": {"name": "adam_step"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_GPU, "tid": 1,
             "args": {"name": "stream_0"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_TRANSFER, "tid": 1,
             "args": {"name": "pcie"}},
        ]
        payload = {"traceEvents": meta_events + self.events}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"[TraceManager] {len(self.events)} events -> {path}")
