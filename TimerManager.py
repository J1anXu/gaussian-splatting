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
PID_CPU = 1
PID_GPU = 2

TID_MAIN     = 1
TID_PIPELINE = 2
TID_ADAM     = 3


class TraceManager:
    """
    Chrome Trace Event Format logger (chrome://tracing / Perfetto UI).

    - Zero overhead when disabled (enabled=False)
    - GPU spans use torch.cuda.Event for accurate device-side timing
    - CPU spans use time.time_ns() (~20 ns overhead each)
    - Records every iteration when enabled
    - Thread-safe

    Typical usage in train loop:
        tm = TraceManager(enabled=config.TIMELINE)
        for iteration in range(...):
            tm.step(iteration)
            with tm.span("frustum_culling"):
                ...
            with tm.gpu_span("render_nograd", block_id=sub_id):
                ...
            tm.counter("pts", {"visible": n_vis, "total": n_total})
        tm.export("timeline/trace.json")
    """

    def __init__(self, enabled: bool = True):
        self.enabled      = enabled
        self._iter        = 0
        self._active      = enabled
        self._lock        = threading.Lock()
        self.events       = []
        self._gpu_pending = []   # (e_start, e_end, meta)

    # ── iteration gate ────────────────────────────────────────────────────────

    def step(self, iteration: int):
        self._iter   = iteration
        self._active = self.enabled

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

    # ── GPU span ──────────────────────────────────────────────────────────────

    @contextmanager
    def gpu_span(self, name: str, block_id=None, **kwargs):
        """
        Wraps CUDA work with torch.cuda.Event markers.
        GPU duration is resolved lazily in flush_gpu_events() after synchronize.
        The event is placed at the CPU timestamp of the record() call so it
        lines up visually with the CPU timeline.
        """
        if not self._active or not _TORCH_AVAILABLE:
            yield
            return
        e_start = torch.cuda.Event(enable_timing=True)
        e_end   = torch.cuda.Event(enable_timing=True)
        cpu_ts  = time.time_ns() // 1000
        e_start.record()
        yield
        e_end.record()
        meta = {"name": name, "block_id": block_id,
                "cpu_ts": cpu_ts, "iter": self._iter, **kwargs}
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
        Resolve pending GPU events. Call after torch.cuda.synchronize().
        Converts elapsed_time (ms) -> microseconds for chrome trace.
        """
        if not self._gpu_pending:
            return
        resolved = []
        for e_start, e_end, meta in self._gpu_pending:
            try:
                gpu_us = int(e_start.elapsed_time(e_end) * 1000)
                ev = {
                    "name": meta["name"], "ph": "X",
                    "ts":  meta["cpu_ts"],
                    "dur": max(gpu_us, 1),
                    "pid": PID_GPU, "tid": 1,
                    "args": {k: v for k, v in meta.items()
                             if k not in ("name", "cpu_ts")},
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
             "args": {"name": "GPU (via CUDA Event)"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_CPU, "tid": TID_MAIN,
             "args": {"name": "main_loop"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_CPU, "tid": TID_PIPELINE,
             "args": {"name": "pipeline_grad_sync"}},
            {"name": "thread_name",  "ph": "M", "pid": PID_CPU, "tid": TID_ADAM,
             "args": {"name": "adam_step"}},
        ]
        payload = {"traceEvents": meta_events + self.events}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"[TraceManager] {len(self.events)} events -> {path}")
