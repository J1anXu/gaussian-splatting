import time
import json
import torch


class TraceManager:
    """Three-tier tracing → chrome://tracing JSON.

    Tiers:
        "main"   → 1-Main Thread   (CPU perf_counter, big pipeline stages)
        "worker" → 2-Work Thread   (CPU perf_counter, _make_pending tasks)
        "cuda"   → 3-CUDA Stream   (torch.cuda.Event, accurate GPU timing)

    Usage:
        tracer = TraceManager()
        tracer.set_iteration(iteration)

        # CPU-timed span (tier "main" or "worker")
        ev = tracer.begin("frustum_culling")
        ...
        tracer.end(ev)

        # CUDA-timed span (always tier "cuda")
        ev = tracer.begin_cuda("rendering", block_id=0)
        render(...)
        tracer.end_cuda(ev)

        tracer.export("trace.json")
    """

    _TID = {"main": "1-Main Thread", "worker": "2-Work Thread"}

    def __init__(self):
        self._events = []
        self._cuda_pending = []
        self._t0 = time.perf_counter_ns()
        self._iter = 0

    def set_iteration(self, iteration: int):
        self._iter = iteration

    def _us(self, ns: int) -> float:
        """Offset from anchor in microseconds."""
        return (ns - self._t0) / 1000.0

    # ── CPU-timed spans ─────────────────────────────────────
    def begin(self, name: str, tier: str = "main", **kwargs):
        kwargs["iteration"] = self._iter
        return {"name": name, "tier": tier, "ts": time.perf_counter_ns(), "args": kwargs}

    def end(self, handle: dict):
        dur = time.perf_counter_ns() - handle["ts"]
        self._events.append({
            "name": handle["name"],
            "ph": "X",
            "ts": self._us(handle["ts"]),
            "dur": dur / 1000.0,
            "pid": "Training",
            "tid": self._TID[handle["tier"]],
            "args": handle["args"],
        })

    # ── CUDA-timed spans (always "3-CUDA Stream") ──────────
    def begin_cuda(self, name: str, stream=None, **kwargs):
        kwargs["iteration"] = self._iter
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(stream)
        return {
            "name": name,
            "start": ev,
            "stream": stream,
            "wall_ns": time.perf_counter_ns(),
            "args": kwargs,
        }

    def end_cuda(self, handle: dict):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(handle.get("stream"))
        self._cuda_pending.append({
            "name": handle["name"],
            "start": handle["start"],
            "end": ev,
            "wall_ns": handle["wall_ns"],
            "args": handle["args"],
        })

    # ── export ──────────────────────────────────────────────
    def export(self, path: str = "trace.json"):
        torch.cuda.synchronize()
        for p in self._cuda_pending:
            dur_ms = p["start"].elapsed_time(p["end"])
            self._events.append({
                "name": p["name"],
                "ph": "X",
                "ts": self._us(p["wall_ns"]),
                "dur": dur_ms * 1000.0,   # ms → µs
                "pid": "Training",
                "tid": "3-CUDA Stream",
                "args": p["args"],
            })
        self._cuda_pending.clear()

        with open(path, "w") as f:
            json.dump({"traceEvents": self._events}, f)
        print(f"Trace exported → {path}  ({len(self._events)} events)")
