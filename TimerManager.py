import time
import json
import torch


# Fixed tier → tid mapping (top to bottom in chrome://tracing)
_TIER_TID = {
    "main":   "1-Main",
    "block":  "2-Per-block",
    "worker": "3-Worker",
}
_CUDA_TID = "4-CUDA"


class TraceManager:
    """Fixed 4-row tracing → chrome://tracing JSON.

    Rows (top → bottom):
      1-Main       : high-level phases (frustum_culling, preparation, merge, traversal, flush_last)
      2-Per-block  : per-block leaf ops (subset_on, composition, loss, join_worker …)
      3-Worker     : background-thread work (scatter_grad, opt_step …)
      4-CUDA       : GPU-timed spans (rendering, backward, d2h_copy …)

    Colors are NOT specified — chrome assigns them automatically by event name,
    so same-name events share a color and different names get different colors.
    """

    def __init__(self):
        self._events = []
        self._cuda_pending = []
        self._t0 = time.perf_counter_ns()
        self._iter = 0

    def set_iteration(self, iteration: int):
        self._iter = iteration

    def _us(self, ns: int) -> float:
        return (ns - self._t0) / 1000.0

    # ── CPU-timed spans ─────────────────────────────────────
    def begin(self, name: str, tier: str = "main", **kwargs):
        kwargs.setdefault("iteration", self._iter)
        return {"name": name, "tier": tier, "ts": time.perf_counter_ns(), "args": kwargs}

    def end(self, handle: dict):
        dur = time.perf_counter_ns() - handle["ts"]
        tid = _TIER_TID.get(handle["tier"], _TIER_TID["main"])
        self._events.append({
            "name": handle["name"],
            "ph": "X",
            "ts": self._us(handle["ts"]),
            "dur": dur / 1000.0,
            "pid": "Training",
            "tid": tid,
            "args": handle["args"],
        })

    # ── CUDA-timed spans (always on 4-CUDA row) ────────────
    def begin_cuda(self, name: str, stream=None, **kwargs):
        kwargs.setdefault("iteration", self._iter)
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
                "dur": dur_ms * 1000.0,
                "pid": "Training",
                "tid": _CUDA_TID,
                "args": p["args"],
            })
        self._cuda_pending.clear()

        with open(path, "w") as f:
            json.dump({"traceEvents": self._events}, f)
        print(f"Trace exported → {path}  ({len(self._events)} events)")
