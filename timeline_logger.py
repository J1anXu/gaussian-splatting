"""
Pipeline Timeline Logger — Chrome Trace JSON + terminal text summary.

Usage:
    tl = TimelineLogger()
    tl.set_iteration(1001)
    with tl.scope("backward", tid="GPU", cat="gpu"):
        ...
    tl.async_begin("copy_grad.xyz", async_id="copy_b0_xyz")
    tl.async_end("copy_grad.xyz", async_id="copy_b0_xyz")
    tl.save_chrome_trace("timeline.json")
    tl.print_text_summary()
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TimelineEvent:
    name: str       # e.g. "backward", "copy_grad.xyz"
    cat: str        # "gpu", "cpu", "async_copy", "deferred"
    ph: str         # Chrome trace phase: "B"/"E", "b"/"e", "i"
    ts: float       # microseconds relative to _t0
    tid: str        # track: "GPU", "CPU", "CopyStream"
    pid: str        # "iter_{N}"
    args: dict = field(default_factory=dict)
    id: str = ""    # async correlation id


class TimelineLogger:
    def __init__(self, enabled: bool = True):
        self._events: List[TimelineEvent] = []
        self._t0: float = time.perf_counter()
        self.enabled: bool = enabled
        self._iteration: int = 0

    # ---- iteration control ----
    def set_iteration(self, iteration: int) -> None:
        self._iteration = iteration

    def _us(self) -> float:
        """Current timestamp in microseconds relative to _t0."""
        return (time.perf_counter() - self._t0) * 1e6
    def _pid(self) -> str:
        return f"iter_{self._iteration}"

    def _record(self, name: str, cat: str, ph: str, tid: str,
                async_id: str = "", **kwargs) -> None:
        if not self.enabled:
            return
        self._events.append(TimelineEvent(
            name=name, cat=cat, ph=ph, ts=self._us(),
            tid=tid, pid=self._pid(),
            args=kwargs, id=async_id,
        ))

    # ---- synchronous scope (B/E duration events) ----
    @contextmanager
    def scope(self, name: str, tid: str = "CPU", cat: str = "cpu", **kwargs):
        self._record(name, cat, "B", tid, **kwargs)
        try:
            yield
        finally:
            self._record(name, cat, "E", tid, **kwargs)

    # ---- async event pairs (b/e) ----
    def async_begin(self, name: str, async_id: str,
                    tid: str = "CopyStream", cat: str = "async_copy", **kwargs) -> None:
        self._record(name, cat, "b", tid, async_id=async_id, **kwargs)

    def async_end(self, name: str, async_id: str,
                  tid: str = "CopyStream", cat: str = "async_copy", **kwargs) -> None:
        self._record(name, cat, "e", tid, async_id=async_id, **kwargs)

    # ---- instant event (i) ----
    def instant(self, name: str, tid: str = "CPU", cat: str = "cpu", **kwargs) -> None:
        self._record(name, cat, "i", tid, **kwargs)

    # ---- output: Chrome Trace JSON ----
    def save_chrome_trace(self, path: str) -> None:
        trace = []
        for ev in self._events:
            entry = {
                "name": ev.name,
                "cat": ev.cat,
                "ph": ev.ph,
                "ts": ev.ts,
                "pid": ev.pid,
                "tid": ev.tid,
                "args": ev.args,
            }
            if ev.id:
                entry["id"] = ev.id
            if ev.ph == "i":
                entry["s"] = "t"  # thread-scoped instant
            trace.append(entry)
        with open(path, "w") as f:
            json.dump(trace, f)
        print(f"[TimelineLogger] Chrome trace saved to {path} ({len(trace)} events)")

    # ---- output: terminal text summary ----
    def print_text_summary(self, iteration: Optional[int] = None) -> None:
        """Print a compact ASCII timeline for one or all iterations."""
        by_iter: dict = {}
        for ev in self._events:
            by_iter.setdefault(ev.pid, []).append(ev)

        pids = sorted(by_iter.keys(), key=lambda p: int(p.split("_")[-1]))
        if iteration is not None:
            target = f"iter_{iteration}"
            pids = [p for p in pids if p == target]

        for pid in pids:
            self._print_iter_summary(pid, by_iter[pid])

    def _print_iter_summary(self, pid: str, events: List[TimelineEvent]) -> None:
        # collect duration spans from B/E pairs
        spans = []
        stack: dict = {}
        for ev in events:
            if ev.ph == "B":
                stack.setdefault(ev.tid, []).append(ev)
            elif ev.ph == "E":
                tid_stack = stack.get(ev.tid, [])
                if tid_stack:
                    b_ev = tid_stack.pop()
                    spans.append((b_ev.name, b_ev.tid, b_ev.cat,
                                  b_ev.ts, ev.ts, b_ev.args))

        # collect async spans from b/e pairs
        async_open: dict = {}
        async_spans = []
        for ev in events:
            if ev.ph == "b":
                async_open[ev.id] = ev
            elif ev.ph == "e" and ev.id in async_open:
                b_ev = async_open.pop(ev.id)
                async_spans.append((b_ev.name, b_ev.tid, b_ev.cat,
                                    b_ev.ts, ev.ts, b_ev.args))

        if not spans and not async_spans:
            return

        all_starts = [s[3] for s in spans] + [s[3] for s in async_spans]
        t0 = min(all_starts) if all_starts else 0.0

        iter_num = pid.split("_")[-1]
        print(f"\n=== Iteration {iter_num} ===")

        def block_key(args):
            return args.get("block_id", -1)

        all_spans = [(n, tid, cat, ts0, ts1, a, "sync")
                     for n, tid, cat, ts0, ts1, a in spans]
        all_spans += [(n, tid, cat, ts0, ts1, a, "async")
                      for n, tid, cat, ts0, ts1, a in async_spans]
        all_spans.sort(key=lambda x: x[3])

        current_block = None
        for name, tid, cat, ts_start, ts_end, args, kind in all_spans:
            bid = block_key(args)
            if bid != current_block:
                current_block = bid
                if bid >= 0:
                    print(f"  Block {bid}:")
            ms_s = (ts_start - t0) / 1000.0
            ms_e = (ts_end - t0) / 1000.0
            dur = ms_e - ms_s
            suffix = f"  ({tid}, {kind})" if kind == "async" else f"  ({tid})"
            print(f"    [{ms_s:7.1f} - {ms_e:7.1f} ms] ({dur:6.1f}ms) {name}{suffix}")

        # overlap detection: CPU deferred vs GPU work
        cpu_spans = [(n, ts0, ts1) for n, tid, cat, ts0, ts1, a, k in all_spans if tid == "CPU"]
        gpu_spans = [(n, ts0, ts1) for n, tid, cat, ts0, ts1, a, k in all_spans if tid == "GPU"]
        for cn, cs, ce in cpu_spans:
            for gn, gs, ge in gpu_spans:
                overlap_start = max(cs, gs)
                overlap_end = min(ce, ge)
                if overlap_end > overlap_start:
                    ov_ms = (overlap_end - overlap_start) / 1000.0
                    print(f"  Overlap: {cn}(CPU) overlapped {ov_ms:.1f}ms with {gn}(GPU)")
