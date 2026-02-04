import time
from collections import OrderedDict
from contextlib import contextmanager
class TimerManager:
    def __init__(self):
        # name -> dict
        self.records = OrderedDict()

    def register(self, name, comment=""):
        if name not in self.records:
            self.records[name] = {
                "time": 0.0,
                "comment": comment,
                "start": None,
            }

    def start(self, name):
        if name not in self.records:
            self.register(name)
        self.records[name]["start"] = time.perf_counter()

    def end(self, name):
        rec = self.records[name]
        assert rec["start"] is not None, f"Timer '{name}' was not started"
        rec["time"] += time.perf_counter() - rec["start"]
        rec["start"] = None

    @contextmanager
    def scope(self, name, comment=""):
        self.register(name, comment)
        self.start(name)
        try:
            yield
        finally:
            self.end(name)


    def summary(self, sort_by_time=True):
        lines = []
        total_sec = sum(r["time"] for r in self.records.values())
        total_ms = total_sec * 1000.0

        items = self.records.items()
        if sort_by_time:
            items = sorted(items, key=lambda x: x[1]["time"], reverse=True)

        lines.append("=" * 90)
        lines.append(f"{'Name':30s} {'Time(ms)':>12s} {'%':>8s}  Comment")
        lines.append("-" * 90)

        for name, r in items:
            t_ms = r["time"] * 1000.0
            pct = (r["time"] / total_sec * 100) if total_sec > 0 else 0.0
            lines.append(
                f"{name:30s} {t_ms:12.3f} {pct:7.2f}%  {r['comment']}"
            )

        lines.append("-" * 90)
        lines.append(f"{'TOTAL':30s} {total_ms:12.3f} 100.00%")
        lines.append("=" * 90)
        res = "\n".join(lines)
        print(res)
        return res


    def reset(self):
        for r in self.records.values():
            r["time"] = 0.0
            r["start"] = None
