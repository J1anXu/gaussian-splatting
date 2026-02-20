import time
import threading
import json

class TraceManager:
    def __init__(self):
        self.events = []
        self.pid = 1

    def begin(self, name, cat="cpu", block_id=None):
        return {
            "name": name,
            "cat": cat,
            "ts": time.time_ns() / 1000,
            "tid": threading.get_ident(),
            "block": block_id
        }

    def end(self, start_event):
        dur = time.time_ns() / 1000 - start_event["ts"]
        self.events.append({
            "name": start_event["name"],
            "cat": start_event["cat"],
            "ph": "X",
            "ts": start_event["ts"],
            "dur": dur,
            "pid": self.pid,
            "tid": start_event["tid"],
            "args": {"block": start_event["block"]}
        })

    def export(self, path="trace.json"):
        with open(path, "w") as f:
            json.dump({"traceEvents": self.events}, f)