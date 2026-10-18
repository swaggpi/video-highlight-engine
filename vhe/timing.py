from __future__ import annotations

import time
from contextlib import contextmanager


class Timings:
    """Wall-clock per stage, with an optional item count for ms-per-item. Stages are listed in the order they start.
    A part ("sub": True, e.g. the model calls inside a stage) is reported indented and not added to the total."""

    def __init__(self) -> None:
        self.stages: list[dict] = []

    @contextmanager
    def stage(self, name: str, items: int | None = None):
        start = time.perf_counter()
        record = {"stage": name, "seconds": 0.0, "items": items}
        self.stages.append(record)
        try:
            yield record
        finally:
            record["seconds"] = round(time.perf_counter() - start, 3)

    def add(self, name: str, seconds: float, sub: bool = True) -> None:
        """Add `seconds` to the part (or stage) `name`, created on first use."""
        for s in self.stages:
            if s["stage"] == name and bool(s.get("sub")) == sub:
                s["seconds"] = round(s["seconds"] + seconds, 3)
                return
        record = {"stage": name, "seconds": round(seconds, 3), "items": None}
        self.stages.append(record | {"sub": True} if sub else record)

    def report(self, video_seconds: float | None = None) -> str:
        return format_report(self.stages, video_seconds)


def format_report(stages: list[dict], video_seconds: float | None = None) -> str:
    lines = [f"{'stage':<22}{'seconds':>9}{'ms/item':>10}"]
    total = 0.0
    for s in stages:
        per = f"{1000 * s['seconds'] / s['items']:.1f}" if s.get("items") else "-"
        if s.get("sub"):
            lines.append(f"{'  ' + s['stage']:<22}{s['seconds']:>9.2f}{per:>10}")
            continue
        total += s["seconds"]
        lines.append(f"{s['stage']:<22}{s['seconds']:>9.2f}{per:>10}")
    lines.append(f"{'total':<22}{total:>9.2f}")
    if video_seconds:
        lines.append(f"{60 * total / video_seconds:.1f} min per hour of video "
                     f"({video_seconds / max(total, 1e-9):.2f}x realtime)")
    return "\n".join(lines)


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}h{s // 60 % 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Counter:
    """'label done/total (pct), rate, ETA' messages for long stages, at most one every `every_s` seconds."""

    def __init__(self, say, label: str, unit: str = "", every_s: float = 5.0) -> None:
        self.say, self.label, self.unit, self.every_s = say, label, unit, every_s
        self.start = time.perf_counter()
        self.last = -every_s

    def __call__(self, done: float, total: float) -> None:
        now = time.perf_counter()
        if done < total and now - self.last < self.every_s:
            return
        self.last = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        msg = f"{self.label} {done:.0f}/{total:.0f}{self.unit} ({100 * done / max(total, 1e-9):.0f}%)"
        if done < total and rate > 0:
            msg += f", {rate:.2f}{self.unit}/s, eta {hms(eta)}"
        self.say(msg)
