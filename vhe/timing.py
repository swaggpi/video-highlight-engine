from __future__ import annotations

import time
from contextlib import contextmanager


class Timings:
    """Wall-clock per stage, with an optional item count for ms-per-item."""

    def __init__(self) -> None:
        self.stages: list[dict] = []

    @contextmanager
    def stage(self, name: str, items: int | None = None):
        start = time.perf_counter()
        record = {"stage": name, "seconds": 0.0, "items": items}
        try:
            yield record
        finally:
            record["seconds"] = round(time.perf_counter() - start, 3)
            self.stages.append(record)

    def report(self, video_seconds: float | None = None) -> str:
        return format_report(self.stages, video_seconds)


def format_report(stages: list[dict], video_seconds: float | None = None) -> str:
    lines = [f"{'stage':<22}{'seconds':>9}{'ms/item':>10}"]
    total = 0.0
    for s in stages:
        total += s["seconds"]
        per = f"{1000 * s['seconds'] / s['items']:.1f}" if s.get("items") else "-"
        lines.append(f"{s['stage']:<22}{s['seconds']:>9.2f}{per:>10}")
    lines.append(f"{'total':<22}{total:>9.2f}")
    if video_seconds:
        lines.append(f"{60 * total / video_seconds:.1f} min per hour of video "
                     f"({video_seconds / max(total, 1e-9):.2f}x realtime)")
    return "\n".join(lines)
