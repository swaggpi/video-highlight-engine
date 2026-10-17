"""Candidate clips from below the shot level: sub-shots (cuts + content change points), sliding windows of
several lengths, windows centred on signal peaks, and proposals from the highlight model. The query's length
limits shape which candidates exist, not just which survive."""
from __future__ import annotations

import numpy as np

from vhe.signals.shots import shots_from_cuts

LENGTHS = (3, 4, 6, 8, 10, 15, 20, 30, 45, 60, 90)


def peak_times(x: np.ndarray, fps: float, min_gap_s: float = 3.0, top: int = 60, pct: float = 80) -> list[float]:
    if len(x) == 0:
        return []
    thresh = np.percentile(x, pct)
    gap = max(1, int(min_gap_s * fps))
    out: list[int] = []
    for i in np.argsort(-x):
        if x[i] < thresh or len(out) >= top:
            break
        if all(abs(i - j) >= gap for j in out):
            out.append(int(i))
    return [i / fps for i in out]


def generate(duration: float, cuts: list[float], change_points: list[float], min_len: float, max_len: float,
             peaks: list[float] = (), proposals: list[tuple] = ()) -> list[tuple[float, float, str]]:
    lengths = [L for L in LENGTHS if min_len <= L <= max_len] or [float(np.clip(8.0, min_len, max_len))]
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float, str]] = []

    def add(s: float, e: float, src: str) -> None:
        s, e = max(0.0, s), min(duration, e)
        if e - s < min_len - 1e-6 or e - s > max_len + 1e-6:
            return
        key = (round(s * 2) / 2, round(e * 2) / 2)
        if key not in seen:
            seen.add(key)
            out.append((float(s), float(e), src))

    for a, b in shots_from_cuts(sorted(set(cuts) | set(change_points)), duration):
        add(a, b, "shot")
        for L in lengths:
            if L <= b - a:
                for s in np.arange(a, b - L + 1e-6, max(1.0, L / 2)):
                    add(s, s + L, "window")
    # Windows across cuts cover runs of very short shots (edited footage); scoring penalises each cut inside.
    for L in lengths:
        for s in np.arange(0.0, max(duration - L, 0.0) + 1e-6, max(1.0, L / 2)):
            add(s, s + L, "span")
    for p in peaks:
        for L in lengths:
            add(min(max(0.0, p - L / 2), max(0.0, duration - L)), min(max(0.0, p - L / 2), max(0.0, duration - L)) + L,
                "peak")
    for s, e, *_ in proposals:
        c, L = (s + e) / 2, float(np.clip(e - s, min_len, max_len))
        start = min(max(0.0, c - L / 2), max(0.0, duration - L))
        add(start, start + L, "model")
    return out


def cuts_inside(s: float, e: float, cuts: np.ndarray, margin: float = 0.3) -> int:
    return int(np.count_nonzero((cuts > s + margin) & (cuts < e - margin)))
