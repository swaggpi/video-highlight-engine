"""Hard-cut detection on the tiny full-frame-rate stream (adaptive threshold, flash suppression).

At 3 fps camera motion looks like a cut and cuts land up to 333 ms off, so this runs at the source rate.
TransNetV2 / OmniShotCut (MIT) are the planned upgrade for dissolves and fades.
"""
from __future__ import annotations

import numpy as np


def frame_diffs(tiny: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """d[k] = mean |frame_k - frame_{k-1}| in grey levels; d[0] = 0."""
    n = len(tiny)
    d = np.zeros(n, dtype=np.float32)
    for start in range(1, n, chunk):
        end = min(n, start + chunk)
        a = tiny[start - 1:end - 1].astype(np.int16)
        b = tiny[start:end].astype(np.int16)
        d[start:end] = np.abs(b - a).mean(axis=(1, 2))
    return d


def detect_cuts(tiny: np.ndarray, src_fps: float, ratio: float = 3.0, min_abs: float = 8.0,
                window: int = 8, min_shot_s: float = 0.6, d: np.ndarray | None = None) -> list[float]:
    """Return cut times in seconds (time of the first frame of the new shot)."""
    if d is None:
        d = frame_diffs(tiny)
    n = len(d)
    if n < 3:
        return []
    # Mean of the neighbourhood excluding the frame itself, via cumulative sums.
    c = np.concatenate([[0.0], np.cumsum(d, dtype=np.float64)])
    k = np.arange(n)
    lo = np.clip(k - window, 1, n)
    hi = np.clip(k + window + 1, 1, n)
    total = c[hi] - c[lo] - np.where((k >= lo) & (k < hi), d, 0.0)
    count = np.maximum((hi - lo) - ((k >= lo) & (k < hi)), 1)
    base = total / count
    candidates = np.flatnonzero((d >= min_abs) & (d / (base + 1.0) >= ratio))

    cuts: list[float] = []
    last = -np.inf
    for i in candidates:
        if i < 1:
            continue
        # A flash or strobe returns to the previous picture within two frames: not a cut.
        if i + 2 < n and np.abs(tiny[i + 2].astype(np.int16) - tiny[i - 1]).mean() < 0.4 * d[i]:
            continue
        # ...and the frame after a one-frame flash matches the frame before it.
        if i >= 2 and np.abs(tiny[i].astype(np.int16) - tiny[i - 2]).mean() < 0.4 * d[i]:
            continue
        t = i / src_fps
        if t - last >= min_shot_s:
            cuts.append(round(float(t), 3))
            last = t
    return cuts


def shots_from_cuts(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    edges = [0.0, *[c for c in cuts if 0.0 < c < duration], duration]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]
