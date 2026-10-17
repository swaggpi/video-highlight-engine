"""Cut refinement: move clip edges to hard cuts, speech pauses or word boundaries, and away from shaky
frames, then enforce the length limits. A clip should never start or end mid-word."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_GUARD = 0.04  # stay one frame clear of a cut


@dataclass
class Context:
    duration: float
    cuts: list[float]
    vad: list[tuple[float, float]]
    words: list[dict]
    shake: np.ndarray | None = None
    fps: float = 3.0


def _nearest(points: list[float], t: float, radius: float) -> float | None:
    best = None
    for p in points:
        if abs(p - t) <= radius and (best is None or abs(p - t) < abs(best - t)):
            best = p
    return best


def _word_at(words: list[dict], t: float) -> dict | None:
    for w in words:
        if w["start"] < t < w["end"]:
            return w
    return None


def _in_speech(vad: list[tuple[float, float]], t: float) -> tuple[float, float] | None:
    for s, e in vad:
        if s < t < e:
            return s, e
    return None


def _steadier(ctx: Context, t: float, radius: float = 0.5) -> float:
    if ctx.shake is None or len(ctx.shake) == 0:
        return t
    i = int(round(t * ctx.fps))
    r = max(1, int(radius * ctx.fps))
    lo, hi = max(0, i - r), min(len(ctx.shake), i + r + 1)
    if lo >= hi:
        return t
    window = ctx.shake[lo:hi]
    if ctx.shake[min(max(i, 0), len(ctx.shake) - 1)] <= np.percentile(ctx.shake, 75):
        return t
    return (lo + int(np.argmin(window))) / ctx.fps


def snap_start(t: float, ctx: Context, radius: float = 1.0, pad: float = 0.2) -> float:
    cut = _nearest(ctx.cuts, t, radius)
    if cut is not None:
        return cut + FRAME_GUARD
    w = _word_at(ctx.words, t)
    if w is not None:
        return max(0.0, w["start"] - 0.1)
    seg = _in_speech(ctx.vad, t)
    if seg is not None and seg[0] >= t - radius:
        return max(0.0, seg[0] - 0.15)
    return max(0.0, _steadier(ctx, t) - pad)


def snap_end(t: float, ctx: Context, radius: float = 1.0, pad: float = 0.2) -> float:
    cut = _nearest(ctx.cuts, t, radius)
    if cut is not None:
        return cut - FRAME_GUARD
    w = _word_at(ctx.words, t)
    if w is not None:
        return min(ctx.duration, w["end"] + 0.1)
    seg = _in_speech(ctx.vad, t)
    if seg is not None and seg[1] <= t + radius:
        return min(ctx.duration, seg[1] + 0.15)
    return min(ctx.duration, _steadier(ctx, t) + pad)


def refine(start: float, end: float, ctx: Context, min_len: float, max_len: float) -> tuple[float, float]:
    s, e = snap_start(start, ctx), snap_end(end, ctx)
    if e - s < min_len:
        e = min(ctx.duration, s + min_len)
        s = max(0.0, min(s, e - min_len))
    if e - s > max_len:
        e = s + max_len
        w = _word_at(ctx.words, e)
        if w is not None and w["start"] - s >= min_len:
            e = w["start"] - 0.05
    return round(max(0.0, s), 3), round(min(ctx.duration, e), 3)
