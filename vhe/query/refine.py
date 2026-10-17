"""Cut refinement: move clip edges to hard cuts, speech pauses or word boundaries, and away from shaky
frames, then enforce the length limits. A clip should never start or end mid-word."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

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
    _starts: list[float] = field(init=False, repr=False)
    _free_cuts: list[float] = field(init=False, repr=False)
    _shaky: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.words = sorted(self.words, key=lambda w: w["start"])
        self._starts = [w["start"] for w in self.words]
        # A cut detected inside a word would cut the word: only cuts between words attract clip edges.
        self._free_cuts = [c for c in self.cuts if _word_at(self, c) is None]
        self._shaky = np.percentile(self.shake, 75) if self.shake is not None and len(self.shake) else np.inf


def _nearest(points: list[float], t: float, radius: float) -> float | None:
    best = None
    for p in points:
        if abs(p - t) <= radius and (best is None or abs(p - t) < abs(best - t)):
            best = p
    return best


def _word_index(ctx: Context, t: float) -> int:
    """Index of the last word starting at or before t (-1 if none); words are sorted by start."""
    return bisect.bisect_right(ctx._starts, t) - 1


def _word_at(ctx: Context, t: float) -> dict | None:
    i = _word_index(ctx, t)
    return ctx.words[i] if i >= 0 and ctx.words[i]["start"] < t < ctx.words[i]["end"] else None


def _before_word(ctx: Context, w: dict, pad: float) -> float:
    """Just before word w, but not inside the previous word."""
    i = _word_index(ctx, w["start"] - 1e-9)
    prev_end = ctx.words[i]["end"] if i >= 0 else 0.0
    return max(0.0, prev_end, w["start"] - pad) if prev_end <= w["start"] else max(0.0, w["start"] - pad)


def _after_word(ctx: Context, w: dict, pad: float) -> float:
    """Just after word w, but not inside the next word."""
    i = _word_index(ctx, w["start"])
    nxt = ctx.words[i + 1]["start"] if 0 <= i + 1 < len(ctx.words) and ctx.words[i + 1]["start"] >= w["end"] \
        else ctx.duration
    return min(ctx.duration, nxt, w["end"] + pad)


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
    if ctx.shake[min(max(i, 0), len(ctx.shake) - 1)] <= ctx._shaky:
        return t
    return (lo + int(np.argmin(window))) / ctx.fps


def snap_start(t: float, ctx: Context, radius: float = 1.0, pad: float = 0.2) -> float:
    cut = _nearest(ctx._free_cuts, t, radius)
    if cut is not None:
        return cut + FRAME_GUARD
    w = _word_at(ctx, t)
    if w is not None:
        return _before_word(ctx, w, 0.1)
    seg = _in_speech(ctx.vad, t)
    if seg is not None and seg[0] >= t - radius:
        return max(0.0, seg[0] - 0.15)
    return max(0.0, _steadier(ctx, t) - pad)


def snap_end(t: float, ctx: Context, radius: float = 1.0, pad: float = 0.2) -> float:
    cut = _nearest(ctx._free_cuts, t, radius)
    if cut is not None:
        return cut - FRAME_GUARD
    w = _word_at(ctx, t)
    if w is not None:
        return _after_word(ctx, w, 0.1)
    seg = _in_speech(ctx.vad, t)
    if seg is not None and seg[1] <= t + radius:
        return min(ctx.duration, seg[1] + 0.15)
    return min(ctx.duration, _steadier(ctx, t) + pad)


def refine(start: float, end: float, ctx: Context, min_len: float, max_len: float) -> tuple[float, float]:
    s, e = snap_start(start, ctx), snap_end(end, ctx)
    if e - s < min_len:
        # Grow towards the end first, without crossing the next cut; then backwards to the previous cut;
        # only if the shot itself is too short, ignore cuts.
        next_cut = min([c for c in ctx.cuts if c > e] + [ctx.duration + FRAME_GUARD]) - FRAME_GUARD
        e = min(next_cut, s + min_len, ctx.duration)
        if e - s < min_len:
            prev_cut = max([c for c in ctx.cuts if c < s - FRAME_GUARD] + [-FRAME_GUARD]) + FRAME_GUARD
            s = max(prev_cut, e - min_len, 0.0)
        if e - s < min_len:
            e = min(ctx.duration, s + min_len)
            s = max(0.0, min(s, e - min_len))
    if e - s > max_len:
        e = s + max_len
        # The trim must not leave a few frames of the next shot: end before a cut just inside the clip.
        cut = max([c for c in ctx.cuts if s + min_len < c - FRAME_GUARD and c < e], default=None)
        if cut is not None and e - cut < 1.0:
            e = cut - FRAME_GUARD
    # Never start or end inside a word: start before it; end after it, or before it if that would overrun max_len.
    w = _word_at(ctx, s)
    if w is not None:
        s = _before_word(ctx, w, 0.05)
    w = _word_at(ctx, e)
    if w is not None:
        after, before = _after_word(ctx, w, 0.05), _before_word(ctx, w, 0.05)
        e = after if after - s <= max_len or before - s < min_len else before
    return round(max(0.0, s), 3), round(min(ctx.duration, e), 3)
