"""Cut refinement: move clip edges to hard cuts, speech pauses or word boundaries, and away from shaky
frames, then enforce the length limits. A clip should never start or end mid-word; a visual clip should start and
end on steady, usable frames (not inside a whip-pan, on a black or blocked frame); a speech clip starts where a
sentence starts and ends where one ends."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from vhe.query.candidates import sentence_spans
from vhe.signals.quality import REASONS

FRAME_GUARD = 0.04  # stay one frame clear of a cut
# Clean edges: the first and last EDGE_STEPS grid steps (2 at 3 fps: 0.67 s) of a visual clip must be usable, not
# bad framing, and not mid-swing. Calibrated on the irl-event judge notes (2026-10-18): steady holds have cam_speed
# ~0.000-0.005 and shake <0.01, deliberate follow pans 0.02-0.06, the whip-pans the judge flagged 0.16-2.6 with shake
# 0.10-0.6 (pooled 90th percentiles: cam_speed 0.08, shake 0.10).
EDGE_STEPS = 2
EDGE_MAX_SPEED = 0.1
EDGE_MAX_SHAKE = 0.1
EDGE_SEARCH_S = 1.5  # how far outward an edge may move to find clean frames when trimming inward leaves too little
SPEECH_PAD = 0.15  # lead-in/out around a sentence, never into the neighbouring word


@dataclass
class Context:
    duration: float
    cuts: list[float]
    vad: list[tuple[float, float]]
    words: list[dict]
    shake: np.ndarray | None = None
    fps: float = 3.0
    clean: np.ndarray | None = None  # per grid step: an acceptable clip edge (clean_steps); None = not checked
    _starts: list[float] = field(init=False, repr=False)
    _free_cuts: list[float] = field(init=False, repr=False)
    _shaky: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.words = sorted(self.words, key=lambda w: w["start"])
        self._starts = [w["start"] for w in self.words]
        # A cut detected inside a word would cut the word: only cuts between words attract clip edges.
        self._free_cuts = [c for c in self.cuts if _word_at(self, c) is None]
        self._shaky = np.percentile(self.shake, 75) if self.shake is not None and len(self.shake) else np.inf
        # sentence of each word, and each sentence's first / last word
        self._sentences = sentence_spans(self.words)
        self._sent_of = np.zeros(len(self.words), dtype=int)
        for k, (a, b) in enumerate(self._sentences):
            self._sent_of[a:b + 1] = k


def clean_steps(grid: dict, n: int | None = None) -> np.ndarray:
    """Per grid step: fit to start or end a visual clip on. Usable (not covered, black, blown, blurry, very shaky or
    obstructed), not pointed at the ground or sky, and the camera not swinging (cam_speed, shake)."""
    n = len(grid["t"]) if n is None else n
    usable = np.asarray(grid["usable"], dtype=bool) if "usable" in grid else np.ones(n, dtype=bool)
    ok = usable.copy()
    if "unusable" in grid:
        ok &= (np.asarray(grid["unusable"], dtype=np.uint8) & np.uint8(1 << REASONS.index("bad_framing"))) == 0
    for key, limit in (("cam_speed", EDGE_MAX_SPEED), ("shake", EDGE_MAX_SHAKE)):
        if key in grid:
            ok &= np.nan_to_num(np.asarray(grid[key], dtype=float), nan=0.0) <= limit
    return ok


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
    if ctx.clean is not None:
        s, e = trim_edges(s, e, ctx, min_len, max_len)
    # Never start or end inside a word: start before it; end after it, or before it if that would overrun max_len.
    w = _word_at(ctx, s)
    if w is not None:
        s = _before_word(ctx, w, 0.05)
    w = _word_at(ctx, e)
    if w is not None:
        after, before = _after_word(ctx, w, 0.05), _before_word(ctx, w, 0.05)
        e = after if after - s <= max_len or before - s < min_len else before
    return round(max(0.0, s), 3), round(min(ctx.duration, e), 3)


# ---- clean visual edges ----------------------------------------------------------------------------------------------
def _edge_ok(ctx: Context, i: int, end: bool = False) -> bool:
    """The EDGE_STEPS steps from step i on (or, for an end, before step i) are clean."""
    a, b = (i - EDGE_STEPS, i) if end else (i, i + EDGE_STEPS)
    return 0 <= a and b <= len(ctx.clean) and bool(ctx.clean[a:b].all())


def edges_clean(s: float, e: float, ctx: Context) -> bool:
    """True if a clip s..e starts and ends on clean steps (always True when the context has no clean mask)."""
    if ctx.clean is None or not len(ctx.clean):
        return True
    i0 = min(int(np.floor(s * ctx.fps + 1e-6)), len(ctx.clean) - 1)
    i1 = max(int(np.ceil(e * ctx.fps - 1e-6)), i0 + 1)
    return _edge_ok(ctx, i0) and _edge_ok(ctx, min(i1, len(ctx.clean)), end=True)


def trim_edges(s: float, e: float, ctx: Context, min_len: float, max_len: float) -> tuple[float, float]:
    """Move the edges of s..e onto clean steps: inward first (drop a whip-pan, black or blocked frames at either
    end), outward by up to EDGE_SEARCH_S when inward would leave less than min_len. Unchanged if no clean pair of
    edges keeps the clip within min_len..max_len (the caller sees edges_clean() False)."""
    n = len(ctx.clean)
    if not n or edges_clean(s, e, ctx):
        return s, e
    fps = ctx.fps
    i0 = min(int(np.floor(s * fps + 1e-6)), n - 1)
    i1 = min(max(int(np.ceil(e * fps - 1e-6)), i0 + 1), n)
    reach = int(round(EDGE_SEARCH_S * fps))
    starts = [i for i in range(max(0, i0 - reach), i1) if _edge_ok(ctx, i)]
    ends = [j for j in range(i0 + 1, min(n, i1 + reach) + 1) if _edge_ok(ctx, j, end=True)]
    best = None
    for i in starts:
        for j in ends:
            L = (j - i) / fps
            if L < min_len - 1e-6 or L > max_len + 1e-6:
                continue
            # inward first (the bad frames are dropped, not kept in the middle); then keep as much of the original
            # as possible and move as little as possible
            inward = i >= i0 and j <= i1
            kept = min(j, i1) - max(i, i0)
            moved = abs(i - i0) + abs(j - i1)
            key = (inward, kept, -moved)
            if best is None or key > best[0]:
                best = (key, i, j)
    if best is None:
        return s, e
    _, i, j = best
    s2 = s if i == i0 else i / fps
    e2 = e if j == i1 else min(ctx.duration, j / fps)
    return s2, e2


# ---- whole sentences -------------------------------------------------------------------------------------------------
def refine_speech(start: float, end: float, ctx: Context, min_len: float, max_len: float) -> tuple[float, float]:
    """A speech clip on sentence boundaries: it starts just before the first word of a sentence and ends just after
    the last word of one (never inside a neighbouring word), keeping the sentences that lie mostly inside
    start..end. A sentence cut by start or end is taken whole if the clip stays within max_len, else left out; a
    single sentence longer than max_len keeps its piece inside start..end, between words. Without word timings the
    span is returned as it is."""
    words = ctx.words
    mids = [0.5 * (w["start"] + w["end"]) for w in words]
    inside = [k for k, m in enumerate(mids) if start <= m <= end]
    if not inside:
        return round(max(0.0, start), 3), round(min(ctx.duration, end), 3)
    first_s, last_s = int(ctx._sent_of[inside[0]]), int(ctx._sent_of[inside[-1]])
    sents = ctx._sentences

    def span(a: int, b: int) -> tuple[float, float]:
        return words[sents[a][0]]["start"], words[sents[b][1]]["end"]

    a, b = first_s, last_s
    # a sentence only partly inside: whole if it fits, else dropped (unless it is the only one)
    if span(a, b)[1] - span(a, b)[0] > max_len + 1e-6:
        while a < b and span(a, b)[1] - span(a, b)[0] > max_len + 1e-6:
            # drop the end that lies less inside the requested span
            out_a = max(0.0, start - span(a, a)[0])
            out_b = max(0.0, span(b, b)[1] - end)
            if out_b >= out_a:
                b -= 1
            else:
                a += 1
    ws, we = span(a, b)
    first, last = sents[a][0], sents[b][1]
    if we - ws > max_len + 1e-6:  # one sentence longer than the clip may be: its piece inside start..end (speech_spans
        first, last = inside[0], inside[-1]  # split it at a clause mark or pause), between words
        while last > first and words[last]["end"] - words[first]["start"] > max_len + 1e-6:
            last -= 1
        ws, we = words[first]["start"], words[last]["end"]
    prev_end = words[first - 1]["end"] if first > 0 else 0.0
    next_start = words[last + 1]["start"] if last + 1 < len(words) else ctx.duration
    s = max(prev_end + 0.02, ws - SPEECH_PAD, 0.0) if prev_end <= ws else ws
    e = min(next_start - 0.02, we + SPEECH_PAD, ctx.duration) if next_start >= we else we
    if e - s < min_len:  # pad into the surrounding pauses, never into a word
        need = min_len - (e - s)
        room_l, room_r = max(0.0, s - max(prev_end + 0.02, 0.0)), max(0.0, min(next_start - 0.02, ctx.duration) - e)
        grow_l = min(room_l, max(need / 2, need - room_r))
        s, e = s - grow_l, e + min(room_r, need - grow_l)
    return round(max(0.0, s), 3), round(min(ctx.duration, e), 3)
