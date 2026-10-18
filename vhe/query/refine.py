"""Cut refinement: move clip edges to hard cuts, speech pauses or word boundaries, and away from shaky
frames, then enforce the length limits. A clip should never start or end mid-word; a visual clip should start and
end on steady, usable frames (not inside a whip-pan, on a black or blocked frame); a speech clip starts where a
sentence starts and ends where one ends."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from vhe.query.candidates import sentence_spans
from vhe.signals.quality import reason_mask

FRAME_GUARD = 0.04  # stay one frame clear of a cut
# Clean edges: the first and last EDGE_STEPS grid steps (2 at 3 fps: 0.67 s) of a visual clip must be usable, not
# bad framing, and not mid-swing. Calibrated on the irl-event judge notes (2026-10-18): steady holds have cam_speed
# ~0.000-0.005 and shake <0.01, deliberate follow pans 0.02-0.06, the whip-pans the judge flagged 0.16-2.6 with shake
# 0.10-0.6 (pooled 90th percentiles: cam_speed 0.08, shake 0.10).
EDGE_STEPS = 2
EDGE_MAX_SPEED = 0.1
EDGE_MAX_SHAKE = 0.1
EDGE_SEARCH_S = 1.5  # how far outward an edge may move to find clean frames when trimming inward leaves too little
# Clip edges near speech sit in the pause between words, so a clip's audio holds its first and last word whole. The
# r3 judge heard talking-track clips "cut mid-word" whose edges were 0.05-0.15 s off Whisper's word times: Whisper's
# word ends are often early and onsets soft, so the lead-in is SPEECH_LEAD and the tail a more conservative
# SPEECH_TAIL, each at most to the middle of the pause (never into the neighbouring word). WORD_GUARD: no edge lies
# inside a word extended by this much (or by half the pause, when the pause is shorter): word_clear().
SPEECH_LEAD = 0.25
SPEECH_TAIL = 0.4
WORD_GUARD = 0.15
# A talking-track clip (the brief wants speech) shorter than its minimum length grows into the pauses around it by at
# most this much per side beyond its lead/tail: a long silent tail is dead air in a talking track (r3 B2).
TALK_MAX_SILENCE = 0.35


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


def clean_steps(grid: dict) -> np.ndarray:
    """Per grid step: fit to start or end a visual clip on. Usable (not covered, black, blown, blurry, very shaky or
    obstructed), not pointed at the ground or sky, and the camera not swinging (cam_speed, shake)."""
    ok = np.asarray(grid["usable"], dtype=bool) & ~reason_mask(grid["unusable"], "bad_framing")
    for key, limit in (("cam_speed", EDGE_MAX_SPEED), ("shake", EDGE_MAX_SHAKE)):
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


def _pauses(words: list[dict], j: int) -> tuple[float, float]:
    """(pause before word j, pause after it) in seconds; inf at the ends, 0 where ASR words overlap."""
    before = words[j]["start"] - words[j - 1]["end"] if j > 0 else np.inf
    after = words[j + 1]["start"] - words[j]["end"] if j + 1 < len(words) else np.inf
    return max(0.0, before), max(0.0, after)


def _index_of(ctx: Context, w: dict) -> int:
    i = bisect.bisect_left(ctx._starts, w["start"])
    while i + 1 < len(ctx.words) and ctx.words[i] is not w and ctx.words[i + 1]["start"] == w["start"]:
        i += 1
    return i


def _before_word(ctx: Context, w: dict, pad: float = SPEECH_LEAD) -> float:
    """A clip start that holds word w whole: `pad` before it, at most to the middle of the pause before it (never
    into the previous word)."""
    gap, _ = _pauses(ctx.words, _index_of(ctx, w))
    return max(0.0, w["start"] - min(pad, gap / 2))


def _after_word(ctx: Context, w: dict, pad: float = SPEECH_TAIL) -> float:
    """A clip end that holds word w whole: `pad` after it, at most to the middle of the pause after it (never into
    the next word)."""
    _, gap = _pauses(ctx.words, _index_of(ctx, w))
    return min(ctx.duration, w["end"] + min(pad, gap / 2))


def word_near(words: list[dict], t: float, guard: float = WORD_GUARD, starts: list[float] | None = None) -> dict | None:
    """The word (start-sorted `words`) that t lies inside, each word extended by `guard` seconds on either side (by at
    most half the pause to its neighbour); None when t is clear of every word. 1 ms tolerance: edges are rounded."""
    starts = [w["start"] for w in words] if starts is None else starts
    i = bisect.bisect_right(starts, t) - 1
    for j in (i, i + 1):
        if 0 <= j < len(words):
            before, after = _pauses(words, j)
            lo = words[j]["start"] - min(guard, before / 2)
            hi = words[j]["end"] + min(guard, after / 2)
            if lo + 1e-3 < t < hi - 1e-3:
                return words[j]
    return None


def word_clear(words: list[dict], t: float, guard: float = WORD_GUARD) -> bool:
    """True when a clip edge at t cuts no word (word_near is None): the check every speech-aware edge passes."""
    return word_near(sorted(words, key=lambda w: w["start"]), t, guard) is None


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
        return _before_word(ctx, w)
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
        return _after_word(ctx, w)
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
    return _clear_of_words(s, e, ctx, min_len, max_len)


def _clear_of_words(s: float, e: float, ctx: Context, min_len: float, max_len: float) -> tuple[float, float]:
    """Never start or end in a word (WORD_GUARD around it): start before it (or after it, when the start is past its
    end); end after it, or before it if that would overrun max_len."""
    w = word_near(ctx.words, s, starts=ctx._starts)
    if w is not None:
        s = _before_word(ctx, w) if s < w["end"] else _after_word(ctx, w)
    w = word_near(ctx.words, e, starts=ctx._starts)
    if w is not None:
        after, before = _after_word(ctx, w), _before_word(ctx, w)
        e = after if e > w["start"] and (after - s <= max_len or before - s < min_len) else before
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
    return bool(spans_edges_clean(ctx.clean, np.zeros(1, int), np.array([len(ctx.clean)]), np.array([s]),
                                  np.array([e]), ctx.fps)[0])


def spans_edges_clean(clean: np.ndarray, lo: np.ndarray, n: np.ndarray, s: np.ndarray, e: np.ndarray,
                      fps: float) -> np.ndarray:
    """edges_clean for many clips at once: clip j spans s[j]..e[j] seconds of a file whose clean mask is
    clean[lo[j]:lo[j] + n[j]] (several files' masks concatenated, as a corpus grid)."""
    i0 = np.minimum(np.floor(np.asarray(s) * fps + 1e-6).astype(int), n - 1)
    j = np.minimum(np.maximum(np.ceil(np.asarray(e) * fps - 1e-6).astype(int), i0 + 1), n)
    cum = np.concatenate([[0], np.cumsum(np.asarray(clean, dtype=np.int64))])

    def ok(a, b):  # _edge_ok: steps a..b inside the file, all clean
        inside = (a >= 0) & (b <= n)
        return inside & (cum[lo + np.clip(b, 0, n)] - cum[lo + np.clip(a, 0, n)] == b - a)

    return ok(i0, i0 + EDGE_STEPS) & ok(j - EDGE_STEPS, j)


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
def refine_speech(start: float, end: float, ctx: Context, min_len: float, max_len: float,
                  max_silence: float | None = None) -> tuple[float, float]:
    """A speech clip on sentence boundaries: it starts SPEECH_LEAD before the first word of a sentence and ends
    SPEECH_TAIL after the last word of one (each at most to the middle of the pause, never inside a neighbouring
    word), keeping the sentences that lie mostly inside start..end. A sentence cut by start or end is taken whole if
    the clip stays within max_len, else left out; a single sentence longer than max_len keeps its piece inside
    start..end, between words. A clip shorter than min_len grows into the pauses around it, by at most max_silence
    per side when given (a talking track: no long silent tails). Without word timings the span is returned as it
    is."""
    words = ctx.words
    mids = [0.5 * (w["start"] + w["end"]) for w in words]
    inside = [k for k, m in enumerate(mids) if start <= m <= end]
    if not inside:  # no word in it: only kept clear of the words around it
        return _clear_of_words(start, end, ctx, min_len, max_len)
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
    s, e = _before_word(ctx, words[first]), _after_word(ctx, words[last])
    if e - s < min_len:  # pad into the surrounding pauses, never into a word (nor past its guard)
        need = min_len - (e - s)
        lo = prev_end + min(WORD_GUARD, max(0.0, ws - prev_end) / 2) if first > 0 else 0.0
        hi = next_start - min(WORD_GUARD, max(0.0, next_start - we) / 2) if last + 1 < len(words) else ctx.duration
        room_l, room_r = max(0.0, s - lo), max(0.0, hi - e)
        if max_silence is not None:
            room_l, room_r = min(room_l, max_silence), min(room_r, max_silence)
        grow_l = min(room_l, max(need / 2, need - room_r))
        s, e = s - grow_l, e + min(room_r, need - grow_l)
    return round(max(0.0, s), 3), round(min(ctx.duration, e), 3)
