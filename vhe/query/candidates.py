"""Candidate clips from below the shot level: sub-shots (cuts + content change points), sliding windows of
several lengths, windows centred on signal peaks, proposals from the highlight model, and utterances (transcript
sentences, or voice-activity spans when there is no transcript) so speech clips start and end with the speaker.
The query's length limits shape which candidates exist, not just which survive."""
from __future__ import annotations

import numpy as np

from vhe.signals.shots import shots_from_cuts

LENGTHS = (3, 4, 6, 8, 10, 15, 20, 30, 45, 60, 90)
SPEECH_GAP = 0.6  # utterances closer than this belong to one clip
SPEECH_PAD = 0.15  # lead-in/out around speech, never more than half the pause to the neighbour
SENTENCE_END = (".", "!", "?", "…", "。")


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


def _sentences(segments: list[dict], limit: float) -> list[tuple[float, float]]:
    """Transcript sentences as (start, end); a sentence longer than limit is split at word boundaries."""
    out: list[tuple[float, float]] = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            out.append((float(seg["start"]), float(seg["end"])))
            continue
        a = words[0]["start"]
        for k, w in enumerate(words):
            last = k == len(words) - 1
            if last or w["word"].rstrip("\"')»”").endswith(SENTENCE_END) or words[k + 1]["end"] - a > limit:
                out.append((float(a), float(w["end"])))
                if not last:
                    a = words[k + 1]["start"]
    return out


def speech_spans(segments: list[dict], vad: list[tuple[float, float]], max_len: float,
                 gap: float = SPEECH_GAP, pad: float = SPEECH_PAD) -> list[tuple[float, float]]:
    """Utterance clips: sentences (or VAD spans without a transcript) merged across pauses shorter than gap, split
    at sentence ends so no clip exceeds max_len, padded by up to pad seconds on each side."""
    limit = max(0.5, max_len - 2 * pad)
    if segments:
        units = _sentences(segments, limit)
    else:  # no words: split over-long voice spans evenly
        units = []
        for s, e in vad:
            k = int(np.ceil((e - s) / limit))
            units += [(s + (e - s) * j / k, s + (e - s) * (j + 1) / k) for j in range(k)] if k > 1 else [(s, e)]
    runs: list[list[float]] = []
    for s, e in sorted(units):
        if runs and s - runs[-1][1] < gap and e - runs[-1][0] <= limit:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    out = []
    for k, (s, e) in enumerate(runs):
        before = (s - runs[k - 1][1]) / 2 if k > 0 else pad
        after = (runs[k + 1][0] - e) / 2 if k + 1 < len(runs) else pad
        out.append((max(0.0, s - min(pad, max(before, 0.0))), e + min(pad, max(after, 0.0))))
    return out


def generate(duration: float, cuts: list[float], change_points: list[float], min_len: float, max_len: float,
             peaks: list[float] = (), proposals: list[tuple] = (),
             speech: list[tuple[float, float]] = ()) -> list[tuple[float, float, str]]:
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

    # First, so an identical window does not take the utterance's place.
    for s, e in speech:  # utterances shorter than min_len grow evenly into the surrounding pause
        L = max(e - s, min_len)
        start = min(max(0.0, (s + e) / 2 - L / 2), max(0.0, duration - L)) if L > e - s else s
        add(start, start + L, "speech")
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
