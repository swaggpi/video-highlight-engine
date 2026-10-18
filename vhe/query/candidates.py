"""Candidate clips from below the shot level: sub-shots (cuts + content change points), sliding windows of
several lengths, windows centred on signal peaks, proposals from the highlight model, and utterances (transcript
sentences, or voice-activity spans when there is no transcript) so speech clips start and end with the speaker.
The query's length limits shape which candidates exist, not just which survive."""
from __future__ import annotations

import numpy as np

from vhe.config import CANDIDATE_LENGTHS
from vhe.signals import smooth
from vhe.signals.shots import shots_from_cuts

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


SENTENCE_PAUSE = 1.2  # a pause this long between words ends a sentence even without punctuation (ASR omits it)
CLAUSE_END = (",", ";", ":", "—", "-")
CLAUSE_PAUSE = 0.4  # an over-long sentence splits at a clause mark, else at its longest pause from this on


def _ends(word: str, marks: tuple) -> bool:
    return word.strip().rstrip("\"')»”").endswith(marks)


# A pause does not end a sentence on a word that leaves it hanging ("... and", "... I'm going to") or before one that
# cannot open one ("of music I don't know"): the r3 judge heard talking-track clips end on "and" / "so" / "but" and
# start on "of". Up to DANGLING_PAUSE the sentence runs on; punctuation still ends it.
DANGLING_END = {"and", "or", "but", "so", "because", "cause", "the", "a", "an", "to", "of", "with", "for", "that",
                "than", "if", "i", "we", "you", "he", "she", "they", "my", "your", "our", "his", "her", "their", "i'm",
                "we're", "you're", "they're", "gonna", "going", "is", "are", "was", "were", "like", "um", "uh", "this",
                "these", "in", "on", "at", "from", "about", "as", "let", "let's", "just"}
DANGLING_START = {"of", "to", "for", "with", "than", "from", "and", "or", "but"}
DANGLING_PAUSE = 3.0


def _bare(word: str) -> str:
    return word.strip().strip("\"'»”“,").lower()


def sentence_spans(words: list[dict], pause: float = SENTENCE_PAUSE) -> list[tuple[int, int]]:
    """Sentences as (first, last) word indices over start-sorted words: a sentence ends at a word ending in . ! ? …
    or before a pause of at least `pause` seconds, unless the pause leaves it hanging (DANGLING_END / DANGLING_START,
    pauses under DANGLING_PAUSE)."""
    out, a = [], 0
    for k, w in enumerate(words):
        last = k == len(words) - 1
        if last or _ends(w["word"], SENTENCE_END):
            out.append((a, k))
            a = k + 1
            continue
        gap = words[k + 1]["start"] - w["end"]
        hanging = gap < DANGLING_PAUSE and (_bare(w["word"]) in DANGLING_END
                                             or _bare(words[k + 1]["word"]) in DANGLING_START)
        if gap >= pause and not hanging:
            out.append((a, k))
            a = k + 1
    return out


def _split_long(words: list[dict], a: int, b: int, limit: float) -> list[tuple[int, int]]:
    """Word range a..b (one sentence) in pieces no longer than limit: at clause marks or long pauses where possible,
    else at the word boundary that keeps the piece longest."""
    out = []
    while words[b]["end"] - words[a]["start"] > limit:
        fits = [k for k in range(a, b) if words[k]["end"] - words[a]["start"] <= limit]
        if not fits:  # one word longer than the limit
            fits = [a]
        clause = [k for k in fits if _ends(words[k]["word"], CLAUSE_END)
                  or words[k + 1]["start"] - words[k]["end"] >= CLAUSE_PAUSE]
        k = max(clause or fits)
        out.append((a, k))
        a = k + 1
    out.append((a, b))
    return out


def _sentences(segments: list[dict], limit: float) -> list[tuple[float, float]]:
    """Transcript sentences as (start, end). Sentences run across ASR segments (a segment often starts mid-sentence);
    one longer than limit is split at a clause mark or pause, else at a word boundary. Segments without word
    timings count as one sentence each."""
    out: list[tuple[float, float]] = []
    words = sorted((w for seg in segments for w in seg.get("words") or []), key=lambda w: w["start"])
    for seg in segments:
        if not seg.get("words"):
            out.append((float(seg["start"]), float(seg["end"])))
    for a, b in sentence_spans(words):
        out += [(float(words[i]["start"]), float(words[j]["end"])) for i, j in _split_long(words, a, b, limit)]
    return sorted(out)


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


def fit_utterances(spans: list[tuple[float, float]], lo: float, hi: float, duration: float,
                   guard: float = 0.05) -> list[tuple[float, float]]:
    """Utterance clips (speech_spans, in time order) of at least lo seconds without cutting a neighbour: a short one
    grows evenly into the pauses around it (never into the next utterance), then takes whole neighbouring
    utterances in while it stays within hi. One still shorter than lo, or longer than hi, is dropped."""
    spans = sorted(spans)
    out = []
    for j, (s, e) in enumerate(spans):
        a, b = j, j
        while e - s < lo:
            left = spans[a - 1][1] + guard if a > 0 else 0.0
            right = spans[b + 1][0] - guard if b + 1 < len(spans) else duration
            need = lo - (e - s)
            grow_l = min(max(0.0, s - left), max(need / 2, need - max(0.0, right - e)))
            s, e = s - grow_l, min(right, e + need - grow_l)
            if e - s >= lo - 1e-6:
                break
            # the pauses are too short: take the nearer whole neighbour in, if it fits
            nxt = spans[b + 1] if b + 1 < len(spans) and spans[b + 1][1] - s <= hi else None
            prv = spans[a - 1] if a > 0 and e - spans[a - 1][0] <= hi else None
            if nxt is not None and (prv is None or nxt[0] - e <= s - prv[1]):
                b, e = b + 1, nxt[1]
            elif prv is not None:
                a, s = a - 1, prv[0]
            else:
                break
        if lo - 1e-6 <= e - s <= hi + 1e-6:
            out.append((max(0.0, s), min(duration, e)))
    return out


def generate(duration: float, cuts: list[float], change_points: list[float], min_len: float, max_len: float,
             peaks: list[float] = (), proposals: list[tuple] = (),
             speech: list[tuple[float, float]] = ()) -> list[tuple[float, float, str]]:
    lengths = [L for L in CANDIDATE_LENGTHS if min_len <= L <= max_len] or [float(np.clip(8.0, min_len, max_len))]
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
    # Utterances shorter than min_len grow into the surrounding pause, then by whole neighbours: never by a word of
    # another sentence.
    for s, e in fit_utterances(list(speech), min_len, max_len, duration):
        add(s, e, "speech")
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


def corpus_candidates(corpus, per_step: np.ndarray, p, proposals: list[list]) -> tuple:
    """generate() per file of a corpus, on the file's own timeline (peaks of its part of per_step, its cuts, change
    points and utterances; proposals: the highlight model's, per file), then shifted: (start, end, file, source)
    arrays in virtual seconds. p: the ParsedQuery (min_len, max_len)."""
    c, fps = corpus, corpus.fps
    out = []
    for k, ix in enumerate(c.members):
        speech = speech_spans((ix.transcript or {}).get("segments", []), ix.vad, p.max_len)
        peaks = peak_times(smooth(per_step[c.bounds[k]:c.bounds[k + 1]], fps), fps)
        out += [(c.offsets[k] + s, c.offsets[k] + e, k, src)
                for s, e, src in generate(float(c.durations[k]), ix.cuts, ix.change_points, p.min_len, p.max_len,
                                          peaks=peaks, proposals=proposals[k], speech=speech)]
    if not out:
        return np.zeros(0), np.zeros(0), np.zeros(0, int), []
    S, E, K, src = zip(*out, strict=True)
    return np.array(S), np.array(E), np.array(K), list(src)


def cuts_inside(s, e, cuts: np.ndarray, margin: float = 0.3):
    """Cuts strictly inside (s + margin, e - margin); s and e may be arrays (cuts sorted)."""
    cuts = np.sort(np.asarray(cuts, dtype=float))
    n = np.searchsorted(cuts, np.asarray(e) - margin, "left") - np.searchsorted(cuts, np.asarray(s) + margin, "right")
    return np.maximum(n, 0) if np.ndim(n) else int(max(n, 0))
