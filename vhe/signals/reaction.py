"""Reaction signals, derived at query time from what every index already holds (no re-index): an audio REACTION
curve and a speech EXCITEMENT curve, per grid step.

reaction: loudness and onset-strength bursts over the file's own running baseline (a running median over
REACTION_BASE_S), so a shout, a scream, a sudden game sting or a cheering crowd stands out against whatever the file's
normal level is (a streamer who talks all the time, a quiet street). In physical units, not z-scored (a z-score would
blow a constant file's noise up to the same +3 as a real scream): one unit is REACTION_DB of loudness over the
baseline (or REACTION_ONSET_LOG of log onset strength), clipped to +-3. Measured 2026-10-18 on the stream-vod set's
85-min livestream (constant talking, game audio): bursts over 4 s are +6.8 dB at the 90th percentile, +12.8 dB at the
99th; log onset +0.22 / +0.53. Muted steps (digital silence) are left out of the baseline and read 0; 0 where a file
has no audio.

excitement: the transcript's excited speech: exclamation marks, laughter tokens ("haha", "lol"), strong words and
interjections ("oh my god", "no", "what", "bro", "insane"), shouted ALL-CAPS words, a word said again right away ("run
run run", "no no no"), plus bursts of the speech rate over the file's running baseline. Robust z over the pool,
clipped to +-3; all 0 without a transcript.

Both feed the highlight prior of "the best moments" and of reaction briefs ("the funniest moments", "every jump scare",
"a cold open"; vhe.query.model.ParsedQuery.reaction) through the heuristic scorer, and are feature columns (features
v4: reaction_mean, reaction_max, excitement_mean)."""
from __future__ import annotations

import re

import numpy as np

from vhe.signals import per_file, robust_z, smooth

REACTION_BASE_S = 90.0  # running-median window of the baseline a burst is measured against
REACTION_WIN_S = 4.0  # a burst is the mean excess over this window (a scream lasts a second or two)
REACTION_DB = 6.0  # one reaction unit: loudness this many dB over the running baseline
REACTION_ONSET_LOG = 0.25  # ... or log onset strength this much over its running baseline
REACTION_ONSET_W = 0.25  # share of the onset-strength burst in reaction (the rest is loudness)
MUTED_DB = -60.0  # loudness below this is digital silence (vhe.rank.signals.LIVE_MIN_DB): no baseline, no reaction
EXCITE_WIN_S = 8.0  # excited speech is counted over this window (a reaction is a few sentences)

# Per-word excitement points (English and German): interjections and strong words.
STRONG_WORDS = {
    "oh", "no", "nope", "what", "wow", "whoa", "woah", "yo", "bro", "dude", "omg", "wtf", "god", "gosh", "holy",
    "damn", "dang", "freaking", "frick", "fricking", "insane", "crazy", "kidding", "stop", "run", "help", "yes",
    "yeah", "let's", "lets", "go", "wait", "hey", "ah", "aah", "ahh", "ow", "ouch", "ugh", "nah", "scared", "scary",
    "dead", "died", "die", "killed", "unbelievable", "incredible", "amazing", "nein", "krass", "alter", "digga",
    "geil", "wahnsinn", "hilfe", "boah", "gott", "verdammt", "scheiße", "scheisse",
}
LAUGH = re.compile(r"^(?:a?ha(?:ha)+h?|he(?:he)+|hi(?:hi)+|lol+|lmao+|lmfao|rofl|haha\w*|xd)$")
POINTS = {"exclaim": 1.0, "laugh": 1.5, "caps": 1.0, "strong": 0.5, "repeat": 0.5}


def running_median(x: np.ndarray, width: int, block: int = 1) -> np.ndarray:
    """Centred running median over `width` samples (edges repeat the end values), computed on means of `block`
    consecutive samples (a 90 s window on 1 s blocks instead of 270 grid steps) and repeated back to len(x)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pad = (-n) % block
    coarse = np.pad(x, (0, pad), mode="edge").reshape(-1, block).mean(1)
    w = max(1, width // block) | 1
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(coarse, w // 2, mode="edge"), w)
    return np.repeat(np.median(windows, axis=1), block)[:n]


def burst(x: np.ndarray, fps: float, base_s: float = REACTION_BASE_S, win_s: float = REACTION_WIN_S,
          file_id: np.ndarray | None = None, valid: np.ndarray | None = None) -> np.ndarray:
    """Per step: the mean excess of x over its running median (base_s window) within win_s, per file (x's units).
    valid: steps that count; the others take the file's median of the valid ones first (no burst at a mute edge)."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    ok = np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)

    def one(v: np.ndarray, good: np.ndarray) -> np.ndarray:
        if not len(v):
            return v
        v = np.where(good, v, np.median(v[good]) if good.any() else 0.0)
        base = running_median(v, max(1, int(base_s * fps)), max(1, int(round(fps))))
        return smooth(v - base, int(win_s * fps))

    if file_id is None:
        return one(x, ok)
    out = np.empty_like(x)
    edges = np.flatnonzero(np.diff(file_id)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(x)], strict=True):
        out[a:b] = one(x[a:b], ok[a:b])
    return out


def reaction_steps(loudness: np.ndarray, onset: np.ndarray, audio: np.ndarray, fps: float,
                   file_id: np.ndarray | None = None) -> np.ndarray:
    """The audio reaction curve (module docstring): loudness (dB) and log onset-strength bursts over each file's running
    baseline in REACTION_DB / REACTION_ONSET_LOG units, mixed at REACTION_ONSET_W, clipped to +-3; 0 where there is no
    audio or the sound is muted."""
    loudness = np.nan_to_num(np.asarray(loudness, dtype=float), nan=-120.0)
    audio = np.broadcast_to(np.asarray(audio, dtype=bool), loudness.shape) & (loudness >= MUTED_DB)
    if not audio.any():
        return np.zeros(len(loudness))
    loud = burst(loudness, fps, file_id=file_id, valid=audio) / REACTION_DB
    ons = burst(np.log(np.maximum(np.nan_to_num(np.asarray(onset, dtype=float)), 0.0) + 1e-3), fps, file_id=file_id,
                valid=audio) / REACTION_ONSET_LOG
    r = (1 - REACTION_ONSET_W) * loud + REACTION_ONSET_W * ons
    return np.where(audio, np.clip(r, -3, 3), 0.0)


def word_points(words: list[dict]) -> list[float]:
    """Excitement points per transcript word (POINTS: exclamation, laughter, shouted caps, strong word, repeat)."""
    out, prev = [], None
    for w in words:
        raw = str(w.get("word", "")).strip()
        key = re.sub(r"[^\w']", "", raw.lower())
        pts = 0.0
        if "!" in raw:
            pts += POINTS["exclaim"]
        if LAUGH.match(key):
            pts += POINTS["laugh"]
        letters = re.sub(r"[^\w]", "", raw)
        if len(letters) >= 2 and letters.isalpha() and letters.isupper():
            pts += POINTS["caps"]
        if key in STRONG_WORDS:
            pts += POINTS["strong"]
        if key and key == prev:
            pts += POINTS["repeat"]
        prev = key
        out.append(pts)
    return out


def excitement_steps(words: list[dict], n: int, fps: float, file_id: np.ndarray | None = None) -> np.ndarray:
    """The speech excitement curve (module docstring) over n steps from transcript words (virtual seconds): excitement
    points per step and the speech-rate burst over the file's running baseline, each robust-z'd over the pool,
    averaged and clipped to +-3. All 0 without words."""
    if not words or n == 0:
        return np.zeros(n)
    pts, rate = np.zeros(n), np.zeros(n)
    idx = np.array([int(w["start"] * fps) for w in words])
    ok = (idx >= 0) & (idx < n)
    np.add.at(pts, idx[ok], np.asarray(word_points(words), dtype=float)[ok])
    np.add.at(rate, idx[ok], 1.0)
    win = int(EXCITE_WIN_S * fps)
    excited = robust_z(per_file(lambda v: smooth(v, win), pts, file_id))
    faster = robust_z(burst(rate, fps, win_s=EXCITE_WIN_S, file_id=file_id))
    return np.clip((excited + faster) / 2, -3, 3)
