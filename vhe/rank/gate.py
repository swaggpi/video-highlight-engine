"""Hard gates on candidate spans: what a clip may not contain, whatever its score. A span is dropped when too many
of its grid steps are unusable (covered, blown, blurry: config.ACTIVE_REASONS), muted where the brief needs live
sound, showing an excluded item or camera defect, holding a forbidden word on screen, or (when the brief wants none)
speech. The per-step masks come from the brief's signals (vhe.rank.brief_signals, vhe.query.exclusions)."""
from __future__ import annotations

import numpy as np

MAX_UNUSABLE = 0.2  # candidates with more unusable steps (the active reasons: covered, blown, blurry) are dropped
LIVE_MAX_MUTED = 0.1  # need_live_audio: clips with a larger muted (or audio-less) fraction are dropped
EXCLUDE_MAX_FRAC = 0.1  # a clip with more of its steps showing an excluded item is dropped
DEFECT_MAX_FRAC = 0.1  # ... or with more of its steps showing an excluded camera defect ("no whip-pans")
# "no host talking over it", "it goes under music", "music bed to be added": clips with more voice activity are
# dropped. 0.3 let the r3 judge hear host speech under p3-newsedit-2's and p1-videographer-2's picks; on the
# irl-event pools 81-99 % of candidates have no voice-activity step at all, so a near-zero bar costs little.
NO_SPEECH_MAX_FRAC = 0.05


def step_fraction(corpus, S: np.ndarray, E: np.ndarray, K: np.ndarray, bad: np.ndarray | None = None) -> np.ndarray:
    """Fraction of unusable grid steps per span (or of the steps flagged in `bad`)."""
    i0, i1 = corpus.steps(S, E, K)
    bad = np.concatenate([[0], np.cumsum(~corpus.usable if bad is None else bad)])
    return (bad[i1] - bad[i0]) / (i1 - i0)


def gate(corpus, S: np.ndarray, E: np.ndarray, K: np.ndarray, p, muted: np.ndarray,
         sig: dict | None = None) -> np.ndarray:
    """Candidates to keep: few unusable steps, live sound where the brief asks for it (muted: per step), and (with the
    brief's signals) nothing it forbids: excluded content or camera defects, or speech when it wants none."""
    if not len(S):
        return np.zeros(0, bool)
    keep = step_fraction(corpus, S, E, K) <= MAX_UNUSABLE
    if p.need_live_audio:
        keep &= step_fraction(corpus, S, E, K, muted) <= LIVE_MAX_MUTED
    if sig is not None:
        if sig.get("excluded") is not None:
            keep &= step_fraction(corpus, S, E, K, sig["excluded"]) <= EXCLUDE_MAX_FRAC
        if sig.get("excluded_text") is not None:  # a forbidden word read on screen: any sighting
            keep &= step_fraction(corpus, S, E, K, sig["excluded_text"]) <= 0.0
        if sig.get("defect") is not None:
            keep &= step_fraction(corpus, S, E, K, sig["defect"]) <= DEFECT_MAX_FRAC
        if p.no_speech:
            keep &= step_fraction(corpus, S, E, K, sig["speech"]) <= NO_SPEECH_MAX_FRAC
    return keep
