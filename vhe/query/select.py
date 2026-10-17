"""Deterministic selection: greedy maximal marginal relevance with no temporal overlap and an optional total
duration budget. Near-duplicates (same scene, similar look) are penalised by embedding similarity.

The stopping rule comes from the query: a fixed count, a runtime budget (min, max) filled in MMR order, or every
candidate above a threshold ("every moment someone speaks"). Candidates should already carry their final
(refined) edges, so budget and overlap checks hold for the clips that are returned."""
from __future__ import annotations

import numpy as np

from vhe.query.parser import MAX_CLIPS

EXHAUSTIVE_MIN_TOTAL = 0.5  # in score units (about half a robust z above the length-bucket median)
EXHAUSTIVE_MIN_SPEECH = 0.5  # speech-intent exhaustive queries: fraction of the clip with voice activity


def overlap_fraction(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))


def select(cands: list[dict], vectors: np.ndarray, n: int, lam: float = 1.0, sim_floor: float = 0.6,
           max_overlap: float = 0.2, budget_s: float | tuple[float, float] | None = None, n_max: int | None = None,
           min_total: float | None = None) -> list[dict]:
    """cands: dicts with 'start', 'end', 'total'; vectors: L2-normalised mean embedding per candidate (same order).
    budget_s: a number is an upper bound on the summed length; (lo, hi) keeps picking past n (up to n_max) until
    the sum reaches lo, and never exceeds hi. min_total: candidates scoring below it are never picked.
    Callers pass a pre-filtered pool; this is O(n * len(cands))."""
    if not cands:
        return []
    lo, hi = (0.0, np.inf) if budget_s is None else \
        (float(budget_s[0]), float(budget_s[1])) if isinstance(budget_s, (tuple, list)) else (0.0, float(budget_s))
    n_max = n if n_max is None else max(n, n_max)
    totals = np.array([c["total"] for c in cands], dtype=float)
    starts = np.array([c["start"] for c in cands], dtype=float)
    ends = np.array([c["end"] for c in cands], dtype=float)
    lengths = ends - starts
    alive = np.ones(len(cands), dtype=bool) if min_total is None else totals >= min_total
    alive &= lengths <= hi + 1e-6
    max_sim = np.full(len(cands), -1.0)
    chosen: list[int] = []
    used = 0.0
    while alive.any() and len(chosen) < n_max and (len(chosen) < n or used < lo):
        penalty = lam * np.clip((max_sim - sim_floor) / (1.0 - sim_floor), 0.0, None)
        val = np.where(alive, totals - penalty, -np.inf)
        i = int(np.argmax(val))
        if not np.isfinite(val[i]):
            break
        alive[i] = False
        if used + lengths[i] > hi + 1e-6:
            continue
        chosen.append(i)
        used += lengths[i]
        inter = np.clip(np.minimum(ends, ends[i]) - np.maximum(starts, starts[i]), 0.0, None)
        alive &= inter / np.maximum(np.minimum(lengths, lengths[i]), 1e-9) <= max_overlap
        max_sim = np.maximum(max_sim, vectors @ vectors[i])
    return [dict(cands[i], rank=r) for r, i in enumerate(chosen)]


def select_for(cands: list[dict], vectors: np.ndarray, p) -> list[dict]:
    """Selection for a parsed query (vhe.query.parser.ParsedQuery) over refined, final-edge candidates: clips
    never overlap; a budget is filled to at least its minimum when the pool allows, and never past its maximum."""
    if p.exhaustive:  # every match: no diversity re-ordering, the budget (if any) only caps the total
        if p.weights.get("speech", 0.0) > 0:  # "every moment someone speaks": only clips that are mostly speech
            keep = [k for k, c in enumerate(cands) if c.get("speech_frac", 1.0) >= EXHAUSTIVE_MIN_SPEECH]
            cands, vectors = [cands[k] for k in keep], vectors[keep]
        cap = None if p.budget is None else p.budget[1]
        return select(cands, vectors, p.count, lam=0.0, max_overlap=0.0, budget_s=cap,
                      min_total=EXHAUSTIVE_MIN_TOTAL)
    n_max = MAX_CLIPS if p.budget is not None and not p.count_explicit else p.count
    return select(cands, vectors, p.count, max_overlap=0.0, budget_s=p.budget, n_max=n_max)
