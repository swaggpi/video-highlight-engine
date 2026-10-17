"""Deterministic selection: greedy maximal marginal relevance with no temporal overlap and an optional total
duration budget. Near-duplicates (same scene, similar look) are penalised by embedding similarity."""
from __future__ import annotations

import numpy as np


def overlap_fraction(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))


def select(cands: list[dict], vectors: np.ndarray, n: int, lam: float = 1.0, sim_floor: float = 0.6,
           max_overlap: float = 0.2, budget_s: float | None = None) -> list[dict]:
    """cands: dicts with 'start', 'end', 'total'; vectors: L2-normalised mean embedding per candidate (same order).
    Callers pass a pre-filtered pool; this is O(n * len(cands))."""
    if not cands:
        return []
    totals = np.array([c["total"] for c in cands], dtype=float)
    starts = np.array([c["start"] for c in cands], dtype=float)
    ends = np.array([c["end"] for c in cands], dtype=float)
    lengths = ends - starts
    alive = np.ones(len(cands), dtype=bool)
    max_sim = np.full(len(cands), -1.0)
    chosen: list[int] = []
    used = 0.0
    while alive.any() and len(chosen) < n:
        penalty = lam * np.clip((max_sim - sim_floor) / (1.0 - sim_floor), 0.0, None)
        val = np.where(alive, totals - penalty, -np.inf)
        i = int(np.argmax(val))
        if not np.isfinite(val[i]):
            break
        alive[i] = False
        if budget_s is not None and used + lengths[i] > budget_s:
            continue
        chosen.append(i)
        used += lengths[i]
        inter = np.clip(np.minimum(ends, ends[i]) - np.maximum(starts, starts[i]), 0.0, None)
        alive &= inter / np.maximum(np.minimum(lengths, lengths[i]), 1e-9) <= max_overlap
        max_sim = np.maximum(max_sim, vectors @ vectors[i])
    return [dict(cands[i], rank=r) for r, i in enumerate(chosen)]
