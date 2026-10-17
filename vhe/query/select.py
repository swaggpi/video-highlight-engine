"""Deterministic selection: greedy maximal marginal relevance with no temporal overlap and an optional total
duration budget. Near-duplicates (same scene, similar look) are penalised by embedding similarity."""
from __future__ import annotations

import numpy as np


def overlap_fraction(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))


def select(cands: list[dict], vectors: np.ndarray, n: int, lam: float = 1.0, sim_floor: float = 0.6,
           max_overlap: float = 0.2, budget_s: float | None = None, pool: int = 400) -> list[dict]:
    """cands: dicts with 'start', 'end', 'total'; vectors: L2-normalised mean embedding per candidate."""
    order = np.argsort([-c["total"] for c in cands])[:pool]
    chosen: list[int] = []
    used = 0.0
    remaining = list(order)
    while remaining and len(chosen) < n:
        best, best_val = None, -np.inf
        for i in remaining:
            c = cands[i]
            if any(overlap_fraction((c["start"], c["end"]), (cands[j]["start"], cands[j]["end"])) > max_overlap
                   for j in chosen):
                continue
            if budget_s is not None and used + (c["end"] - c["start"]) > budget_s:
                continue
            penalty = 0.0
            if chosen:
                sim = float(np.max(vectors[chosen] @ vectors[i]))
                penalty = lam * max(0.0, sim - sim_floor) / (1.0 - sim_floor)
            val = c["total"] - penalty
            if val > best_val:
                best, best_val = i, val
        if best is None:
            break
        chosen.append(best)
        used += cands[best]["end"] - cands[best]["start"]
        remaining.remove(best)
    return [dict(cands[i], mmr=round(float(cands[i]["total"]), 4)) for i in chosen]
