"""A shot list's relevance per item. The brief's relevance is the best item per step minus a bounded avoid penalty
(combine); a span serves the item it matches best, centred by each item's corpus median, if it matches it well enough
(assign); selection uses that to cover distinct items."""
from __future__ import annotations

import numpy as np

from vhe.rank.signals import span_top_mean

# Relevance units (vhe.rank.signals.relevance): 0 is log-odds -8, one unit 2.5 nats, clipped to +-3.
AVOID_FLOOR = 1.2  # an avoid item matching below this (log-odds about -5) is noise and never penalised
AVOID_MAX = 1.0  # at most one unit off: avoid never sinks a clearly relevant shot
# A clip serves an item only above this (log-odds about -6.75, a partial SigLIP match). On the irl-event footage
# specific items peak at +0.5..+1.5 ("dancers in traditional costumes") while generic ones sit at +1.8 everywhere
# ("wide of the empty avenue"), so which item a clip serves is decided on relevance centred by each item's corpus
# median (how unusual the match is for that item), and only this floor is absolute.
ITEM_MIN_REL = 0.5
ITEM_MARGIN = 0.5  # exhaustive per-item picks: the clip must match this item nearly as well as its best item
# An item's words read on screen (vhe.query.text_match.ocr_item_scores, 0..1) are evidence of which item a span shows,
# in relevance units at a full match: SigLIP cannot read "PASARGAD" on a float. Matching only (which item a span
# serves, the exhaustive relevance gates); scores never see it here (HeuristicScorer has its own on-screen bonuses).
ITEM_OCR_EVIDENCE = 1.0


def evidence(rel: np.ndarray, ocr: np.ndarray | None) -> np.ndarray:
    """(spans, items) how strongly each span shows each item: its SigLIP relevance plus ITEM_OCR_EVIDENCE times the
    share of the item's words read on screen inside it (ocr None: no OCR in the index)."""
    return rel if ocr is None else rel + ITEM_OCR_EVIDENCE * ocr


def combine(rel_items: np.ndarray, rel_avoid: np.ndarray | None = None) -> np.ndarray:
    """rel_items: (items, steps) relevance per wanted item; rel_avoid: (avoid items, steps) or None. The best item
    per step, minus how far the best avoid item beats max(AVOID_FLOOR, that wanted relevance), capped at AVOID_MAX."""
    want = np.max(rel_items, axis=0)
    if rel_avoid is None or not len(rel_avoid):
        return want
    return want - np.clip(np.max(rel_avoid, axis=0) - np.maximum(AVOID_FLOOR, want), 0.0, AVOID_MAX)


def item_scores(rel_items: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    """(spans, items): each item's relevance over each grid-step span [i0, i1), as the mean of its best steps
    (vhe.rank.signals.span_top_mean, the segment_scores rule)."""
    if not len(rel_items):
        return np.zeros((len(i0), 0))
    return np.stack([span_top_mean(row, i0, i1) for row in rel_items], axis=1)


def item_centre(rel_items: np.ndarray) -> np.ndarray:
    """Each item's median relevance over the corpus: its baseline, subtracted before items are compared."""
    return np.median(rel_items, axis=1) if rel_items.size else np.zeros(len(rel_items))


def eligible(scores: np.ndarray, centre: np.ndarray, margin: float = np.inf) -> np.ndarray:
    """(spans, items) bool: the span matches the item above ITEM_MIN_REL, and (centred) within `margin` of the
    item it matches best."""
    lift = scores - centre[None, :]
    return (scores >= ITEM_MIN_REL) & (lift >= lift.max(1, keepdims=True) - margin)


def assign(scores: np.ndarray, centre: np.ndarray | None = None) -> np.ndarray:
    """The item each span serves: among the items it matches above ITEM_MIN_REL, the best centred match; -1 when
    it matches none."""
    if not scores.size:
        return np.full(len(scores), -1)
    centre = np.zeros(scores.shape[1]) if centre is None else centre
    lift = np.where(scores >= ITEM_MIN_REL, scores - centre[None, :], -np.inf)
    best = lift.argmax(1)
    return np.where(np.isfinite(lift[np.arange(len(scores)), best]), best, -1)
