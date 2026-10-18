"""Deterministic selection: greedy maximal marginal relevance with no temporal overlap and an optional total
duration budget. Near-duplicates are never picked twice: a clip in the same file as a pick and within DUP_GAP_S of it
(the same continuous action) or looking nearly the same (embedding cosine >= DUP_SIM) is dropped; other similar clips
are penalised by embedding similarity (MMR).

The stopping rule comes from the query: a fixed count, a runtime budget (min, max) filled in MMR order, or every
candidate above a threshold ("every moment someone speaks"). Candidates should already carry their final
(refined) edges, so budget and overlap checks hold for the clips that are returned. Across a folder of takes, a
per-file cap keeps one take from dominating; it defers rather than forbids, so a budget can still be filled."""
from __future__ import annotations

import numpy as np

import math

from vhe.query.parser import MAX_CLIPS

EXHAUSTIVE_MIN_TOTAL = 0.5  # in score units (about half a robust z above the length-bucket median)
# Near-duplicates, calibrated on the irl-event judge's "redundant with" labels (r2 report, 114 flagged pairs out of
# 2688 within-brief pairs, 2026-10-18): same file and gap <= 5 s, or same file and mean-embedding cosine >= 0.92,
# catches 86 of the 92 flagged same-file pairs and 28 of ~2600 unflagged ones (most of those adjacent talking-track
# sentences, which are exempt). Across files a cosine rule is not worth it (>= 0.95: 7 flagged vs 63 not), so the
# MMR penalty stays the only cross-file diversity.
DUP_GAP_S = 5.0
DUP_SIM = 0.92
EXHAUSTIVE_MIN_SPEECH = 0.5  # speech-intent exhaustive queries: fraction of the clip with voice activity
CAP_MIN_FILES = 4  # the per-file cap applies from this many files on


def per_file_cap(count: int, n_files: int) -> int | None:
    """At most max(2, ceil(count / 3)) clips per file; None (no cap) when there are only a few files."""
    return max(2, math.ceil(count / 3)) if n_files >= CAP_MIN_FILES else None


def overlap_fraction(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))


def duplicates(cands: list[dict], vectors: np.ndarray, o: dict, v: np.ndarray, gap: float = DUP_GAP_S,
               sim: float = DUP_SIM, talk_track: bool = False) -> np.ndarray:
    """bool per candidate: a near-duplicate of clip `o` (vector v). Same file and either within `gap` seconds of it
    (overlapping included: the same continuous action) or embedding cosine >= `sim`. talk_track (a brief that wants
    speech): two speech clips are distinct sentences unless they overlap (a talking track is consecutive sentences);
    in a visual edit a speech clip is a picture like any other."""
    if not cands:
        return np.zeros(0, dtype=bool)
    files = np.array([c.get("file", 0) for c in cands])
    starts = np.array([c["start"] for c in cands], dtype=float)
    ends = np.array([c["end"] for c in cands], dtype=float)
    speech = np.array([c.get("source") == "speech" for c in cands])
    between = np.maximum(starts, o["start"]) - np.minimum(ends, o["end"])  # < 0: overlap
    near = (between <= gap) | (vectors @ v >= sim)
    if talk_track and o.get("source") == "speech":
        near = np.where(speech, between < 0, near)
    return (files == o.get("file", 0)) & near


def select(cands: list[dict], vectors: np.ndarray, n: int, lam: float = 1.0, sim_floor: float = 0.6,
           max_overlap: float = 0.2, budget_s: float | tuple[float, float] | None = None, n_max: int | None = None,
           min_total: float | None = None, groups: np.ndarray | None = None, cap: int | None = None,
           items: np.ndarray | None = None, taken: list[dict] = (), taken_vectors: np.ndarray | None = None,
           dedupe: bool = True, talk_track: bool = False) -> list[dict]:
    """cands: dicts with 'start', 'end', 'total' (and 'file', 'source', 'penalty' when known); vectors:
    L2-normalised mean embedding per candidate (same order). budget_s: a number is an upper bound on the summed
    length; (lo, hi) keeps picking past n (up to n_max) until the sum reaches lo, and never exceeds hi. min_total:
    candidates scoring below it are never picked. groups/cap: once a group (file) has `cap` picks, its candidates are
    only used when nothing else is left. items: the shot item each candidate serves (-1: none). Picks go round robin
    over the items that still have candidates: every item once before any item twice, and so on; candidates serving
    no item only come in when no item has one left (coverage first, then the per-file cap, then MMR). A candidate's
    'penalty' (unclean edges) is subtracted from its total for the choice only. dedupe: near-duplicates
    (duplicates(), with talk_track) of a pick, or of a clip in `taken` (opener/closer, with taken_vectors), are never
    picked.
    Callers pass a pre-filtered pool; this is O(n * len(cands))."""
    if not cands:
        return []
    lo, hi = (0.0, np.inf) if budget_s is None else \
        (float(budget_s[0]), float(budget_s[1])) if isinstance(budget_s, (tuple, list)) else (0.0, float(budget_s))
    n_max = n if n_max is None else max(n, n_max)
    totals = np.array([c["total"] - c.get("penalty", 0.0) for c in cands], dtype=float)
    starts = np.array([c["start"] for c in cands], dtype=float)
    ends = np.array([c["end"] for c in cands], dtype=float)
    lengths = ends - starts
    alive = np.ones(len(cands), dtype=bool) if min_total is None else totals >= min_total
    alive &= lengths <= hi + 1e-6
    max_sim = np.full(len(cands), -1.0)
    for t, tv in zip(taken, taken_vectors if taken_vectors is not None else [None] * len(taken)):
        inter = np.clip(np.minimum(ends, t["end"]) - np.maximum(starts, t["start"]), 0.0, None)
        alive &= ~((np.array([c.get("file", 0) for c in cands]) == t.get("file", 0)) & (inter > 0))
        if tv is not None:
            if dedupe:
                alive &= ~duplicates(cands, vectors, t, tv, talk_track=talk_track)
            max_sim = np.maximum(max_sim, vectors @ tv)
    deferred = np.zeros(len(cands), dtype=bool)
    picks: dict[int, int] = {}
    chosen: list[int] = []
    used = 0.0
    items = None if items is None else np.asarray(items)
    item_picks: dict[int, int] = {}
    while alive.any() and len(chosen) < n_max and (len(chosen) < n or used < lo):
        penalty = lam * np.clip((max_sim - sim_floor) / (1.0 - sim_floor), 0.0, None)
        pool = alive
        if items is not None:  # round robin: an item with fewer picks first (every item once before any twice)
            live = np.unique(items[alive & (items >= 0)])
            if len(live):
                least = min(item_picks.get(int(i), 0) for i in live)
                pool = alive & np.isin(items, [i for i in live if item_picks.get(int(i), 0) == least])
        val = np.where(pool & ~deferred, totals - penalty, -np.inf)
        if not np.isfinite(val).any():
            val = np.where(pool, totals - penalty, -np.inf)
        i = int(np.argmax(val))
        if not np.isfinite(val[i]):
            break
        alive[i] = False
        if used + lengths[i] > hi + 1e-6:
            continue
        chosen.append(i)
        used += lengths[i]
        if items is not None and items[i] >= 0:
            item_picks[int(items[i])] = item_picks.get(int(items[i]), 0) + 1
        if groups is not None and cap is not None:
            picks[groups[i]] = picks.get(groups[i], 0) + 1
            if picks[groups[i]] >= cap:
                deferred |= groups == groups[i]
        inter = np.clip(np.minimum(ends, ends[i]) - np.maximum(starts, starts[i]), 0.0, None)
        alive &= inter / np.maximum(np.minimum(lengths, lengths[i]), 1e-9) <= max_overlap
        if dedupe:
            alive &= ~duplicates(cands, vectors, cands[i], vectors[i], talk_track=talk_track)
        max_sim = np.maximum(max_sim, vectors @ vectors[i])
    return [dict(cands[i], rank=r) for r, i in enumerate(chosen)]


def select_per_item(cands: list[dict], item_rel: np.ndarray, p, item_centre: np.ndarray | None = None,
                    vectors: np.ndarray | None = None) -> list[dict]:
    """"Every float: A, B, C ...": the best clip for each named item; an item with no clip matching it above
    ITEM_MIN_REL (and, centred, nearly as well as the clip's best item) is skipped rather than filled with another
    clip. A clip's score for item i is its score with the relevance part replaced by its relevance to i; (item,
    clip) pairs are taken best first, clips never overlap or duplicate one another (with vectors), and the budget
    (if any) caps the total."""
    from vhe.query.plan import ITEM_MARGIN, eligible

    if not cands:
        return []
    w = p.weights.get("relevance", 1.0)
    base = np.array([c["total"] - c.get("penalty", 0.0) - w * c.get("relevance", 0.0) for c in cands])
    centre = np.zeros(item_rel.shape[1]) if item_centre is None else np.asarray(item_centre)
    score = np.where(eligible(item_rel, centre, ITEM_MARGIN), base[:, None] + w * item_rel, -np.inf)
    hi = np.inf if p.budget is None else p.budget[1]
    picked: list[dict] = []
    used = 0.0
    while len(picked) < p.count and np.isfinite(score).any():
        j, i = np.unravel_index(int(np.argmax(score)), score.shape)
        c = cands[j]
        score[j, :] = -np.inf
        if used + c["end"] - c["start"] > hi + 1e-6 or \
                any(min(c["end"], o["end"]) > max(c["start"], o["start"]) for o in picked):
            continue
        picked.append(dict(c, item=int(i), rank=len(picked)))
        used += c["end"] - c["start"]
        score[:, i] = -np.inf
        if vectors is not None:
            score[duplicates(cands, vectors, c, vectors[j]), :] = -np.inf
    return picked


def select_for(cands: list[dict], vectors: np.ndarray, p, groups: np.ndarray | None = None,
               item_rel: np.ndarray | None = None, item_centre: np.ndarray | None = None,
               taken: list[dict] = (), taken_vectors: np.ndarray | None = None) -> list[dict]:
    """Selection for a parsed query (vhe.query.parser.ParsedQuery) over refined, final-edge candidates: clips
    never overlap or repeat a moment (duplicates()); a budget is filled to at least its minimum when the pool allows,
    and never past its maximum. groups: file index per candidate, for the per-file cap (not applied to exhaustive
    queries). item_rel: (candidates, shot items) relevance to each item of p.shots. With two or more items, selection
    covers distinct items before a second clip of one item. Exhaustive briefs: per named item ("every float: A, B")
    the best clip per item; otherwise ("every shot where ...") every distinct match. With a budget, "every X" only
    asks for coverage: the budget is filled the normal way. taken: clips already placed (opener/closer)."""
    covering = item_rel is not None and len(getattr(p, "shots", [])) >= 2 and len(item_rel) == len(cands)
    talk = p.weights.get("speech", 0.0) > 0  # the brief wants speech: consecutive sentences are distinct clips
    exhaustive = p.exhaustive and p.budget is None
    if covering and exhaustive and getattr(p, "per_item", True):
        return select_per_item(cands, item_rel, p, item_centre, vectors)
    items = np.array([c.get("item", -1) for c in cands]) if covering else None
    if exhaustive:  # every match: no diversity re-ordering, the budget (if any) only caps the total
        keep = list(range(len(cands)))
        if p.weights.get("speech", 0.0) > 0:  # "every moment someone speaks": only clips that are mostly speech
            keep = [k for k in keep if cands[k].get("speech_frac", 1.0) >= EXHAUSTIVE_MIN_SPEECH]
        if items is not None:  # "every shot where ...": clips that match one of the items
            keep = [k for k in keep if items[k] >= 0]
        cands, vectors = [cands[k] for k in keep], vectors[keep]
        return select(cands, vectors, p.count, lam=0.0, max_overlap=0.0, min_total=EXHAUSTIVE_MIN_TOTAL,
                      taken=taken, taken_vectors=taken_vectors, talk_track=talk)
    # A count that cannot fill the budget minimum even at the longest clips is a part of the brief ("only two or three
    # parade moments", "at least two shots of dancers"), not the number of clips in the edit.
    binding = p.count_explicit and (p.budget is None or p.count * getattr(p, "max_len", math.inf) >= p.budget[0])
    n_max = p.count if binding else MAX_CLIPS if p.budget is not None else p.count
    cap = None if groups is None else per_file_cap(p.count, len(set(groups.tolist())))
    return select(cands, vectors, p.count, max_overlap=0.0, budget_s=p.budget, n_max=n_max, groups=groups, cap=cap,
                  items=items, taken=taken, taken_vectors=taken_vectors, talk_track=talk)
