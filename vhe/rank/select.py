"""Deterministic selection: greedy maximal marginal relevance with no temporal overlap and an optional total
duration budget. Near-duplicates are never picked twice: a clip in the same file as a pick and within DUP_GAP_S of it
(the same continuous action) or looking nearly the same (embedding cosine >= DUP_SIM) is dropped; other similar clips
are penalised by embedding similarity (MMR).

The stopping rule comes from the query: a fixed count, a runtime budget (min, max) filled in MMR order, or every
candidate above a threshold ("every moment someone speaks"; a visual "every shot of X" keeps only the strong matches
of each item, strong_matches, and never two back-to-back takes of one view). Candidates should already carry their final
(refined) edges, so budget and overlap checks hold for the clips that are returned. Across a folder of takes, a
per-file cap keeps one take from dominating; it defers rather than forbids, so a budget can still be filled.

Scores are compared, never read as absolute values: the exhaustive thresholds and the MMR penalty are in units of the
brief's score spread (ScoreScale: the median and robust spread of the stage-1 scores of every gated candidate), so a
scorer on any scale (a learned model, or the heuristic times 7 plus 100) picks what the heuristic picks when it ranks
alike. Calibration (2026-10-18, irl-event dev briefs): the heuristic's stage-1 spread is 1.6-3.2 score units (median
2.4; 1.7 and 2.2 on p3-newsedit-1 and p1-videographer-2, the briefs the exhaustive margins were tuned on) and its
median about 0 (-0.1..0.24), so 1 spread = 2 heuristic units: EXHAUSTIVE_MIN_Z 0.25 is the former 0.5 units,
EXHAUSTIVE_SCORE_MARGIN 1.25 the former 2.5, MMR_LAMBDA 0.5 the former 1.0."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vhe.query.model import MAX_CLIPS
from vhe.query.refine import refine
from vhe.rank.items import ITEM_MARGIN, eligible


@dataclass(frozen=True)
class ScoreScale:
    """Where a brief's scores sit: loc (median) and spread (robust standard deviation) of its stage-1 scores."""

    loc: float = 0.0
    spread: float = 1.0


def score_scale(scores) -> ScoreScale:
    """ScoreScale of a pool of scores: the median, and the interquartile range / 1.349 (the standard deviation when
    that is 0, 1 when every score is equal)."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if not len(s):
        return ScoreScale()
    q25, q50, q75 = np.percentile(s, [25, 50, 75])
    spread = (q75 - q25) / 1.349
    if not spread > 0:
        spread = float(s.std())
    return ScoreScale(float(q50), float(spread) if spread > 0 else 1.0)


# The heuristic's typical stage-1 pool (calibration above): the scale callers without a pool get, in which every
# threshold is its former absolute value in heuristic score units.
HEURISTIC_SCALE = ScoreScale(0.0, 2.0)
EXHAUSTIVE_MIN_Z = 0.25  # exhaustive lists: totals at least this many spreads above the pool's median
MMR_LAMBDA = 0.5  # spreads off a candidate at full similarity to a pick (above the similarity floor)
# Near-duplicates, calibrated on the irl-event judge's "redundant with" labels (r2 report, 114 flagged pairs out of
# 2688 within-brief pairs, 2026-10-18): same file and gap <= 5 s, or same file and mean-embedding cosine >= 0.92,
# catches 86 of the 92 flagged same-file pairs and 28 of ~2600 unflagged ones (most of those adjacent talking-track
# sentences, which are exempt). Across files a cosine rule is not worth it (>= 0.95: 7 flagged vs 63 not), so the
# MMR penalty stays the only cross-file diversity.
DUP_GAP_S = 5.0
DUP_SIM = 0.92
EXHAUSTIVE_MIN_SPEECH = 0.5  # speech-intent exhaustive queries: fraction of the clip with voice activity
# "every usable shot of X" / "every shot where ...", visual: precision over recall. A blind judge failed r3's long
# lists (p1-videographer-2: 29 picks, mean fit 0.86, hit rate 0.24; Opus 5 picks at 2.4) for padding with
# non-matches. A span is kept when
#   item relevance >= EXHAUSTIVE_MIN_REL (absolute) and >= the item's best match - EXHAUSTIVE_ITEM_MARGIN, and
#   total >= the item's best clean span - EXHAUSTIVE_SCORE_MARGIN spreads (and >= the pool median + EXHAUSTIVE_MIN_Z
#   spreads), with clean edges and at most EXHAUSTIVE_MAX_UNUSABLE unusable steps (the quality bar).
# Relevance here is the scorer-independent match (vhe.rank.items.evidence: SigLIP plus on-screen names); without a
# shot list the content text is the one item (relevance: best-60 % of the brief's per-step relevance). Calibrated
# 2026-10-18 on the r3 judge's fit scores (both systems' clips, matched to this pool) of p1-videographer-2 and
# p3-newsedit-1, as (spans kept, mean fit 0-3, share fit >= 2, judged fit >= 2 spans kept), in heuristic units:
#   margins 1.0 / 1.0: 7 / 2.57 / 1.00 / 7 of 11  and 13 / 2.62 / 0.92 / 12 of 36  (precise, misses good shots)
#   margins 1.5 / 2.5: 13 / 2.31 / 0.85 / 11 of 11 and 20 / 2.55 / 0.90 / 18 of 36  (chosen)
#   margins 2.0 / 3.0: 14 / 2.21 / 0.79 / 11 of 11 and 29 / 2.45 / 0.86 / 25 of 36  (recall: more, weaker)
# The fit-3 dancers sat 0.6-1.3 below their item's best match, the fit-3 float names of the fact-check list down
# to 2.5 score units below their item's best: larger margins buy recall (a fact-check list) with precision.
EXHAUSTIVE_MIN_REL = 0.5  # = vhe.rank.items.ITEM_MIN_REL: below it a span serves no item at all
EXHAUSTIVE_ITEM_MARGIN = 1.5  # relevance units below the item's best match
EXHAUSTIVE_SCORE_MARGIN = 1.25  # spreads below the item's best clean span (2.5 heuristic units at spread 2)
EXHAUSTIVE_MAX_UNUSABLE = 0.05  # fraction of unusable steps (the candidate gate allows 0.2)
EXHAUSTIVE_VISUAL_CAP = 20  # at most this many picks for a visual "every X" list without a count (Opus: 5-18)
# Exhaustive lists also never hold two back-to-back takes of the same view: same file, within EXHAUSTIVE_DUP_GAP_S
# and mean-embedding cosine >= EXHAUSTIVE_DUP_SIM. On the r3 judge labels, same-file pairs 5-10 s apart at cosine
# >= 0.9 were flagged redundant 10 times in 18 (at 10-15 s: 4 in 14), against 114 in 2688 over all pairs.
EXHAUSTIVE_DUP_GAP_S = 10.0
EXHAUSTIVE_DUP_SIM = 0.9
CAP_MIN_FILES = 4  # the per-file cap applies from this many files on
EXTEND_STEP_S = 0.5  # budget fill: picks grow in steps of this many seconds


def per_file_cap(count: int, n_files: int) -> int | None:
    """At most max(2, ceil(count / 3)) clips per file; None (no cap) when there are only a few files."""
    return max(2, math.ceil(count / 3)) if n_files >= CAP_MIN_FILES else None


def disjoint(o: dict, others) -> bool:
    """True if span `o` overlaps none of `others` (touching is fine). Spans are in virtual seconds, so spans of
    different files never overlap."""
    return all(min(o["end"], t["end"]) <= max(o["start"], t["start"]) for t in others)


def duplicates(cands: list[dict], vectors: np.ndarray, o: dict, v: np.ndarray, gap: float = DUP_GAP_S,
               sim: float = DUP_SIM, talk_track: bool = False,
               back_to_back: tuple[float, float] | None = None) -> np.ndarray:
    """bool per candidate: a near-duplicate of clip `o` (vector v). Same file and either within `gap` seconds of it
    (overlapping included: the same continuous action) or embedding cosine >= `sim`. back_to_back (gap2, sim2):
    also same file, within gap2 seconds and cosine >= sim2 (exhaustive lists: two takes of the same view in a row).
    talk_track (a brief that wants speech): two speech clips are distinct sentences unless they overlap (a talking
    track is consecutive sentences); in a visual edit a speech clip is a picture like any other."""
    if not cands:
        return np.zeros(0, dtype=bool)
    files = np.array([c.get("file", 0) for c in cands])
    starts = np.array([c["start"] for c in cands], dtype=float)
    ends = np.array([c["end"] for c in cands], dtype=float)
    speech = np.array([c.get("source") == "speech" for c in cands])
    between = np.maximum(starts, o["start"]) - np.minimum(ends, o["end"])  # < 0: overlap
    cos = vectors @ v
    near = (between <= gap) | (cos >= sim)
    if back_to_back is not None:
        near |= (between <= back_to_back[0]) & (cos >= back_to_back[1])
    if talk_track and o.get("source") == "speech":
        near = np.where(speech, between < 0, near)
    return (files == o.get("file", 0)) & near


def select(cands: list[dict], vectors: np.ndarray, n: int, lam: float = 1.0, sim_floor: float = 0.6,
           max_overlap: float = 0.2, budget_s: float | tuple[float, float] | None = None, n_max: int | None = None,
           min_total: float | None = None, groups: np.ndarray | None = None, cap: int | None = None,
           items: np.ndarray | None = None, taken: list[dict] = (), taken_vectors: np.ndarray | None = None,
           dedupe: bool = True, talk_track: bool = False,
           back_to_back: tuple[float, float] | None = None) -> list[dict]:
    """cands: dicts with 'start', 'end', 'total' (and 'file', 'source' when known); vectors: L2-normalised mean
    embedding per candidate (same order). lam: the MMR penalty at full similarity, in score units (select_for passes
    MMR_LAMBDA spreads). budget_s: a number is an upper bound on the summed length; (lo, hi) keeps picking past n (up
    to n_max) until the sum reaches lo, and never exceeds hi. min_total: candidates scoring below it are never picked.
    groups/cap: once a group (file) has `cap` picks, its candidates are only used when nothing else is left. items:
    the shot item each candidate serves (-1: none). Picks go round robin over the items that still have candidates:
    every item once before any item twice, and so on; candidates serving no item only come in when no item has one
    left (coverage first, then the per-file cap, then MMR). dedupe: near-duplicates (duplicates(), with talk_track and
    back_to_back) of a pick, or of a clip in `taken` (opener/closer, with taken_vectors), are never picked.
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
    for t, tv in zip(taken, taken_vectors if taken_vectors is not None else [None] * len(taken), strict=True):
        inter = np.clip(np.minimum(ends, t["end"]) - np.maximum(starts, t["start"]), 0.0, None)
        alive &= ~((np.array([c.get("file", 0) for c in cands]) == t.get("file", 0)) & (inter > 0))
        if tv is not None:
            if dedupe:
                alive &= ~duplicates(cands, vectors, t, tv, talk_track=talk_track, back_to_back=back_to_back)
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
            alive &= ~duplicates(cands, vectors, cands[i], vectors[i], talk_track=talk_track,
                                 back_to_back=back_to_back)
        max_sim = np.maximum(max_sim, vectors @ vectors[i])
    return [dict(cands[i], rank=r) for r, i in enumerate(chosen)]


def select_per_item(cands: list[dict], item_rel: np.ndarray, p, item_centre: np.ndarray | None = None,
                    vectors: np.ndarray | None = None, item_scores: np.ndarray | None = None) -> list[dict]:
    """"Every float: A, B, C ...": the best clip for each named item; an item with no clip matching it above
    ITEM_MIN_REL (and, centred, nearly as well as the clip's best item) is skipped rather than filled with another
    clip. item_rel: (clips, items) evidence of each item (vhe.rank.items.evidence); item_scores: (clips, items) the
    scorer's score of each clip for each item (Scorer.score_items; default: the candidates' "item_scores", else
    their totals). (item, clip) pairs are taken best first, clips never overlap or duplicate one another (with
    vectors), and the budget (if any) caps the total."""
    if not cands:
        return []
    if item_scores is None:
        item_scores = np.array([c["item_scores"] if "item_scores" in c else np.full(item_rel.shape[1], c["total"])
                                for c in cands], dtype=float).reshape(item_rel.shape)
    centre = np.zeros(item_rel.shape[1]) if item_centre is None else np.asarray(item_centre)
    score = np.where(eligible(item_rel, centre, ITEM_MARGIN), np.asarray(item_scores, dtype=float), -np.inf)
    hi = np.inf if p.budget is None else p.budget[1]
    picked: list[dict] = []
    used = 0.0
    while len(picked) < p.count and np.isfinite(score).any():
        j, i = np.unravel_index(int(np.argmax(score)), score.shape)
        c = cands[j]
        score[j, :] = -np.inf
        if used + c["end"] - c["start"] > hi + 1e-6 or not disjoint(c, picked):
            continue
        picked.append(dict(c, item=int(i), rank=len(picked)))
        used += c["end"] - c["start"]
        score[:, i] = -np.inf
        if vectors is not None:
            score[duplicates(cands, vectors, c, vectors[j]), :] = -np.inf
    return picked


def strong_matches(cands: list[dict], items: np.ndarray | None = None, item_rel: np.ndarray | None = None,
                   scale: ScoreScale | None = None) -> list[int]:
    """Indices of the candidates a visual "every X" list keeps (see EXHAUSTIVE_*): clean edges ('clean_edges'), at
    most EXHAUSTIVE_MAX_UNUSABLE unusable steps, and per item (the shot item each candidate serves, -1: none, dropped;
    no items: one item, the content text, with the candidate's 'relevance') a relevance above EXHAUSTIVE_MIN_REL and
    within EXHAUSTIVE_ITEM_MARGIN of the item's best match, and a total within EXHAUSTIVE_SCORE_MARGIN spreads (scale;
    default HEURISTIC_SCALE) of the item's best clean candidate. item_rel: (candidates, items) evidence of each
    shot item."""
    if not cands:
        return []
    n = len(cands)
    item = np.zeros(n, dtype=int) if items is None else np.asarray(items, dtype=int)
    if items is None or item_rel is None:
        rel = np.array([c.get("relevance", 0.0) for c in cands], dtype=float)
        floor = -np.inf if items is None else EXHAUSTIVE_MIN_REL
    else:
        rel = np.asarray(item_rel, dtype=float)[np.arange(n), np.clip(item, 0, None)]
        floor = EXHAUSTIVE_MIN_REL
    total = np.array([c["total"] for c in cands], dtype=float)
    margin = EXHAUSTIVE_SCORE_MARGIN * (scale or HEURISTIC_SCALE).spread
    clean = np.array([c.get("clean_edges", True) and c.get("unusable_frac", 0.0) <= EXHAUSTIVE_MAX_UNUSABLE
                      for c in cands])
    keep = np.zeros(n, dtype=bool)
    for i in np.unique(item[item >= 0]):
        mine = item == i
        good = mine & clean
        if not good.any():
            continue
        keep |= good & (rel >= floor) & (rel >= rel[mine].max() - EXHAUSTIVE_ITEM_MARGIN) \
            & (total >= total[good].max() - margin)
    return [int(j) for j in np.flatnonzero(keep)]


def select_for(cands: list[dict], vectors: np.ndarray, p, groups: np.ndarray | None = None,
               item_rel: np.ndarray | None = None, item_centre: np.ndarray | None = None,
               taken: list[dict] = (), taken_vectors: np.ndarray | None = None,
               scale: ScoreScale | None = None) -> list[dict]:
    """Selection for a parsed query (vhe.query.model.ParsedQuery) over refined, final-edge candidates: clips
    never overlap or repeat a moment (duplicates()); a budget is filled to at least its minimum when the pool allows,
    and never past its maximum. groups: file index per candidate, for the per-file cap (not applied to exhaustive
    queries). item_rel: (candidates, shot items) relevance to each item of p.shots. With two or more items, selection
    covers distinct items before a second clip of one item. Exhaustive briefs: per named item ("every float: A, B")
    the best clip per item; a speech brief ("every moment someone speaks") every mostly-speech clip; otherwise
    ("every shot where ...") the strong, distinct matches (strong_matches; back-to-back takes of one view count as
    duplicates), at most EXHAUSTIVE_VISUAL_CAP without an explicit count. With a budget, "every X" only asks for
    coverage: the budget is filled the normal way. taken: clips already placed (opener/closer). scale: the brief's
    ScoreScale (the engine passes its stage-1 pool's; default HEURISTIC_SCALE): every score threshold and the MMR
    penalty are in its spreads. The exhaustive floor never rises above the best kept candidate, so a list with
    matches is never empty on any score scale."""
    covering = item_rel is not None and len(getattr(p, "shots", [])) >= 2 and len(item_rel) == len(cands)
    talk = p.weights.get("speech", 0.0) > 0  # the brief wants speech: consecutive sentences are distinct clips
    exhaustive = p.exhaustive and p.budget is None
    scale = scale or HEURISTIC_SCALE
    if covering and exhaustive and getattr(p, "per_item", True):
        return select_per_item(cands, item_rel, p, item_centre, vectors)
    items = np.array([c.get("item", -1) for c in cands]) if covering else None
    if exhaustive:  # every match: no diversity re-ordering, the budget (if any) only caps the total
        if talk:  # "every moment someone speaks": every sentence that is mostly speech (each says something else)
            keep = [k for k in range(len(cands)) if cands[k].get("speech_frac", 1.0) >= EXHAUSTIVE_MIN_SPEECH
                    and (items is None or items[k] >= 0)]
            n, b2b = p.count, None
        else:  # "every shot where ...": only strong, distinct matches of an item, a capped list
            keep = strong_matches(cands, items, item_rel if items is not None else None, scale)
            n = p.count if getattr(p, "count_explicit", False) else min(p.count, EXHAUSTIVE_VISUAL_CAP)
            b2b = (EXHAUSTIVE_DUP_GAP_S, EXHAUSTIVE_DUP_SIM)
        cands, vectors = [cands[k] for k in keep], vectors[keep]
        floor = scale.loc + EXHAUSTIVE_MIN_Z * scale.spread
        if cands:
            floor = min(floor, max(c["total"] for c in cands))
        return select(cands, vectors, n, lam=0.0, max_overlap=0.0, min_total=floor,
                      taken=taken, taken_vectors=taken_vectors, talk_track=talk, back_to_back=b2b)
    # A count that cannot fill the budget minimum even at the longest clips is a part of the brief ("only two or three
    # parade moments", "at least two shots of dancers"), not the number of clips in the edit.
    binding = p.count_explicit and (p.budget is None or p.count * getattr(p, "max_len", math.inf) >= p.budget[0])
    n_max = p.count if binding else MAX_CLIPS if p.budget is not None else p.count
    cap = None if groups is None else per_file_cap(p.count, len(set(groups.tolist())))
    return select(cands, vectors, p.count, lam=MMR_LAMBDA * scale.spread, max_overlap=0.0, budget_s=p.budget,
                  n_max=n_max, groups=groups, cap=cap, items=items, taken=taken, taken_vectors=taken_vectors,
                  talk_track=talk)


def fill_budget(body: list[dict], edges: list[dict], p, corpus, context, keep, other_s: float = 0.0) -> list[dict]:
    """Below the budget minimum once the pool is used up: lengthen the visual picks (best first) toward the
    brief's per-clip maximum, in EXTEND_STEP_S steps forward then backward, within their shot (no cut crossed),
    keeping DUP_GAP_S from other picks in the file, passing the gates (keep(S, E, K) -> bool per span: vhe.rank.gate)
    and ending on clean frames. context(k): file k's refine Context. other_s: runtime already taken outside body and
    edges (a soundbite)."""
    lo, hi = p.budget
    used = sum(o["end"] - o["start"] for o in [*body, *edges]) + other_s
    if used >= lo - 1e-6 or not body:
        return body
    c = corpus
    others = [*body, *edges]
    for o in sorted(body, key=lambda o: -o["total"]):
        if used >= lo - 1e-6:
            break
        if o.get("source") == "speech":  # sentences are already whole
            continue
        k, off = o["file"], c.offsets[o["file"]]
        ctx = context(k)
        cuts = [off + x for x in ctx.cuts]
        near = [x for x in others if x is not o and x["file"] == k]
        limit_hi = min([off + ctx.duration] + [x["start"] - DUP_GAP_S for x in near if x["start"] >= o["end"]]
                       + [x for x in cuts if x > o["end"] + 1e-6])
        limit_lo = max([off] + [x["end"] + DUP_GAP_S for x in near if x["end"] <= o["start"]]
                       + [x for x in cuts if x < o["start"] - 1e-6])
        s, e = o["start"], o["end"]
        base = used - (e - s)  # runtime without this clip
        for direction in (1, -1):
            while base + (e - s) < lo and e - s + EXTEND_STEP_S <= p.max_len + 1e-6 \
                    and base + (e - s) + EXTEND_STEP_S <= hi + 1e-6:
                s2, e2 = (s, e + EXTEND_STEP_S) if direction > 0 else (s - EXTEND_STEP_S, e)
                if e2 > limit_hi + 1e-6 or s2 < limit_lo - 1e-6:
                    break
                if not keep(np.array([s2]), np.array([e2]), np.array([k]))[0]:
                    break
                s, e = s2, e2
        if (s, e) == (o["start"], o["end"]):
            continue
        ls, le = refine(s - off, e - off, ctx, p.min_len, p.max_len)  # back onto clean frames and word gaps
        ls, le = max(ls, limit_lo - off), min(le, limit_hi - off)
        if le - ls > o["end"] - o["start"]:
            used = base + (le - ls)
            o["start"], o["end"], o["extended"] = off + ls, off + le, True
    return body
