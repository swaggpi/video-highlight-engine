"""The two-stage ranking of gated candidates, and the pools of the special slots.

shortlist: stage 1, vectorised over every candidate, featurises (vhe.rank.features) and asks the Scorer for score();
stage 2 takes the best `size` (with a shot list also each item's own best), refines their edges (vhe.query.refine:
whole sentences, clean frames), gates the refined spans again, featurises them and asks rescore() for the totals
selection runs on. edge_clip: the opener / closer / hook slot from its own pool. topic_picks: transcript sentences
that talk about the brief's spoken topics.

context(k) -> file k's refine Context, muted -> per-step muted mask, span_vectors(spans) -> mean frame embeddings:
the engine's caches."""
from __future__ import annotations

import dataclasses

import numpy as np

from vhe.query.candidates import fit_utterances, speech_spans
from vhe.query.model import EDGE_WEIGHTS, HOOK_WEIGHTS, ParsedQuery
from vhe.query.refine import TALK_MAX_SILENCE, edges_clean, refine, refine_speech
from vhe.query.text_match import OCR_BONUS, OCR_GENERIC, ocr_item_scores
from vhe.query.topics import topic_scores
from vhe.rank.features import INDEX, featurize, query_vector
from vhe.rank.gate import LIVE_MAX_MUTED, gate, step_fraction
from vhe.rank.items import assign, item_scores
from vhe.rank.select import disjoint, duplicates
from vhe.rank.signals import span_mean

EDGE_POOL = 30
ITEM_POOL = 60  # with a shot list, each item's own best candidates join the pool
EDGE_PENALTY = 1.5  # score units off a visual clip whose first or last frames are not clean (whip-pan, black, blocked)


def shortlist(corpus, scorer, context, muted: np.ndarray, sig: dict, weights: dict, cands: tuple, p: ParsedQuery,
              size: int, persona: str | None = None, trace: dict | None = None) -> list[dict]:
    """Stage 1 and 2 (module docstring) on the best `size` candidates. Each returned candidate carries its total, its
    refined features ("x") and, with a shot list, the item it serves. cands: (S, E, K, src[, allowed]); allowed: the
    candidates the brief allows (hard exclusions): the rest are still scored (the scorer's length-bucket statistics
    are over all usable footage) but never shortlisted. trace: filled with the stage-1 features, scores, spans and
    query vector (for a full candidate dump)."""
    c = corpus
    S, E, K, src = cands[:4]
    allowed = cands[4] if len(cands) > 4 else np.ones(len(S), dtype=bool)
    q = query_vector(p, persona, scorer.personas, weights)
    X0 = featurize(c, sig, S, E, K, src, p)
    s0 = np.asarray(scorer.score(X0, q), dtype=float)
    if trace is not None:
        trace.update(X=X0, s=s0, S=S, E=E, K=K, src=src, q=q)
    ranked = np.where(allowed, s0, -np.inf)
    chosen = np.argsort(-ranked)[:min(size, int(allowed.sum()))]
    if sig.get("items") is not None and len(sig["items"]) > 1:
        # A generic item ("wide of the avenue", relevant everywhere) would fill the pool by itself: also take
        # each item's own best candidates, ranked with that item's relevance in place of the max.
        i0, i1 = c.steps(S, E, K)
        w = weights.get("relevance", 0.0)
        h_mean = X0[:, INDEX["h_rel_mean"]]
        extra = [np.argsort(-(ranked + w * (span_mean(row, i0, i1) - h_mean)))[:min(ITEM_POOL, int(allowed.sum()))]
                 for row in sig["items"]]
        chosen = np.concatenate([chosen, *extra])
        chosen = chosen[np.sort(np.unique(chosen, return_index=True)[1])]
    out, rows = [], []
    for j in chosen:
        k, off = int(K[j]), c.offsets[K[j]]
        ctx = context(k)
        # speech clips are whole sentences (a talking track without long silent tails); visual clips move onto
        # clean frames
        if src[j] == "speech":
            s, e = refine_speech(float(S[j] - off), float(E[j] - off), ctx, p.min_len, p.max_len,
                                 TALK_MAX_SILENCE if p.weights.get("speech", 0.0) > 0 else None)
        else:
            s, e = refine(float(S[j] - off), float(E[j] - off), ctx, p.min_len, p.max_len)
        clean = src[j] == "speech" or edges_clean(s, e, ctx)
        out.append({"start": off + s, "end": off + e, "file": k, "source": src[j],
                    "cuts_inside": int(X0[j, INDEX["n_cuts_inside"]]),
                    "relevance": float(X0[j, INDEX["h_rel_top60"]]), "clean_edges": bool(clean),
                    "penalty": 0.0 if clean else EDGE_PENALTY})
        rows.append(j)
    if not out:
        return out
    # Refinement moves edges by up to a second or so: gate the final spans again.
    keep = gate(c, *(np.array([o[key] for o in out]) for key in ("start", "end", "file")), p, muted, sig)
    out = [o for o, ok in zip(out, keep, strict=True) if ok]
    rows = np.array([j for j, ok in zip(rows, keep, strict=True) if ok], dtype=int)
    if not out:
        return out
    S1, E1, K1 = (np.array([o[key] for o in out]) for key in ("start", "end", "file"))
    scores = None
    if sig.get("items") is not None:  # which shot item each final span serves
        r0, r1 = c.steps(S1, E1, K1)
        scores = item_scores(sig["items"], r0, r1)
        # legible on-screen text naming an item (no-op without OCR); a readable-text brief counts its names only
        ocr = ocr_item_scores(c, p.shots, S1, E1, K1, OCR_GENERIC if p.legible_text else frozenset())
        if ocr is not None:
            scores = scores + OCR_BONUS * ocr
        for o, row, it in zip(out, scores, assign(scores, sig["item_centre"]), strict=True):
            o["item_rel"], o["item"] = row, int(it)
    X1 = featurize(c, sig, S1, E1, K1, [o["source"] for o in out], p, item_rel=scores)
    totals = scorer.rescore(s0[rows], X0[rows], X1, q)
    for o, t, x in zip(out, totals, X1, strict=True):
        o["total"], o["x"], o["unusable_frac"] = float(t), x, float(x[INDEX["unusable_frac"]])
    return out


def edge_clip(corpus, scorer, context, muted: np.ndarray, span_vectors, sig: dict, rel: np.ndarray | None,
              avoided: np.ndarray | None, cands: tuple, p: ParsedQuery, taken: list[dict], max_len: float,
              persona: str | None = None, hook: bool = False) -> dict | None:
    """Best clip for the opener/closer description: its own relevance (rel; avoided: the same minus the brief's avoid
    penalty; both None without a description), without the brief's highlight saliency, steady and clean, not
    overlapping `taken`, framed as the brief asks (a wide brief's closer is a wide). The description's own framing
    words ("an establishing wide") work through its relevance: a shot-scale bonus on top moved p1-videographer-1's
    opener from the avenue with the crowd to a later, emptier wide. hook: the hook slot, scored with HOOK_WEIGHTS and
    framing "close" (shot scale, the largest face, something coming at the lens), on the description's relevance or,
    without one, the brief's, and at least the brief's minimum clip length (a hook fills its first second)."""
    if rel is not None:
        edge_sig = dict(sig, relevance=avoided, siglip=rel, avoid_pen=rel - avoided, nc_sal=None, items=None,
                        item_centre=None)
    else:
        edge_sig = dict(sig, items=None, item_centre=None)
    # ocr_item_match measures the shot items' names, not the slot's description: no readable-text term here
    framing = "close" if hook else p.framing
    slot_p = p if (framing, p.legible_text) == (p.framing, False) else \
        dataclasses.replace(p, framing=framing, legible_text=False)
    pool = shortlist(corpus, scorer, context, muted, edge_sig, HOOK_WEIGHTS if hook else EDGE_WEIGHTS, cands, slot_p,
                     EDGE_POOL, persona)
    pool = [o for o in pool if o["end"] - o["start"] <= max_len + 1e-6 and disjoint(o, taken)]
    if hook:
        pool = [o for o in pool if o["end"] - o["start"] >= p.min_len - 1e-6] or pool
    if pool and taken:  # the opener and the closer are not the same moment
        vec = span_vectors(pool)
        for t, tv in zip(taken, span_vectors(taken), strict=True):
            pool = [o for o, dup in zip(pool, duplicates(pool, vec, t, tv), strict=True) if not dup]
            vec = span_vectors(pool)
    return max(pool, key=lambda o: o["total"] - o.get("penalty", 0.0)) if pool else None


def topic_picks(corpus, context, muted: np.ndarray, p: ParsedQuery, n: int | None,
                length: tuple[float, float]) -> tuple[list[dict], int, str]:
    """Transcript-sentence spans (speech_spans) whose words match p.topics: (picks in time order, number of spans
    scored, match method). n=None keeps every span above the threshold; otherwise the best n above it. Picture
    defects do not gate speech (the speaker may be off camera); muted spans are dropped when the brief asks for live
    sound. Each pick: {"start", "end", "file", "text", "total", "source": "speech", "item": the topic}."""
    c = corpus
    lo, hi = length
    spans = []
    for k, ix in enumerate(c.members):
        dur = float(c.durations[k])
        for s, e in fit_utterances(speech_spans((ix.transcript or {}).get("segments", []), ix.vad, hi), lo, hi,
                                   dur):
            s, e = refine_speech(s, e, context(k), lo, hi, TALK_MAX_SILENCE)  # whole sentences
            text = c.words_between(c.offsets[k] + s, c.offsets[k] + e)
            if text:
                spans.append({"start": c.offsets[k] + s, "end": c.offsets[k] + e, "file": k, "text": text})
    if p.need_live_audio and spans:
        frac = step_fraction(c, *(np.array([o[key] for o in spans]) for key in ("start", "end", "file")), muted)
        spans = [o for o, m in zip(spans, frac, strict=True) if m <= LIVE_MAX_MUTED]
    score, best, method, thresh = topic_scores([o["text"] for o in spans], p.topics)
    order = [j for j in np.argsort(-score, kind="stable") if score[j] >= thresh]
    picked: list[dict] = []
    for j in order:
        o = spans[j]
        if n is not None and len(picked) >= n:
            break
        if disjoint(o, picked):
            picked.append(dict(o, total=float(score[j]), source="speech", item=p.topics[int(best[j])]))
    picked.sort(key=lambda o: o["start"])
    return picked, len(spans), method
