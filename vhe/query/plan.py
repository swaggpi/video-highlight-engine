"""The query side of a brief plan (vhe.query.parser.ParsedQuery shots/avoid/topics/need_live_audio).

Relevance with a shot list is the best item per step (each item on the same absolute log-odds scale as a single
text, vhe.rank.signals.relevance), minus a bounded penalty where an avoid item matches better than both a floor and
the wanted item. A clip serves the item it matches best, if it matches it well enough; selection uses that to cover
distinct items. Spoken topics are matched against the transcript text of speech candidates: Ollama bge-m3 embeddings
when the server is reachable, otherwise IDF-weighted keyword overlap. need_live_audio drops clips whose sound is
muted."""
from __future__ import annotations

import json
import math
import re
import urllib.request
from datetime import datetime, timezone

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

LIVE_MIN_DB = -60.0  # loudness (dBFS) below this is muted: phone mutes are digital silence at -120 dB
LIVE_MAX_MUTED = 0.1  # need_live_audio: clips with a larger muted (or audio-less) fraction are dropped

EMBED_MODEL = "bge-m3"
EMBED_TIMEOUT = 30.0
EMBED_KEEP_ALIVE = "0s"  # unload at once (as the parser LLM): the next query's SigLIP / CG-DETR share the 6 GB GPU
# bge-m3 cosines between transcript sentences and topic descriptions are compressed (on the irl-event footage:
# median 0.49 over all host speech, on-topic sentences 0.51-0.70), and an embedding alone misses literal cues
# ("copyright"); so the score is cosine + KEYWORD_BONUS * keyword overlap, kept from TOPIC_MIN_SIM on.
TOPIC_MIN_SIM = 0.53
KEYWORD_BONUS = 0.1
TOPIC_MIN_KEYWORD = 0.25  # fallback without embeddings: IDF-weighted share of the topic's words in the span
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "these", "those", "his", "her", "their", "our", "you",
             "your", "are", "was", "were", "is", "be", "been", "being", "have", "has", "had", "him", "she", "they",
             "them", "what", "where", "when", "why", "how", "who", "which", "about", "into", "onto", "out", "some",
             "any", "all", "just", "going", "gonna", "here", "there", "not", "but", "can", "will", "would", "one",
             "its", "it's", "i'm", "we're", "you're", "der", "die", "das", "und", "mit", "ein", "eine", "von", "auf"}


# ---- shots and avoid ------------------------------------------------------------------------------------------------
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


def muted_steps(loudness: np.ndarray, has_audio: np.ndarray) -> np.ndarray:
    """True where there is no usable live sound: no audio track, or digital silence (a streamer's mute)."""
    loud = np.nan_to_num(np.asarray(loudness, dtype=float), nan=-120.0)
    return ~np.asarray(has_audio, dtype=bool) | (loud < LIVE_MIN_DB)


# ---- spoken topics --------------------------------------------------------------------------------------------------
def _keep_alive(model: str, host: str) -> str:
    """A short keep-alive, unless another client already holds the model loaded longer (keep its expiry)."""
    try:
        with urllib.request.urlopen(f"{host}/api/ps", timeout=2) as resp:
            for m in json.loads(resp.read()).get("models", []):
                if m.get("name", "").split(":")[0] == model.split(":")[0] and m.get("expires_at"):
                    left = (datetime.fromisoformat(m["expires_at"]) - datetime.now(timezone.utc)).total_seconds()
                    if left > 30:
                        return f"{int(left)}s"
    except Exception:
        pass
    return EMBED_KEEP_ALIVE


def embed(texts: list[str], model: str = EMBED_MODEL, host: str | None = None) -> np.ndarray | None:
    """L2-normalised Ollama embeddings (one batch), or None when the server or model is unavailable."""
    from vhe.query.parser import OLLAMA_HOST

    host = host or OLLAMA_HOST
    if not texts:
        return np.zeros((0, 1))
    body = {"model": model, "input": texts, "keep_alive": _keep_alive(model, host)}
    try:
        req = urllib.request.Request(f"{host}/api/embed", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            v = np.asarray(json.loads(resp.read())["embeddings"], dtype=np.float32)
    except Exception:
        return None
    if v.ndim != 2 or len(v) != len(texts):
        return None
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _terms(text: str) -> list[str]:
    """Content words, crudely stemmed (plural/-ing/-ed) so 'muting' meets 'mute' and 'streets' meets 'street'."""
    out = []
    for w in re.findall(r"[^\W\d_][\w']*", text.lower()):
        if len(w) < 3 or w in STOPWORDS:
            continue
        w = re.sub(r"(?:'s|ing|ed|es|s)$", "", w) if len(w) > 4 else w
        out.append(w.rstrip("e")[:6])
    return out


def keyword_scores(spans: list[str], topics: list[str]) -> np.ndarray:
    """(spans, topics): IDF-weighted share of each topic's terms that occur in each span (0..1)."""
    docs = [set(_terms(s)) for s in spans]
    n = len(docs)
    out = np.zeros((n, len(topics)))
    for k, topic in enumerate(topics):
        terms = sorted(set(_terms(topic)))
        if not terms:
            continue
        idf = np.array([math.log((n + 1) / (sum(t in d for d in docs) + 0.5)) for t in terms])
        idf = np.clip(idf, 0.1, None)
        for j, d in enumerate(docs):
            out[j, k] = float(sum(w for t, w in zip(terms, idf) if t in d) / idf.sum())
    return out


def topic_scores(spans: list[str], topics: list[str], embed_fn=None) -> tuple[np.ndarray, np.ndarray, str, float]:
    """(best score per span, best topic per span, method, threshold). method is 'bge-m3' or 'keywords'.
    embed_fn: texts -> normalised vectors or None (default: embed)."""
    if not spans or not topics:
        return np.zeros(len(spans)), np.zeros(len(spans), int), "none", math.inf
    kw = keyword_scores(spans, topics)
    vecs = (embed_fn or embed)(list(topics) + list(spans))
    if vecs is not None and len(vecs) == len(topics) + len(spans):
        sims = vecs[len(topics):] @ vecs[:len(topics)].T + KEYWORD_BONUS * kw
        method, thresh = EMBED_MODEL, TOPIC_MIN_SIM
    else:
        sims, method, thresh = kw, "keywords", TOPIC_MIN_KEYWORD
    return sims.max(1), sims.argmax(1), method, thresh


# ---- hard exclusions ------------------------------------------------------------------------------------------------
# "no protest signs", "absolutely no sanitation trucks": a clip showing an excluded item is dropped, not penalised.
# SigLIP relevance is not comparable across phrases (on the irl-event footage "empty street" has a corpus median of
# +0.75, "portraits of victims" never exceeds -0.7), so an item counts as present at a step when its relevance is in
# the item's own top decile over the corpus and above an absolute floor (log-odds about -7.4). Calibrated 2026-10-18
# on clips the judge failed for showing an excluded thing: protest signs +1.25..+2.17 (gate 0.5), sanitation trucks
# +2.27 (0.68), floats +2.34 (0.84), flag +0.65 (0.55); judged-fine clips of the same brief stayed below their gates
# except one flag-waving float at 0.59 for "protest signs". Abstract phrases ("portraits of victims") are not
# detected at all at this scale: the soft avoid penalty still applies to them.
EXCLUDE_QUANTILE = 0.9
EXCLUDE_MIN_REL = 0.25
EXCLUDE_CLEAR_REL = 1.5  # a clear match (log-odds about -4.3) always counts, also for an item common in the footage
EXCLUDE_MIN_RUN = 2  # grid steps (0.67 s at 3 fps)
EXCLUDE_MAX_FRAC = 0.1  # a clip with more of its steps showing an excluded item is dropped
DEFECT_MAX_FRAC = 0.1  # ... or with more of its steps showing an excluded camera defect ("no whip-pans")
NO_SPEECH_MAX_FRAC = 0.3  # "no host talking over it": clips with more voice activity are dropped
WHIP_SPEED = 0.15  # cam_speed of a whip-pan (follow pans are 0.02-0.06; vhe.query.refine EDGE_MAX_SPEED)
# Exclude items about the camera, not the content: the index's own defect signals answer them, not SigLIP.
DEFECTS = {
    "whip": r"\b(?:whip\w*|swish\w*|fast pans?|reißschwenks?)\b",
    "shaky": r"\b(?:shak\w*|wobbl\w*|jitter\w*|unsteady|verwackel\w*|wackler)\b",
    "blocked": r"\b(?:block\w*|cover\w*|obstruct\w*|lens|linse|finger\w*|hands?|in front of)\b",
    "framing": (r"\b(?:ground|floor|pavement|asphalt|boden|straight up)\b|"
                r"\b(?:camera|pointing|points|pointed)\b.*\bsky\b"),
}
GENERIC_WORDS = {"shot", "shots", "clip", "clips", "frame", "frames", "anything", "something", "any", "the", "with",
                 "filling", "fills", "fill", "across", "close", "closeup", "close-up", "close-ups", "view", "scene"}


def defect_kind(item: str) -> str | None:
    """'whip', 'shaky', 'blocked' or 'framing' for an exclude item about a camera defect, else None."""
    low = item.lower()
    return next((k for k, pat in DEFECTS.items() if re.search(pat, low)), None)


def defect_steps(grid: dict, kind: str, n: int) -> np.ndarray:
    """bool per grid step where the camera defect `kind` shows (from the usable-mask reasons and motion)."""
    from vhe.query.refine import EDGE_MAX_SHAKE
    from vhe.signals.quality import REASONS

    bits = np.asarray(grid["unusable"] if "unusable" in grid else np.zeros(n), dtype=np.uint8)

    def reason(*names: str) -> np.ndarray:
        return (bits & np.uint8(sum(1 << REASONS.index(r) for r in names))) > 0

    def above(key: str, limit: float) -> np.ndarray:
        return np.nan_to_num(np.asarray(grid[key] if key in grid else np.zeros(n), dtype=float), nan=0.0) > limit

    if kind == "whip":
        return above("cam_speed", WHIP_SPEED)
    if kind == "shaky":
        return reason("shaky") | above("shake", EDGE_MAX_SHAKE)
    if kind == "blocked":
        return reason("covered", "uniform", "obstructed")
    return reason("bad_framing")


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[^\W\d_][\w'-]*", text.lower()) if len(w) > 2 and w not in STOPWORDS
            and w not in GENERIC_WORDS}


def conflicts(item: str, wanted: list[str]) -> bool:
    """The exclude item names only things a wanted shot asks for ('frame with a barricade' against 'establishing wide
    with the barricades'): it then stays a soft penalty, never a gate that would remove the wanted shot."""
    stems = _stems(item)
    return bool(stems) and any(stems <= _stems(w) for w in wanted if w)


def exclusion_steps(rel_excl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rel_excl: (items, steps) SigLIP relevance of each exclude item. (bool per step where any item shows, the
    per-item thresholds): an item shows where its relevance is in its corpus top decile and above EXCLUDE_MIN_REL,
    or above EXCLUDE_CLEAR_REL in any case, for at least EXCLUDE_MIN_RUN steps in a row (a one-frame blip is not a
    sighting)."""
    rel_excl = np.atleast_2d(np.asarray(rel_excl, dtype=float))
    if not len(rel_excl) or not rel_excl.shape[1]:
        return np.zeros(rel_excl.shape[1], dtype=bool), np.zeros(len(rel_excl))
    thr = np.clip(np.quantile(rel_excl, EXCLUDE_QUANTILE, axis=1), EXCLUDE_MIN_REL, EXCLUDE_CLEAR_REL)
    hit = (rel_excl >= thr[:, None]).any(0)
    edges = np.diff(np.r_[0, hit.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < EXCLUDE_MIN_RUN:
            hit[a:b] = False
    return hit, thr


# ---- on-screen text (hook) ------------------------------------------------------------------------------------------
OCR_BONUS = 1.0  # relevance units for a span whose legible text contains all of an item's words


def ocr_item_scores(corpus, items: list[str], S: np.ndarray, E: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Hook for on-screen text: (spans, items) the share of each item's words found in the text read inside
    each span (0..1, matched without spaces), or None when the corpus exposes no text (indexes before on-screen text,
    or OCR not run): callers then change nothing.

    Reads `corpus.text_items` when it exists: [{"t": virtual seconds, "text", "conf", "file", ...}] (vhe.corpus, from
    each index's ocr.json). The engine adds OCR_BONUS times this to each span's item relevance, so a float whose name
    is legible serves its item ("PASARGAD float"). A legibility prior for "every shot where text is readable" briefs
    can come from the grid's text_area / text_conf the same way. S, E: virtual seconds; K: file index per span."""
    texts = getattr(corpus, "text_items", None)
    if not texts or not items or not len(S):
        return None
    t = np.array([float(x["t"]) for x in texts])
    order = np.argsort(t, kind="stable")
    t = t[order]
    read = [_squash(str(texts[j].get("text", ""))) for j in order]
    files = np.array([int(texts[j].get("file", 0)) for j in order])
    # OCR often drops spaces ("TOMBof CYRUS"): words are matched as substrings of the span's text with spaces removed
    terms = [[_squash(w) for w in re.findall(r"[^\W\d_][\w'-]*", x.lower()) if len(w) > 2 and w not in STOPWORDS]
             for x in items]
    out = np.zeros((len(S), len(items)))
    for i, (s, e, k) in enumerate(zip(S, E, K)):
        a, b = np.searchsorted(t, s, "left"), np.searchsorted(t, e, "right")
        text = "|".join(r for r, f in zip(read[a:b], files[a:b]) if f == k)
        if text:
            out[i] = [sum(w in text for w in ws) / len(ws) if ws else 0.0 for ws in terms]
    return out


LEGIBLE_CUE = (r"\b(?:legible|readable|read(?:s|ing)? the|on-screen text|exact (?:text|wording)|spelling|fact-check|"
               r"lesbar|schrift)\b")
LEGIBLE_BONUS = 1.0  # relevance units at full OCR confidence, for briefs that ask for readable text


def legibility_steps(grid: dict, raw: str) -> np.ndarray | None:
    """Per step 0..1: confident legible text in frame (grid text_conf where text_area > 0, index v3), for a brief that
    asks for readable text ("every shot where the text is legible"); None otherwise or without OCR in the index."""
    if "text_conf" not in grid or "text_area" not in grid or not re.search(LEGIBLE_CUE, raw.lower()):
        return None
    conf = np.nan_to_num(np.asarray(grid["text_conf"], dtype=float), nan=0.0)
    area = np.nan_to_num(np.asarray(grid["text_area"], dtype=float), nan=0.0)
    return np.clip(np.where(area > 0, conf, 0.0), 0.0, 1.0)


def _squash(text: str) -> str:
    return re.sub(r"[^0-9a-zäöüß]", "", text.lower())
