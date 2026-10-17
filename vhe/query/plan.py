"""The query side of a brief plan (vhe.query.parser.ParsedQuery shots/avoid/topics/need_live_audio).

Relevance with a shot list is the best item per step (each item on the same absolute log-odds scale as a single
text, score.relevance), minus a bounded penalty where an avoid item matches better than both a floor and the wanted
item. A clip serves the item it matches best, if it matches it well enough; selection uses that to cover distinct
items. Spoken topics are matched against the transcript text of speech candidates: Ollama bge-m3 embeddings when
the server is reachable, otherwise IDF-weighted keyword overlap. need_live_audio drops clips whose sound is muted."""
from __future__ import annotations

import json
import math
import re
import urllib.request
from datetime import datetime, timezone

import numpy as np

# Relevance units (score.relevance): 0 is log-odds -8, one unit 2.5 nats, clipped to +-3.
AVOID_FLOOR = 1.2  # an avoid item matching below this (log-odds about -5) is noise and never penalised
AVOID_MAX = 1.0  # at most one unit off: avoid never sinks a clearly relevant shot
# A clip serves an item only above this (log-odds about -6.75, a partial SigLIP match). On the irl-event footage
# specific items peak at +0.5..+1.5 ("dancers in traditional costumes") while generic ones sit at +1.8 everywhere
# ("wide of the empty avenue"), so which item a clip serves is decided on relevance centred by each item's corpus
# median (how unusual the match is for that item), and only this floor is absolute.
ITEM_MIN_REL = 0.5
ITEM_MARGIN = 0.5  # exhaustive per-item picks: the clip must match this item nearly as well as its best item
TOP_FRAC = 0.6  # a span is as relevant as its best 60 % of steps (same rule as score.segment_scores)

LIVE_MIN_DB = -60.0  # loudness (dBFS) below this is muted: phone mutes are digital silence at -120 dB
LIVE_MAX_MUTED = 0.1  # need_live_audio: clips with a larger muted (or audio-less) fraction are dropped

EMBED_MODEL = "bge-m3"
EMBED_TIMEOUT = 30.0
EMBED_KEEP_ALIVE = "30s"
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
    """(spans, items): each item's relevance over each grid-step span [i0, i1), as the mean of its best steps."""
    out = np.zeros((len(i0), len(rel_items)))
    for j, (a, b) in enumerate(zip(np.asarray(i0, int), np.asarray(i1, int))):
        block = np.sort(rel_items[:, a:max(b, a + 1)], axis=1)
        out[j] = block[:, -max(1, int(np.ceil(TOP_FRAC * block.shape[1]))):].mean(1)
    return out


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
