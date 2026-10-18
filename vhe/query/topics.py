"""Spoken topics (ParsedQuery.topics) against transcript sentences: Ollama bge-m3 embeddings when the server is
reachable, plus IDF-weighted keyword overlap (alone when it is not)."""
from __future__ import annotations

import json
import math
import re
import urllib.request
from datetime import datetime, timezone

import numpy as np

from vhe.query.llm import OLLAMA_HOST
from vhe.query.text_match import STOPWORDS

EMBED_MODEL = "bge-m3"
EMBED_TIMEOUT = 30.0
EMBED_KEEP_ALIVE = "0s"  # unload at once (as the parser LLM): the next query's SigLIP / CG-DETR share the 6 GB GPU
# bge-m3 cosines between transcript sentences and topic descriptions are compressed (on the irl-event footage:
# median 0.49 over all host speech, on-topic sentences 0.51-0.70), and an embedding alone misses literal cues
# ("copyright"); so the score is cosine + KEYWORD_BONUS * keyword overlap, kept from TOPIC_MIN_SIM on.
TOPIC_MIN_SIM = 0.53
KEYWORD_BONUS = 0.1
TOPIC_MIN_KEYWORD = 0.25  # fallback without embeddings: IDF-weighted share of the topic's words in the span


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
            out[j, k] = float(sum(w for t, w in zip(terms, idf, strict=True) if t in d) / idf.sum())
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
