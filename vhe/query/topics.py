"""Spoken topics (ParsedQuery.topics) against transcript sentences: Ollama bge-m3 embeddings when the server is
reachable, plus IDF-weighted keyword overlap (alone when it is not)."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vhe.query.llm import OLLAMA_HOST
from vhe.query.text_match import STOPWORDS

EMBED_MODEL = "bge-m3"
EMBED_TIMEOUT = 30.0
EMBED_CACHE = Path.home() / ".cache" / "vhe" / "embed.sqlite"  # raw vectors per (model, digest, sha256 of the text)
EMBED_KEEP_ALIVE = "0s"  # unload at once (as the parser LLM): the next query's SigLIP / CG-DETR share the 6 GB GPU
EMBED_BATCH = 16  # texts per request (vectors do not depend on the batch: see embed)
EMBED_HOLD_S = 60  # keep-alive between the requests of one embed() call
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


def _digest(model: str, host: str) -> str | None:
    """The installed model's digest (Ollama /api/tags), or None when the server or model is unavailable."""
    name = model if ":" in model else model + ":latest"
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            for m in json.loads(resp.read()).get("models", []):
                if name in (m.get("name"), m.get("model")) and m.get("digest"):
                    return str(m["digest"])
    except Exception:
        pass
    return None


def _batches(texts: list[str], model: str, host: str):
    """(texts, raw float32 embeddings or None on any failure) per request of EMBED_BATCH texts, each with its own
    EMBED_TIMEOUT: one request for a corpus's ~50 sentences took 20-31 s on a shared GPU and timed out into the keyword
    fallback. The model stays loaded between the requests and unloads (or keeps another client's expiry) after the
    last."""
    keep = _keep_alive(model, host)
    hold = f"{max(int(keep.rstrip('s') or 0), EMBED_HOLD_S)}s"
    for a in range(0, len(texts), EMBED_BATCH):
        chunk = texts[a:a + EMBED_BATCH]
        body = {"model": model, "input": chunk, "keep_alive": keep if a + EMBED_BATCH >= len(texts) else hold}
        try:
            req = urllib.request.Request(f"{host}/api/embed", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
                v = np.asarray(json.loads(resp.read())["embeddings"], dtype=np.float32)
        except Exception:
            v = None
        yield chunk, (v if v is not None and v.ndim == 2 and len(v) == len(chunk) else None)


class _Store:
    """Raw embeddings per (model, digest, text): in memory, and in an SQLite file shared by processes (writes that
    fail are skipped: the cache is an optimisation)."""

    def __init__(self) -> None:
        self.mem: dict[tuple[str, str, str], np.ndarray] = {}
        self.lock = threading.Lock()

    def _db(self):
        path = Path(EMBED_CACHE)
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=5)
        db.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, digest TEXT, key TEXT, vec BLOB, "
                   "PRIMARY KEY (model, digest, key))")
        return db

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, model: str, digest: str, texts: list[str]) -> dict[str, np.ndarray]:
        with self.lock:
            found = {t: self.mem[(model, digest, t)] for t in texts if (model, digest, t) in self.mem}
        rest = list(dict.fromkeys(t for t in texts if t not in found))
        if rest:
            keys = {self._key(t): t for t in rest}
            try:
                db = self._db()
                try:
                    for a in range(0, len(rest), 500):
                        chunk = list(keys)[a:a + 500]
                        rows = db.execute(f"SELECT key, vec FROM emb WHERE model=? AND digest=? AND key IN "
                                          f"({','.join('?' * len(chunk))})", [model, digest, *chunk]).fetchall()
                        for key, blob in rows:
                            found[keys[key]] = np.frombuffer(blob, dtype=np.float32).copy()
                finally:
                    db.close()
            except sqlite3.Error:
                pass
            with self.lock:
                self.mem.update({(model, digest, t): found[t] for t in rest if t in found})
        return found

    def put(self, model: str, digest: str, vectors: dict[str, np.ndarray]) -> None:
        with self.lock:
            self.mem.update({(model, digest, t): v for t, v in vectors.items()})
        try:
            db = self._db()
            try:
                with db:
                    db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?, ?)",
                                   [(model, digest, self._key(t), np.asarray(v, np.float32).tobytes())
                                    for t, v in vectors.items()])
            finally:
                db.close()
        except sqlite3.Error:
            pass


_STORE = _Store()


def embed(texts: list[str], model: str = EMBED_MODEL, host: str | None = None) -> np.ndarray | None:
    """L2-normalised Ollama embeddings, or None when the server or model is unavailable. Texts embedded before with
    the same model (by digest) come from the cache (EMBED_CACHE): the transcript sentences of a corpus are embedded
    once, not on every topic brief (22-31 s each on the irl-event footage, 2026-10-18). Ollama's bge-m3 vectors do
    not depend on the batch (a text alone, in a batch of 60 or in reverse order: bit-identical), so a cached vector
    is the one a fresh batch would give."""
    host = host or OLLAMA_HOST
    if not texts:
        return np.zeros((0, 1))
    digest = _digest(model, host)  # None: no digest to key the cache on (or no server), nothing cached
    have = _STORE.get(model, digest, texts) if digest else {}
    missing = list(dict.fromkeys(t for t in texts if t not in have))
    for chunk, v in _batches(missing, model, host):
        if v is None:
            return None
        new = dict(zip(chunk, v, strict=True))
        if digest:
            _STORE.put(model, digest, new)  # each batch as it arrives: a later timeout keeps the earlier ones
        have.update(new)
    v = np.stack([have[t] for t in texts]).astype(np.float32)
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
