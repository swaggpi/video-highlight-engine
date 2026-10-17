"""Query side: free-text search over the index, and ranked clips for a query."""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np

from vhe.config import resolve_device
from vhe.index.store import Index
from vhe.query.candidates import LENGTHS, cuts_inside, generate, peak_times
from vhe.query.parser import ParsedQuery, parse, parse_with_llm
from vhe.query.refine import Context, refine
from vhe.query.score import robust_z, segment_scores, smooth, timestep_signals
from vhe.query.select import select

CUT_PENALTY = 0.15
LENGTH_PRIOR = 0.15
TAG_MIN_PROB = 0.3
EXTRA_PICKS = 3  # spares in case refinement makes two picks collide
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "der", "die", "das", "und", "mit", "ein", "eine",
             "von", "auf", "für", "den", "dem", "des", "a", "an", "of", "on", "in", "at", "to", "is"}

# One model lock per process: UI requests and indexing must not run inference concurrently (MPS/CUDA memory).
MODEL_LOCK = threading.RLock()


class Engine:
    def __init__(self, index_dir: Path, device: str = "auto", use_nc: bool = True) -> None:
        self.index = Index(index_dir)
        self.device = resolve_device(device)
        self.use_nc = use_nc and bool(self.index.meta.get("nc_highlight"))
        self._encoder = None
        self._nc = None
        self._text_cache: dict[str, np.ndarray] = {}
        emb = self.index.grid["emb"].astype(np.float32)
        self._emb = emb
        self._emb_cum = np.concatenate([np.zeros((1, emb.shape[1]), np.float32), np.cumsum(emb, axis=0)])
        self.load_seconds = 0.0

    @property
    def encoder(self):
        with MODEL_LOCK:
            if self._encoder is None:
                from vhe.signals.frames import Encoder

                t0 = time.perf_counter()
                self._encoder = Encoder(self.index.meta["encoder"], self.device)
                self.load_seconds += time.perf_counter() - t0
            return self._encoder

    def text_vector(self, text: str) -> np.ndarray:
        if text not in self._text_cache:
            with MODEL_LOCK:
                self._text_cache[text] = self.encoder.texts([text])[0]
        return self._text_cache[text]

    def _nc_scores(self, text: str):
        if not self.use_nc or not text:
            return None, []
        with MODEL_LOCK:
            if self._nc is None:
                from vhe.signals.highlight_nc import NCHighlighter

                t0 = time.perf_counter()
                self._nc = NCHighlighter(self.index.meta["config"]["nc_checkpoint"], self.device)
                self.load_seconds += time.perf_counter() - t0
            meta = self.index.meta["nc_highlight"]
            return self._nc.predict(text, meta["chunks"], self.index.root / "nc", self.index.n, self.index.fps)

    def _mean_vectors(self, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
        v = (self._emb_cum[i1] - self._emb_cum[i0]) / np.maximum(i1 - i0, 1)[:, None]
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    # ---- manual search -------------------------------------------------------------------------------
    def search(self, text: str, k: int = 10, half_width_s: float = 2.0) -> list[dict]:
        text = text.strip()
        if not text:
            return []
        idx, fps = self.index, self.index.fps
        curve = smooth(self._emb @ self.text_vector(text), int(fps))
        z = robust_z(curve)
        results = [{"start": round(max(0.0, t - half_width_s), 2), "end": round(min(idx.duration, t + half_width_s), 2),
                    "score": round(float(z[int(round(t * fps))]), 3), "source": "visual"}
                   for t in peak_times(curve, fps, min_gap_s=2 * half_width_s, top=k, pct=0)]
        terms = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2 and w not in STOPWORDS]
        for seg in (idx.transcript or {}).get("segments", []):
            words = set(re.findall(r"\w+", seg["text"].lower()))
            hits = sum(term in words for term in terms)
            if hits:
                # A spoken match is strong evidence: rank it like a visual match about 2.5 MAD above the median.
                results.append({"start": seg["start"], "end": seg["end"], "score": round(2.0 + hits / len(terms), 3),
                                "source": "speech", "text": seg["text"]})
        results.sort(key=lambda r: -r["score"])
        return results[:k]

    # ---- highlight clips -----------------------------------------------------------------------------
    def parse(self, query: str, llm: str | None = None, **overrides) -> ParsedQuery:
        return parse_with_llm(query, llm, **overrides) if llm else parse(query, **overrides)

    def clips(self, query: str, count: int | None = None, min_len: float | None = None,
              max_len: float | None = None, llm: str | None = None, budget_s: float | None = None) -> dict:
        load_before = self.load_seconds
        t0 = time.perf_counter()
        idx, fps, grid = self.index, self.index.fps, self.index.grid
        p = self.parse(query, llm, count=count, min_len=min_len, max_len=max_len)
        budget = budget_s if budget_s is not None else p.budget_s
        text_vec = self.text_vector(p.text) if p.text else None
        # The highlight model was trained on full English sentences, so it gets the raw query.
        nc_sal, proposals = self._nc_scores(p.raw if p.text else "")
        sig = timestep_signals(grid, fps, text_vec, nc_sal, has_audio=idx.has_audio)
        per_step = sum(p.weights.get(k, 0.0) * sig[k] for k in ("relevance", "quality", "energy", "steady"))
        cands = generate(idx.duration, idx.cuts, idx.change_points, p.min_len, p.max_len,
                         peaks=peak_times(smooth(per_step, int(fps)), fps), proposals=proposals)
        result = {"query": p.to_dict(), "clips": [], "candidates": len(cands),
                  "noncommercial": idx.meta.get("noncommercial_components", []) if nc_sal is not None else []}
        if not cands:
            return self._finish(result, t0, load_before)

        # Stage 1, vectorised: segment means from prefix sums for every candidate.
        S = np.array([c[0] for c in cands])
        E = np.array([c[1] for c in cands])
        i0 = np.clip(np.floor(S * fps).astype(int), 0, idx.n - 1)
        i1 = np.clip(np.maximum(np.ceil(E * fps).astype(int), i0 + 1), 1, idx.n)
        means = {}
        for key in ("relevance", "quality", "energy", "steady", "speech"):
            c = np.concatenate([[0.0], np.cumsum(sig[key])])
            means[key] = (c[i1] - c[i0]) / (i1 - i0)
        means["speech"] = means["speech"] * 2 - 1
        total = sum(p.weights.get(k, 0.0) * means[k] for k in means)
        # Short windows on a peak have higher means than long ones; compare each candidate within its length
        # bucket, then add a mild prior towards the middle of the allowed range.
        bucket = np.searchsorted(np.asarray(LENGTHS, dtype=float), E - S - 1e-6)
        for b in np.unique(bucket):
            total[bucket == b] -= np.median(total[bucket == b])
        target = (p.min_len + p.max_len) / 2
        cut_arr = np.asarray(idx.cuts)
        n_cuts = np.array([cuts_inside(s, e, cut_arr) for s, e in zip(S, E)])
        total += -LENGTH_PRIOR * np.abs((E - S) - target) / max(target, 1.0) - CUT_PENALTY * n_cuts

        # Stage 2 on a pool: exact relevance (mean of the best 60% of steps), MMR, refinement.
        pool = np.argsort(-total)[:max(400, 40 * p.count)]
        scored = []
        for j in pool:
            parts = segment_scores(sig, int(i0[j]), int(i1[j]), p.weights)
            adj = float(total[j] + p.weights.get("relevance", 0.0) * (parts["relevance"] - means["relevance"][j]))
            scored.append({"start": float(S[j]), "end": float(E[j]), "source": cands[j][2],
                           "cuts_inside": int(n_cuts[j]), "total": adj})
        chosen = select(scored, self._mean_vectors(i0[pool], i1[pool]), p.count + EXTRA_PICKS, budget_s=budget)

        ctx = Context(idx.duration, idx.cuts, idx.vad, idx.words, grid.get("shake"), fps)
        kept: list[tuple[float, float]] = []
        for c in chosen:
            s, e = refine(c["start"], c["end"], ctx, p.min_len, p.max_len)
            s, e = self._avoid_overlap(s, e, kept, p.min_len)
            if s is None:
                continue
            kept.append((s, e))
            j0, j1 = int(np.floor(s * fps)), max(min(int(np.ceil(e * fps)), idx.n), int(np.floor(s * fps)) + 1)
            parts = segment_scores(sig, j0, j1, p.weights)
            result["clips"].append({
                "start": s, "end": e, "rank_score": round(c["total"], 3), "score": round(parts["total"], 3),
                "relevance": round(parts["relevance"], 3), "quality": round(parts["quality"], 3),
                "energy": round(parts["energy"], 3), "labels": self._labels(j0, j1, parts),
                "source": c["source"], "transcript": self._words_between(s, e)})
            if len(result["clips"]) >= p.count:
                break
        return self._finish(result, t0, load_before)

    def _finish(self, result: dict, t0: float, load_before: float) -> dict:
        load = self.load_seconds - load_before
        result["load_seconds"] = round(load, 3)
        result["seconds"] = round(time.perf_counter() - t0 - load, 3)  # ranking time, models already loaded
        return result

    @staticmethod
    def _avoid_overlap(s: float, e: float, kept: list[tuple[float, float]], min_len: float):
        for ks, ke in kept:
            if s < ke and e > ks:
                if ks <= s:  # overlaps the end of a kept clip: start after it
                    s = ke
                else:  # overlaps the start of a kept clip: end before it
                    e = ks
            if e - s < min_len:
                return None, None
        return s, e

    def _labels(self, i0: int, i1: int, parts: dict) -> list[str]:
        probs = self.index.grid["tags"][i0:max(i1, i0 + 1)].astype(np.float32).mean(0)
        labels = [self.index.meta["labels"][j] for j in np.argsort(-probs)[:3] if probs[j] >= TAG_MIN_PROB]
        if parts["energy"] > 0.5:
            labels.append("high energy")
        elif parts["energy"] < -0.5:
            labels.append("calm")
        if parts["speech"] > 0.0:
            labels.append("speech")
        return labels

    def _words_between(self, s: float, e: float) -> str:
        import bisect

        words = self.index.words
        lo = bisect.bisect_left([w["start"] for w in words], s) if words else 0
        out = []
        for w in words[lo:]:
            if w["start"] >= e:
                break
            out.append(w["word"])
        return " ".join(out)
