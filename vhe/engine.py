"""Query side: free-text search over the index, and ranked clips for a query."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vhe.config import resolve_device
from vhe.index.store import Index
from vhe.query.candidates import cuts_inside, generate, peak_times
from vhe.query.parser import ParsedQuery, parse, parse_with_llm
from vhe.query.refine import Context, refine
from vhe.query.score import robust_z, segment_scores, smooth, timestep_signals
from vhe.query.select import select

CUT_PENALTY = 0.15
TAG_MIN_PROB = 0.3


class Engine:
    def __init__(self, index_dir: Path, device: str = "auto", use_nc: bool = True) -> None:
        self.index = Index(index_dir)
        self.device = resolve_device(device)
        self.use_nc = use_nc and bool(self.index.meta.get("nc_highlight"))
        self._encoder = None
        self._nc = None
        self._text_cache: dict[str, np.ndarray] = {}

    @property
    def encoder(self):
        if self._encoder is None:
            from vhe.signals.frames import Encoder

            self._encoder = Encoder(self.index.meta["encoder"], self.device)
        return self._encoder

    def text_vector(self, text: str) -> np.ndarray:
        if text not in self._text_cache:
            self._text_cache[text] = self.encoder.texts([text])[0]
        return self._text_cache[text]

    def _nc_scores(self, text: str):
        if not self.use_nc or not text:
            return None, []
        if self._nc is None:
            from vhe.signals.highlight_nc import NCHighlighter

            self._nc = NCHighlighter(self.index.meta["config"]["nc_checkpoint"], self.device)
        meta = self.index.meta["nc_highlight"]
        return self._nc.predict(text, meta["chunks"], self.index.root / "nc", self.index.n, self.index.fps)

    # ---- manual search -------------------------------------------------------------------------------
    def search(self, text: str, k: int = 10, half_width_s: float = 2.0) -> list[dict]:
        idx, fps = self.index, self.index.fps
        sim = idx.grid["emb"].astype(np.float32) @ self.text_vector(text)
        curve = smooth(sim, int(fps))
        results = [{"start": round(max(0.0, t - half_width_s), 2), "end": round(min(idx.duration, t + half_width_s), 2),
                    "score": round(float(curve[int(round(t * fps))]), 4), "source": "visual"}
                   for t in peak_times(curve, fps, min_gap_s=2 * half_width_s, top=k, pct=0)]
        terms = [w for w in text.lower().split() if len(w) > 2]
        for seg in (idx.transcript or {}).get("segments", []):
            if terms and any(term in seg["text"].lower() for term in terms):
                results.append({"start": seg["start"], "end": seg["end"], "score": 1.0, "source": "speech",
                                "text": seg["text"]})
        results.sort(key=lambda r: -r["score"])
        return results[:k]

    # ---- highlight clips -----------------------------------------------------------------------------
    def parse(self, query: str, llm: str | None = None, **overrides) -> ParsedQuery:
        return parse_with_llm(query, llm, **overrides) if llm else parse(query, **overrides)

    def clips(self, query: str, count: int | None = None, min_len: float | None = None,
              max_len: float | None = None, llm: str | None = None) -> dict:
        t0 = time.perf_counter()
        idx, fps, grid = self.index, self.index.fps, self.index.grid
        p = self.parse(query, llm, count=count, min_len=min_len, max_len=max_len)
        text_vec = self.text_vector(p.text) if p.text else None
        nc_sal, proposals = self._nc_scores(p.text)
        sig = timestep_signals(grid, fps, text_vec, nc_sal)
        per_step = sum(p.weights.get(k, 0.0) * sig[k] for k in ("relevance", "quality", "energy", "steady"))
        cands = generate(idx.duration, idx.cuts, idx.change_points, p.min_len, p.max_len,
                         peaks=peak_times(smooth(per_step, int(fps)), fps), proposals=proposals)

        cut_arr = np.asarray(idx.cuts)
        emb = grid["emb"].astype(np.float32)
        scored, vectors = [], []
        target = (p.min_len + p.max_len) / 2
        for s, e, src in cands:
            i0, i1 = int(np.floor(s * fps)), int(np.ceil(e * fps))
            parts = segment_scores(sig, i0, min(i1, idx.n), p.weights)
            n_cuts = cuts_inside(s, e, cut_arr)
            # Mild preference for lengths near the middle of the allowed range.
            length_prior = -0.1 * abs((e - s) - target) / max(target, 1.0)
            parts["total"] += length_prior - CUT_PENALTY * n_cuts
            scored.append({"start": s, "end": e, "source": src, "cuts_inside": n_cuts, **parts})
            v = emb[i0:max(i1, i0 + 1)].mean(0)
            vectors.append(v / (np.linalg.norm(v) + 1e-9))
        if not scored:
            return {"query": p.to_dict(), "clips": [], "seconds": round(time.perf_counter() - t0, 3)}

        chosen = select(scored, np.stack(vectors), p.count)
        ctx = Context(idx.duration, idx.cuts, idx.vad, idx.words, grid.get("shake"), fps)
        clips = []
        for c in chosen:
            s, e = refine(c["start"], c["end"], ctx, p.min_len, p.max_len)
            i0, i1 = int(np.floor(s * fps)), min(int(np.ceil(e * fps)), idx.n)
            parts = segment_scores(sig, i0, i1, p.weights)
            clips.append({"start": s, "end": e, "score": round(parts["total"], 3),
                          "relevance": round(parts["relevance"], 3), "quality": round(parts["quality"], 3),
                          "energy": round(parts["energy"], 3), "labels": self._labels(i0, i1, parts),
                          "source": c["source"], "transcript": self._words_between(s, e)})
        clips.sort(key=lambda c: -c["score"])
        return {"query": p.to_dict(), "clips": clips, "candidates": len(cands),
                "noncommercial": idx.meta.get("noncommercial_components", []) if self.use_nc else [],
                "seconds": round(time.perf_counter() - t0, 3)}

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
        return " ".join(w["word"] for w in self.index.words if s <= w["start"] < e)


def energy_curve(index: Index) -> np.ndarray:
    g = index.grid
    return (robust_z(g["motion"]) + robust_z(g["loudness"]) + robust_z(g["onset"])) / 3
