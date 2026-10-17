"""Query side: free-text search and ranked clips for an editor's brief, over one video or a folder of raw footage.

Both run on a Corpus (vhe/corpus.py): every file's index on one virtual timeline, so there is one code path.
Candidates are generated and refined per file (no clip spans two files); scoring, selection and the edit order
run over all files together. Clips carry the file and file-local times for playback and export."""
from __future__ import annotations

import bisect
import dataclasses
import functools
import re
import threading
import time
from pathlib import Path

import numpy as np

from vhe.config import resolve_device
from vhe.corpus import Corpus, open_corpus
from vhe.query.candidates import LENGTHS, cuts_inside, generate, peak_times, speech_spans
from vhe.query.parser import ParsedQuery, parse, parse_with_llm
from vhe.query.refine import Context, refine
from vhe.query.score import relevance, segment_scores, smooth, timestep_signals
from vhe.query.select import select_for

CUT_PENALTY = 0.15
LENGTH_PRIOR = 0.15
TAG_MIN_PROB = 0.3
SPEECH_BONUS = 0.3  # utterance-aligned candidates beat windows that start mid-sentence, scaled by speech weight
MAX_UNUSABLE = 0.2  # candidates with more unusable steps (covered, blown, blurry, very shaky) are dropped, any mood
BUCKET_MIN = 5  # length buckets smaller than this are not median-centred (a singleton would become exactly 0)
EDGE_WEIGHTS = {"relevance": 1.0, "steady": 0.8, "quality": 0.6}  # opener/closer: the shot asked for, held steady
EDGE_POOL = 30
SIGNALS = ("relevance", "quality", "energy", "steady", "speech")
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "der", "die", "das", "und", "mit", "ein", "eine",
             "von", "auf", "für", "den", "dem", "des", "a", "an", "of", "on", "in", "at", "to", "is"}

# One model lock per process: UI requests and indexing must not run inference concurrently (MPS/CUDA memory).
MODEL_LOCK = threading.RLock()


class Engine:
    def __init__(self, source: Path | str | Corpus, device: str = "auto", use_nc: bool = True) -> None:
        self.corpus = source if isinstance(source, Corpus) else open_corpus(Path(source))
        c = self.corpus
        self.device = resolve_device(device)
        self.use_nc = use_nc and any(ix.meta.get("nc_highlight") for ix in c.members)
        self._encoder = None
        self._nc = None
        self._text_cache: dict[str, np.ndarray] = {}
        self._contexts: dict[int, Context] = {}
        self._emb = c.grid["emb"].astype(np.float32)
        scale = c.meta.get("logit_scale")
        self._logit = None if scale is None else (float(scale), float(c.meta.get("logit_bias") or 0.0))
        self.load_seconds = 0.0

    @classmethod
    def open(cls, path: Path | str, device: str = "auto", use_nc: bool = True) -> "Engine":
        """A project folder, a per-video index dir, or a video with its index next to it."""
        return cls(open_corpus(Path(path)), device, use_nc)

    @property
    def encoder(self):
        with MODEL_LOCK:
            if self._encoder is None:
                from vhe.signals.frames import Encoder

                t0 = time.perf_counter()
                self._encoder = Encoder(self.corpus.meta["encoder"], self.device)  # model from vhe.models cache
                self.load_seconds += time.perf_counter() - t0
            return self._encoder

    def text_vector(self, text: str) -> np.ndarray:
        if text not in self._text_cache:
            with MODEL_LOCK:
                self._text_cache[text] = self.encoder.texts([text])[0]
        return self._text_cache[text]

    def _relevance(self, text: str, nc_saliency: np.ndarray | None = None) -> np.ndarray:
        c = self.corpus
        return relevance(self._emb @ self.text_vector(text), c.fps, self._logit, nc_saliency, c.file_id)

    def _nc_scores(self, text: str) -> tuple[np.ndarray | None, list[list]]:
        """Highlight-model saliency on the corpus grid (NaN for files without it) and proposals per file."""
        c = self.corpus
        if not self.use_nc or not text:
            return None, [[] for _ in c.members]
        with MODEL_LOCK:
            if self._nc is None:
                from vhe.signals.highlight_nc import NCHighlighter

                t0 = time.perf_counter()
                ckpt = next(ix.meta["config"]["nc_checkpoint"] for ix in c.members if ix.meta.get("nc_highlight"))
                self._nc = NCHighlighter(ckpt, self.device)
                self.load_seconds += time.perf_counter() - t0
            sal, proposals = np.full(c.n, np.nan), []
            for k, ix in enumerate(c.members):
                meta = ix.meta.get("nc_highlight")
                s, props = self._nc.predict(text, meta["chunks"], ix.root / "nc", ix.n, c.fps) if meta else (None, [])
                if s is not None:
                    sal[c.bounds[k]:c.bounds[k + 1]] = s
                proposals.append(props)
            return sal, proposals

    def _context(self, k: int) -> Context:
        if k not in self._contexts:
            ix, c = self.corpus.members[k], self.corpus
            self._contexts[k] = Context(float(c.durations[k]), ix.cuts, ix.vad, ix.words, ix.grid.get("shake"), c.fps)
        return self._contexts[k]

    def _mean_vectors(self, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
        v = np.stack([self._emb[a:b].mean(0) for a, b in zip(i0, i1)]) if len(i0) else np.zeros((0, 1))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def _hit(self, k: int, s: float, e: float, **extra) -> dict:
        """A result in file k: virtual start/end clamped to the file, plus file-local times."""
        c = self.corpus
        s, e = max(s, c.offsets[k]), min(e, c.offsets[k] + c.durations[k])
        return {"file": c.files[k], "path": str(c.sources[k]), "file_start": c.local(k, s), "file_end": c.local(k, e),
                "start": round(float(s), 3), "end": round(float(e), 3), **extra}

    # ---- manual search -------------------------------------------------------------------------------
    def search(self, text: str, k: int = 10, half_width_s: float = 2.0) -> list[dict]:
        text = text.strip()
        if not text:
            return []
        c, fps = self.corpus, self.corpus.fps
        curve = np.where(c.usable, self._relevance(text), -3.0)
        results = [self._hit(c.file_of(t), t - half_width_s, t + half_width_s,
                             score=round(float(curve[min(int(round(t * fps)), c.n - 1)]), 3), source="visual")
                   for t in peak_times(curve, fps, min_gap_s=2 * half_width_s, top=k, pct=0)]
        terms = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2 and w not in STOPWORDS]
        for seg in c.segments:
            words = set(re.findall(r"\w+", seg["text"].lower()))
            hits = sum(term in words for term in terms)
            if hits:
                # A spoken match is strong evidence: rank it like a clear visual match (relevance around +2..+3).
                results.append(self._hit(seg["file"], seg["start"], seg["end"], score=round(2.0 + hits / len(terms), 3),
                                         source="speech", text=seg["text"]))
        results.sort(key=lambda r: -r["score"])
        return results[:k]

    # ---- highlight clips -----------------------------------------------------------------------------
    def parse(self, query: str, llm: str | None = None, **overrides) -> ParsedQuery:
        return parse_with_llm(query, llm, **overrides) if llm else parse(query, **overrides)

    def _candidates(self, per_step: np.ndarray, p: ParsedQuery, proposals: list[list]) -> tuple:
        """Per file, on its own timeline, then shifted: (start, end, file, source) arrays in virtual seconds."""
        c, fps = self.corpus, self.corpus.fps
        out = []
        for k, ix in enumerate(c.members):
            speech = speech_spans((ix.transcript or {}).get("segments", []), ix.vad, p.max_len)
            peaks = peak_times(smooth(per_step[c.bounds[k]:c.bounds[k + 1]], fps), fps)
            out += [(c.offsets[k] + s, c.offsets[k] + e, k, src)
                    for s, e, src in generate(float(c.durations[k]), ix.cuts, ix.change_points, p.min_len, p.max_len,
                                              peaks=peaks, proposals=proposals[k], speech=speech)]
        if not out:
            return np.zeros(0), np.zeros(0), np.zeros(0, int), []
        S, E, K, src = zip(*out)
        return np.array(S), np.array(E), np.array(K), list(src)

    def _unusable(self, S: np.ndarray, E: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Fraction of unusable grid steps per span."""
        i0, i1 = self.corpus.steps(S, E, K)
        bad = np.concatenate([[0], np.cumsum(~self.corpus.usable)])
        return (bad[i1] - bad[i0]) / (i1 - i0)

    def _pool(self, sig: dict, weights: dict, cands: tuple, p: ParsedQuery, size: int) -> list[dict]:
        """Stage 1, vectorised over every candidate: segment means from prefix sums, compared within length
        buckets. Stage 2 on the best `size`: exact relevance (mean of the best 60% of steps) and refined edges, so
        selection (budget, overlap, MMR) runs on the final clips."""
        c = self.corpus
        S, E, K, src = cands
        i0, i1 = c.steps(S, E, K)
        means = {}
        for key in SIGNALS:
            cum = np.concatenate([[0.0], np.cumsum(sig[key])])
            means[key] = (cum[i1] - cum[i0]) / (i1 - i0)
        means["speech"] = means["speech"] * 2 - 1
        total = sum(weights.get(k, 0.0) * means[k] for k in means)
        # Short windows on a peak have higher means than long ones; compare each candidate within its length
        # bucket, then add a mild prior towards the middle of the allowed range.
        bucket = np.searchsorted(np.asarray(LENGTHS, dtype=float), E - S - 1e-6)
        for b in np.unique(bucket):
            if np.count_nonzero(bucket == b) >= BUCKET_MIN:
                total[bucket == b] -= np.median(total[bucket == b])
        target = (p.min_len + p.max_len) / 2
        n_cuts = cuts_inside(S, E, np.asarray(c.cuts))
        total += -LENGTH_PRIOR * np.abs((E - S) - target) / max(target, 1.0) - CUT_PENALTY * n_cuts
        total += SPEECH_BONUS * max(0.0, weights.get("speech", 0.0)) * np.array([s == "speech" for s in src])

        out = []
        for j in np.argsort(-total)[:size]:
            parts = segment_scores(sig, int(i0[j]), int(i1[j]), weights)
            adj = float(total[j] + weights.get("relevance", 0.0) * (parts["relevance"] - means["relevance"][j]))
            k, off = int(K[j]), c.offsets[K[j]]
            s, e = refine(float(S[j] - off), float(E[j] - off), self._context(k), p.min_len, p.max_len)
            out.append({"start": off + s, "end": off + e, "file": k, "source": src[j], "cuts_inside": int(n_cuts[j]),
                        "total": adj})
        if out:  # refinement moves edges by up to a second or so: gate the final spans again
            frac = self._unusable(*(np.array([o[key] for o in out]) for key in ("start", "end", "file")))
            out = [o for o, f in zip(out, frac) if f <= MAX_UNUSABLE]
        return out

    def _edge_clip(self, text: str, sig: dict, cands: tuple, p: ParsedQuery, taken: list[dict],
                   max_len: float) -> dict | None:
        """Best clip for the opener/closer description: its own relevance, steady and clean, not overlapping."""
        pool = self._pool(dict(sig, relevance=self._relevance(text)), EDGE_WEIGHTS, cands, p, EDGE_POOL)
        pool = [o for o in pool if o["end"] - o["start"] <= max_len + 1e-6
                and all(min(o["end"], t["end"]) <= max(o["start"], t["start"]) for t in taken)]
        return max(pool, key=lambda o: o["total"]) if pool else None

    def clips(self, query: str, count: int | None = None, min_len: float | None = None,
              max_len: float | None = None, llm: str | None = None,
              budget_s: float | tuple[float, float] | None = None) -> dict:
        load_before = self.load_seconds
        t0 = time.perf_counter()
        c, fps = self.corpus, self.corpus.fps
        p = self.parse(query, llm, count=count, min_len=min_len, max_len=max_len, budget_s=budget_s)
        # The highlight model was trained on full English sentences, so it gets the raw query.
        nc_sal, proposals = self._nc_scores(p.raw if p.text else "")
        rel = self._relevance(p.text, nc_sal) if p.text else None
        sig = timestep_signals(c.grid, fps, rel, c.audio_steps, c.file_id)
        per_step = sum(p.weights.get(k, 0.0) * sig[k] for k in ("relevance", "quality", "energy", "steady"))
        S, E, K, src = self._candidates(per_step, p, proposals)
        result = {"query": p.to_dict(), "clips": [], "candidates": len(S), "files": len(c.members),
                  "noncommercial": sorted({x for ix in c.members for x in ix.meta.get("noncommercial_components", [])})
                  if nc_sal is not None else []}
        keep = self._unusable(S, E, K) <= MAX_UNUSABLE if len(S) else np.zeros(0, bool)
        result["unusable_dropped"] = int(len(S) - keep.sum())
        cands = (S[keep], E[keep], K[keep], [s for s, k in zip(src, keep) if k])
        if not len(cands[0]):
            return self._finish(result, t0, load_before)

        # Opener and closer first: they take their slots (and budget) before the body is selected.
        edges: dict[str, dict] = {}
        room = np.inf if p.budget is None else p.budget[1]
        for role, text in (("opener", p.opener), ("closer", p.closer)):
            if text:
                e = self._edge_clip(text, sig, cands, p, list(edges.values()), room)
                if e is not None:
                    edges[role] = dict(e, text=text)
                    room -= e["end"] - e["start"]
        used = sum(e["end"] - e["start"] for e in edges.values())
        body_p = dataclasses.replace(
            p, count=max(0, p.count - len(edges)),
            budget=None if p.budget is None else (max(0.0, p.budget[0] - used), p.budget[1] - used))

        scored = [o for o in self._pool(sig, p.weights, cands, p, max(400, 40 * p.count))
                  if all(min(o["end"], t["end"]) <= max(o["start"], t["start"]) for t in edges.values())]
        if scored:
            r0, r1 = c.steps(*(np.array([o[key] for o in scored]) for key in ("start", "end", "file")))
            speech_cum = np.concatenate([[0.0], np.cumsum(sig["speech"])])
            for o, f in zip(scored, (speech_cum[r1] - speech_cum[r0]) / (r1 - r0)):
                o["speech_frac"] = float(f)
            body = select_for(scored, self._mean_vectors(r0, r1), body_p, groups=np.array([o["file"] for o in scored]))
        else:
            body = []
        # Edit order: files in manifest order (= virtual time), then time within the file; opener first, closer last.
        ordered = ([("opener", edges["opener"])] if "opener" in edges else []) + \
            [(None, o) for o in sorted(body, key=lambda o: o["start"])] + \
            ([("closer", edges["closer"])] if "closer" in edges else [])
        for role, o in ordered:
            result["clips"].append(self._clip(o, sig, p, role))
        return self._finish(result, t0, load_before)

    def _clip(self, o: dict, sig: dict, p: ParsedQuery, role: str | None) -> dict:
        c = self.corpus
        k, s, e = o["file"], round(float(o["start"]), 3), round(float(o["end"]), 3)
        j0, j1 = (int(x) for x in c.steps(s, e, k))
        parts = segment_scores(sig, j0, j1, p.weights)
        labels = self._labels(j0, j1, parts)
        return self._hit(k, s, e) | {
            "duration": round(e - s, 3), "rank_score": round(o["total"], 3), "score": round(parts["total"], 3),
            "relevance": round(parts["relevance"], 3), "quality": round(parts["quality"], 3),
            "energy": round(parts["energy"], 3), "labels": labels, "source": o["source"], "role": role,
            "reason": self._reason(parts, p, labels, role, o.get("text")), "transcript": self._words_between(s, e)}

    @staticmethod
    def _reason(parts: dict, p: ParsedQuery, labels: list[str], role: str | None, text: str | None) -> str:
        names = {"relevance": f"matches '{p.text}'", "quality": "clean footage", "steady": "steady camera",
                 "energy": "high energy" if p.weights.get("energy", 0.0) >= 0 else "calm", "speech": "speech"}
        contrib = sorted(((p.weights.get(k, 0.0) * parts[k], k) for k in names), reverse=True)
        bits = [f"{role}: {text}"] if role else []
        bits += [f"{names[k]} ({parts[k]:+.1f})" for v, k in contrib[:2] if v > 0.05]
        return "; ".join(bits + ([", ".join(labels)] if labels else [])) or "best available"

    def _finish(self, result: dict, t0: float, load_before: float) -> dict:
        load = self.load_seconds - load_before
        result["load_seconds"] = round(load, 3)
        result["seconds"] = round(time.perf_counter() - t0 - load, 3)  # ranking time, models already loaded
        return result

    def _labels(self, i0: int, i1: int, parts: dict) -> list[str]:
        probs = self.corpus.grid["tags"][i0:max(i1, i0 + 1)].astype(np.float32).mean(0)
        labels = [self.corpus.meta["labels"][j] for j in np.argsort(-probs)[:3] if probs[j] >= TAG_MIN_PROB]
        if parts["energy"] > 0.5:
            labels.append("high energy")
        elif parts["energy"] < -0.5:
            labels.append("calm")
        if parts["speech"] > 0.0:
            labels.append("speech")
        return labels

    @functools.cached_property
    def _word_starts(self) -> list[float]:
        return [w["start"] for w in self.corpus.words]

    def _words_between(self, s: float, e: float) -> str:
        words = self.corpus.words
        out = []
        for w in words[bisect.bisect_left(self._word_starts, s):]:
            if w["start"] >= e:
                break
            out.append(w["word"])
        return " ".join(out)
