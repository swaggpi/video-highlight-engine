"""Query side: free-text search and ranked clips for an editor's brief, over one video or a folder of raw footage.

Both run on a Corpus (vhe/corpus.py): every file's index on one virtual timeline, so there is one code path.
Candidates are generated and refined per file (no clip spans two files); scoring, selection and the edit order
run over all files together. Clips carry the file and file-local times for playback and export.

The Engine holds the models (SigLIP text encoder, the non-commercial highlight model) and per-corpus caches, and
orchestrates clips(): parse the brief (vhe.query.parser) -> per-step signals (vhe.rank.brief_signals) -> candidates
(vhe.query.candidates) -> gates (vhe.rank.gate) -> features and Scorer.score -> refine the best -> features of the
refined spans -> Scorer.rescore (vhe.rank.pool) -> selection (vhe.rank.select) -> edit order -> result clips
(vhe.results). The scorer is the hand-weighted one unless a learned model is given (vhe.rank.scorer); a run can dump
every candidate's features, score and selection (vhe.rank.dump).

Hard rules on top of the scores: excluded content and camera defects ("no protest signs", "no whip-pans"), forbidden
words read on screen ("nothing with 'executed' on it", "nothing political") and, when the brief says so ("no host
talking", "it goes under music"), speech drop candidates; visual clips start and end on clean frames (else they are
penalised); speech clips are whole sentences, their first and last word whole; selection never repeats a moment
(vhe.rank.select.duplicates); a budget minimum is filled by lengthening picks when the pool runs out, and a result
says in `warnings` when it still falls short. A model that runs out of GPU memory is moved to the CPU (with a
warning, vhe.models.on_device) instead of failing the query."""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import re
import threading
import time
from pathlib import Path

import numpy as np

from vhe import results
from vhe.corpus import Corpus, open_corpus
from vhe.models import MODEL_LOCK, on_device, on_gpu_lost, resolve_device
from vhe.query.candidates import corpus_candidates, peak_times
from vhe.query.model import ParsedQuery
from vhe.query.parser import parse_brief
from vhe.query.refine import Context, clean_steps
from vhe.query.text_match import STOPWORDS
from vhe.rank import dump as dumps
from vhe.rank import pool as pools
from vhe.rank.brief_signals import brief_signals
from vhe.rank.features import featurize, pool_stats
from vhe.rank.gate import gate
from vhe.rank.items import combine
from vhe.rank.scorer import Scorer, load_scorer
from vhe.rank.select import disjoint, fill_budget, score_scale, select_for
from vhe.rank.signals import muted_steps, relevance, timestep_signals
from vhe.timing import Timings

EMB_BLOCK = 1 << 16  # rows cast to float32 per matmul block


class Engine:
    def __init__(self, source: Path | str | Corpus, device: str = "auto", use_nc: bool = True,
                 scorer: Scorer | str | Path | None = None) -> None:
        """scorer: a Scorer, "heuristic" / None (the hand-weighted total) or a learned model file."""
        self.corpus = source if isinstance(source, Corpus) else open_corpus(Path(source))
        c = self.corpus
        self.device = resolve_device(device)
        self.use_nc = use_nc and any(ix.meta.get("nc_highlight") for ix in c.members)
        self._encoder = None
        self._nc = None
        self._text_cache: dict[str, np.ndarray] = {}
        self._nc_cache: dict[str, tuple[np.ndarray | None, list[list]]] = {}  # highlight-model output per query text
        self._contexts: dict[int, Context] = {}
        self._tls = threading.local()  # the running clips() call's Timings (UI requests run in threads)
        self._emb = c.grid["emb"]  # stored dtype (float16 saves ~1 GB on a 40 h corpus); cast per block in _similarity
        self._logit = (float(c.meta["logit_scale"]), float(c.meta.get("logit_bias") or 0.0))
        self.load_seconds = 0.0
        self.scorer = load_scorer(scorer)
        self.warnings: list[str] = []  # e.g. a model moved to the CPU after a GPU out-of-memory error
        on_gpu_lost(self._to_cpu)  # models handed to the LLM parse could not go back to the GPU (vhe.models)

    @classmethod
    def open(cls, path: Path | str, device: str = "auto", use_nc: bool = True,
             scorer: Scorer | str | Path | None = None) -> "Engine":
        """A project folder, a per-video index dir, or a video with its index next to it."""
        return cls(open_corpus(Path(path)), device, use_nc, scorer)

    # ---- models ------------------------------------------------------------------------------------------------
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
        return self.text_vectors([text])[0]

    def text_vectors(self, texts: list[str]) -> np.ndarray:
        """SigLIP text embeddings, the uncached ones in one batch (the encoder truncates to its 64 tokens)."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._text_cache))
        if missing:
            with MODEL_LOCK:
                vectors = self._on_device("SigLIP", lambda: self.encoder.texts(missing))
                self._text_cache.update(zip(missing, vectors, strict=True))
        return np.stack([self._text_cache[t] for t in texts])

    def _on_device(self, what: str, fn):
        """fn() on the engine's device, on the CPU after a GPU out-of-memory error (vhe.models.on_device). fn must
        fetch its model through the engine (self.encoder, self._nc_model())."""
        return on_device(self.device, what, fn, self._to_cpu)

    def _to_cpu(self, msg: str) -> None:
        self.warnings.append(msg)
        self.device, self._encoder, self._nc = "cpu", None, None
        self._nc_cache.clear()

    @contextlib.contextmanager
    def _part(self, name: str):
        """Time a model call as a part of the running clips() stage (Timings.add)."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            clock = getattr(self._tls, "clock", None)
            if clock is not None:
                clock.add(name, time.perf_counter() - t0)

    def _nc_scores(self, text: str) -> tuple[np.ndarray | None, list[list]]:
        """Highlight-model saliency on the corpus grid (NaN for files without it) and proposals per file; computed once
        per query text (the model is deterministic, its window features are fixed), returned as copies."""
        c = self.corpus
        if not self.use_nc or not text:
            return None, [[] for _ in c.members]
        with self._part("highlight model"):
            if text not in self._nc_cache:
                with MODEL_LOCK:
                    self._nc_cache[text] = self._on_device("CG-DETR", lambda: self._nc_predict(text))
            sal, proposals = self._nc_cache[text]
            return sal.copy(), [list(props) for props in proposals]

    def _nc_model(self):
        if self._nc is None:
            from vhe.signals.highlight_nc import NCHighlighter

            t0 = time.perf_counter()
            c = self.corpus
            ckpt = next(ix.meta["config"]["nc_checkpoint"] for ix in c.members if ix.meta.get("nc_highlight"))
            self._nc = NCHighlighter(ckpt, self.device)
            self.load_seconds += time.perf_counter() - t0
        return self._nc

    def _nc_predict(self, text: str) -> tuple[np.ndarray, list[list]]:
        c = self.corpus
        nc = self._nc_model()
        sal, proposals = np.full(c.n, np.nan), []
        for k, ix in enumerate(c.members):
            meta = ix.meta.get("nc_highlight")
            s, props = nc.predict(text, meta["chunks"], ix.root / "nc", ix.n, c.fps) if meta else (None, [])
            if s is not None:
                sal[c.bounds[k]:c.bounds[k + 1]] = s
            proposals.append(props)
        return sal, proposals

    # ---- relevance and the engine's per-corpus caches ---------------------------------------------------------
    def _relevance(self, text: str) -> np.ndarray:
        c = self.corpus
        with self._part("siglip"):
            return relevance(self._similarity(self.text_vector(text)), c.fps, self._logit, c.file_id)

    def _relevances(self, texts: list[str]) -> np.ndarray:
        """(texts, steps) SigLIP relevance per text on the absolute scale. With one text this is text_vector's path,
        so a single-text query scores exactly as before."""
        c = self.corpus
        if len(texts) == 1:
            return self._relevance(texts[0])[None]
        with self._part("siglip"):
            sims = self._similarity(self.text_vectors(texts).T)
            return np.stack([relevance(sims[:, k], c.fps, self._logit, c.file_id) for k in range(len(texts))])

    def _similarity(self, v: np.ndarray) -> np.ndarray:
        """emb @ v in float32 (v: one vector, or (dim, k) for k texts), one block of rows at a time."""
        v = v.astype(np.float32)
        out = np.empty((len(self._emb), *v.shape[1:]), np.float32)
        for a in range(0, len(out), EMB_BLOCK):
            out[a:a + EMB_BLOCK] = self._emb[a:a + EMB_BLOCK].astype(np.float32) @ v
        return out

    def _signals(self, p: ParsedQuery) -> tuple[dict, np.ndarray | None, list[list]]:
        """The brief's per-step signals (vhe.rank.brief_signals) with this engine's models."""
        return brief_signals(self.corpus, p, self._relevances, self._nc_scores, self._muted)

    def _context(self, k: int) -> Context:
        if k not in self._contexts:
            ix, c = self.corpus.members[k], self.corpus
            self._contexts[k] = Context(float(c.durations[k]), ix.cuts, ix.vad, ix.words, ix.grid.get("shake"), c.fps,
                                        clean=clean_steps(ix.grid))
        return self._contexts[k]

    @functools.cached_property
    def _muted(self) -> np.ndarray:
        return muted_steps(self.corpus.grid["loudness"], self.corpus.audio_steps)

    def _gate(self, S: np.ndarray, E: np.ndarray, K: np.ndarray, p: ParsedQuery, sig: dict | None = None) -> np.ndarray:
        return gate(self.corpus, S, E, K, p, self._muted, sig)

    def _mean_vectors(self, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
        v = np.stack([self._emb[a:b].mean(0, dtype=np.float32) for a, b in zip(i0, i1, strict=True)]) if len(i0) \
            else np.zeros((0, 1))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def _span_vectors(self, spans: list[dict]) -> np.ndarray:
        """L2-normalised mean frame embedding of each span ({"start", "end", "file"}, virtual seconds)."""
        if not spans:
            return np.zeros((0, self._emb.shape[1]), np.float32)
        r0, r1 = self.corpus.steps(*(np.array([o[key] for o in spans]) for key in ("start", "end", "file")))
        return self._mean_vectors(r0, r1)

    def _pool(self, sig: dict, weights: dict, cands: tuple, p: ParsedQuery, size: int, persona: str | None = None,
              trace: dict | None = None) -> list[dict]:
        """The scored, refined shortlist (vhe.rank.pool.shortlist)."""
        return pools.shortlist(self.corpus, self.scorer, self._context, self._muted, sig, weights, cands, p, size,
                               persona, trace)

    # ---- manual search -------------------------------------------------------------------------------
    def search(self, text: str, k: int = 10, half_width_s: float = 2.0) -> list[dict]:
        text = text.strip()
        if not text:
            return []
        c, fps = self.corpus, self.corpus.fps
        curve = np.where(c.usable, self._relevance(text), -3.0)
        found = [c.locate(c.file_of(t), t - half_width_s, t + half_width_s) |
                 {"score": round(float(curve[min(int(round(t * fps)), c.n - 1)]), 3), "source": "visual"}
                 for t in peak_times(curve, fps, min_gap_s=2 * half_width_s, top=k, pct=0)]
        terms = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2 and w not in STOPWORDS]
        for seg in c.segments:
            words = set(re.findall(r"\w+", seg["text"].lower()))
            hits = sum(term in words for term in terms)
            if hits:
                # A spoken match is strong evidence: rank it like a clear visual match (relevance around +2..+3).
                found.append(c.locate(seg["file"], seg["start"], seg["end"]) |
                             {"score": round(2.0 + hits / len(terms), 3), "source": "speech", "text": seg["text"]})
        found.sort(key=lambda r: -r["score"])
        return found[:k]

    # ---- highlight clips -----------------------------------------------------------------------------
    def parse(self, query: str, llm: str | None = "auto", **overrides) -> ParsedQuery:
        """llm: an Ollama model, 'auto' (the default parser model when reachable) or None (rules only)."""
        return parse_brief(query, llm, **overrides)

    def clips(self, query: str, count: int | None = None, min_len: float | None = None,
              max_len: float | None = None, llm: str | None = "auto",
              budget_s: float | tuple[float, float] | None = None, persona: str | None = None,
              dump: Path | str | None = None, dump_all: bool = False) -> dict:
        """llm: Ollama model for parsing the brief, 'auto' (default model when reachable) or None (rules).
        persona: who the edit is for (part of the query vector a learned scorer sees). dump: write every scored
        candidate there (vhe.rank.dump: the shortlist and the picks; with dump_all every gated candidate too).
        The result's `warnings` say what the edit could not deliver (a budget the footage cannot fill, a model that
        fell back to the CPU); `timings` is the wall clock per stage (vhe.timing.Timings records; parts such as the
        model calls inside a stage have "sub": True)."""
        load_before, warn_before = self.load_seconds, len(self.warnings)
        t0 = time.perf_counter()
        clock = self._tls.clock = Timings()
        try:
            result = self._clips(clock, query, count, min_len, max_len, llm, budget_s, persona, dump, dump_all)
        finally:
            self._tls.clock = None
        if self.load_seconds > load_before:  # inside the stages above (the first query loads the models)
            clock.add("of which model load", self.load_seconds - load_before)
        result["timings"] = clock.stages
        return self._finish(result, t0, load_before, warn_before)

    def _clips(self, clock: Timings, query: str, count: int | None, min_len: float | None, max_len: float | None,
               llm: str | None, budget_s, persona: str | None, dump, dump_all: bool) -> dict:
        c = self.corpus
        with clock.stage("parse"):
            p = self.parse(query, llm, count=count, min_len=min_len, max_len=max_len, budget_s=budget_s)
        if p.topics and not p.shots:  # "every moment the host talks about ...": transcript spans, not pictures
            with clock.stage("topics"):
                clips, n_spans, method = self._topic_clips(p, None if p.exhaustive else p.count,
                                                           p.soundbite_len or (p.min_len, p.max_len), role=None)
            result = {"query": p.to_dict(), "clips": clips, "candidates": n_spans, "files": len(c.members),
                      "noncommercial": [], "unusable_dropped": 0, "excluded_dropped": 0, "topic_match": method}
            if dump:
                with clock.stage("dump"):
                    sig, _, proposals = self._signals(p)
                    dumps.write_run(dump, c, self.scorer, query, p, persona, sig, topic=clips,
                                    stats=self._pool_stats(sig, self._candidates(sig, p, proposals)[0], p))
            return result
        with clock.stage("signals"):
            sig, nc_raw, proposals = self._signals(p)
        with clock.stage("candidates"):
            cands, n_generated = self._candidates(sig, p, proposals)
        S, E, K, src, allowed = cands
        result = {"query": p.to_dict(), "clips": [], "candidates": n_generated, "files": len(c.members),
                  "noncommercial": sorted({x for ix in c.members for x in ix.meta.get("noncommercial_components", [])})
                  if nc_raw is not None else []}
        result["unusable_dropped"] = int(n_generated - len(S))
        result["excluded_dropped"] = int(len(S) - allowed.sum())
        if not allowed.any():
            if dump:
                with clock.stage("dump"):
                    dumps.write_run(dump, c, self.scorer, query, p, persona, sig)
            result["warnings"] = ["no usable footage matches the brief after its exclusions"]
            return result

        bites = []
        if p.topics:  # "also the best 5-8 s soundbite where the host sums up": a separate deliverable, at the end
            with clock.stage("topics"):
                bites, _, result["topic_match"] = self._topic_clips(p, 1, p.soundbite_len or (p.min_len, p.max_len),
                                                                    role="soundbite")
        # Opener and closer first: they take their slots (and budget, as the soundbite does) before the body.
        edges: dict[str, dict] = {}
        bite_s = sum(b["duration"] for b in bites)
        room = np.inf if p.budget is None else p.budget[1] - bite_s
        with clock.stage("slots"):
            for role, text in (("opener", p.opener), ("closer", p.closer)):
                # "the first clip has to be the hook": the opener slot is a hook pool
                hook = role == "opener" and p.hook
                if text or hook:
                    e = self._edge_clip(text, sig, cands, p, list(edges.values()), room, persona, hook=hook)
                    if e is not None:
                        edges[role] = dict(e, text=(f"hook, {text}" if text else "hook") if hook else text)
                        room -= e["end"] - e["start"]
        used = sum(e["end"] - e["start"] for e in edges.values()) + bite_s
        body_p = dataclasses.replace(
            p, count=max(0, p.count - len(edges)),
            budget=None if p.budget is None else (max(0.0, p.budget[0] - used), p.budget[1] - used))

        trace: dict = {}  # the stage-1 pool: its scores set the brief's score scale, its statistics the features'
        with clock.stage("shortlist"):
            pool = self._pool(sig, p.weights, cands, p, max(400, 40 * p.count), persona, trace)
        for cid, o in enumerate(pool):
            o["cid"] = cid
        scored = [o for o in pool if disjoint(o, edges.values())]
        with clock.stage("select"):
            if scored:
                r0, r1 = c.steps(*(np.array([o[key] for o in scored]) for key in ("start", "end", "file")))
                speech_cum = np.concatenate([[0.0], np.cumsum(sig["speech"])])
                for o, f in zip(scored, (speech_cum[r1] - speech_cum[r0]) / (r1 - r0), strict=True):
                    o["speech_frac"] = float(f)
                item_rel = np.array([o["item_rel"] for o in scored]) if sig["items"] is not None else None
                taken = list(edges.values())
                body = select_for(scored, self._mean_vectors(r0, r1), body_p,
                                  groups=np.array([o["file"] for o in scored]), item_rel=item_rel,
                                  item_centre=sig["item_centre"], taken=taken,
                                  taken_vectors=self._span_vectors(taken), scale=score_scale(trace["s"]))
            else:
                body = []
        if p.budget is not None:
            with clock.stage("budget fill"):
                body = fill_budget(body, list(edges.values()), p, c, self._context,
                                   lambda S, E, K: self._gate(S, E, K, p, sig), bite_s)
                self._rescore_extended(body, sig, p, trace["stats"])
        # Edit order: files in manifest order (= virtual time), then time within the file; opener first, closer last.
        ordered = ([("opener", edges["opener"])] if "opener" in edges else []) + \
            [(None, o) for o in sorted(body, key=lambda o: o["start"])] + \
            ([("closer", edges["closer"])] if "closer" in edges else [])
        with clock.stage("results", items=len(ordered)):
            for role, o in ordered:
                result["clips"].append(results.clip(c, o, sig, p, role, self.scorer))
            result["clips"] += bites
            result["warnings"] = results.shortfall(result["clips"], p)
        if dump:
            with clock.stage("dump"):
                dumps.write_run(dump, c, self.scorer, query, p, persona, sig, pool, edges, body, bites,
                                trace if dump_all else None, stats=trace["stats"])
        return result

    def _candidates(self, sig: dict, p: ParsedQuery, proposals: list[list]) -> tuple[tuple, int]:
        """((S, E, K, src, allowed) of the usable candidates, number generated): candidates from the brief's per-step
        relevance, quality, energy, steadiness and cue curve (vhe.query.candidates), without unusable or muted spans;
        allowed: minus excluded content, camera defects or speech (vhe.rank.gate)."""
        c = self.corpus
        per_step = sum(p.weights.get(k, 0.0) * sig[k] for k in ("relevance", "quality", "energy", "steady"))
        if sig.get("cue") is not None:
            per_step = per_step + sig["cue"]
        S, E, K, src = corpus_candidates(c, per_step, p, proposals)
        n = len(S)
        keep = self._gate(S, E, K, p)
        S, E, K, src = S[keep], E[keep], K[keep], [s for s, k in zip(src, keep, strict=True) if k]
        return (S, E, K, src, self._gate(S, E, K, p, sig)), n

    def _pool_stats(self, sig: dict, cands: tuple, p: ParsedQuery) -> dict:
        """The stage-1 pool statistics of a brief (vhe.rank.features.pool_stats over its usable candidates), as
        clips() computes them: for spans featurised outside a clips() run."""
        S, E, K, src = cands[:4]
        return pool_stats(featurize(self.corpus, sig, S, E, K, src, p))

    def _rescore_extended(self, body: list[dict], sig: dict, p: ParsedQuery, stats: dict) -> None:
        """Clips the budget fill lengthened (vhe.rank.select.fill_budget) get the features and total of their final
        span, so the result and the dump describe the clip that is returned."""
        grown = [o for o in body if o.get("extended")]
        if not grown:
            return
        S, E, K = (np.array([o[key] for o in grown]) for key in ("start", "end", "file"))
        X = featurize(self.corpus, sig, S, E, K, [o["source"] for o in grown], p, stats=stats)
        for o, x in zip(grown, X, strict=True):
            t = self.scorer.rescore(np.array([o["total"]]), o["x"][None], x[None], o["q"])
            o.update(x=x, total=float(np.asarray(t)[0]))

    def _edge_clip(self, text: str | None, sig: dict, cands: tuple, p: ParsedQuery, taken: list[dict],
                   max_len: float, persona: str | None = None, hook: bool = False) -> dict | None:
        """The opener/closer (or hook) slot (vhe.rank.pool.edge_clip) on the description's own SigLIP relevance."""
        rel = avoided = None
        if text:
            rel = self._relevance(text)
            avoided = combine(rel[None], self._relevances(p.avoid)) if p.avoid else rel  # the brief's avoid penalty
        return pools.edge_clip(self.corpus, self.scorer, self._context, self._muted, self._span_vectors, sig, rel,
                               avoided, cands, p, taken, max_len, persona, hook)

    def _topic_clips(self, p: ParsedQuery, n: int | None, length: tuple[float, float],
                     role: str | None) -> tuple[list[dict], int, str]:
        """Clips of the transcript sentences that talk about p.topics (vhe.rank.pool.topic_picks): (clips in edit
        order, number of spans scored, match method)."""
        c = self.corpus
        picked, n_spans, method = pools.topic_picks(c, self._context, self._muted, p, n, length)
        sig = timestep_signals(c.grid, c.fps, None, c.audio_steps, c.file_id)
        out = []
        for o in picked:
            clip = results.clip(c, o, sig, p, role)
            clip["score"] = clip["rank_score"] = round(o["total"], 3)
            clip["reason"] = f"talks about '{o['item']}' ({method} {o['total']:.2f})"
            out.append(clip)
        return out, n_spans, method

    def _finish(self, result: dict, t0: float, load_before: float, warn_before: int = 0) -> dict:
        result["warnings"] = self.warnings[warn_before:] + result.get("warnings", [])
        load = self.load_seconds - load_before
        result["load_seconds"] = round(load, 3)
        result["seconds"] = round(time.perf_counter() - t0 - load, 3)  # ranking time, models already loaded
        return result

    # ---- features for spans picked elsewhere -------------------------------------------------------------------
    def features_for_spans(self, query: str, spans: list[dict], persona: str | None = None, llm: str | None = "auto",
                           **overrides) -> dict:
        """The dump document ("vhe-candidates/1", stage "external", vhe.rank.dump.external_spans) for spans picked
        outside the engine: features from the same signals and stage-1 pool statistics clips() computes for this brief,
        the score the engine's scorer gives a final clip (rescore of the span's own features)."""
        p = self.parse(query, llm, **overrides)
        sig, _, proposals = self._signals(p)
        stats = self._pool_stats(sig, self._candidates(sig, p, proposals)[0], p)
        return dumps.external_spans(self.corpus, self.scorer, query, p, persona, sig, spans, stats)
