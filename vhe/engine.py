"""Query side: free-text search and ranked clips for an editor's brief, over one video or a folder of raw footage.

Both run on a Corpus (vhe/corpus.py): every file's index on one virtual timeline, so there is one code path.
Candidates are generated and refined per file (no clip spans two files); scoring, selection and the edit order
run over all files together. Clips carry the file and file-local times for playback and export.

clips() orchestrates: parse the brief -> per-step signals -> candidates -> usable gate -> features (vhe.rank.features)
-> Scorer.score -> refine the best -> features of the refined spans -> Scorer.rescore -> selection -> edit order.
The scorer is the hand-weighted one unless a learned model is given (vhe.rank.scorer); a run can dump every
candidate's features, score and selection (vhe.rank.dump)."""
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
from vhe.query.candidates import fit_span, generate, peak_times, speech_spans
from vhe.query.parser import ParsedQuery, parse_brief
from vhe.query.plan import LIVE_MAX_MUTED, assign, combine, item_centre, item_scores, muted_steps, topic_scores
from vhe.query.refine import Context, refine
from vhe.rank import dump as dumps
from vhe.rank.features import INDEX, featurize, query_names, query_vector
from vhe.rank.scorer import Scorer, load_scorer
from vhe.rank.select import select_for
from vhe.rank.signals import blend_nc, nc_signal, relevance, segment_scores, smooth, span_mean, timestep_signals

TAG_MIN_PROB = 0.3
MAX_UNUSABLE = 0.2  # candidates with more unusable steps (covered, blown, blurry, very shaky) are dropped, any mood
EDGE_WEIGHTS = {"relevance": 1.0, "steady": 0.8, "quality": 0.6}  # opener/closer: the shot asked for, held steady
EDGE_POOL = 30
ITEM_POOL = 60  # with a shot list, each item's own best candidates join the pool
EMB_BLOCK = 1 << 16  # rows cast to float32 per matmul block
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "der", "die", "das", "und", "mit", "ein", "eine",
             "von", "auf", "für", "den", "dem", "des", "a", "an", "of", "on", "in", "at", "to", "is"}

# One model lock per process: UI requests and indexing must not run inference concurrently (MPS/CUDA memory).
MODEL_LOCK = threading.RLock()


def _disjoint(o: dict, others) -> bool:
    """True if span `o` does not overlap any of `others` (touching is fine)."""
    return all(min(o["end"], t["end"]) <= max(o["start"], t["start"]) for t in others)


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
        self._contexts: dict[int, Context] = {}
        self._emb = c.grid["emb"]  # stored dtype (float16 saves ~1 GB on a 40 h corpus); cast per block in _similarity
        scale = c.meta.get("logit_scale")
        self._logit = None if scale is None else (float(scale), float(c.meta.get("logit_bias") or 0.0))
        self.load_seconds = 0.0
        self.scorer = load_scorer(scorer)

    @classmethod
    def open(cls, path: Path | str, device: str = "auto", use_nc: bool = True,
             scorer: Scorer | str | Path | None = None) -> "Engine":
        """A project folder, a per-video index dir, or a video with its index next to it."""
        return cls(open_corpus(Path(path)), device, use_nc, scorer)

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
                self._text_cache.update(zip(missing, self.encoder.texts(missing)))
        return np.stack([self._text_cache[t] for t in texts])

    def _relevance(self, text: str) -> np.ndarray:
        c = self.corpus
        return relevance(self._similarity(self.text_vector(text)), c.fps, self._logit, c.file_id)

    def _relevances(self, texts: list[str]) -> np.ndarray:
        """(texts, steps) SigLIP relevance per text on the absolute scale. With one text this is text_vector's path,
        so a single-text query scores exactly as before."""
        c = self.corpus
        if len(texts) == 1:
            return self._relevance(texts[0])[None]
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
        """(per-step signals, raw highlight-model saliency or None, its proposals per file) for a parsed brief.

        relevance: the hand-weighted relevance the candidates, the heuristic scorer and the reported scores use: per
        shot item (or the text) SigLIP averaged with the highlight saliency where a file has it, the best item per
        step, minus the bounded avoid penalty. siglip / avoid_pen / nc_sal: the same parts kept apart (features).
        items: SigLIP relevance per shot item (which item a clip serves is decided on SigLIP alone: the query-wide
        saliency says nothing about a single item and would shift the absolute ITEM_MIN_REL scale)."""
        c = self.corpus
        nc_raw, proposals = self._nc_scores(self._nc_query(p))
        nc = nc_signal(nc_raw)
        sig = timestep_signals(c.grid, c.fps, None, c.audio_steps, c.file_id)
        sig.update(siglip=None, avoid_pen=None, nc_sal=nc, items=None, item_centre=None, muted=self._muted)
        items = p.shots or ([p.text] if p.text else [])
        if items:
            rel_items = self._relevances(items)
            rel_avoid = self._relevances(p.avoid) if p.avoid else None
            want = combine(rel_items)
            sig.update(relevance=combine(blend_nc(rel_items, nc), rel_avoid), siglip=want,
                       avoid_pen=want - combine(rel_items, rel_avoid))
            if p.shots:
                sig.update(items=rel_items, item_centre=item_centre(rel_items))
        return sig, nc_raw, proposals

    def _avoided(self, rel: np.ndarray, p: ParsedQuery) -> np.ndarray:
        """Opener/closer relevance with the brief's avoid penalty too."""
        return combine(rel[None], self._relevances(p.avoid)) if p.avoid else rel

    @staticmethod
    def _nc_query(p: ParsedQuery) -> str:
        """The highlight model was trained on short English sentences: the raw query when it is one, the shot list
        when there is one (an LLM's one-line summary such as "news package" names no visible content), else the
        content text. highlight_nc fits whatever it gets into CLIP's 77 tokens."""
        if p.shots:
            return ", ".join(p.shots)
        simple = not (p.avoid or p.topics) and len(p.raw.split()) <= 25
        return p.raw if simple and p.text else p.text

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
        v = np.stack([self._emb[a:b].mean(0, dtype=np.float32) for a, b in zip(i0, i1)]) if len(i0) else np.zeros((0, 1))
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
    def parse(self, query: str, llm: str | None = "auto", **overrides) -> ParsedQuery:
        """llm: an Ollama model, 'auto' (the default parser model when reachable) or None (rules only)."""
        return parse_brief(query, llm, **overrides)

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

    def _unusable(self, S: np.ndarray, E: np.ndarray, K: np.ndarray, bad: np.ndarray | None = None) -> np.ndarray:
        """Fraction of unusable grid steps per span (or of the steps flagged in `bad`)."""
        i0, i1 = self.corpus.steps(S, E, K)
        bad = np.concatenate([[0], np.cumsum(~self.corpus.usable if bad is None else bad)])
        return (bad[i1] - bad[i0]) / (i1 - i0)

    @functools.cached_property
    def _muted(self) -> np.ndarray:
        return muted_steps(self.corpus.grid["loudness"], self.corpus.audio_steps)

    def _gate(self, S: np.ndarray, E: np.ndarray, K: np.ndarray, p: ParsedQuery) -> np.ndarray:
        """Candidates to keep: few unusable steps, and live sound where the brief asks for it."""
        if not len(S):
            return np.zeros(0, bool)
        keep = self._unusable(S, E, K) <= MAX_UNUSABLE
        if p.need_live_audio:
            keep &= self._unusable(S, E, K, self._muted) <= LIVE_MAX_MUTED
        return keep

    def _pool(self, sig: dict, weights: dict, cands: tuple, p: ParsedQuery, size: int, persona: str | None = None,
              trace: dict | None = None) -> list[dict]:
        """Stage 1, vectorised over every candidate: features and Scorer.score. Stage 2 on the best `size` (with a
        shot list, also each item's own best): refined edges, the usable gate again, features of the refined spans
        and Scorer.rescore, so selection (budget, overlap, MMR) runs on the final clips. Each returned candidate
        carries its total, its refined features ("x") and, with a shot list, the item it serves. trace: filled with
        the stage-1 features, scores, spans and query vector (for a full candidate dump)."""
        c = self.corpus
        S, E, K, src = cands
        q = query_vector(p, persona, self.scorer.personas, weights)
        X0 = featurize(c, sig, S, E, K, src, p)
        s0 = np.asarray(self.scorer.score(X0, q), dtype=float)
        if trace is not None:
            trace.update(X=X0, s=s0, S=S, E=E, K=K, src=src, q=q)
        chosen = np.argsort(-s0)[:size]
        if sig.get("items") is not None and len(sig["items"]) > 1:
            # A generic item ("wide of the avenue", relevant everywhere) would fill the pool by itself: also take
            # each item's own best candidates, ranked with that item's relevance in place of the max.
            i0, i1 = c.steps(S, E, K)
            w = weights.get("relevance", 0.0)
            h_mean = X0[:, INDEX["h_rel_mean"]]
            extra = [np.argsort(-(s0 + w * (span_mean(row, i0, i1) - h_mean)))[:ITEM_POOL] for row in sig["items"]]
            chosen = np.concatenate([chosen, *extra])
            chosen = chosen[np.sort(np.unique(chosen, return_index=True)[1])]
        out, rows = [], []
        for j in chosen:
            k, off = int(K[j]), c.offsets[K[j]]
            s, e = refine(float(S[j] - off), float(E[j] - off), self._context(k), p.min_len, p.max_len)
            out.append({"start": off + s, "end": off + e, "file": k, "source": src[j],
                        "cuts_inside": int(X0[j, INDEX["n_cuts_inside"]]),
                        "relevance": float(X0[j, INDEX["h_rel_top60"]])})
            rows.append(j)
        if not out:
            return out
        # Refinement moves edges by up to a second or so: gate the final spans again.
        keep = self._gate(*(np.array([o[key] for o in out]) for key in ("start", "end", "file")), p)
        out, rows = [o for o, ok in zip(out, keep) if ok], np.array([j for j, ok in zip(rows, keep) if ok], dtype=int)
        if not out:
            return out
        S1, E1, K1 = (np.array([o[key] for o in out]) for key in ("start", "end", "file"))
        scores = None
        if sig.get("items") is not None:  # which shot item each final span serves
            r0, r1 = c.steps(S1, E1, K1)
            scores = item_scores(sig["items"], r0, r1)
            for o, row, it in zip(out, scores, assign(scores, sig["item_centre"])):
                o["item_rel"], o["item"] = row, int(it)
        X1 = featurize(c, sig, S1, E1, K1, [o["source"] for o in out], p, item_rel=scores)
        totals = self.scorer.rescore(s0[rows], X0[rows], X1, q)
        for o, t, x in zip(out, totals, X1):
            o["total"], o["x"] = float(t), x
        return out

    def _edge_clip(self, text: str, sig: dict, cands: tuple, p: ParsedQuery, taken: list[dict],
                   max_len: float, persona: str | None = None) -> dict | None:
        """Best clip for the opener/closer description: its own relevance (minus the brief's avoid penalty, without
        the brief's highlight saliency), steady and clean, not overlapping."""
        rel = self._relevance(text)
        avoided = self._avoided(rel, p)
        edge_sig = dict(sig, relevance=avoided, siglip=rel, avoid_pen=rel - avoided, nc_sal=None, items=None,
                        item_centre=None)
        pool = self._pool(edge_sig, EDGE_WEIGHTS, cands, p, EDGE_POOL, persona)
        pool = [o for o in pool if o["end"] - o["start"] <= max_len + 1e-6
                and _disjoint(o, taken)]
        return max(pool, key=lambda o: o["total"]) if pool else None

    def clips(self, query: str, count: int | None = None, min_len: float | None = None,
              max_len: float | None = None, llm: str | None = "auto",
              budget_s: float | tuple[float, float] | None = None, persona: str | None = None,
              dump: Path | str | None = None, dump_all: bool = False) -> dict:
        """llm: Ollama model for parsing the brief, 'auto' (default model when reachable) or None (rules).
        persona: who the edit is for (part of the query vector a learned scorer sees). dump: write every scored
        candidate there (vhe.rank.dump: the shortlist and the picks; with dump_all every gated candidate too)."""
        load_before = self.load_seconds
        t0 = time.perf_counter()
        c = self.corpus
        p = self.parse(query, llm, count=count, min_len=min_len, max_len=max_len, budget_s=budget_s)
        if p.topics and not p.shots:  # "every moment the host talks about ...": transcript spans, not pictures
            clips, n_spans, method = self._topic_clips(p, None if p.exhaustive else p.count,
                                                       p.soundbite_len or (p.min_len, p.max_len), role=None)
            result = {"query": p.to_dict(), "clips": clips, "candidates": n_spans, "files": len(c.members),
                      "noncommercial": [], "unusable_dropped": 0, "topic_match": method}
            if dump:
                self._dump(dump, query, p, persona, self._signals(p)[0], topic=clips)
            return self._finish(result, t0, load_before)
        sig, nc_raw, proposals = self._signals(p)
        per_step = sum(p.weights.get(k, 0.0) * sig[k] for k in ("relevance", "quality", "energy", "steady"))
        S, E, K, src = self._candidates(per_step, p, proposals)
        result = {"query": p.to_dict(), "clips": [], "candidates": len(S), "files": len(c.members),
                  "noncommercial": sorted({x for ix in c.members for x in ix.meta.get("noncommercial_components", [])})
                  if nc_raw is not None else []}
        keep = self._gate(S, E, K, p)
        result["unusable_dropped"] = int(len(S) - keep.sum())
        cands = (S[keep], E[keep], K[keep], [s for s, k in zip(src, keep) if k])
        if not len(cands[0]):
            if dump:
                self._dump(dump, query, p, persona, sig)
            return self._finish(result, t0, load_before)

        # Opener and closer first: they take their slots (and budget) before the body is selected.
        edges: dict[str, dict] = {}
        room = np.inf if p.budget is None else p.budget[1]
        for role, text in (("opener", p.opener), ("closer", p.closer)):
            if text:
                e = self._edge_clip(text, sig, cands, p, list(edges.values()), room, persona)
                if e is not None:
                    edges[role] = dict(e, text=text)
                    room -= e["end"] - e["start"]
        used = sum(e["end"] - e["start"] for e in edges.values())
        body_p = dataclasses.replace(
            p, count=max(0, p.count - len(edges)),
            budget=None if p.budget is None else (max(0.0, p.budget[0] - used), p.budget[1] - used))

        trace = {} if dump and dump_all else None
        pool = self._pool(sig, p.weights, cands, p, max(400, 40 * p.count), persona, trace)
        for cid, o in enumerate(pool):
            o["cid"] = cid
        scored = [o for o in pool if _disjoint(o, edges.values())]
        if scored:
            r0, r1 = c.steps(*(np.array([o[key] for o in scored]) for key in ("start", "end", "file")))
            speech_cum = np.concatenate([[0.0], np.cumsum(sig["speech"])])
            for o, f in zip(scored, (speech_cum[r1] - speech_cum[r0]) / (r1 - r0)):
                o["speech_frac"] = float(f)
            item_rel = np.array([o["item_rel"] for o in scored]) if sig["items"] is not None else None
            body = select_for(scored, self._mean_vectors(r0, r1), body_p, groups=np.array([o["file"] for o in scored]),
                              item_rel=item_rel, item_centre=sig["item_centre"])
        else:
            body = []
        # Edit order: files in manifest order (= virtual time), then time within the file; opener first, closer last.
        ordered = ([("opener", edges["opener"])] if "opener" in edges else []) + \
            [(None, o) for o in sorted(body, key=lambda o: o["start"])] + \
            ([("closer", edges["closer"])] if "closer" in edges else [])
        for role, o in ordered:
            result["clips"].append(self._clip(o, sig, p, role))
        bites = []
        if p.topics:  # "also the best 5-8 s soundbite where the host sums up": a separate deliverable, at the end
            bites, _, result["topic_match"] = self._topic_clips(p, 1, p.soundbite_len or (p.min_len, p.max_len),
                                                                role="soundbite")
            result["clips"] += bites
        if dump:
            self._dump(dump, query, p, persona, sig, pool, edges, body, bites, trace)
        return self._finish(result, t0, load_before)

    # ---- features and candidate dumps ------------------------------------------------------------------------
    def _span(self, k: int, s: float, e: float) -> tuple[str, float, float]:
        """(file label, file-local start, end) of virtual span s..e in file k, exactly as an output clip has it."""
        hit = self._hit(int(k), round(float(s), 3), round(float(e), 3))
        return hit["file"], hit["file_start"], hit["file_end"]

    @staticmethod
    def _item_name(p: ParsedQuery, item) -> str | None:
        if isinstance(item, (int, np.integer)):  # index into p.shots; -1: serves no item
            return p.shots[int(item)] if 0 <= int(item) < len(p.shots) else None
        return item

    def _dump_doc(self, query: str, p: ParsedQuery, persona: str | None, candidates: list[dict]) -> dict:
        c = self.corpus
        personas = self.scorer.personas
        return dumps.build(scorer=self.scorer.name, corpus_root=str(c.root or c.members[0].root),
                           n_files=len(c.members), brief=query, parsed=p.to_dict(), persona=persona,
                           q=query_vector(p, persona, personas), q_names=query_names(personas), candidates=candidates)

    def _dump(self, path: Path | str, query: str, p: ParsedQuery, persona: str | None, sig: dict,
              pool: list[dict] = (), edges: dict[str, dict] | None = None, body: list[dict] = (),
              topic: list[dict] = (), trace: dict | None = None) -> None:
        """Write the run's candidates (vhe.rank.dump): with a trace every gated candidate as generated, the
        shortlist with its refined edges and final totals (selected ones with their selection rank and item), the
        opener/closer and topic clips. Every feature vector uses the brief's signals."""
        c = self.corpus
        out = []
        if trace:
            for j in range(len(trace["S"])):
                out.append(dumps.entry(*self._span(trace["K"][j], trace["S"][j], trace["E"][j]), trace["src"][j],
                                       "candidate", trace["s"][j], trace["X"][j]))
        picked = {o["cid"]: o for o in body}
        for o in pool:
            b = picked.get(o["cid"])
            out.append(dumps.entry(*self._span(o["file"], o["start"], o["end"]), o["source"], "shortlist", o["total"],
                                   o["x"], selected=b is not None, rank=None if b is None else b["rank"],
                                   item=self._item_name(p, (b or o).get("item"))))
        extra = [(o, "edge", role) for role, o in (edges or {}).items()] + \
            [({"file": c.files.index(t["file"]), "start": t["start"], "end": t["end"], "source": t["source"],
               "total": t["score"], "item": t["item"]}, "topic", t["role"]) for t in topic]
        if extra:
            S, E, K = (np.array([o[key] for o, _, _ in extra]) for key in ("start", "end", "file"))
            X = featurize(c, sig, S, E, K, [o["source"] for o, _, _ in extra], p)
            for (o, stage, role), x in zip(extra, X):
                out.append(dumps.entry(*self._span(o["file"], o["start"], o["end"]), o["source"], stage, o["total"],
                                       x, selected=True, role=role, item=self._item_name(p, o.get("item"))))
        dumps.write(path, self._dump_doc(query, p, persona, out))

    def features_for_spans(self, query: str, spans: list[dict], persona: str | None = None, llm: str | None = "auto",
                           **overrides) -> dict:
        """The dump document ("vhe-candidates/1", stage "external") for spans picked outside the engine (an editor,
        another model): {"file", "start", "end"} in file-local seconds; "source", "selected", "rank", "role" and
        "item" are carried over when given. Features come from the same signals clips() computes for this brief;
        the score is the engine scorer's score() over these spans."""
        c = self.corpus
        p = self.parse(query, llm, **overrides)
        sig = self._signals(p)[0]
        K = np.array([self._file_index(sp["file"]) for sp in spans], dtype=int)
        S = np.array([c.offsets[k] + min(max(float(sp["start"]), 0.0), c.durations[k]) for sp, k in zip(spans, K)])
        E = np.array([c.offsets[k] + min(max(float(sp["end"]), 0.0), c.durations[k]) for sp, k in zip(spans, K)])
        src = [str(sp.get("source") or "external") for sp in spans]
        X = featurize(c, sig, S, E, K, src, p)
        scores = self.scorer.score(X, query_vector(p, persona, self.scorer.personas)) if len(X) else []
        out = [dumps.entry(*self._span(k, s, e), source, "external", sc, x, selected=bool(sp.get("selected", False)),
                           rank=sp.get("rank"), role=sp.get("role"), item=sp.get("item"))
               for sp, k, s, e, source, sc, x in zip(spans, K, S, E, src, scores, X)]
        return self._dump_doc(query, p, persona, out)

    def _file_index(self, name: str) -> int:
        """A file label (path relative to the project), or a file name that is unique in the corpus."""
        files = self.corpus.files
        if name in files:
            return files.index(name)
        hits = [k for k, f in enumerate(files) if Path(f).name == Path(name).name]
        if len(hits) != 1:
            raise KeyError(f"file {name!r} is not in the corpus ({len(files)} files)")
        return hits[0]

    def _topic_clips(self, p: ParsedQuery, n: int | None, length: tuple[float, float],
                     role: str | None) -> tuple[list[dict], int, str]:
        """Transcript-sentence spans (speech_spans) whose words match p.topics: (clips in edit order, number of spans
        scored, match method). n=None returns every span above the threshold; otherwise the best n above it.
        Picture defects do not gate speech (the speaker may be off camera); muted spans are dropped when the brief
        asks for live sound."""
        c = self.corpus
        lo, hi = length
        spans = []
        for k, ix in enumerate(c.members):
            dur = float(c.durations[k])
            for s, e in speech_spans((ix.transcript or {}).get("segments", []), ix.vad, hi):
                fitted = fit_span(s, e, lo, hi, dur)
                if fitted is None:
                    continue
                s, e = refine(*fitted, self._context(k), lo, hi)
                text = self._words_between(c.offsets[k] + s, c.offsets[k] + e)
                if text:
                    spans.append({"start": c.offsets[k] + s, "end": c.offsets[k] + e, "file": k, "text": text})
        if p.need_live_audio and spans:
            muted = self._unusable(*(np.array([o[key] for o in spans]) for key in ("start", "end", "file")),
                                   self._muted)
            spans = [o for o, m in zip(spans, muted) if m <= LIVE_MAX_MUTED]
        score, best, method, thresh = topic_scores([o["text"] for o in spans], p.topics)
        order = [j for j in np.argsort(-score, kind="stable") if score[j] >= thresh]
        picked: list[dict] = []
        for j in order:
            o = spans[j]
            if n is not None and len(picked) >= n:
                break
            if _disjoint(o, picked):
                picked.append(dict(o, total=float(score[j]), source="speech", item=p.topics[int(best[j])]))
        picked.sort(key=lambda o: o["start"])
        sig = timestep_signals(c.grid, c.fps, None, c.audio_steps, c.file_id)
        out = []
        for o in picked:
            clip = self._clip(o, sig, p, role)
            clip["score"] = clip["rank_score"] = round(o["total"], 3)
            clip["reason"] = f"talks about '{o['item']}' ({method} {o['total']:.2f})"
            out.append(clip)
        return out, len(spans), method

    def _clip(self, o: dict, sig: dict, p: ParsedQuery, role: str | None) -> dict:
        c = self.corpus
        k, s, e = o["file"], round(float(o["start"]), 3), round(float(o["end"]), 3)
        j0, j1 = (int(x) for x in c.steps(s, e, k))
        parts = segment_scores(sig, j0, j1, p.weights)
        labels = self._labels(j0, j1, parts)
        item = self._item_name(p, o.get("item"))
        return self._hit(k, s, e) | {
            "duration": round(e - s, 3), "rank_score": round(o["total"], 3), "score": round(parts["total"], 3),
            "relevance": round(parts["relevance"], 3), "quality": round(parts["quality"], 3),
            "energy": round(parts["energy"], 3), "labels": labels, "source": o["source"], "role": role,
            "item": item, "reason": self._reason(parts, p, labels, role, o.get("text"), item),
            "transcript": self._words_between(s, e)}

    @staticmethod
    def _reason(parts: dict, p: ParsedQuery, labels: list[str], role: str | None, text: str | None,
                item: str | None = None) -> str:
        what = f"'{item}'" if item else "the shot list" if p.shots else f"'{p.text}'"
        names = {"relevance": f"matches {what}", "quality": "clean footage", "steady": "steady camera",
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
