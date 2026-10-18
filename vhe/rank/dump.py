"""Candidate dumps ("vhe-candidates/1"): every candidate a run scored, with its feature vector, its score and whether
it was selected, so judge labels can be joined onto spans to build a training set.

{"format": "vhe-candidates/1", "feature_version": "v4", "query_version": "v4",
 "run": {"app_sha", "scorer", "created", "corpus_root", "n_files"},
 "query": {"brief", "parsed", "persona", "q": {name: value}},
 "feature_names": [...],
 "candidates": [{"file", "start", "end", "source", "stage", "score", "features", "selected", "rank", "role",
                 "item"}]}

file, start and end are the file label and file-local seconds (3 dp) exactly as in the output clips; a selected row
is the returned clip itself (a clip the budget fill lengthened is dumped with its final edges). stage: "candidate"
(--dump-all: every gated candidate with its edges as generated, scored by score()), "shortlist" (the best candidates
with refined edges, scored by rescore(); selection runs on these), "edge" (the opener/closer picks), "topic"
(spoken-topic clips and soundbites, scored by transcript match), "external" (spans given to `vhe features`). Every
feature vector is computed with the brief's signals and the brief's stage-1 pool statistics (an opener's too), so a
span has the same features whichever way it got into a dump, and every score but a topic row's is the run's scorer
on the row's own features and q: score() for "candidate", rescore() of the span alone for the rest (an opener was
picked in its slot's own pool; its row is scored as the brief would score it). rank: selection order of a body clip;
role: opener/closer/soundbite; item: the shot item or spoken topic a clip serves."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vhe.query.model import ParsedQuery, item_name
from vhe.rank.features import (
    FEATURE_VERSION,
    FEATURES,
    QUERY_VERSION,
    featurize,
    query_names,
    query_vector,
)

FORMAT = "vhe-candidates/1"
STAGES = ("candidate", "shortlist", "edge", "topic", "external")


def app_sha() -> str | None:
    """The git commit of the vhe checkout ('-dirty' with uncommitted changes to tracked files); None outside git."""
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def entry(file: str, start: float, end: float, source: str, stage: str, score: float, x: np.ndarray,
          selected: bool = False, rank: int | None = None, role: str | None = None, item: str | None = None) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return {"file": file, "start": start, "end": end, "source": source, "stage": stage,
            "score": None if score is None else round(float(score), 6),
            "features": np.round(np.asarray(x, dtype=float), 6).tolist(), "selected": bool(selected),
            "rank": None if rank is None else int(rank), "role": role, "item": item}


def build(*, scorer: str, corpus_root: str, n_files: int, brief: str, parsed: dict, persona: str | None,
          q: np.ndarray, q_names: tuple[str, ...], candidates: list[dict]) -> dict:
    return {"format": FORMAT, "feature_version": FEATURE_VERSION, "query_version": QUERY_VERSION,
            "run": {"app_sha": app_sha(), "scorer": scorer,
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "corpus_root": corpus_root, "n_files": int(n_files)},
            "query": {"brief": brief, "parsed": parsed, "persona": persona,
                      "q": {n: float(v) for n, v in zip(q_names, q, strict=True)}},
            "feature_names": list(FEATURES), "candidates": candidates}


def write(path: Path | str, dump: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump, ensure_ascii=False, separators=(",", ":")))
    return path


def read(path: Path | str) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("format") != FORMAT:
        raise ValueError(f"{path}: format {d.get('format')!r}, expected {FORMAT!r}")
    return d


def load_spans(path: Path | str) -> list[dict]:
    """Spans to featurise: a JSON list of {"file", "start", "end", ...} (file-local seconds), or an object with
    such a list under "spans", "clips" (`vhe clips --json`) or "candidates" (a dump)."""
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict):
        d = next((d[k] for k in ("spans", "clips", "candidates") if isinstance(d.get(k), list)), None)
    if not isinstance(d, list) or not all(isinstance(s, dict) and {"file", "start", "end"} <= s.keys() for s in d):
        raise ValueError(f"{path}: expected a list of {{file, start, end}} spans")
    return d


# ---- a run's dump, built from the engine's state ---------------------------------------------------------------------
def span_label(corpus, k: int, s: float, e: float) -> tuple[str, float, float]:
    """(file label, file-local start, end) of virtual span s..e in file k, exactly as an output clip has it."""
    hit = corpus.locate(int(k), round(float(s), 3), round(float(e), 3))
    return hit["file"], hit["file_start"], hit["file_end"]


def document(corpus, scorer, brief: str, p: ParsedQuery, persona: str | None, candidates: list[dict]) -> dict:
    """build() for a run of `scorer` on `corpus`."""
    c = corpus
    personas = scorer.personas
    return build(scorer=scorer.name, corpus_root=str(c.root or c.members[0].root), n_files=len(c.members),
                 brief=brief, parsed=p.to_dict(), persona=persona, q=query_vector(p, persona, personas),
                 q_names=query_names(personas), candidates=candidates)


def final_scores(scorer, X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The scorer's total for final clips with features X, from their own features (rescore of the span alone)."""
    if not len(X):
        return np.zeros(0)
    return np.asarray(scorer.rescore(scorer.score(X, q), X, X, q), dtype=float)


def write_run(path: Path | str, corpus, scorer, brief: str, p: ParsedQuery, persona: str | None, sig: dict,
              pool: list[dict] = (), edges: dict[str, dict] | None = None, body: list[dict] = (),
              topic: list[dict] = (), trace: dict | None = None, stats: dict | None = None) -> None:
    """Write the run's candidates: with a trace every gated candidate as generated, the shortlist with its refined
    edges and final totals (a selected one as returned: its final edges, features and total, selection rank and
    item), the opener/closer and topic clips. Every feature vector uses the brief's signals and stats, the brief's
    stage-1 pool statistics (vhe.rank.features.pool_stats)."""
    c = corpus
    out = []
    if trace:
        for j in range(len(trace["S"])):
            out.append(entry(*span_label(c, trace["K"][j], trace["S"][j], trace["E"][j]), trace["src"][j],
                             "candidate", trace["s"][j], trace["X"][j]))
    picked = {o["cid"]: o for o in body}
    for o in pool:
        b = picked.get(o["cid"])
        r = o if b is None else b  # the clip as returned (the budget fill may have lengthened it)
        out.append(entry(*span_label(c, r["file"], r["start"], r["end"]), r["source"], "shortlist", r["total"],
                         r["x"], selected=b is not None, rank=None if b is None else b["rank"],
                         item=item_name(p, r.get("item"))))
    extra = [(o, "edge", role) for role, o in (edges or {}).items()] + \
        [({"file": c.files.index(t["file"]), "start": t["start"], "end": t["end"], "source": t["source"],
           "total": t["score"], "item": t["item"]}, "topic", t["role"]) for t in topic]
    if extra:
        S, E, K = (np.array([o[key] for o, _, _ in extra]) for key in ("start", "end", "file"))
        X = featurize(c, sig, S, E, K, [o["source"] for o, _, _ in extra], p, stats=stats)
        brief_q = query_vector(p, persona, scorer.personas)
        scores = final_scores(scorer, X, brief_q)
        for (o, stage, role), x, sc in zip(extra, X, scores, strict=True):
            out.append(entry(*span_label(c, o["file"], o["start"], o["end"]), o["source"], stage,
                             sc if stage == "edge" else o["total"], x, selected=True, role=role,
                             item=item_name(p, o.get("item"))))
    write(path, document(c, scorer, brief, p, persona, out))


def external_spans(corpus, scorer, brief: str, p: ParsedQuery, persona: str | None, sig: dict,
                   spans: list[dict], stats: dict | None = None) -> dict:
    """The dump document (stage "external") for spans picked outside the engine (an editor, another model):
    {"file", "start", "end"} in file-local seconds; "source", "selected", "rank", "role" and "item" are carried over
    when given. sig: the brief's signals; stats: its stage-1 pool statistics (None: these spans are the pool); the
    score is the scorer's total for each span as a final clip (final_scores)."""
    c = corpus
    K = np.array([c.file_index(sp["file"]) for sp in spans], dtype=int)
    S = np.array([c.offsets[k] + min(max(float(sp["start"]), 0.0), c.durations[k])
                  for sp, k in zip(spans, K, strict=True)])
    E = np.array([c.offsets[k] + min(max(float(sp["end"]), 0.0), c.durations[k])
                  for sp, k in zip(spans, K, strict=True)])
    src = [str(sp.get("source") or "external") for sp in spans]
    X = featurize(c, sig, S, E, K, src, p, stats=stats)
    scores = final_scores(scorer, X, query_vector(p, persona, scorer.personas))
    out = [entry(*span_label(c, k, s, e), source, "external", sc, x, selected=bool(sp.get("selected", False)),
                 rank=sp.get("rank"), role=sp.get("role"), item=sp.get("item"))
           for sp, k, s, e, source, sc, x in zip(spans, K, S, E, src, scores, X, strict=True)]
    return document(c, scorer, brief, p, persona, out)
