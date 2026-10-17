"""Candidate dumps ("vhe-candidates/1"): every candidate a run scored, with its feature vector, its score and whether
it was selected, so judge labels can be joined onto spans to build a training set.

{"format": "vhe-candidates/1", "feature_version": "v1",
 "run": {"app_sha", "scorer", "created", "corpus_root", "n_files"},
 "query": {"brief", "parsed", "persona", "q": {name: value}},
 "feature_names": [...],
 "candidates": [{"file", "start", "end", "source", "stage", "score", "features", "selected", "rank", "role",
                 "item"}]}

file, start and end are the file label and file-local seconds (3 dp) exactly as in the output clips. stage:
"candidate" (--dump-all: every gated candidate with its edges as generated, scored by score()), "shortlist" (the
best candidates with refined edges, scored by rescore(); selection runs on these), "edge" (the opener/closer picks,
scored in their own pool), "topic" (spoken-topic clips and soundbites, scored by transcript match), "external"
(spans given to `vhe features`). Every feature vector is computed with the brief's signals (an opener's too), so a
span has the same features whichever way it got into a dump. rank: selection order of a body clip; role:
opener/closer/soundbite; item: the shot item or spoken topic a clip serves."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vhe.rank.features import FEATURE_VERSION, FEATURES

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
    return {"format": FORMAT, "feature_version": FEATURE_VERSION,
            "run": {"app_sha": app_sha(), "scorer": scorer,
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "corpus_root": corpus_root, "n_files": int(n_files)},
            "query": {"brief": brief, "parsed": parsed, "persona": persona,
                      "q": {n: float(v) for n, v in zip(q_names, q)}},
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
