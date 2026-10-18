"""Training rows (the harness's labels.py output, eval/dataset/rows.jsonl) -> arrays for train_linear.py and
eval_rank.py. Not part of the vhe package; numpy only (rows are JSONL; there is no parquet path: pyarrow is not a
dependency).

  python tools/train/build_dataset.py ROWS.jsonl [--split train|heldout|all] [--out data.npz] [--keep-all-features]

A row is one span under one query (set, run_id, brief_id) with its feature vector (f_<name>, FEATURE_VERSION), its query
vector (q: {name: value}), the judge label (y_point, label_kind, weight) and bookkeeping (origin, selected_by,
dump_score). The learned feature mask drops src_* (external spans, e.g. Opus picks, have no generator source, so the
column would encode who picked the clip) and h_rel_* (the hand-weighted relevance, kept only so HeuristicScorer is an
exact function of X) unless --keep-all-features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vhe.rank.features import FEATURE_VERSION, FEATURES, PERSONAS, query_names

DROP_PREFIXES = ("src_", "h_rel_")
LABELLED = ("judged", "label_transfer")
MIN_DY = 1 / 3          # a pair needs labels at least one fit step apart
POSITIVE_Y = 2 / 3      # fit >= 2 and usable: compared against unlabelled pool rows
N_POOL, W_POOL = 5, 0.3


def learned_mask(drop: tuple[str, ...] = DROP_PREFIXES) -> np.ndarray:
    """(F,) 0/1 over FEATURES: 1 for the columns a learned model may use."""
    return np.array([0.0 if n.startswith(drop) else 1.0 for n in FEATURES])


@dataclass
class Dataset:
    X: np.ndarray            # (n, F) FEATURES order
    Q: np.ndarray            # (n, len(query_names(PERSONAS)))
    y: np.ndarray            # (n,) y_point, NaN for pool_negative / unjudged rows
    w: np.ndarray            # (n,) row weight
    kind: np.ndarray         # label_kind
    query: np.ndarray        # "set/run_id/brief_id": rows ranked against each other
    group: np.ndarray        # "set/brief_id": the CV unit (the same brief in two runs is one group)
    origin: np.ndarray       # app_dump | app_picks | opus
    selected_by: np.ndarray  # app | opus | none
    dump_score: np.ndarray   # (n,) the score recorded at dump time, NaN if none
    split: np.ndarray
    mask: np.ndarray         # (F,) learned feature mask
    rows: list = field(default_factory=list, repr=False)
    source: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def labelled(self) -> np.ndarray:
        return np.isin(self.kind, LABELLED) & np.isfinite(self.y)

    def subset(self, idx) -> "Dataset":
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        arrays = {k: getattr(self, k)[idx] for k in ("X", "Q", "y", "w", "kind", "query", "group", "origin",
                                                     "selected_by", "dump_score", "split")}
        return Dataset(**arrays, mask=self.mask, rows=[self.rows[i] for i in idx] if self.rows else [],
                       source=self.source)


def load_rows(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build(rows: list[dict], split: str | None = "train", drop: tuple[str, ...] = DROP_PREFIXES,
          keep_unjudged: bool = False, source: dict | None = None) -> Dataset:
    """Rows of `split` (None: all) with this engine's feature version -> Dataset. Rows of another feature version
    are refused (counted in source["skipped"]); unjudged rows (weight 0) are dropped unless keep_unjudged."""
    qn = query_names(PERSONAS)
    skipped: Counter = Counter()
    keep = []
    for r in rows:
        if split is not None and r.get("split") != split:
            continue
        if r.get("feature_version") != FEATURE_VERSION:
            skipped[f"feature_version {r.get('feature_version')}"] += 1
            continue
        if r.get("label_kind") == "unjudged" and not keep_unjudged:
            skipped["unjudged"] += 1
            continue
        keep.append(r)
    missing = Counter(n for r in keep[:1] for n in FEATURES if f"f_{n}" not in r)
    if missing:
        raise ValueError(f"rows lack feature columns {sorted(missing)[:5]} (features {FEATURE_VERSION})")
    n = len(keep)
    X = np.array([[float(r[f"f_{name}"]) for name in FEATURES] for r in keep]).reshape(n, len(FEATURES))
    Q = np.array([[float((r.get("q") or {}).get(name, 0.0)) for name in qn] for r in keep]).reshape(n, len(qn))

    def num(v):
        return np.nan if v is None else float(v)

    def col(key, default=""):
        return np.array([str(r.get(key) if r.get(key) is not None else default) for r in keep], dtype=object)

    return Dataset(
        X=X, Q=Q, y=np.array([num(r.get("y_point")) for r in keep]), w=np.array([num(r.get("weight")) for r in keep]),
        kind=col("label_kind"),
        query=np.array([f"{r['set']}/{r['run_id']}/{r['brief_id']}" for r in keep], dtype=object),
        group=np.array([f"{r['set']}/{r['brief_id']}" for r in keep], dtype=object), origin=col("origin"),
        selected_by=col("selected_by", "none"), dump_score=np.array([num(r.get("dump_score")) for r in keep]),
        split=col("split"), mask=learned_mask(drop), rows=keep,
        source={**(source or {}), "split": split, "skipped": dict(skipped)})


def pairs(ds: Dataset, seed: int = 0, n_pool: int = N_POOL, pool_w: float = W_POOL,
          min_dy: float = MIN_DY, positive_y: float = POSITIVE_Y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training pairs (i better than j, weight) within each query: every two labelled rows whose y differ by at least
    min_dy (weight: the smaller row weight: 1 judged, 0.5 transferred), and each positive labelled row (y >=
    positive_y) against up to n_pool random unlabelled pool rows of the same query (weight pool_w x its row weight)."""
    rng = np.random.default_rng(seed)
    win, lose, W = [], [], []
    lab = ds.labelled
    pool = ds.kind == "pool_negative"
    for q in np.unique(ds.query):
        rows = np.flatnonzero(ds.query == q)
        L, P = rows[lab[rows]], rows[pool[rows]]
        if len(L) > 1:
            dy = ds.y[L][:, None] - ds.y[L][None, :]
            a, b = np.nonzero(dy >= min_dy - 1e-9)
            win += L[a].tolist()
            lose += L[b].tolist()
            W += np.minimum(ds.w[L[a]], ds.w[L[b]]).tolist()
        if len(P):
            for i in L[ds.y[L] >= positive_y - 1e-9]:
                for j in rng.choice(P, size=min(n_pool, len(P)), replace=False):
                    win.append(i)
                    lose.append(j)
                    W.append(pool_w * ds.w[i])
    return np.array(win, dtype=int), np.array(lose, dtype=int), np.array(W, dtype=float)


def summary(ds: Dataset) -> dict:
    lab = ds.labelled
    win, _, _ = pairs(ds)
    return {"rows": len(ds), "queries": int(len(np.unique(ds.query))), "briefs": int(len(np.unique(ds.group))),
            "label_kind": dict(Counter(ds.kind.tolist())), "origin": dict(Counter(ds.origin.tolist())),
            "split": dict(Counter(ds.split.tolist())),
            "y_point": {f"{k:.3f}": v for k, v in sorted(Counter(np.round(ds.y[lab], 3).tolist()).items())},
            "pairs": int(len(win)), "learned_features": int(ds.mask.sum()), "skipped": ds.source.get("skipped", {})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--split", default="train", help="train, heldout or all")
    ap.add_argument("--out", type=Path, default=None, help="write the arrays as .npz")
    ap.add_argument("--keep-all-features", action="store_true", help="do not mask src_* and h_rel_*")
    a = ap.parse_args()
    ds = build(load_rows(a.rows), None if a.split == "all" else a.split, () if a.keep_all_features else DROP_PREFIXES,
               source={"rows": str(a.rows), "sha": file_sha(a.rows)})
    print(json.dumps(summary(ds), indent=1))
    if a.out:
        np.savez_compressed(a.out, X=ds.X, Q=ds.Q, y=ds.y, w=ds.w, kind=ds.kind.astype(str),
                            query=ds.query.astype(str), group=ds.group.astype(str), origin=ds.origin.astype(str),
                            selected_by=ds.selected_by.astype(str), dump_score=ds.dump_score, mask=ds.mask,
                            features=np.array(FEATURES), query_names=np.array(query_names(PERSONAS)))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
