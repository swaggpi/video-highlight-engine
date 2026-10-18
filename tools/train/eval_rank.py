"""Offline ranking metrics of learned models against the hand-weighted scorer, on any split of the training rows.

  python tools/train/eval_rank.py ROWS.jsonl --model models/linear_v2.json [--model ...] [--split heldout|train|all]
                                  [--boot 2000] [--json OUT]

Per query (set, run, brief) over its labelled rows (judged and label_transfer; unlabelled pool rows carry no y and
are left out): pairwise accuracy over the pairs whose y differ by >= 1/3 (a tie in score counts 1/2) and NDCG@5
with gain y. Each metric is averaged over queries (briefs) with a 95 % bootstrap CI over queries, and every model is
compared with the heuristic by the bootstrap CI of the paired difference.

The heuristic bar, two ways:
  heuristic        HeuristicScorer recomputed from the rows' features (score(), then rescore()'s top-60 relevance
                   swap) with all rows of a query as one pool (its length-bucket centring depends on the pool), so
                   app and Opus rows are scored alike. Rows of one query with different parsed briefs (q) are scored
                   per q.
  heuristic_dump   the score recorded with each app row (dump_score): its score in the run for app_dump rows, score()
                   over the app's picks alone for app_picks rows (runs without a dump). Only on the "app rows" table:
                   an Opus row's recorded score comes from `vhe features` over the Opus picks alone, a different pool.
Tables: "all rows" (models vs heuristic) and "app rows" (rows the app featurised in its run: models vs both).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import MIN_DY, Dataset, build, file_sha, load_rows  # noqa: E402

from vhe.rank.scorer import HeuristicScorer, load_scorer  # noqa: E402

K_NDCG = 5


def pairwise(y: np.ndarray, s: np.ndarray, min_dy: float = MIN_DY) -> tuple[float, int]:
    """(correctly ordered pairs, number of pairs) over pairs with y_i - y_j >= min_dy; a score tie counts 1/2."""
    a, b = np.nonzero(y[:, None] - y[None, :] >= min_dy - 1e-9)
    if not len(a):
        return 0.0, 0
    d = s[a] - s[b]
    return float(np.sum(d > 0) + 0.5 * np.sum(d == 0)), int(len(a))


def ndcg(y: np.ndarray, s: np.ndarray, k: int = K_NDCG) -> float | None:
    """NDCG@k with linear gain max(y, 0); ties in s are broken against the row (worst case among equals is not
    modelled: order is stable by row). None when no row has a positive gain."""
    g = np.maximum(y, 0.0)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(g)[::-1][:k]
    idcg = float(np.sum(ideal * disc[:len(ideal)]))
    if idcg <= 0:
        return None
    top = g[np.argsort(-s, kind="stable")][:k]
    return float(np.sum(top * disc[:len(top)])) / idcg


def q_groups(ds: Dataset, per_query: bool) -> list[np.ndarray]:
    """Row indices sharing one query vector (and, with per_query, one query)."""
    groups: dict = {}
    for i, (qid, row) in enumerate(zip(ds.query, ds.Q)):
        groups.setdefault((qid if per_query else None, row.tobytes()), []).append(i)
    return [np.array(v) for v in groups.values()]


def heuristic_scores(ds: Dataset) -> np.ndarray:
    """HeuristicScorer total per row (score() + rescore()'s top-60 swap), with each query's rows (per distinct q)
    as one pool."""
    h = HeuristicScorer()
    out = np.full(len(ds), np.nan)
    for idx in q_groups(ds, per_query=True):
        X, q = ds.X[idx], ds.Q[idx[0]]
        s = h.score(X, q)
        out[idx] = h.rescore(s, X, X, q)
    return out


def queries_with_several_q(ds: Dataset) -> int:
    n = 0
    for q in np.unique(ds.query):
        idx = ds.query == q
        n += len(np.unique(ds.Q[idx], axis=0)) > 1
    return n


def per_query(ds: Dataset, scores: dict[str, np.ndarray], rows: np.ndarray | None = None) -> list[dict]:
    """One record per query: n labelled rows, pairs, and per scorer the pairwise accuracy (None without pairs) and
    NDCG@5 (None without a positive). rows: a boolean row filter applied on top of 'labelled'."""
    lab = ds.labelled if rows is None else ds.labelled & rows
    out = []
    for q in np.unique(ds.query[lab]):
        idx = np.flatnonzero(lab & (ds.query == q))
        y = ds.y[idx]
        rec = {"query": str(q), "group": str(ds.group[idx[0]]), "n": int(len(idx)), "acc": {}, "ndcg": {}}
        for name, s in scores.items():
            sv = s[idx]
            if not np.all(np.isfinite(sv)):
                rec["acc"][name] = rec["ndcg"][name] = None
                continue
            c, n = pairwise(y, sv)
            rec["pairs"] = n
            rec["acc"][name] = c / n if n else None
            rec["ndcg"][name] = ndcg(y, sv)
        out.append(rec)
    return out


def _boot(vals: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    if len(vals) < 2:
        return float("nan"), float("nan")
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(recs: list[dict], names: list[str], base: str | None = "heuristic", n_boot: int = 2000,
              seed: int = 0) -> dict:
    """Per scorer and metric: mean over queries that have the metric for every scorer, bootstrap 95 % CI, and the CI
    of the paired difference to `base`."""
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ("acc", "ndcg"):
        ok = [r for r in recs if all(r[metric].get(n) is not None for n in names)]
        m = {"queries": len(ok), "pairs": int(sum(r.get("pairs", 0) for r in ok)) if metric == "acc" else None}
        for n in names:
            v = np.array([r[metric][n] for r in ok], dtype=float)
            lo, hi = _boot(v, n_boot, rng)
            m[n] = {"mean": float(v.mean()) if len(v) else None, "ci95": [lo, hi]}
            if base and n != base and base in names and len(v):
                d = v - np.array([r[metric][base] for r in ok], dtype=float)
                dlo, dhi = _boot(d, n_boot, rng)
                m[n]["diff_vs_" + base] = {"mean": float(d.mean()), "ci95": [dlo, dhi]}
        out[metric] = m
    return out


def evaluate(ds: Dataset, models: dict[str, object], n_boot: int = 2000) -> dict:
    """Both tables for a dataset and {name: scorer with .score(X, q)} models."""
    scores = {name: score_rows(m, ds) for name, m in models.items()}
    scores["heuristic"] = heuristic_scores(ds)
    names_all = [*models, "heuristic"]
    app = np.isin(ds.origin, ("app_dump", "app_picks"))
    res = {"all_rows": aggregate(per_query(ds, scores), names_all, n_boot=n_boot)}
    if app.any():
        scores_app = {**scores, "heuristic_dump": ds.dump_score}
        names_app = [*names_all, "heuristic_dump"]
        res["app_rows"] = aggregate(per_query(ds, scores_app, rows=app), names_app, n_boot=n_boot)
    res["queries_with_several_q"] = queries_with_several_q(ds)
    return res


def score_rows(model, ds: Dataset) -> np.ndarray:
    """model.score over the rows, per distinct query vector."""
    out = np.full(len(ds), np.nan)
    for idx in q_groups(ds, per_query=False):
        out[idx] = model.score(ds.X[idx], ds.Q[idx[0]])
    return out


def format_table(res: dict) -> str:
    lines = []
    for table in ("all_rows", "app_rows"):
        if table not in res:
            continue
        for metric, label in (("acc", "pairwise acc"), ("ndcg", f"NDCG@{K_NDCG}")):
            m = res[table][metric]
            extra = f", {m['pairs']} pairs" if m.get("pairs") is not None else ""
            lines.append(f"{table} {label} ({m['queries']} briefs{extra}):")
            for name, v in m.items():
                if not isinstance(v, dict) or v.get("mean") is None:
                    continue
                d = next((x for k, x in v.items() if k.startswith("diff_vs_")), None)
                lines.append(f"  {name:<28} {v['mean']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
                             + (f"   vs heuristic {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                                if d else ""))
    if res.get("queries_with_several_q"):
        lines.append(f"note: {res['queries_with_several_q']} queries have rows with different parsed briefs (q)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--model", action="append", default=[], help="model file (repeatable)")
    ap.add_argument("--split", default="heldout", help="train, heldout or all")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    ds = build(load_rows(a.rows), None if a.split == "all" else a.split,
               source={"rows": str(a.rows), "sha": file_sha(a.rows)})
    if not ds.labelled.any():
        raise SystemExit(f"no labelled rows in split {a.split!r} of {a.rows}")
    models = {Path(p).name: load_scorer(p) for p in a.model}
    res = {"rows": str(a.rows), "rows_sha": ds.source["sha"], "split": a.split, "n_rows": len(ds),
           "n_labelled": int(ds.labelled.sum()), **evaluate(ds, models, a.boot)}
    print(format_table(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
