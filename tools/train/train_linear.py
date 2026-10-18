"""Train the first learned scorer: a linear pairwise ranker (RankNet with a linear score) on
phi(x, q) = [z, outer(z, q_small)], z the standardised features (learned mask applied), q_small the brief's signal
weights; plus a pointwise ridge baseline on the same phi. numpy only.

  python tools/train/train_linear.py ROWS.jsonl [--name linear_v3] [--learner ranknet|ridge] [--split train]
                                     [--l2 1e-3,1e-2,0.1,1,10] [--no-cv] [--keep-all-features]
                                     [--no-interactions] [--seed 0] [--report OUT.json]

Pairs (build_dataset.pairs), within one query (set, run, brief): labelled rows whose y differ by >= 1/3 (weight 1,
0.5 with a transferred label), and each positive (y >= 2/3) against up to 5 random unlabelled pool rows (weight 0.3).
Loss: weighted mean of log(1 + exp(-(s_i - s_j))) + l2 * |w|^2, minimised by damped Newton steps (convex).
Ridge: weighted least squares of y on phi (labelled rows), same l2 scale.
The L2 strength is picked by leave-one-brief-out cross-validation inside the training split only (a brief seen in
several runs is one fold): for every l2, each brief is scored by a model trained on the others; the l2 with the best
mean pairwise accuracy over briefs wins (ties: the stronger l2). The held-out split is never read here.
Writes models/<name>.json (format vhe-linear-rank/1, with the feature mask, the CV table and what it was trained on),
loadable by vhe.rank.scorer.load_scorer / `vhe clips --scorer`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DROP_PREFIXES, Dataset, build, file_sha, load_rows, pairs, summary  # noqa: E402
from eval_rank import aggregate, format_table, heuristic_scores, per_query, score_rows  # noqa: E402

from vhe.rank.features import FEATURE_VERSION, FEATURES  # noqa: E402
from vhe.rank.scorer import Q_SMALL, LinearRankScorer  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
LEARNERS = ("ranknet", "ridge")


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0) if len(X) else np.zeros(X.shape[1])
    sd = X.std(0) if len(X) else np.ones(X.shape[1])
    return mu, np.where(sd > 1e-12, sd, 1.0)


def design(ds: Dataset, mu: np.ndarray, sd: np.ndarray, interactions: bool = True) -> np.ndarray:
    """(n, F * (1 + len(Q_SMALL))) phi of every row (n, F without the query interactions), computed by
    LinearRankScorer.phi itself (per distinct q), so training and inference use one feature map."""
    D = len(FEATURES) * (1 + len(Q_SMALL)) if interactions else len(FEATURES)
    sc = LinearRankScorer(mu, sd, np.zeros(D), mask=ds.mask)
    out = np.zeros((len(ds), D))
    groups: dict = {}
    for i, row in enumerate(ds.Q):
        groups.setdefault(row.tobytes(), []).append(i)
    for idx in groups.values():
        out[idx] = sc.phi(ds.X[idx], ds.Q[idx[0]])
    return out


def fit_ranknet(Phi: np.ndarray, win: np.ndarray, lose: np.ndarray, W: np.ndarray, l2: float, iters: int = 100,
                tol: float = 1e-10) -> np.ndarray:
    """argmin_w  sum_k Wn_k log(1 + exp(-(phi_win - phi_lose) w)) + l2 |w|^2,  Wn = W / sum W  (damped Newton)."""
    d = Phi.shape[1]
    w = np.zeros(d)
    if not len(win):
        return w
    Dm = Phi[win] - Phi[lose]
    Wn = W / W.sum()

    def loss(v):
        m = Dm @ v
        return float(Wn @ np.logaddexp(0.0, -m) + l2 * v @ v)

    f = loss(w)
    for _ in range(iters):
        m = Dm @ w
        p = 1.0 / (1.0 + np.exp(m))                     # sigmoid(-m)
        g = -Dm.T @ (Wn * p) + 2 * l2 * w
        H = (Dm * (Wn * p * (1 - p))[:, None]).T @ Dm + 2 * l2 * np.eye(d)
        step = np.linalg.solve(H + 1e-12 * np.eye(d), g)
        t = 1.0
        while t > 1e-8:
            f_new = loss(w - t * step)
            if f_new <= f - 1e-4 * t * (g @ step):
                break
            t /= 2
        w, f_old, f = w - t * step, f, f_new
        if abs(f_old - f) < tol:
            break
    return w


def fit_ridge(Phi: np.ndarray, y: np.ndarray, wt: np.ndarray, l2: float) -> np.ndarray:
    """Weighted ridge of y on phi, intercept unpenalised (and dropped: it does not change a ranking)."""
    if not len(y):
        return np.zeros(Phi.shape[1])
    sw = wt / wt.sum()
    m_phi, m_y = sw @ Phi, sw @ y
    C = Phi - m_phi
    A = (C * sw[:, None]).T @ C + l2 * np.eye(Phi.shape[1])
    return np.linalg.solve(A, C.T @ (sw * (y - m_y)))


def fit(ds: Dataset, learner: str, l2: float, seed: int = 0, meta: dict | None = None,
        interactions: bool = True) -> LinearRankScorer:
    mu, sd = standardise(ds.X)
    Phi = design(ds, mu, sd, interactions)
    if learner == "ranknet":
        win, lose, W = pairs(ds, seed=seed)
        w = fit_ranknet(Phi, win, lose, W, l2)
    elif learner == "ridge":
        lab = ds.labelled
        w = fit_ridge(Phi[lab], ds.y[lab], ds.w[lab], l2)
    else:
        raise ValueError(f"unknown learner {learner!r}")
    return LinearRankScorer(mu, sd, w, mask=ds.mask, name=learner, meta=meta)


def cross_validate(ds: Dataset, lambdas=LAMBDAS, learners=LEARNERS, seed: int = 0, n_boot: int = 2000,
                   interactions: bool = True) -> dict:
    """Leave-one-brief-out predictions for every (learner, l2), metrics against the heuristic on the same rows, and
    the chosen l2 per learner."""
    groups = np.unique(ds.group)
    if len(groups) < 2:
        return {"folds": int(len(groups)), "skipped": "fewer than 2 briefs: no cross-validation"}
    scores = {}
    for learner in learners:
        for l2 in lambdas:
            pred = np.full(len(ds), np.nan)
            for g in groups:
                te = ds.group == g
                model = fit(ds.subset(~te), learner, l2, seed, interactions=interactions)
                pred[te] = score_rows(model, ds.subset(te))
            scores[f"{learner}@{l2:g}"] = pred
    scores["heuristic"] = heuristic_scores(ds)
    names = list(scores)
    recs = per_query(ds, scores)
    table = {"all_rows": aggregate(recs, names, n_boot=n_boot, seed=seed)}
    app = np.isin(ds.origin, ("app_dump", "app_picks"))
    if app.any():
        table["app_rows"] = aggregate(per_query(ds, {**scores, "heuristic_dump": ds.dump_score}, rows=app),
                                      [*names, "heuristic_dump"], n_boot=n_boot, seed=seed)
    chosen = {}
    acc = table["all_rows"]["acc"]
    for learner in learners:
        cands = [(acc[f"{learner}@{l2:g}"]["mean"], l2) for l2 in lambdas
                 if acc[f"{learner}@{l2:g}"]["mean"] is not None]
        chosen[learner] = max(cands)[1] if cands else max(lambdas)
    return {"folds": int(len(groups)), "lambdas": list(lambdas), "chosen_l2": chosen, **table}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--name", default=f"linear_{FEATURE_VERSION}")
    ap.add_argument("--learner", default="ranknet", choices=LEARNERS, help="which learner's model is written")
    ap.add_argument("--split", default="train")
    ap.add_argument("--l2", default=",".join(f"{x:g}" for x in LAMBDAS), help="comma-separated L2 grid")
    ap.add_argument("--no-cv", action="store_true", help="fit with the largest --l2 value, no cross-validation")
    ap.add_argument("--keep-all-features", action="store_true", help="do not mask src_* and h_rel_*")
    ap.add_argument("--no-interactions", action="store_true",
                    help="phi = z only (F weights instead of F * (1 + len(q_small))): for small datasets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=REPO / "models")
    ap.add_argument("--report", type=Path, default=None, help="write the CV table as JSON")
    a = ap.parse_args()
    if a.split == "heldout":
        raise SystemExit("refusing to train on the held-out split")
    lambdas = tuple(float(x) for x in a.l2.split(","))
    sha = file_sha(a.rows)
    ds = build(load_rows(a.rows), a.split, () if a.keep_all_features else DROP_PREFIXES,
               source={"rows": str(a.rows), "sha": sha})
    info = summary(ds)
    print("dataset: " + json.dumps(info))
    if not ds.labelled.any():
        raise SystemExit(f"no labelled rows in split {a.split!r}")
    cv = {"skipped": "--no-cv"} if a.no_cv else cross_validate(ds, lambdas, seed=a.seed, n_boot=a.boot,
                                                                     interactions=not a.no_interactions)
    if "all_rows" in cv:
        print(f"leave-one-brief-out CV over {cv['folds']} briefs (chosen l2 {cv['chosen_l2']}):")
        print(format_table(cv))
    else:
        print(f"CV: {cv['skipped']}")
    l2 = cv.get("chosen_l2", {}).get(a.learner, max(lambdas))
    meta = {"trained_on": {"rows": str(a.rows), "rows_sha": sha, "split": a.split, "n_rows": info["rows"],
                           "n_pairs": info["pairs"], "label_kind": info["label_kind"],
                           "queries": sorted(set(ds.query.tolist()))},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "learner": a.learner, "l2": l2,
            "interactions": not a.no_interactions,
            "cv": {k: v for k, v in cv.items() if k in ("folds", "lambdas", "chosen_l2", "skipped", "all_rows")}}
    model = fit(ds, a.learner, l2, a.seed, meta=meta, interactions=not a.no_interactions)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.name}.json"
    out.write_text(json.dumps(model.to_json(), indent=1))
    print(f"{a.learner} l2={l2:g} on {info['rows']} rows / {info['pairs']} pairs -> {out}")
    if a.report:
        a.report.write_text(json.dumps({"dataset": info, "cv": cv}, indent=1))


if __name__ == "__main__":
    main()
