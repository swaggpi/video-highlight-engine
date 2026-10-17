"""tools/train (not shipped in the package): dataset building, the linear pairwise ranker and ridge baseline,
offline metrics, and the model file the engine loads, on synthetic rows with a known ranking direction."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
import build_dataset as bd  # noqa: E402
import eval_rank as er  # noqa: E402
import train_linear as tl  # noqa: E402

from vhe.rank.features import FEATURES, INDEX, PERSONAS, query_names  # noqa: E402
from vhe.rank.scorer import HeuristicScorer, LinearRankScorer, load_scorer  # noqa: E402

F = len(FEATURES)
TRUE = {"rel_mean": 1.0, "quality_mean": 0.6, "unusable_frac": -0.8, "energy_mean": 0.4}


def true_w() -> np.ndarray:
    w = np.zeros(F)
    for k, v in TRUE.items():
        w[INDEX[k]] = v
    return w


def make_rows(n_queries=40, per_query=20, seed=0, split="train", set_name="syn", leak_src=True, n_pool=4):
    """Labelled rows whose y is a quantised function of TRUE; with leak_src the masked src_window column carries the
    label too (a model that used it would look better in training), and h_rel_mean is noise."""
    rng = np.random.default_rng(seed)
    qn = query_names(PERSONAS)
    rows = []
    for b in range(n_queries):
        q = {n: 0.0 for n in qn}
        q.update(w_relevance=1.0 + 0.2 * rng.standard_normal(), w_quality=0.5, w_energy=0.3, w_steady=0.2,
                 w_speech=0.0, target_len=5.0, log_target_len=float(np.log(5.0)))
        X = rng.standard_normal((per_query + n_pool, F))
        lat = X @ true_w()
        cuts = np.quantile(lat[:per_query], [0.25, 0.5, 0.75])
        y = np.searchsorted(cuts, lat) / 3
        if leak_src:
            X[:, INDEX["src_window"]] = 5 * y
        for i in range(len(X)):
            pool = i >= per_query
            rows.append({"row_id": f"{set_name}/r1/B{b}/app_dump/{i}", "set": set_name, "run_id": "r1",
                         "brief_id": f"B{b}", "q": q, "origin": "app_dump" if i % 2 else "opus",
                         "selected_by": "none" if pool else ("app" if i % 2 else "opus"),
                         "dump_score": float(lat[i]) if i % 2 else 0.0,
                         "label_kind": "pool_negative" if pool else "judged", "y_point": None if pool else float(y[i]),
                         "weight": 0.3 if pool else 1.0, "split": split, "feature_version": "v1",
                         **{f"f_{n}": float(X[i, j]) for j, n in enumerate(FEATURES)}})
    return rows


def effective_direction(model: LinearRankScorer, q: np.ndarray) -> np.ndarray:
    """d score / d z_j for each feature under query q."""
    base = model.score(model.mu[None, :], q)[0]
    return np.array([model.score((model.mu + model.sd * np.eye(F)[j])[None, :], q)[0] - base for j in range(F)])


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_mask_drops_src_and_h_rel():
    m = bd.learned_mask()
    for n in FEATURES:
        assert m[INDEX[n]] == (0.0 if n.startswith(("src_", "h_rel_")) else 1.0)


def test_pairs_within_query_with_label_and_pool_weights():
    rows = make_rows(n_queries=2, per_query=4, n_pool=3)
    rows[1]["label_kind"], rows[1]["weight"] = "label_transfer", 0.5
    ds = bd.build(rows)
    I, J, W = bd.pairs(ds, seed=1)
    assert len(I) and np.all(ds.query[I] == ds.query[J])
    lab = ds.labelled
    for i, j, w in zip(I, J, W):
        if ds.kind[j] == "pool_negative":
            assert ds.y[i] >= 2 / 3 - 1e-9 and w == pytest.approx(0.3 * ds.w[i])
        else:
            assert lab[i] and lab[j] and ds.y[i] - ds.y[j] >= 1 / 3 - 1e-9 and w == min(ds.w[i], ds.w[j])
    assert 0.5 in W                                              # the transferred row's pairs are half weight
    assert np.sum(ds.kind[J] == "pool_negative") <= 5 * np.sum(lab & (ds.y >= 2 / 3 - 1e-9))


def test_ranknet_recovers_the_true_direction_and_ignores_masked_columns():
    ds = bd.build(make_rows())
    model = tl.fit(ds, "ranknet", l2=1e-2)
    q = ds.Q[0]
    d = effective_direction(model, q)
    assert cosine(d, true_w()) > 0.95
    masked = bd.learned_mask() == 0
    assert np.all(d[masked] == 0)                                # src_window carried the label; the model cannot see it
    X = np.random.default_rng(5).standard_normal((20, F))
    X2 = X.copy()
    X2[:, masked] = 1e6 * np.random.default_rng(6).standard_normal((20, int(masked.sum())))
    assert np.allclose(model.score(X, q), model.score(X2, q))


def test_ridge_baseline_recovers_the_direction():
    ds = bd.build(make_rows())
    d = effective_direction(tl.fit(ds, "ridge", l2=1e-2), ds.Q[0])
    assert cosine(d, true_w()) > 0.9


def test_without_interactions_the_model_has_one_weight_per_feature():
    ds = bd.build(make_rows(n_queries=10))
    model = tl.fit(ds, "ranknet", l2=1e-2, interactions=False)
    assert model.w.shape == (F,) and cosine(effective_direction(model, ds.Q[0]), true_w()) > 0.9


def test_train_cli_writes_a_model_the_engine_loads_and_ranks_with(tmp_path, monkeypatch, capsys):
    rows_path = tmp_path / "rows.jsonl"
    rows = make_rows(n_queries=8) + make_rows(n_queries=3, seed=9, split="heldout", set_name="festival-commons")
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["train_linear.py", str(rows_path), "--name", "t1", "--out-dir", str(tmp_path),
                                      "--l2", "0.01,1", "--boot", "200", "--report", str(tmp_path / "cv.json")])
    tl.main()
    out = capsys.readouterr().out
    assert "leave-one-brief-out CV over 8 briefs" in out and "heuristic" in out
    m = json.loads((tmp_path / "t1.json").read_text())
    assert m["format"] == "vhe-linear-rank/1" and m["mask"] == bd.learned_mask().astype(int).tolist()
    assert m["learner"] == "ranknet" and m["l2"] in (0.01, 1.0) and m["interactions"] is True
    assert all(q.startswith("syn/") for q in m["trained_on"]["queries"])   # the held-out set is never trained on
    cv = json.loads((tmp_path / "cv.json").read_text())["cv"]
    assert cv["folds"] == 8 and set(cv["chosen_l2"]) == {"ranknet", "ridge"}
    best = cv["all_rows"]["acc"][f"ranknet@{m['l2']:g}"]["mean"]
    assert best > 0.9
    sc = load_scorer(tmp_path / "t1.json")
    assert isinstance(sc, LinearRankScorer) and sc.mask is not None
    # ranks unseen queries by the true direction
    held = bd.build(rows, split="heldout")
    res = er.evaluate(held, {"t1": sc}, n_boot=200)
    assert res["all_rows"]["acc"]["t1"]["mean"] > 0.9
    assert res["app_rows"]["acc"]["heuristic_dump"]["mean"] == pytest.approx(1.0)  # dump_score = latent on app rows
    # eval CLI on the held-out split
    monkeypatch.setattr(sys, "argv", ["eval_rank.py", str(rows_path), "--model", str(tmp_path / "t1.json"),
                                      "--boot", "100", "--json", str(tmp_path / "ev.json")])
    er.main()
    assert "t1.json" in capsys.readouterr().out
    assert json.loads((tmp_path / "ev.json").read_text())["split"] == "heldout"


def test_train_refuses_the_heldout_split(tmp_path, monkeypatch):
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in make_rows(n_queries=2)))
    monkeypatch.setattr(sys, "argv", ["train_linear.py", str(p), "--split", "heldout", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        tl.main()


def test_metrics_and_heuristic_bar():
    y = np.array([1.0, 2 / 3, 0.0, 0.0])
    assert er.pairwise(y, np.array([3.0, 2.0, 1.0, 0.0])) == (5.0, 5)
    assert er.pairwise(y, np.zeros(4)) == (2.5, 5)                # ties count one half
    assert er.ndcg(y, np.array([3.0, 2.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert er.ndcg(y, np.array([0.0, 1.0, 2.0, 3.0])) < 0.6
    assert er.ndcg(np.zeros(3), np.arange(3.0)) is None
    ds = bd.build(make_rows(n_queries=2))
    h = er.heuristic_scores(ds)
    idx = np.flatnonzero(ds.query == ds.query[0])
    X, q = ds.X[idx], ds.Q[idx[0]]
    s = HeuristicScorer().score(X, q)
    assert np.allclose(h[idx], HeuristicScorer().rescore(s, X, X, q))


def test_rows_of_another_feature_version_are_refused():
    rows = make_rows(n_queries=1)
    rows[0]["feature_version"] = "v0"
    ds = bd.build(rows)
    assert len(ds) == len(rows) - 1 and ds.source["skipped"] == {"feature_version v0": 1}


def test_linear_scorer_mask_validation(tmp_path):
    with pytest.raises(ValueError, match="mask"):
        LinearRankScorer(np.zeros(F), np.ones(F), np.zeros(F), mask=[1, 0])
    sc = LinearRankScorer(np.zeros(F), np.ones(F), np.ones(F), mask=bd.learned_mask())
    p = tmp_path / "m.json"
    p.write_text(json.dumps(sc.to_json()))
    assert np.array_equal(load_scorer(p).mask, bd.learned_mask())
    no_mask = LinearRankScorer(np.zeros(F), np.ones(F), np.ones(F))
    assert "mask" not in no_mask.to_json()
