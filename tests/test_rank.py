"""The ranking seam (vhe/rank): the hand-weighted scorer reproduces the engine as it was, features are deterministic
and versioned, a learned linear model plugs in, and runs dump candidates (and given spans) in one schema."""
import json
from pathlib import Path

import numpy as np
import pytest

from heuristic_cases import _dirs, _emb, _engine, _nc_mock, run_all
from test_corpus import make_project
from vhe.cli import main
from vhe.engine import Engine
from vhe.query import topics
from vhe.query.model import ParsedQuery
from vhe.rank import dump as dumps
from vhe.rank.features import (
    FEATURE_VERSION,
    FEATURES,
    INDEX,
    PERSONAS,
    featurize,
    query_names,
    query_vector,
)
from vhe.rank.scorer import HeuristicScorer, LinearRankScorer, Scorer, load_scorer
from vhe.rank.signals import (
    blend_nc,
    nc_signal,
    span_max,
    span_mean,
    span_min,
    span_top_mean,
)

DATA = Path(__file__).parent / "data"
ENTRY_KEYS = {"file", "start", "end", "source", "stage", "score", "features", "selected", "rank", "role", "item"}
BRIEF = "4 clips. Open with a wide shot. Show the dogs, the cats and the horses. Leave out shots of the birds."


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _shots_project(tmp_path):
    dogs, cats, horses, birds, wide = _dirs(5)
    dirs = (dogs, cats, horses, birds, wide)
    rng = np.random.default_rng(0)
    specs = [("a.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(5, 50, dogs)])}),
             ("b.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(20, 30, cats), (40, 50, birds)])}),
             ("c.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(5, 25, dogs), (30, 40, birds), (45, 55, wide)])}),
             ("d.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(0, 12, horses), (30, 58, cats)])})]
    vec = {"dogs": dogs, "cats": cats, "horses": horses, "birds": birds, "wide shot": wide}
    return make_project(tmp_path / "shots", specs), vec


def _vec_patch(monkeypatch, vec):
    from heuristic_cases import _text_vec

    monkeypatch.setattr(Engine, "text_vectors", lambda self, texts: np.stack(
        [vec.get(t, _text_vec(t)) for t in texts]).astype(np.float32))


# ---- regression: the hand-weighted scorer is the engine as it was --------------------------------------------------
def test_heuristic_scorer_reproduces_pre_seam_clips(tmp_path):
    """tests/data/heuristic_clips.json: Engine.clips on the same cases (see heuristic_cases for its provenance)."""
    expected = json.loads((DATA / "heuristic_clips.json").read_text())
    got = json.loads(json.dumps(run_all(tmp_path), default=_jsonable, ensure_ascii=False))
    assert set(got) == set(expected)
    for name in expected:
        assert got[name] == expected[name], name


def test_engine_default_scorer_is_heuristic(tmp_path):
    root, _ = _shots_project(tmp_path)
    assert isinstance(Engine.open(root, device="cpu", use_nc=False).scorer, HeuristicScorer)
    assert isinstance(load_scorer("heuristic"), Scorer) and load_scorer(None).name == "heuristic"
    with pytest.raises(FileNotFoundError):
        load_scorer(tmp_path / "missing.json")


# ---- signals: SigLIP relevance and the highlight saliency are separate ------------------------------------------
def test_nc_saliency_is_its_own_signal_and_blend_matches_the_old_rule():
    rng = np.random.default_rng(0)
    rel = np.clip(rng.normal(size=(2, 60)), -3, 3)
    raw = rng.normal(size=60)
    raw[:20] = np.nan  # a file without the highlight model
    nc = nc_signal(raw)
    assert np.isnan(nc[:20]).all() and np.isfinite(nc[20:]).all()
    assert nc_signal(None) is None and nc_signal(np.full(5, np.nan)) is None
    b = blend_nc(rel, nc)
    np.testing.assert_array_equal(b[:, :20], rel[:, :20])
    np.testing.assert_allclose(b[:, 20:], 0.5 * rel[:, 20:] + 0.5 * nc[20:])
    assert blend_nc(rel, None) is rel


def test_span_statistics_match_the_per_span_definitions():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    i0 = rng.integers(0, 400, 200)
    i1 = i0 + rng.integers(1, 90, 200)
    top = span_top_mean(x, i0, i1)
    for j in range(len(i0)):
        seg = x[i0[j]:i1[j]]
        assert top[j] == float(np.sort(seg)[-max(1, int(np.ceil(0.6 * len(seg)))):].mean())  # bit-exact
        assert span_min(x, i0, i1)[j] == seg.min() and span_max(x, i0, i1)[j] == seg.max()
    np.testing.assert_allclose(span_mean(x, i0, i1), [x[a:b].mean() for a, b in zip(i0, i1, strict=True)])


# ---- features --------------------------------------------------------------------------------------------------
def test_feature_names_are_stable_and_unique():
    assert FEATURE_VERSION == "v2" and len(set(FEATURES)) == len(FEATURES)  # v2: test_cues checks the appended columns
    assert FEATURES[:3] == ("rel_mean", "rel_top60", "rel_max")
    for name in ("item_rel", "avoid_pen", "nc_sal_mean", "nc_present", "h_rel_mean", "h_rel_top60", "quality_min",
                 "blown_frac", "unusable_frac", "motion_z", "cam_speed_z", "loud_z", "onset_z", "speech_frac",
                 "has_audio", "muted_frac", "length_s", "length_rel", "n_cuts_inside", "pos_in_file",
                 "src_speech", "tag_city_street", "tag_lake_or_river"):
        assert name in INDEX
    names = query_names()
    assert names[:5] == ("w_relevance", "w_quality", "w_energy", "w_steady", "w_speech")
    assert {"has_shots", "n_shots", "has_topics", "need_live_audio", "target_len"} <= set(names)
    assert names[-len(PERSONAS):] == tuple(f"persona_{x}" for x in PERSONAS)


def test_featurize_is_deterministic_and_describes_the_spans(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    eng = Engine.open(root, device="cpu", use_nc=False)
    c = eng.corpus
    p = eng.parse(BRIEF, None)
    S = np.array([5.0, 60.0 + 20.0, 120.0 + 30.0, 180.0])
    E = S + np.array([10.0, 8.0, 6.0, 12.0])
    K, src = np.array([0, 1, 2, 3]), ["window", "shot", "peak", "speech"]
    X = featurize(c, eng._signals(p)[0], S, E, K, src, p)
    again = Engine.open(root, device="cpu", use_nc=False)
    X2 = featurize(again.corpus, again._signals(again.parse(BRIEF, None))[0], S, E, K, src, p)
    assert X.shape == (4, len(FEATURES)) and np.isfinite(X).all()
    np.testing.assert_array_equal(X, X2)
    col = lambda n: X[:, INDEX[n]]  # noqa: E731
    np.testing.assert_array_equal(col("length_s"), E - S)
    np.testing.assert_array_equal(X[:, [INDEX[f"src_{s}"] for s in src]], np.eye(4))
    assert col("rel_mean")[0] > 1.0 and col("rel_mean")[2] < col("rel_mean")[0]  # a: dogs; c 30-36 s: birds
    assert col("avoid_pen")[2] > 0.0 and col("avoid_pen")[0] == 0.0  # birds are avoided
    assert col("item_rel")[0] > 0.5  # serves the dogs, well above the dogs' median
    assert (col("nc_present") == 0).all() and (col("nc_sal_mean") == 0).all()  # no highlight model
    np.testing.assert_allclose(col("h_rel_mean"), col("rel_mean") - col("avoid_pen"), atol=1e-12)
    q = query_vector(p, "p2-tiktok")
    names = query_names()
    assert q[names.index("persona_p2-tiktok")] == 1.0 and q[names.index("n_shots")] == 3.0
    assert q[names.index("has_avoid")] == 1.0 and q[names.index("w_relevance")] == p.weights["relevance"]


def test_nc_saliency_feeds_its_own_features_not_relevance(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    eng = _engine(root, vec, _nc_mock(3, absent=2))
    p = eng.parse(BRIEF, None)
    sig = eng._signals(p)[0]
    S = np.array([5.0, 125.0])  # file a (saliency), file c (none)
    K = np.array([0, 2])
    X = featurize(eng.corpus, sig, S, S + 8.0, K, ["window", "window"], p)
    col = lambda n: X[:, INDEX[n]]  # noqa: E731
    assert list(col("nc_present")) == [1.0, 0.0] and col("nc_sal_mean")[1] == 0.0
    plain = featurize(eng.corpus, dict(sig, nc_sal=None), S, S + 8.0, K, ["window", "window"], p)
    np.testing.assert_array_equal(col("rel_mean"), plain[:, INDEX["rel_mean"]])  # SigLIP only either way
    assert col("h_rel_mean")[0] != col("rel_mean")[0] - col("avoid_pen")[0]  # the heuristic blend uses it


# ---- scorers ---------------------------------------------------------------------------------------------------
def _linear(w_index: dict, **kw) -> LinearRankScorer:
    F = len(FEATURES)
    w = np.zeros(F)
    for name, v in w_index.items():
        w[INDEX[name]] = v
    return LinearRankScorer(np.zeros(F), np.ones(F), w, **kw)


def test_linear_one_hot_on_rel_mean_ranks_by_rel_mean(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, len(FEATURES)))
    q = query_vector(ParsedQuery(raw="", text="dogs", weights={"relevance": 1.0}, min_len=3.0, max_len=10.0))
    m = _linear({"rel_mean": 1.0})
    s = m.score(X, q)
    assert list(np.argsort(-s)) == list(np.argsort(-X[:, INDEX["rel_mean"]]))
    np.testing.assert_array_equal(m.rescore(s, X[::-1], X, q), s)  # rescore: the refined spans' own features
    # Interactions: w over [z, outer(z, q_small)]; a weight on rel_mean x w_relevance scales with the query weight.
    F, Qs = len(FEATURES), len(m.q_small)
    w = np.zeros(F * (1 + Qs))
    w[F + INDEX["rel_mean"] * Qs + 0] = 2.0
    inter = LinearRankScorer(np.zeros(F), np.ones(F), w)
    np.testing.assert_allclose(inter.score(X, q), 2.0 * X[:, INDEX["rel_mean"]])
    # File round trip and version checks.
    path = tmp_path / "m.json"
    path.write_text(json.dumps(dict(inter.to_json(), trained_on="unit", created="2026-10-18")))
    loaded = load_scorer(path)
    assert isinstance(loaded, LinearRankScorer) and loaded.name == "linear:m.json"
    np.testing.assert_array_equal(loaded.score(X, q), inter.score(X, q))
    path.write_text(json.dumps(dict(inter.to_json(), feature_version="v0")))
    with pytest.raises(ValueError, match="features"):
        load_scorer(path)
    path.write_text(json.dumps(dict(inter.to_json(), w=[1.0, 2.0])))
    with pytest.raises(ValueError, match="w needs"):
        load_scorer(path)


def test_heuristic_scores_follow_from_dumped_features(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    out = tmp_path / "dump.json"
    Engine.open(root, device="cpu", use_nc=False).clips(BRIEF, llm=None, dump=out, dump_all=True)
    d = dumps.read(out)
    rows = [c for c in d["candidates"] if c["stage"] == "candidate"]
    X = np.array([c["features"] for c in rows])
    q = np.array(list(d["query"]["q"].values()))
    np.testing.assert_allclose(HeuristicScorer().score(X, q), [c["score"] for c in rows], atol=1e-4)


def test_engine_runs_with_a_learned_model_file(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    model = tmp_path / "linear.json"
    model.write_text(json.dumps(_linear({"rel_top60": 1.0, "quality_mean": 0.3}).to_json()))
    res = Engine.open(root, device="cpu", use_nc=False, scorer=model).clips("3 clips of dogs", llm=None,
                                                                          dump=tmp_path / "d.json")
    assert len(res["clips"]) == 3 and all(c["relevance"] > 1.0 for c in res["clips"])
    assert dumps.read(tmp_path / "d.json")["run"]["scorer"] == "linear:linear.json"


# ---- dumps -----------------------------------------------------------------------------------------------------
def test_dump_schema_and_selected_spans_match_the_clips(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    eng = Engine.open(root, device="cpu", use_nc=False)
    out = tmp_path / "dump.json"
    res = eng.clips(BRIEF, llm=None, persona="p3-newsedit", dump=out)
    d = json.loads(out.read_text())
    assert d["format"] == "vhe-candidates/1" and d["feature_version"] == FEATURE_VERSION
    assert set(d["run"]) == {"app_sha", "scorer", "created", "corpus_root", "n_files"}
    assert d["run"]["scorer"] == "heuristic" and d["run"]["n_files"] == 4 and d["run"]["corpus_root"] == str(root)
    assert set(d["query"]) == {"brief", "parsed", "persona", "q"}
    assert d["query"]["brief"] == BRIEF and d["query"]["persona"] == "p3-newsedit"
    assert d["query"]["parsed"]["shots"] == ["dogs", "cats", "horses"]
    assert list(d["query"]["q"]) == list(query_names()) and d["query"]["q"]["persona_p3-newsedit"] == 1.0
    assert d["feature_names"] == list(FEATURES)
    cands = d["candidates"]
    assert cands and all(set(c) == ENTRY_KEYS and len(c["features"]) == len(FEATURES) for c in cands)
    assert {c["stage"] for c in cands} == {"shortlist", "edge"}  # no dump_all: no stage-1 rows
    picked = [c for c in cands if c["selected"]]
    clips = [(c["file"], c["file_start"], c["file_end"], c["role"], c["item"]) for c in res["clips"]]
    assert sorted(clips, key=str) == sorted(((c["file"], c["start"], c["end"], c["role"], c["item"]) for c in picked),
                                            key=str)
    body = [c for c in picked if c["stage"] == "shortlist"]
    assert sorted(c["rank"] for c in body) == list(range(len(body)))
    assert [c["role"] for c in picked if c["stage"] == "edge"] == ["opener"]
    assert all(c["rank"] is None for c in cands if not c["selected"])
    # dump_all adds every gated candidate before refinement, scored by score()
    eng.clips(BRIEF, llm=None, dump=tmp_path / "all.json", dump_all=True)
    full = dumps.read(tmp_path / "all.json")["candidates"]
    n = sum(c["stage"] == "candidate" for c in full)
    assert res["excluded_dropped"] > 0  # "Leave out shots of the birds": scored, but never shortlisted
    assert n == res["candidates"] - res["unusable_dropped"] and len(full) == n + len(cands)


def test_dump_for_a_spoken_topic_brief(tmp_path):
    from heuristic_cases import _speech_index

    root = _speech_index(tmp_path / "talk.mp4.vhe")
    plan_embed = topics.embed
    topics.embed = lambda texts, **kw: None
    try:
        res = _engine(root).clips("Find every moment where the host talks about where he is (Madison Avenue).",
                                  llm=None, dump=tmp_path / "t.json")
    finally:
        topics.embed = plan_embed
    cands = dumps.read(tmp_path / "t.json")["candidates"]
    assert [c["stage"] for c in cands] == ["topic"] * len(res["clips"]) and res["clips"]
    assert [(c["start"], c["item"]) for c in cands] == [(c["file_start"], c["item"]) for c in res["clips"]]


def test_features_for_external_spans_match_the_run_dump(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    eng = Engine.open(root, device="cpu", use_nc=False)
    eng.clips(BRIEF, llm=None, dump=tmp_path / "run.json")
    shortlist = [c for c in dumps.read(tmp_path / "run.json")["candidates"] if c["stage"] == "shortlist"][:40]
    spans = [{"file": c["file"], "start": c["start"], "end": c["end"], "rank": k} for k, c in enumerate(shortlist)]
    doc = eng.features_for_spans(BRIEF, spans, persona="p1-videographer", llm=None)
    assert doc["format"] == "vhe-candidates/1" and doc["query"]["persona"] == "p1-videographer"
    ext = doc["candidates"]
    assert [c["stage"] for c in ext] == ["external"] * len(spans) and [c["rank"] for c in ext] == list(range(40))
    assert [(c["file"], c["start"], c["end"]) for c in ext] == [(s["file"], s["start"], s["end"]) for s in spans]
    src = [INDEX[f"src_{s}"] for s in ("shot", "window", "peak", "model", "speech")]
    a = np.delete(np.array([c["features"] for c in ext]), src, axis=1)  # the source is unknown for external spans
    b = np.delete(np.array([c["features"] for c in shortlist]), src, axis=1)
    np.testing.assert_allclose(a, b, atol=2e-6)  # same spans, same brief: same features (3 dp times)


def test_cli_features_on_synthetic_spans(tmp_path, monkeypatch, capsys):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    spans = tmp_path / "opus.json"
    spans.write_text(json.dumps({"clips": [{"file": "a.mp4", "start": 10.0, "end": 16.0},
                                           {"file": "d.mp4", "start": 31.0, "end": 36.5, "item": "cats"}]}))
    out = tmp_path / "feat.json"
    main(["features", str(root), "--brief", BRIEF, "--spans", str(spans), "--out", str(out), "--no-llm",
          "--persona", "p4-vlogger", "--device", "cpu"])
    d = dumps.read(out)
    assert [(c["file"], c["start"], c["end"], c["item"]) for c in d["candidates"]] == \
        [("a.mp4", 10.0, 16.0, None), ("d.mp4", 31.0, 36.5, "cats")]
    assert d["candidates"][0]["features"][INDEX["rel_mean"]] > 1.0  # dogs at 5-50 s in a.mp4
    assert "2 spans" in capsys.readouterr().out
    main(["clips", str(root), BRIEF, "--no-llm", "--device", "cpu", "--persona", "p4-vlogger",
          "--dump-candidates", str(tmp_path / "c.json")])
    assert dumps.read(tmp_path / "c.json")["query"]["persona"] == "p4-vlogger"
    spans.write_text(json.dumps([{"file": "zzz.mp4", "start": 0, "end": 3}]))
    with pytest.raises(KeyError):
        main(["features", str(root), "--brief", BRIEF, "--spans", str(spans), "--out", str(out), "--no-llm"])
