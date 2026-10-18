"""The ranking seam (vhe/rank): the hand-weighted scorer's output is pinned, features are deterministic, raw and
versioned, selection is the same on any score scale, a learned linear model plugs in, and runs dump candidates (and
given spans) in one schema, every score a function of its row."""
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
    FEATURES_V3,
    FEATURES_V4,
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


# ---- regression: the hand-weighted scorer's pinned output (provenance: heuristic_cases) ---------------------------
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
    assert FEATURE_VERSION == "v4" and len(set(FEATURES)) == len(FEATURES)  # v2 block: test_cues
    assert FEATURES[:3] == ("rel_mean", "rel_top60", "rel_max")
    assert FEATURES[95:] == FEATURES_V4 == (  # v4, appended after v3's 95
        "reaction_mean", "reaction_max", "excitement_mean", "quote_match", "bucket_reaction", "bucket_excitement")
    assert FEATURES[82:95] == FEATURES_V3 == (  # v3, appended after v2's 82
        "edges_clean", "legibility_mean", "highlight_prior_mean", "detail_z_mean", "framing_frac", "bucket_size",
        "bucket_h_rel", "bucket_quality", "bucket_energy", "bucket_steady", "bucket_speech", "bucket_legibility",
        "bucket_highlight")
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


def _assert_scores_follow_from_features(d: dict) -> int:
    """Every row of a dump but topic rows: HeuristicScorer on the row's own features and the dump's q reproduces the
    dumped score (score() for stage-1 candidates, the final total for the rest). Returns the rows checked."""
    q = np.array(list(d["query"]["q"].values()))
    h = HeuristicScorer()
    rows = [c for c in d["candidates"] if c["stage"] != "topic"]
    for stage in {c["stage"] for c in rows}:
        mine = [c for c in rows if c["stage"] == stage]
        X = np.array([c["features"] for c in mine])
        want = h.score(X, q) if stage == "candidate" else dumps.final_scores(h, X, q)
        np.testing.assert_allclose(want, [c["score"] for c in mine], atol=1e-4, err_msg=stage)
    return len(rows)


def test_heuristic_scores_follow_from_dumped_features(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    out = tmp_path / "dump.json"
    eng = Engine.open(root, device="cpu", use_nc=False)
    res = eng.clips(BRIEF, llm=None, dump=out, dump_all=True)
    d = dumps.read(out)
    assert {c["stage"] for c in d["candidates"]} == {"candidate", "shortlist", "edge"}
    assert _assert_scores_follow_from_features(d) == len(d["candidates"])
    # a body clip's dumped score is its displayed score (the scorer's total)
    body = {(c["file"], c["start"], c["end"]): c["score"] for c in d["candidates"] if c["selected"]
            and c["stage"] == "shortlist"}
    for c in res["clips"]:
        if c["role"] is None:
            assert c["score"] == pytest.approx(body[(c["file"], c["file_start"], c["file_end"])], abs=6e-4)
    spans = [{"file": c["file"], "start": c["file_start"], "end": c["file_end"]} for c in res["clips"]]
    _assert_scores_follow_from_features(eng.features_for_spans(BRIEF, spans, llm=None))


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


# ---- v3: the seam is a drop-in for any scorer ---------------------------------------------------------------------
class _Affine:
    """A scorer whose every score is the heuristic's times a plus b: ranks exactly like it, on another scale."""

    def __init__(self, a: float, b: float) -> None:
        self.inner, self.a, self.b = HeuristicScorer(), a, b
        self.name, self.feature_version, self.personas = f"heuristic*{a}+{b}", FEATURE_VERSION, PERSONAS

    def score(self, X, q):
        return self.a * self.inner.score(X, q) + self.b

    def rescore(self, s, X_pre, X_post, q):
        return self.a * self.inner.rescore((s - self.b) / self.a, X_pre, X_post, q) + self.b

    def score_items(self, X_items, q, final):
        return self.a * self.inner.score_items(X_items, q, final) + self.b


def _span(c):
    return c["file"], c["file_start"], c["file_end"], c["role"], c["item"]


def test_selection_does_not_depend_on_the_score_scale(tmp_path):
    """Every pinned case picks the same clips with the heuristic on another affine scale (x7 + 100, and x0.2 - 5,
    which an absolute threshold would empty: dogs_every and speaks_every), and a linear model with random weights
    never returns nothing where the heuristic returns clips."""
    assert isinstance(_Affine(1.0, 0.0), Scorer)
    base = run_all(tmp_path / "h")
    for a, b in ((7.0, 100.0), (0.2, -5.0)):
        got = run_all(tmp_path / f"{a}", scorer=_Affine(a, b))
        for name in base:
            assert [_span(c) for c in got[name]["clips"]] == [_span(c) for c in base[name]["clips"]], (name, a, b)
            for c, h in zip(got[name]["clips"], base[name]["clips"], strict=True):
                if not h["reason"].startswith("talks about"):  # spoken-topic clips: transcript match, no scorer
                    assert c["score"] == pytest.approx(a * h["score"] + b, abs=2e-3 * max(1.0, a))
    F = len(FEATURES)
    for seed in (0, 1):
        lin = LinearRankScorer(np.zeros(F), np.ones(F), np.random.default_rng(seed).normal(size=F * 6))
        got = run_all(tmp_path / f"r{seed}", scorer=lin)
        assert not [name for name in base if base[name]["clips"] and not got[name]["clips"]], seed


def test_extended_clips_are_dumped_as_returned(tmp_path, monkeypatch):
    """A clip the budget fill lengthens is dumped with its final edges, the features of the final span and the total
    the scorer gives that span (and the result shows the same total)."""
    import vhe.engine as engine

    grown = []

    def fill(body, *a, **kw):
        out = engine_fill(body, *a, **kw)
        grown.extend(o for o in out if o.get("extended"))
        return out

    engine_fill = engine.fill_budget
    monkeypatch.setattr(engine, "fill_budget", fill)
    root = make_project(tmp_path / "recap", [("a.mp4", 61.0, {}), ("b/c.mp4", 45.0, {"has_audio": False}),
                                             ("d.mp4", 50.0, {})])
    eng = _engine(root)
    brief = "5 calm B-roll shots 3-6 s each"
    res = eng.clips(brief, llm=None, budget_s=(20.0, 30.0), dump=tmp_path / "d.json")
    assert grown
    d = dumps.read(tmp_path / "d.json")
    picked = [c for c in d["candidates"] if c["selected"]]
    assert sorted((c["file"], c["start"], c["end"]) for c in picked) == \
        sorted((c["file"], c["file_start"], c["file_end"]) for c in res["clips"])
    for o in grown:  # the extended span is the dumped one, with features of that span
        f, s, e = dumps.span_label(eng.corpus, o["file"], o["start"], o["end"])
        row = next(c for c in picked if (c["file"], c["start"], c["end"]) == (f, s, e))
        assert row["features"][INDEX["length_s"]] == pytest.approx(e - s, abs=2e-3)
    _assert_scores_follow_from_features(d)
    ext = eng.features_for_spans(brief, [{"file": c["file"], "start": c["start"], "end": c["end"],
                                          "source": c["source"]} for c in picked], llm=None, budget_s=(20.0, 30.0))
    np.testing.assert_allclose([c["features"] for c in ext["candidates"]], [c["features"] for c in picked],
                               atol=2e-6)
    np.testing.assert_allclose([c["score"] for c in ext["candidates"]], [c["score"] for c in picked], atol=2e-5)


def test_edges_clean_feature_is_the_refine_rule(tmp_path):
    from vhe.query.refine import _edge_ok, edges_clean
    from vhe.rank.features import span_edges_clean

    def reference(s, e, ctx):  # the per-step rule, one clip at a time (refine's _edge_ok)
        n = len(ctx.clean)
        i0 = min(int(np.floor(s * ctx.fps + 1e-6)), n - 1)
        i1 = max(int(np.ceil(e * ctx.fps - 1e-6)), i0 + 1)
        return _edge_ok(ctx, i0) and _edge_ok(ctx, min(i1, n), end=True)


    n = 270
    usable = np.ones(n, bool)
    usable[90:150] = False
    usable[200:201] = False
    root = make_project(tmp_path / "e", [("a.mp4", 90.0, {"usable": usable}), ("b.mp4", 30.0, {})])
    eng = _engine(root)
    c = eng.corpus
    sig = eng._signals(eng.parse("5 clips", None))[0]
    rng = np.random.default_rng(0)
    K = rng.integers(0, 2, 400)
    S = c.offsets[K] + rng.uniform(0, 1, 400) * (c.durations[K] - 5.0)
    E = S + rng.uniform(1.0, 5.0, 400)
    got = span_edges_clean(c, sig["clean"], S, E, K)
    want = [reference(s - c.offsets[k], e - c.offsets[k], eng._context(int(k)))
            for s, e, k in zip(S, E, K, strict=True)]
    assert got.tolist() == want and 0 < got.sum() < len(got)
    assert [edges_clean(s - c.offsets[k], e - c.offsets[k], eng._context(int(k)))
            for s, e, k in zip(S, E, K, strict=True)] == want
    X = featurize(c, sig, S[:3], E[:3], K[:3], ["window", "speech", "shot"], eng.parse("5 clips", None))
    assert X[1, INDEX["edges_clean"]] == 1.0  # speech clips end in word gaps


def test_score_items_and_explain_follow_the_features(tmp_path, monkeypatch):
    from vhe.rank.features import item_features, ocr_matrix
    from vhe.rank.scorer import OCR_ITEM_BONUS

    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    eng = Engine.open(root, device="cpu", use_nc=False)
    c = eng.corpus
    p = eng.parse(BRIEF, None)
    sig = eng._signals(p)[0]
    S = np.array([5.0, 60.0 + 20.0, 120.0 + 30.0, 180.0, 185.0])
    E = S + np.array([10.0, 8.0, 6.0, 12.0, 4.0])
    K = np.array([0, 1, 2, 3, 3])
    X = featurize(c, sig, S, E, K, ["window"] * 5, p)
    q = query_vector(p)
    items = list(item_features(c, sig, S, E, K, X, ocr_matrix(c, p, S, E, K)))
    assert len(items) == len(p.shots)
    for Xi, row in zip(items, sig["items"], strict=True):  # only the relevance columns change, to the item's own
        changed = np.flatnonzero((Xi != X).any(0))
        assert {FEATURES[j] for j in changed} <= {"rel_mean", "rel_top60", "rel_max", "item_rel", "avoid_pen",
                                                  "h_rel_mean", "h_rel_top60", "ocr_item_match"}
        np.testing.assert_allclose(Xi[:, INDEX["rel_mean"]], span_mean(row, *c.steps(S, E, K)))
    h = HeuristicScorer()
    for final in (False, True):
        per = h.score_items(iter(items), q, final)
        total = h.rescore(h.score(X, q), X, X, q) if final else h.score(X, q)
        np.testing.assert_allclose(sum(h.explain(X, q, final).values()), total)
        want = np.stack([(h.rescore(None, Xi, Xi, q) if final else h.score(Xi, q))
                         + p.weights["relevance"] * OCR_ITEM_BONUS * Xi[:, INDEX["ocr_item_match"]]
                         for Xi in items], axis=1)
        np.testing.assert_allclose(per, want)
    dogs, cats = h.score_items(iter(items), q, True)[0, :2]
    assert dogs > cats  # a.mp4 5-15 s shows the dogs
    F = len(FEATURES)
    lin = LinearRankScorer(np.zeros(F), np.ones(F), np.random.default_rng(1).normal(size=F * 6))
    np.testing.assert_allclose(lin.score_items(iter(items), q, True), np.stack([lin.score(Xi, q) for Xi in items], 1))
    np.testing.assert_allclose(sum(lin.explain(X, q).values()), lin.score(X, q))
    assert h.score_items(iter([]), q, True).shape == (0, 0)


def test_ocr_one_rule_and_legibility_bonuses_live_in_the_heuristic(tmp_path):
    import dataclasses

    from heuristic_cases import CUE_BRIEFS, cue_engine
    from vhe.rank.features import ocr_matrix

    eng = cue_engine(tmp_path / "cues")
    c = eng.corpus
    legible = eng.parse(CUE_BRIEFS["cue_legible"], None)  # "the Pasargad float with its name readable"
    plain = dataclasses.replace(legible, legible_text=False)  # the same shots, no readable-text cue
    assert legible.legible_text and legible.shots[0].startswith("pasargad float")
    S, E, K = np.array([60.0 + 20.0, 60.0 + 0.0]), np.array([86.0, 4.0 + 60.0]), np.array([1, 1])
    # "PASARGAD" read: a full match ("float", "name", "readable" never count, whatever the brief's cues)
    np.testing.assert_array_equal(ocr_matrix(c, legible, S, E, K)[:, 0], [1.0, 0.0])
    sig_l, sig_p = eng._signals(legible)[0], eng._signals(plain)[0]
    Xl = featurize(c, sig_l, S, E, K, ["window"] * 2, legible)
    Xp = featurize(c, sig_p, S, E, K, ["window"] * 2, plain)
    for name in ("h_rel_mean", "h_rel_top60", "legibility_mean", "ocr_item_match", "text_conf_mean"):
        np.testing.assert_allclose(Xl[:, INDEX[name]], Xp[:, INDEX[name]], err_msg=name)  # raw: brief-independent
    assert Xl[0, INDEX["legibility_mean"]] == pytest.approx(0.95)
    h = HeuristicScorer()
    terms = h.explain(Xl, query_vector(legible))
    assert terms["ocr"][0] == pytest.approx(0.6) and "ocr" not in h.explain(Xl, query_vector(plain))
    gain = h.relevance(Xl, query_vector(legible), True)[0] - h.relevance(Xl, query_vector(plain), True)[0]
    np.testing.assert_allclose(gain, Xl[:, INDEX["legibility_mean"]])  # LEGIBLE_BONUS 1.0 per unit legibility


def test_search_uses_the_one_stopword_list(tmp_path):
    from heuristic_cases import _speech_index

    eng = _engine(_speech_index(tmp_path / "talk.mp4.vhe"))
    spoken = [r["text"] for r in eng.search("here we are with the weather", k=20) if r["source"] == "speech"]
    # "here" and "are" are stopwords now (search used to skip only articles and prepositions)
    assert spoken == ["The weather is nice today, really sunny."]


def test_reason_names_the_largest_contributions(tmp_path, monkeypatch):
    root, vec = _shots_project(tmp_path)
    _vec_patch(monkeypatch, vec)
    res = Engine.open(root, device="cpu", use_nc=False).clips("3 clips of dogs", llm=None)
    assert all(c["reason"].startswith("matches 'dogs' (+") for c in res["clips"])
    model = tmp_path / "linear.json"
    model.write_text(json.dumps(_linear({"rel_top60": 1.0, "steady_mean": 0.01}).to_json()))
    res = Engine.open(root, device="cpu", use_nc=False, scorer=model).clips("3 clips of dogs", llm=None)
    for c in res["clips"]:
        assert c["reason"].startswith("matches 'dogs' (+") and c["score"] == c["rank_score"]
