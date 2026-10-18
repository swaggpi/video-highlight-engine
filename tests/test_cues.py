"""Visual cues of a brief (framing close/wide, faces, hook, readable text, centring): parsed by the rules and the
LLM, scored through the v2 features by the hand-weighted scorer, the hook slot, the v2 feature/query schema and the
refusal of v1 model files."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from heuristic_cases import CUE_BRIEFS, CUE_SPANS, cue_engine
from test_plan import _answer, _mock_ollama
from vhe.query.parser import LLM_SCHEMA, PROMPT_VERSION, framing_of, parse, parse_with_llm
from vhe.rank import dump as dumps
from vhe.rank.features import (
    FEATURE_VERSION,
    FEATURES,
    FEATURES_V2,
    INDEX,
    PERSONAS,
    QUERY_BASE,
    QUERY_V2,
    QUERY_VERSION,
    featurize,
    query_names,
    query_vector,
)
from vhe.rank.scorer import HeuristicScorer, LinearRankScorer, cue_bonus, load_scorer

TIKTOK = ("need a 20 sec vertical edit from the street fair. first clip has to be the hook, something huge coming "
          "right at the camera in the first second. then 1-2 sec cuts, faces, jugglers, keep everything centered cause "
          "im cropping to 9:16. no empty street. 15-25 sec total")
SLOW = ("The slow version, about two minutes, where the street itself is the main character. Long holds, mostly the "
        "wide view up the avenue, the market stalls rolling slowly through the wide frame rather than close-ups.")
NAMES = ("Six hero moments, 2-3 s each: the stall with its name legible, a juggler, the brass band with the "
         "banner readable.")
V1_LEN = 71  # features v1 (the prefix of v2)
V1_TAIL = ("tag_party", "tag_screen_or_computer", "tag_text_or_sign", "tag_cartoon_or_animation")


def _flags(p):
    return p.framing, p.want_faces, p.hook, p.legible_text, p.centered


# ---- parser: rules -------------------------------------------------------------------------------------------
def test_rules_read_close_hook_faces_and_centring():
    p = parse(TIKTOK)
    assert _flags(p) == ("close", True, True, False, True)
    assert p.budget == (15.0, 25.0)  # "9:16" is an aspect ratio, not 9 min 16 s
    assert not any("centered" in s or "hook" in s for s in p.shots)  # cue-only clauses give no shot items
    assert "jugglers" in p.shots and "faces" in p.shots


def test_rules_read_wide_and_negated_close_ups():
    p = parse(SLOW)
    assert p.framing == "wide" and not p.want_faces and not p.hook and not p.centered


def test_rules_read_legible_text_and_square_crop():
    p = parse(NAMES + " Croppable to square with the subject in the middle.")
    assert p.legible_text and p.centered
    assert parse("I need every shot where on-screen text is legible: stall names, banners.").legible_text


def test_cues_in_the_opener_or_an_avoid_clause_are_not_the_briefs():
    p = parse("60-90 s recap of the fair. Open on an establishing wide of the square. Show the band, the "
              "stalls and the jugglers. No bored faces.")
    assert _flags(p) == ("none", False, False, False, False)
    assert framing_of(p.opener) == "wide"  # the phrase reads wide; it only describes the opener
    assert parse("5 clips: one establishing wide of the square, the band, the stalls.").framing == "none"


def test_a_list_of_hooks_ranks_every_clip_like_the_hook():
    p = parse("dont cut anything. just give me the 3 best possible first-shots for the fair, ranked. 2-3 sec each max")
    assert p.count == 3 and p.hook and p.framing == "close" and (p.min_len, p.max_len) == (2.0, 3.0)
    assert p.weights["energy"] == 0.8 and p.weights["relevance"] <= 0.5
    one = parse("a 20 s reel of the fair, 3 clips. the first clip has to be the hook")
    assert one.hook and one.weights["energy"] == 0.0  # one hook slot; the rest is ranked as the brief says


def test_open_with_a_hook_is_the_hook_slot():
    p = parse("30 s reel of the fair, open with a hook. Show the band and the stalls.")
    assert p.hook and p.opener is None


@pytest.mark.parametrize("brief", ["60-90 s energetic recap", "5 calm B-roll shots 3-6 s each", "3 clips of dogs",
                                   "5 clips where someone talks to camera", "every moment someone speaks",
                                   "45-60 s recap, open with a wide shot, end with the crowd"])
def test_briefs_without_cues_keep_the_defaults(brief):
    assert _flags(parse(brief)) == ("none", False, False, False, False)


# ---- parser: LLM (mocked) ------------------------------------------------------------------------------------
def test_llm_schema_has_the_cue_fields_and_a_new_prompt_version():
    assert PROMPT_VERSION >= 4
    assert {"framing", "want_faces", "hook", "legible_text", "centered"} <= set(LLM_SCHEMA["required"])
    assert LLM_SCHEMA["properties"]["framing"]["enum"] == ["none", "close", "wide"]


def test_llm_cues_count_where_the_brief_grounds_them(monkeypatch):
    brief = ("a 20 s reel of the fair: jugglers and the brass band up close, the first shot must grab, subject at the "
             "center, text on the banners must be clear")
    assert _flags(parse(brief)) == ("none", False, False, False, False)  # the rules read none of these
    _mock_ollama(monkeypatch, _answer(content="fair", shots=["jugglers", "brass band"], framing="close",
                                      want_faces=True, hook=True, legible_text=True, centered=True))
    p = parse_with_llm(brief, model="m")
    # want_faces is not grounded (the brief names no faces); the rest is
    assert p.parser == "llm:m" and _flags(p) == ("close", False, True, True, True)


def test_llm_cues_without_grounding_are_ignored_and_rules_cues_stand(monkeypatch):
    # The model calls the whole edit wide although "wide" is only the opener's; and invents readable text.
    _mock_ollama(monkeypatch, _answer(content="fair recap", shots=["the band", "the stalls"], framing="wide",
                                      legible_text=True, opener="establishing wide of the square"))
    p = parse_with_llm("60-90 s recap of the fair. Open on an establishing wide of the square. Show the band and the "
                       "stalls.", model="m2")
    assert _flags(p) == ("none", False, False, False, False)
    _mock_ollama(monkeypatch, _answer(content="fair", framing="none"))
    p = parse_with_llm(TIKTOK, model="m3")  # the model misses every cue: the rules' stand
    assert _flags(p) == ("close", True, True, False, True)


# ---- scoring: the cue bonus through the v2 features ------------------------------------------------------------
def _X(n=3, **cols):
    X = np.zeros((n, len(FEATURES)))
    X[:, INDEX["scale_close_mean"]] = -6.0  # an ordinary medium shot (features.GRID_NEUTRAL)
    for name, v in cols.items():
        X[:, INDEX[name]] = v
    return X


def _q(**cues):
    p = SimpleNamespace(weights={"relevance": 1.0}, min_len=2.0, max_len=4.0, mood=None, exhaustive=False, text="x",
                        budget=None, shots=[], avoid=[], topics=[], need_live_audio=False,
                        **{"framing": "none", "want_faces": False, "hook": False, "legible_text": False,
                           "centered": False, **cues})
    return query_vector(p)


def test_no_cues_no_bonus_and_close_ranks_close_ups_first():
    X = _X(scale_close_mean=[-6.0, 0.0, 2.0], face_area_max_mean=[0.0, 0.01, 0.05], approach_max=[0.0, 0.0, 0.9])
    h = HeuristicScorer()
    np.testing.assert_array_equal(h.cue_bonus(X, _q()), 0.0)
    base = h.score(X, _q())
    close = h.score(X, _q(framing="close"))
    assert list(np.argsort(-(close - base))) == [2, 1, 0] and close[0] == base[0]
    faces = h.cue_bonus(X, _q(want_faces=True))
    assert faces[2] > faces[1] > faces[0] == 0.0  # want_faces alone: the face term only
    np.testing.assert_allclose(faces[2], 0.5)


def test_wide_rewards_wides_and_penalises_close_ups():
    X = _X(scale_wide_mean=[5.0, 1.0, -2.0], scale_close_mean=[-8.0, -6.0, 1.0])
    b = HeuristicScorer().cue_bonus(X, _q(framing="wide"))
    np.testing.assert_allclose(b, [0.8, 0.0, -0.45])  # scale_close 1.0: 0.75 of the close ramp


def test_centered_and_legible_terms():
    X = _X(face_center_mean=[0.9, 0.1, 0.0], ocr_item_match=[0.0, 1.0, 0.5])
    np.testing.assert_allclose(HeuristicScorer().cue_bonus(X, _q(centered=True)), [0.27, 0.03, 0.0])
    np.testing.assert_allclose(HeuristicScorer().cue_bonus(X, _q(legible_text=True)), [0.0, 0.6, 0.3])
    # the same function on per-step values (the engine's candidate peaks)
    steps = {n: X[:, INDEX[n]] for n in FEATURES_V2}
    np.testing.assert_allclose(cue_bonus(steps, "none", False, True, False), [0.27, 0.03, 0.0])


def _overlaps(clip, spans):
    return any(clip["file_start"] < b and clip["file_end"] > a for a, b in spans)


def test_engine_cue_briefs_pick_the_cue_footage(tmp_path):
    eng = cue_engine(tmp_path / "cues")
    close_spans = {f"take{k}.mp4": v.get("close", []) for k, v in CUE_SPANS.items()}
    run = {name: eng.clips(q, llm=None)["clips"] for name, q in CUE_BRIEFS.items()}
    n_close = {name: sum(_overlaps(c, close_spans[c["file"]]) for c in clips) for name, clips in run.items()}
    assert n_close["cue_close"] == 2 and n_close["cue_none"] < 2 and n_close["cue_wide"] == 0
    assert any(c["file"] == "take0.mp4" and _overlaps(c, CUE_SPANS[0]["wide"]) for c in run["cue_wide"])
    assert any(c["file"] == "take1.mp4" and _overlaps(c, CUE_SPANS[1]["text"]) for c in run["cue_legible"])


def test_hook_slot_is_the_first_clip_and_the_best_hook(tmp_path):
    eng = cue_engine(tmp_path / "cues")
    res = eng.clips(CUE_BRIEFS["cue_hook"], llm=None)
    first = res["clips"][0]
    assert first["role"] == "opener" and first["reason"].startswith("opener: hook")
    assert first["file"] == "take0.mp4" and _overlaps(first, CUE_SPANS[0]["close"])  # approach + close-up
    assert res["query"]["min_len"] - 1e-6 <= first["duration"] <= res["query"]["max_len"] + 1e-6
    plain = eng.clips("a 20 s edit of the parade, 2-3 s per clip", llm=None)["clips"]
    assert all(c["role"] is None for c in plain)  # no hook cue, no opener: no slot


# ---- schema: features and query v2, dumps, v1 models ---------------------------------------------------------
def test_features_v2_append_to_v1_in_order():
    assert FEATURE_VERSION == "v2" and QUERY_VERSION == "v2" and len(set(FEATURES)) == len(FEATURES)
    assert len(FEATURES) == V1_LEN + len(FEATURES_V2) and FEATURES[V1_LEN - len(V1_TAIL):V1_LEN] == V1_TAIL
    assert FEATURES[V1_LEN:] == FEATURES_V2 == (
        "scale_close_mean", "scale_close_max", "scale_wide_mean", "face_area_max_mean", "face_area_max_max",
        "face_center_mean", "face_n_mean", "approach_max", "text_area_mean", "text_conf_mean", "ocr_item_match")
    names = query_names()
    assert names[:len(QUERY_BASE)] == QUERY_BASE and names[len(QUERY_BASE):len(QUERY_BASE) + len(QUERY_V2)] == (
        "framing_none", "framing_close", "framing_wide", "want_faces", "hook", "legible_text", "centered")
    assert names[-len(PERSONAS):] == tuple(f"persona_{x}" for x in PERSONAS)
    q = query_vector(parse(TIKTOK), "p2-tiktok")
    assert len(q) == len(names) and q[names.index("framing_close")] == 1.0 and q[names.index("framing_none")] == 0.0
    assert q[names.index("hook")] == q[names.index("centered")] == q[names.index("want_faces")] == 1.0


def test_featurize_v2_columns_on_the_cue_project(tmp_path):
    eng = cue_engine(tmp_path / "cues")
    c = eng.corpus
    p = eng.parse(CUE_BRIEFS["cue_legible"], None)
    S = np.array([10.0, 30.0, 0.0, 60.0 + 20.0, 60.0 + 0.0])
    E = S + np.array([4.0, 4.0, 4.0, 6.0, 4.0])
    K = np.array([0, 0, 0, 1, 1])
    X = featurize(c, eng._signals(p)[0], S, E, K, ["window"] * 5, p)
    col = lambda n: X[:, INDEX[n]]  # noqa: E731
    assert np.isfinite(X).all() and X.shape == (5, len(FEATURES))
    np.testing.assert_allclose(col("scale_close_max"), [3.0, -8.0, -6.0, -6.0, -6.0])
    np.testing.assert_allclose(col("scale_wide_mean"), [-3.0, 5.0, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(col("face_area_max_max"), [0.05, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(col("face_n_mean"), [1, 0, 0, 0, 0])
    np.testing.assert_allclose(col("approach_max"), [0.8, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(col("text_conf_mean"), [0, 0, 0, 0.95, 0], atol=1e-6)
    np.testing.assert_allclose(col("ocr_item_match"), [0, 0, 0, 1.0, 0])  # "PASARGAD" read inside the span


def test_featurize_v2_is_neutral_without_index_v3_signals(tmp_path):
    from heuristic_cases import _dirs, _emb, _engine
    from test_corpus import make_project

    (dogs,) = _dirs(1, seed=2)
    root = make_project(tmp_path / "old", [("a.mp4", 30.0, {"emb": _emb(np.random.default_rng(0), 90, (dogs,))})])
    eng = _engine(root, {"dogs": dogs})
    p = eng.parse("3 clips of dogs, close-ups", None)
    X = featurize(eng.corpus, eng._signals(p)[0], np.array([0.0, 10.0]), np.array([4.0, 16.0]), np.array([0, 0]),
                  ["window"] * 2, p)
    v2 = X[:, V1_LEN:]
    np.testing.assert_array_equal(v2[:, FEATURES_V2.index("scale_close_mean")], -6.0)
    assert not np.delete(v2, [FEATURES_V2.index("scale_close_mean"), FEATURES_V2.index("scale_close_max")], 1).any()
    np.testing.assert_array_equal(HeuristicScorer().cue_bonus(X, query_vector(p)), 0.0)


def test_dumps_carry_v2(tmp_path):
    eng = cue_engine(tmp_path / "cues")
    eng.clips(CUE_BRIEFS["cue_close"], llm=None, dump=tmp_path / "d.json", dump_all=True)
    d = dumps.read(tmp_path / "d.json")
    assert d["feature_version"] == "v2" and d["query_version"] == "v2" and d["feature_names"] == list(FEATURES)
    assert d["query"]["q"]["framing_close"] == 1.0 and d["query"]["parsed"]["framing"] == "close"
    rows = [c for c in d["candidates"] if c["stage"] == "candidate"]
    X = np.array([c["features"] for c in rows])
    q = np.array(list(d["query"]["q"].values()))
    np.testing.assert_allclose(HeuristicScorer().score(X, q), [c["score"] for c in rows], atol=1e-4)


def test_v1_model_files_are_refused_clearly(tmp_path):
    F = len(FEATURES)
    v2 = LinearRankScorer(np.zeros(F), np.ones(F), np.zeros(F)).to_json()
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(v2))
    assert isinstance(load_scorer(path), LinearRankScorer)
    v1 = dict(v2, feature_version="v1", query_version="v1", features=list(FEATURES[:V1_LEN]), mu=[0.0] * V1_LEN,
              sd=[1.0] * V1_LEN, w=[0.0] * V1_LEN)
    path.write_text(json.dumps(v1))
    with pytest.raises(ValueError, match=r"features 'v1'.*this engine computes features 'v2'.*Retrain"):
        load_scorer(path)
    with pytest.raises(ValueError, match="query vector 'v1'"):
        path.write_text(json.dumps(dict(v2, query_version="v1")))
        load_scorer(path)
