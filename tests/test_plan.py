"""Real editor briefs as plans: shot lists, things to avoid, spoken topics, live sound; LLM parsing with grounding
and a disk cache; coverage-aware selection; and the CLIP context crash."""
import io
import json

import numpy as np
import pytest

from vhe.query import parser, plan
from vhe.query.parser import parse, parse_brief, parse_with_llm
from vhe.query.select import select, select_for

# The irl-event field-test briefs (vhe-field-test/eval/sets/irl-event/briefs.json, B1-B5).
B1 = ('Cut a 60-90 s energetic recap of the Persian Day Parade. Open on an establishing wide of Madison Avenue with '
      'barricades. Build through the mounted police, the NYPD Police Band, dancers in traditional costumes and the '
      'most colourful floats. End on the giant flag and the chanting crowd at the end of the route. Leave out shots '
      'where the camera points at the ground or sky, or where something blocks the lens.')
B2 = ('Find every moment where the host talks directly to his viewers about the stream: where he is (street names, '
      'Madison Avenue), why he is muting the music (copyright), and what he plans next ("explore some of '
      'Manhattan", "you\'re going to see water"). Give the clip and the start and end time of each, so they can be '
      'used as a talking track.')
B3 = ('List every parade float that passes the camera: the Nowruz/Haft-Sin float, Gate of the Nations, Pasargad, '
      'Statue of Cyrus, the white-statue float with the big flag, SHIRAZ, and the portrait floats. Give the clip '
      'and time range where each float is best seen, with one recommended 3-5 s shot per float.')
B4 = ('Pick 5 calm B-roll shots of 3-6 s each to sit under a voice-over: steady or locked-off framing, no '
      'whip-pans, nothing blocking the lens. For example: the wide of the empty avenue with barricades, a float '
      'rolling slowly past, the Statue of Cyrus close-up, the sanitation workers, the sidewalk under the '
      'scaffolding.')
B5 = ('For a news package, find the strongest 20-30 s stretch with live natural sound (not muted): the giant '
      'lion-and-sun flag carried past the camera, portrait protest signs and the crowd chanting at the end of the '
      'parade. Also give the best 5-8 s soundbite where the host sums up the parade ("short parade ... I think this '
      'is the end").')


def _has(items, *words):
    """Some item contains all the words."""
    return any(all(w in x.lower() for w in words) for x in items)


# ---- rules parser on the five briefs ------------------------------------------------------------------------------
def test_rules_b1_recap_with_shot_list_and_avoid():
    p = parse(B1)
    assert p.parser == "rules" and p.budget == (60.0, 90.0) and p.mood == "energetic" and not p.exhaustive
    assert "establishing wide" in p.opener and "madison avenue" in p.opener
    assert "giant flag" in p.closer and "chanting crowd" in p.closer  # "and ..." and "end of the route" kept
    for words in (("mounted", "police"), ("police", "band"), ("dancers", "costumes"), ("floats",)):
        assert _has(p.shots, *words), (words, p.shots)
    assert len(p.shots) == 4 and not _has(p.shots, "persian")  # the subject is context, not a shot
    assert _has(p.avoid, "camera", "ground") and _has(p.avoid, "camera", "sky") and _has(p.avoid, "lens")
    assert not p.topics and not p.need_live_audio
    assert len(p.text.split()) <= 12 and p.weights["relevance"] >= 2.0  # shot lists weigh content over mood


def test_rules_b2_spoken_topics():
    p = parse(B2)
    assert p.exhaustive and p.weights["speech"] == 1.0 and not p.shots and not p.avoid
    assert len(p.topics) == 3
    assert _has(p.topics, "madison avenue") and _has(p.topics, "muting", "copyright") and _has(p.topics, "water")
    assert not _has(p.topics, "talking track")  # the output instruction is not a topic


def test_rules_b3_named_floats_exhaustive():
    p = parse(B3)
    assert p.exhaustive and (p.min_len, p.max_len) == (3.0, 5.0) and p.budget is None
    assert len(p.shots) == 7
    for words in (("nowruz",), ("gate of the nations", "float"), ("pasargad", "float"), ("statue of cyrus", "float"),
                  ("white-statue", "flag"), ("shiraz", "float"), ("portrait",)):
        assert _has(p.shots, *words), (words, p.shots)
    assert not _has(p.shots, "clip") and not _has(p.shots, "time range")


def test_rules_b4_calm_broll_examples():
    p = parse(B4)
    assert p.count == 5 and p.count_explicit and (p.min_len, p.max_len) == (3.0, 6.0) and p.mood == "calm"
    for words in (("empty avenue",), ("float", "rolling"), ("statue of cyrus",), ("sanitation",), ("scaffolding",)):
        assert _has(p.shots, *words), (words, p.shots)
    assert len(p.shots) == 5 and not _has(p.shots, "locked") and not _has(p.shots, "voice")
    assert _has(p.avoid, "whip") and _has(p.avoid, "lens")


def test_rules_b5_live_stretch_plus_soundbite():
    p = parse(B5)
    assert p.need_live_audio and p.count == 1 and (p.min_len, p.max_len) == (20.0, 30.0) and p.budget is None
    assert p.soundbite_len == (5.0, 8.0)
    for words in (("flag",), ("protest signs",), ("crowd", "chanting")):
        assert _has(p.shots, *words), (words, p.shots)
    assert len(p.topics) == 1 and _has(p.topics, "sums up", "end")
    assert not _has(p.shots, "news package") and not _has(p.shots, "host")
    assert p.weights["speech"] == 0.0  # the visual stretch is not a speech search; the soundbite is separate


def test_simple_briefs_have_no_plan_lists():
    for q in ("3 clips of dogs", "bride and groom dancing", "every moment someone speaks to camera",
              "60-90 s energetic recap, open with an establishing shot", "the host says on stage"):
        p = parse(q)
        assert not p.shots and not p.topics and not p.avoid and not p.need_live_audio, q
    assert parse("mountains, lakes and forests").shots == ["mountains", "lakes", "forests"]
    assert parse("dogs playing, no people").avoid == ["people"]


# ---- LLM path -------------------------------------------------------------------------------------------------------
def _answer(**kw):
    a = {"content": "", "shots": [], "avoid": [], "topics": [], "need_live_audio": False, "mood": "neutral",
         "speech": "any", "min_seconds": 0, "max_seconds": 0, "count": 0, "budget_min_seconds": 0,
         "budget_max_seconds": 0, "soundbite_min_seconds": 0, "soundbite_max_seconds": 0, "exhaustive": False,
         "opener": "", "closer": ""}
    a.update(kw)
    return a


def _mock_ollama(monkeypatch, answer):
    calls = []

    def urlopen(req, timeout):
        calls.append(json.loads(req.data))
        return io.BytesIO(json.dumps({"message": {"content": json.dumps(answer)}}).encode())

    monkeypatch.setattr(parser.urllib.request, "urlopen", urlopen)
    return calls


def test_llm_plan_is_grounded_and_cached(monkeypatch):
    answer = _answer(content="Persian Day Parade recap", mood="energetic",
                     shots=["Madison Avenue with barricades", "mounted police", "NYPD Police Band performing",
                            "dancers in traditional costumes", "colorful floats", "Rosewood float"],
                     avoid=["camera pointing at the ground", "finger over the lens", "camera pointing at the sky"],
                     min_seconds=10, max_seconds=30, count=7, budget_min_seconds=60, budget_max_seconds=90,
                     need_live_audio=True, opener="Madison Avenue with barricades",
                     closer="giant flag and chanting crowd")
    calls = _mock_ollama(monkeypatch, answer)
    p = parse_with_llm(B1, model="qwen3:4b")
    assert p.parser == "llm:qwen3:4b" and p.parser_note is None and len(calls) == 1
    body = calls[0]
    assert body["think"] is False and body["options"]["temperature"] == 0 and body["keep_alive"] == "30s"
    assert body["format"]["required"] == parser.LLM_SCHEMA["required"] and body["messages"][-1]["content"] == B1
    # Numbers the brief does not contain (10-30 s per clip, 7 clips) are ignored; the budget is in it.
    assert p.budget == (60.0, 90.0) and (p.min_len, p.max_len) == (2.0, 4.0) and not p.count_explicit
    assert not p.need_live_audio  # the brief asks for no sound at all
    assert "Rosewood float" not in p.shots and "finger over the lens" not in p.avoid  # invented, not in the brief
    assert "Madison Avenue with barricades" not in p.shots  # restates the opener
    assert _has(p.shots, "nypd") and _has(p.shots, "colorful floats") and p.opener == "Madison Avenue with barricades"
    assert _has(p.avoid, "ground") and _has(p.avoid, "lens")  # the rules' avoid items are added
    # Second run: from the disk cache, no request, same plan.
    again = parse_with_llm(B1, model="qwen3:4b")
    assert len(calls) == 1 and again.parser_note == "cached" and again.shots == p.shots


def test_llm_fills_what_rules_miss_and_categorises_names(monkeypatch):
    _mock_ollama(monkeypatch, _answer(shots=["Nowruz/Haft-Sin float", "Gate of the Nations", "Pasargad",
                                             "Statue of Cyrus", "SHIRAZ", "portrait floats"], exhaustive=True))
    p = parse_with_llm(B3, model="m")
    assert p.exhaustive and (p.min_len, p.max_len) == (3.0, 5.0)  # the length came from the rules
    assert "Pasargad float" in p.shots and "SHIRAZ float" in p.shots and "Gate of the Nations float" in p.shots
    assert "portrait floats" in p.shots


def test_llm_unreachable_falls_back_to_rules(monkeypatch):
    def down(req, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(parser.urllib.request, "urlopen", down)
    p = parse_with_llm(B4, model="qwen3:4b")
    assert p.parser == "rules" and "unavailable" in p.parser_note and p.count == 5 and len(p.shots) == 5


def test_default_parser_is_llm_unless_disabled(monkeypatch):
    calls = _mock_ollama(monkeypatch, _answer(content="dogs"))
    assert parse_brief("dogs").parser == "rules" and not calls  # conftest: VHE_LLM=none
    monkeypatch.delenv("VHE_LLM")
    assert parser.default_llm() == parser.DEFAULT_LLM == "qwen3:4b"
    assert parse_brief("dogs").parser == "llm:qwen3:4b" and calls[0]["model"] == "qwen3:4b"
    assert parse_brief("dogs", llm=None).parser == "rules"


# ---- relevance from a plan ------------------------------------------------------------------------------------------
def test_avoid_penalty_is_bounded_and_spares_clear_matches():
    want = np.array([[3.0, 0.5, 0.5, -2.0]])
    avoid = np.array([[3.0, 3.0, 0.8, 3.0]])
    rel = plan.combine(want, avoid)
    assert rel[0] == 3.0  # clearly relevant: the avoid match does not beat it
    assert rel[1] == pytest.approx(0.5 - plan.AVOID_MAX)  # avoid dominates a weak match, by at most AVOID_MAX
    assert rel[2] == 0.5  # avoid below its floor is noise
    assert rel[3] == pytest.approx(-2.0 - plan.AVOID_MAX)
    np.testing.assert_array_equal(plan.combine(np.array([[1.0, 2.0], [3.0, 0.0]])), [3.0, 2.0])


def test_item_assignment_is_centred_per_item():
    # Item 0 is generic (matches everything at +1.8), item 1 specific: a +2.5 match for item 1 serves item 1.
    scores = np.array([[2.0, 2.5], [2.0, -1.0], [0.2, 0.3]])
    centre = np.array([1.8, -2.0])
    assert list(plan.assign(scores, centre)) == [1, 0, -1]


# ---- coverage-aware selection ---------------------------------------------------------------------------------------
def test_round_robin_covers_items_before_repeats():
    # Item 0 has five great candidates, item 1 two modest ones, item 2 one weak; -1 serves nothing.
    totals = [5.0, 4.9, 4.8, 4.7, 4.6, 2.0, 1.9, 1.0, 6.0]
    items = [0, 0, 0, 0, 0, 1, 1, 2, -1]
    cands = [{"start": 10.0 * k, "end": 10.0 * k + 3, "total": t} for k, t in enumerate(totals)]
    picked = select(cands, np.eye(len(cands)), 6, max_overlap=0.0, items=np.array(items))
    order = [items[int(c["start"] // 10)] for c in picked]
    assert sorted(order[:3]) == [0, 1, 2]  # every item once first, best item first
    assert order[3:5] == [0, 1] and order[5] == 0  # then round robin while candidates last
    assert -1 not in order  # a candidate serving no item waits until no item has one left
    picked = select(cands, np.eye(len(cands)), 9, max_overlap=0.0, items=np.array(items))
    assert [items[int(c["start"] // 10)] for c in picked][-1] == -1


def test_exhaustive_named_items_best_clip_each_and_skips_missing():
    p = parse(B3)
    cands, rows = [], []
    for k in range(8):  # item k has a strong clip at 10k s and a weak one at 10k + 5 s; item 5 has no footage
        for off, rel in ((0.0, 2.5), (5.0, 1.0)):
            row = np.full(len(p.shots), -2.0)
            if k < len(p.shots) and k != 5:
                row[k] = rel
            cands.append({"start": 10.0 * k + off, "end": 10.0 * k + off + 4, "total": rel + 0.1,
                          "relevance": float(row.max())})
            rows.append(row)
    item_rel = np.array(rows)
    picked = select_for(cands, np.eye(len(cands)), p, item_rel=item_rel, item_centre=np.full(len(p.shots), -2.0))
    assert sorted(c["item"] for c in picked) == [0, 1, 2, 3, 4, 6]  # one per item, item 5 skipped
    assert all(c["start"] % 10 == 0 for c in picked)  # the strong clip of each


# ---- spoken topics --------------------------------------------------------------------------------------------------
def test_topic_scores_with_embeddings_and_keyword_fallback():
    spans = ["I'm on Madison Avenue at 31st Street right now", "the weather is nice", "we mute the music for copyright"]
    topics = ["where he is, Madison Avenue", "why he mutes the music, copyright"]
    basis = {"where": [1, 0, 0], "mute": [0, 1, 0], "other": [0, 0, 1]}

    def fake_embed(texts):
        v = np.array([basis["where"] if "Madison" in t else basis["mute"] if "mute" in t else basis["other"]
                      for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    score, best, method, thresh = plan.topic_scores(spans, topics, embed_fn=fake_embed)
    assert method == "bge-m3" and list(score >= thresh) == [True, False, True] and best[0] == 0 and best[2] == 1
    score, best, method, thresh = plan.topic_scores(spans, topics, embed_fn=lambda texts: None)
    assert method == "keywords" and list(score >= thresh) == [True, False, True] and best[2] == 1


def test_embed_unreachable_returns_none(monkeypatch):
    def down(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(plan.urllib.request, "urlopen", down)
    assert plan.embed(["a"]) is None


# ---- CLIP context crash (CG-DETR query path) -------------------------------------------------------------------------
def test_long_brief_fits_clip_context():
    clip = pytest.importorskip("clip")
    from vhe.signals.highlight_nc import CLIP_CONTEXT, fit_clip_query

    brief = (B1 + " " + B4)[:600]
    assert len(brief) == 600
    with pytest.raises(RuntimeError, match="too long"):
        clip.tokenize(brief)  # what Lighthouse does with the raw query: the crash on real briefs
    fitted = fit_clip_query(brief)
    assert brief.startswith(fitted) and len(fitted) > 100
    assert clip.tokenize(fitted).shape[1] == CLIP_CONTEXT  # no exception
    assert fit_clip_query("mounted police") == "mounted police"


def test_nc_predict_with_600_char_brief_does_not_crash():
    clip = pytest.importorskip("clip")
    from vhe.signals.highlight_nc import NCHighlighter

    class FakePredictor:  # tokenizes like Lighthouse's CLIP text encoder, no model
        def predict(self, query, feats):
            clip.tokenize(query)
            return {"pred_saliency_scores": [0.5] * 10, "pred_relevant_windows": [[1.0, 3.0, 0.9]]}

    nc = object.__new__(NCHighlighter)
    nc.predictor, nc.clip_len, nc.device = FakePredictor(), 2.0, "cpu"
    nc._feats = {"/tmp/x/chunk_000.pt": {}}
    from pathlib import Path

    sal, props = nc.predict((B1 + " " + B5)[:600], [{"start": 0.0, "length": 20.0, "path": "chunk_000.pt"}],
                            Path("/tmp/x"), 60, 3.0)
    assert np.isfinite(sal).any() and props == [(1.0, 3.0, 0.9)]


# ---- engine end to end on synthetic footage --------------------------------------------------------------------------
FPS, DIM = 3.0, 64


def _project(tmp_path, specs):
    from test_corpus import make_project

    return make_project(tmp_path, specs)


def _directions(n, seed=7):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(DIM, n)))
    return q.T  # orthonormal


def _emb(rng, n_steps, dirs, at=None, d=None, avoid=()):
    e = rng.normal(size=(n_steps, DIM))
    for v in dirs:
        e -= np.outer(e @ v, v)  # nothing matches any item by chance
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    if at is not None:
        a, b = int(at[0] * FPS), int(at[1] * FPS)
        e[a:b] = 0.2 * d + np.sqrt(1 - 0.04) * e[a:b]  # cosine 0.2: a clear SigLIP match
    return e


def test_engine_shot_list_covers_items_and_reports_them(tmp_path, monkeypatch):
    from vhe.engine import Engine

    dogs, cats, horses = _directions(3)
    rng = np.random.default_rng(0)
    dirs = (dogs, cats, horses)
    specs = [("a.mp4", 60.0, {"emb": _emb(rng, 180, dirs, (5, 50), dogs)}),
             ("b.mp4", 60.0, {"emb": _emb(rng, 180, dirs, (20, 30), cats)}),
             ("c.mp4", 60.0, {"emb": _emb(rng, 180, dirs, (5, 50), dogs)})]
    root = _project(tmp_path, specs)
    vec = {"dogs": dogs, "cats": cats, "horses": horses}
    monkeypatch.setattr(Engine, "text_vectors",
                        lambda self, texts: np.stack([vec[t] for t in texts]).astype(np.float32))
    eng = Engine.open(root, device="cpu", use_nc=False)

    res = eng.clips("3 clips. Show the dogs, the cats and the horses.")
    assert res["query"]["shots"] == ["dogs", "cats", "horses"]
    items = [c["item"] for c in res["clips"]]
    assert len(items) == 3 and "cats" in items and "dogs" in items and "horses" not in items
    cat = next(c for c in res["clips"] if c["item"] == "cats")
    assert cat["file"] == "b.mp4" and min(cat["file_end"], 30.0) - max(cat["file_start"], 20.0) >= 0.6 * cat["duration"]
    assert "matches 'cats'" in cat["reason"]

    res = eng.clips("Every animal: the dogs, the cats and the horses.")
    assert res["query"]["exhaustive"]
    assert sorted(c["item"] for c in res["clips"]) == ["cats", "dogs"]  # no junk clip for the horses


def test_engine_need_live_audio_drops_muted(tmp_path, monkeypatch):
    from vhe.engine import Engine

    (dogs,) = _directions(1)
    rng = np.random.default_rng(1)
    loud = rng.random(180)
    loud[:90] = -120.0  # 0-30 s: the streamer muted the sound
    root = _project(tmp_path, [("a.mp4", 60.0, {"emb": _emb(rng, 180, (dogs,), (0, 60), dogs), "loudness": loud})])
    monkeypatch.setattr(Engine, "text_vector", lambda self, t: dogs.astype(np.float32))
    eng = Engine.open(root, device="cpu", use_nc=False)
    res = eng.clips("4 clips of dogs with natural sound")
    assert res["query"]["need_live_audio"] and res["clips"]
    assert all(c["file_start"] >= 30.0 - 0.1 * c["duration"] for c in res["clips"])
    res = eng.clips("8 clips of dogs")
    assert any(c["file_start"] < 27.0 for c in res["clips"])  # without the requirement the muted part is fine


def test_engine_topics_match_transcript_spans(tmp_path, monkeypatch):
    from vhe.engine import Engine
    from test_corpus import make_index
    from vhe.index.store import save_index

    lines = [(30.0, "I'm on Madison Avenue right now."), (80.0, "The weather is nice today."),
             (150.0, "We mute the music because of copyright.")]
    segs = []
    for s, text in lines:
        words = text.split()
        segs.append({"start": s, "end": s + 0.5 * len(words), "text": text,
                     "words": [{"start": s + 0.5 * k, "end": s + 0.5 * k + 0.4, "word": w}
                               for k, w in enumerate(words)]})
    root = tmp_path / "v.mp4.vhe"
    ix = make_index(root, 200.0)
    meta = dict(ix.meta, vad=[[s["start"], s["end"]] for s in segs])
    meta.pop("version", None)
    save_index(root, meta, ix.grid, {"language": "en", "segments": segs})
    monkeypatch.setattr(plan, "embed", lambda texts, **kw: None)  # keyword fallback, no server
    res = Engine(root, device="cpu", use_nc=False).clips(
        "Find every moment where the host talks about where he is (Madison Avenue), and why he mutes the music "
        "(copyright).")
    assert res["query"]["topics"] and res["topic_match"] == "keywords"
    clips = res["clips"]
    assert len(clips) == 2 and clips[0]["start"] < clips[1]["start"]  # time order, the weather line left out
    assert 28.0 <= clips[0]["start"] <= 30.0 and 148.0 <= clips[1]["start"] <= 150.0
    assert "Madison" in clips[0]["transcript"] and "copyright" in clips[1]["transcript"]
    assert all(c["item"] in res["query"]["topics"] for c in clips)
