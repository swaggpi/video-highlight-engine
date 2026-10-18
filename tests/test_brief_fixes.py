"""The query-side fixes after the irl-event field test (r2): one focused, synthetic test per item.

1 reliability (LLM unloads, CUDA OOM falls back to the CPU), 2 near-duplicates, 3 clean edges (whole sentences,
steady frames), 4 hard exclusions, 5 budget minimum, 6 multilingual / vague briefs, 7 exhaustive lists and the
on-screen text hook."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from heuristic_cases import _dirs, _emb, _engine, _text_vec
from test_corpus import FPS, make_project
from vhe.engine import Engine
from vhe.query import plan
from vhe.query.candidates import fit_utterances, speech_spans
from vhe.query.parser import parse, parse_with_llm
from vhe.query.refine import Context, edges_clean, refine, refine_speech
from vhe.rank.select import duplicates, select, select_for
from test_plan import _answer, _mock_ollama


def _words(spec):
    """[(start, end, word)] -> transcript words."""
    return [{"start": s, "end": e, "word": w} for s, e, w in spec]


# ---- 1. reliability ---------------------------------------------------------------------------------------------
def test_llm_parse_unloads_the_model_immediately(monkeypatch):
    calls = _mock_ollama(monkeypatch, _answer(content="dogs"))
    parse_with_llm("three clips of dogs", model="qwen3:4b")
    assert calls[0]["keep_alive"] == 0  # not resident while SigLIP / CG-DETR load on the same GPU


def test_topic_embeddings_unload_unless_another_client_holds_the_model(monkeypatch):
    import io

    bodies = []

    def urlopen(req, timeout):
        if isinstance(req, str):  # /api/ps: nothing loaded
            return io.BytesIO(json.dumps({"models": []}).encode())
        bodies.append(json.loads(req.data))
        return io.BytesIO(json.dumps({"embeddings": [[1.0, 0.0]]}).encode())

    monkeypatch.setattr(plan.urllib.request, "urlopen", urlopen)
    assert plan.embed(["a sentence"]) is not None and bodies[0]["keep_alive"] == "0s"


def test_cuda_out_of_memory_falls_back_to_cpu(tmp_path, monkeypatch):
    import torch

    from vhe.signals import frames

    built = []

    class FakeEncoder:
        def __init__(self, name, device):
            built.append(device)
            if device == "cuda":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
            self.device = device

        def texts(self, texts):
            return np.stack([_text_vec(t) for t in texts]).astype(np.float32)

    monkeypatch.setattr(frames, "Encoder", FakeEncoder)
    root = make_project(tmp_path / "p", [("a.mp4", 40.0, {})])
    eng = Engine.open(root, device="cpu", use_nc=False)
    eng.device = "cuda"  # as if --device auto had picked a GPU that another process fills
    with pytest.warns(RuntimeWarning, match="out of GPU memory"):
        res = eng.clips("3 clips of a street", llm=None)
    assert built == ["cuda", "cpu"] and eng.device == "cpu"
    assert res["clips"] and any("CPU" in w for w in res["warnings"])
    # another error is not swallowed
    eng2 = Engine.open(root, device="cpu", use_nc=False)
    eng2.device = "cuda"
    monkeypatch.setattr(FakeEncoder, "__init__", lambda self, n, d: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError):
        eng2.text_vectors(["x"])


# ---- 2. near-duplicates -----------------------------------------------------------------------------------------
def _cand(file, s, e, total, source="window", item=-1):
    return {"file": file, "start": s, "end": e, "total": total, "source": source, "item": item}


def test_same_action_and_lookalikes_are_picked_once():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(8, 16))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[6] = v[5]  # same file, 60 s apart, but the same static view
    cands = [_cand(0, 10, 14, 9.0), _cand(0, 15, 19, 8.9), _cand(0, 18.5, 22.5, 8.8), _cand(0, 6, 9.5, 8.7),  # one pass
             _cand(1, 40, 44, 5.0), _cand(2, 100, 104, 7.0), _cand(2, 160, 164, 6.9), _cand(3, 200, 204, 4.0)]
    picked = select(cands, v, n=5, max_overlap=0.0)
    files = sorted((c["file"], c["start"]) for c in picked)
    assert sum(f == 0 for f, _ in files) == 1  # one clip of the mounted-police-like pass, not four
    assert sum(f == 2 for f, _ in files) == 1  # the static twin is dropped
    assert len(picked) == 4  # nothing else left: stops short instead of padding with duplicates
    assert len(select(cands, v, n=5, max_overlap=0.0, dedupe=False)) == 5  # the old behaviour, for contrast


def test_adjacent_sentences_of_a_talking_track_are_not_duplicates():
    v = np.tile(np.eye(4)[:1], (3, 1))  # same picture throughout
    cands = [_cand(0, 0, 4, 3.0, "speech"), _cand(0, 4.5, 9, 2.9, "speech"), _cand(0, 9.5, 12, 2.8, "speech")]
    assert len(select(cands, v, n=3, max_overlap=0.0, talk_track=True)) == 3
    assert not duplicates(cands[1:], v[1:], cands[0], v[0], talk_track=True).any()
    # in a visual edit the operator talking over the same shot does not make three clips of it
    assert len(select(cands, v, n=3, max_overlap=0.0)) == 1


def test_round_robin_moves_on_instead_of_padding_an_item():
    v = np.eye(6)
    cands = [_cand(0, 10, 14, 9.0, item=0), _cand(0, 15, 19, 8.9, item=0), _cand(0, 18.5, 22.5, 8.8, item=0),
             _cand(1, 30, 34, 5.0, item=1), _cand(2, 40, 44, 4.0, item=1), _cand(3, 50, 54, 1.0, item=-1)]
    picked = select(cands, v, n=4, max_overlap=0.0, items=np.array([c["item"] for c in cands]))
    assert [c["file"] for c in picked].count(0) == 1 and len(picked) == 4  # item 1 twice, then the unassigned clip


# ---- 3. clean edges ---------------------------------------------------------------------------------------------
SPEECH = _words([(10.0, 10.4, "So"), (10.5, 11.0, "we"), (11.1, 11.6, "are"), (11.7, 12.3, "here."),
                 (12.6, 13.0, "The"), (13.1, 13.6, "music"), (13.7, 14.0, "is"), (14.1, 14.8, "muted"),
                 (14.9, 15.6, "because"), (15.7, 16.0, "of"), (16.1, 17.0, "copyright."),
                 (17.4, 17.8, "Next"), (17.9, 18.2, "we"), (18.3, 18.8, "go."), (25.0, 25.6, "Bye.")])


def test_speech_clip_is_whole_sentences():
    ctx = Context(60.0, [], [], SPEECH)
    s, e = refine_speech(13.05, 15.0, ctx, 2.0, 10.0)  # starts on "music", ends on "because": both mid-sentence
    assert s <= 12.6 and s >= 12.3  # before "The", after "here."
    assert 17.0 <= e <= 17.4  # after "copyright.", before "Next"
    s, e = refine_speech(13.05, 15.0, ctx, 2.0, 3.0)  # the sentence does not fit 3 s: it is still not cut mid-word
    assert not any(w["start"] < s < w["end"] or w["start"] < e < w["end"] for w in SPEECH)
    assert e - s <= 3.0 + 2 * 0.15 + 1e-6  # and the clip stays within the length limit (plus the lead-in/out)


def test_sentences_run_across_asr_segments_and_short_ones_grow_by_whole_neighbours():
    # ASR split the second sentence over two segments: the second segment starts mid-sentence.
    segs = [{"start": 10.0, "end": 14.0, "words": SPEECH[:7]}, {"start": 14.1, "end": 26.0, "words": SPEECH[7:]}]
    spans = speech_spans(segs, vad=[], max_len=30, gap=0.2)
    assert any(s <= 12.6 and e >= 17.0 for s, e in spans)  # "The music ... copyright." is one unit
    assert not any(12.6 < s < 17.0 for s, e in spans)  # nothing starts inside it
    fitted = fit_utterances(spans, lo=5.0, hi=12.0, duration=60.0)
    for s, e in fitted:  # every fitted clip starts and ends between words
        assert not any(w["start"] < s < w["end"] or w["start"] < e < w["end"] for w in SPEECH)
        assert e - s >= 5.0 - 1e-6


def test_visual_clip_does_not_end_inside_a_whip_pan():
    n = int(60 * FPS)
    clean = np.ones(n, dtype=bool)
    clean[int(19.5 * FPS):int(21.5 * FPS)] = False  # whip-pan 19.5-21.5 s
    clean[int(30 * FPS):int(30.7 * FPS)] = False  # black frames at 30 s
    ctx = Context(60.0, [], [], [], None, FPS, clean=clean)
    s, e = refine(15.0, 21.0, ctx, 3.0, 8.0)
    assert e <= 19.5 + 1e-6 and e - s >= 3.0 and edges_clean(s, e, ctx)
    s, e = refine(30.0, 35.0, ctx, 3.0, 8.0)
    assert s >= 30.66 and edges_clean(s, e, ctx)
    assert not edges_clean(15.0, 21.0, ctx)


# ---- 4. hard exclusions -----------------------------------------------------------------------------------------
def test_exclusion_wording_is_hard_or_soft():
    hard = parse("5 clips of dogs. No birds, never cats and absolutely no horses.")
    assert set(hard.exclude) >= {"birds", "cats", "horses"} and set(hard.exclude) <= set(hard.avoid)
    soft = parse("5 clips of dogs. Avoid birds, preferably no cats.")
    assert soft.avoid and not soft.exclude
    talk = parse("8 cutaways, steady, no host talking over it.")
    assert talk.no_speech and talk.weights["speech"] < 0
    host = parse("A 90 s recap. Exclude anything shaky and the host's talking (it goes under music).")
    assert host.no_speech and not any("talk" in x for x in host.avoid) and "shaky" in " ".join(host.exclude)
    assert plan.defect_kind("whip-pans") == "whip" and plan.defect_kind("something blocking the lens") == "blocked"
    assert plan.defect_kind("camera points at the sky") == "framing" and plan.defect_kind("protest signs") is None
    assert plan.conflicts("frame with a barricade", ["establishing wide with the barricades"])


def _mixed_project(tmp_path):
    dogs, birds = _dirs(2, seed=3)
    rng = np.random.default_rng(1)
    mixed = (dogs + birds) / np.sqrt(2)  # dogs with birds in the same frame
    spec = [("a.mp4", 60.0, {"emb": _emb(rng, 180, (dogs, birds), [(0, 25, dogs), (30, 60, mixed)])}),
            ("b.mp4", 60.0, {"emb": _emb(rng, 180, (dogs, birds), [(0, 60, birds)])})]
    return make_project(tmp_path / "mixed", spec), {"dogs": dogs, "birds": birds}


def test_no_x_drops_clips_showing_x_while_soft_wording_only_penalises(tmp_path):
    root, vec = _mixed_project(tmp_path)
    eng = _engine(root, vec)
    hard = eng.clips("6 clips of dogs, 3-6 s each. No birds.", llm=None)
    assert hard["excluded_dropped"] > 0 and len(hard["clips"]) >= 2
    assert all(c["file"] == "a.mp4" and c["file_end"] <= 30.0 + 0.7 for c in hard["clips"])  # never the mixed part
    soft = eng.clips("6 clips of dogs, 3-6 s each. Preferably no birds.", llm=None)
    assert soft["excluded_dropped"] == 0
    assert any(c["file"] == "a.mp4" and c["file_start"] >= 29.0 for c in soft["clips"])


def test_no_talking_drops_speech_heavy_clips(tmp_path):
    n = int(60 * FPS)
    speech = np.zeros(n)
    speech[: n // 2] = 1.0  # someone talks for the first 30 s
    root = make_project(tmp_path / "talk", [("a.mp4", 60.0, {"speech": speech})])
    res = _engine(root).clips("4 clips of the street, no talking", llm=None)
    assert res["clips"] and all(c["file_start"] >= 30.0 - 1.0 * 4 for c in res["clips"])
    assert all(np.mean(speech[int(c["file_start"] * FPS):int(c["file_end"] * FPS)]) <= 0.3 for c in res["clips"])


# ---- 5. budget minimum ------------------------------------------------------------------------------------------
def test_a_count_that_cannot_fill_the_budget_does_not_cap_it():
    cands = [_cand(k, 20.0 * k, 20.0 * k + 10, 10 - 0.01 * k) for k in range(30)]  # one 10 s moment per file
    p = SimpleNamespace(count=3, count_explicit=True, budget=(100.0, 140.0), max_len=12.0, exhaustive=False,
                        weights={}, shots=[])
    picked = select_for(cands, np.eye(30), p, groups=np.arange(30))
    assert sum(c["end"] - c["start"] for c in picked) >= 100.0


def test_budget_is_filled_by_longer_holds_or_reported(tmp_path):
    root = make_project(tmp_path / "short", [("a.mp4", 30.0, {"cam_speed": np.zeros(90)}),
                                             ("b.mp4", 30.0, {"cam_speed": np.zeros(90)})])
    eng = _engine(root)
    res = eng.clips("a 40-50 s calm piece of the street, long holds of 6-12 seconds", llm=None)
    total = sum(c["duration"] for c in res["clips"])
    assert total >= 40.0 or res["warnings"]
    assert all(c["duration"] <= 12.0 + 1e-6 for c in res["clips"])
    res = eng.clips("a 300-400 s piece of the street", llm=None)  # 60 s of footage: cannot be filled
    assert any("below the requested" in w for w in res["warnings"])


def test_a_soundbite_counts_toward_the_budget(tmp_path):
    from heuristic_cases import _speech_index

    eng = _engine(_speech_index(tmp_path / "talk.mp4.vhe"))
    res = eng.clips("Show the crowd, the stage and the band. Also give the best 5-8 s soundbite where the host sums up "
                    "the stream (\"that is it for today\").", budget_s=(20.0, 25.0), llm=None)
    assert any(c["role"] == "soundbite" for c in res["clips"])
    assert sum(c["duration"] for c in res["clips"]) <= 25.0 + 1e-6


# ---- 6. multilingual and vague briefs ---------------------------------------------------------------------------
GERMAN = ("hey kannst du mir die besten momente von der parade raussuchen, so die pferde, die riesen fahne und die "
          "tänzerinnen, nich zu lang so 30 sek für insta. keine leeren strassen")


def test_german_rules_read_budget_length_and_negation():
    p = parse(GERMAN)
    assert p.budget == (25.5, 34.5)  # "so 30 sek für insta": the whole post
    assert p.max_len <= 6.0  # "nich zu lang" is short, not "lang"
    assert p.exclude and "leeren strassen" in " ".join(p.exclude)
    assert parse("3 clips ohne autos").exclude == ["autos"]
    assert parse("szenen von 4-6 sekunden, nicht zu lang").max_len == 6.0


def test_llm_items_are_translated_and_kept(monkeypatch):
    calls = _mock_ollama(monkeypatch, _answer(content="parade highlights", shots=["horses", "giant flag", "dancers"],
                                              exclude=["empty streets"]))
    p = parse_with_llm(GERMAN, model="qwen3:4b")
    assert p.shots == ["horses", "giant flag", "dancers"]  # English, not grounded word by word in German
    assert "empty streets" in p.exclude and p.budget == (25.5, 34.5)
    assert "English" in calls[0]["messages"][0]["content"]
    english = parse_with_llm("Show the dogs and the cats.", model="other")  # English briefs stay grounded
    assert "horses" not in english.shots


def test_best_moments_without_a_shot_list_use_a_highlight_prior(tmp_path):
    p = parse("the best moments of the day")
    assert p.highlight and p.weights["relevance"] > 0 and p.weights["energy"] > 0 and p.max_len <= 10.0
    n = int(60 * FPS)
    tags = np.zeros((n, 2), np.float16)
    tags[int(40 * FPS):int(50 * FPS), 0] = 0.9  # people dancing at 40-50 s
    root = make_project(tmp_path / "hl", [("a.mp4", 60.0, {"tags": tags})])
    for ix_meta in (root / ".vhe" / "files").rglob("meta.json"):
        meta = json.loads(ix_meta.read_text())
        meta["labels"] = ["dancing", "b"]
        ix_meta.write_text(json.dumps(meta))
    res = _engine(root).clips("the best moments of the day", count=2, llm=None)
    assert any(c["file_start"] < 50.0 and c["file_end"] > 40.0 for c in res["clips"])
    assert all(c["duration"] <= 10.0 + 1e-6 for c in res["clips"])


# ---- 7. exhaustive lists and the on-screen text hook ------------------------------------------------------------
def test_every_shot_where_returns_every_distinct_match_not_one_per_item():
    assert not parse("I need every shot where text is legible: float names, banners, street signs.").per_item
    assert parse("List every float: the Lotus, the Dragon and the Tower.").per_item
    cands = [_cand(k, 10.0 * k, 10.0 * k + 4, 3.0 - 0.1 * k, item=k % 2) for k in range(8)]
    item_rel = np.array([[2.0, 0.0] if k % 2 == 0 else [0.0, 2.0] for k in range(8)])
    base = dict(count=50, count_explicit=False, budget=None, max_len=6.0, exhaustive=True, weights={"relevance": 1.0},
                shots=["float names", "banners"])
    every = select_for(cands, np.eye(8), SimpleNamespace(**base, per_item=False), item_rel=item_rel)
    assert len(every) == 8  # all distinct matches above the threshold
    per = select_for(cands, np.eye(8), SimpleNamespace(**base, per_item=True), item_rel=item_rel)
    assert len(per) == 2  # one per named item


def test_legibility_prior_only_for_readable_text_briefs_with_ocr():
    grid = {"text_conf": np.array([0.0, 0.9, 0.8, np.nan]), "text_area": np.array([0.0, 0.1, 0.0, 0.2])}
    leg = plan.legibility_steps(grid, "every shot where on-screen text is legible")
    np.testing.assert_allclose(leg, [0.0, 0.9, 0.0, 0.0])
    assert plan.legibility_steps(grid, "5 clips of dogs") is None
    assert plan.legibility_steps({}, "every shot where text is legible") is None  # index without OCR


def test_ocr_hook_scores_items_when_the_corpus_has_text_and_is_a_no_op_otherwise():
    S, E, K = np.array([0.0, 10.0]), np.array([5.0, 15.0]), np.array([0, 0])
    assert plan.ocr_item_scores(SimpleNamespace(), ["pasargad float"], S, E, K) is None
    corpus = SimpleNamespace(text_items=[{"t": 2.0, "text": "TOMBof CYRUS PASARGAD", "conf": 0.9, "file": 0},
                                         {"t": 12.0, "text": "RETAILSPACEFORLEASE", "conf": 0.8, "file": 0}])
    s = plan.ocr_item_scores(corpus, ["Pasargad float", "Tomb of Cyrus"], S, E, K)
    assert s.shape == (2, 2) and s[0, 0] > 0 and s[0, 1] > 0.5 and not s[1].any()
