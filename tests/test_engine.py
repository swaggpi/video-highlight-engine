"""Engine.clips end to end on synthetic projects: the hard gates, budget fill, soundbites, the highlight prior,
talking-track edges and the CPU fallback after a GPU out-of-memory error."""
import json

import numpy as np
import pytest

from heuristic_cases import _dirs, _emb, _engine, _speech_index, _text_vec
from test_corpus import FPS, make_project
from vhe.engine import Engine
from vhe.query.exclusions import (
    OCR_HOLD_S,
)
from vhe.query.refine import (
    SPEECH_LEAD,
    SPEECH_TAIL,
    word_clear,
)
from vhe.query.rules import parse
from vhe.rank.gate import NO_SPEECH_MAX_FRAC


# ---- reliability -----------------------------------------------------------------------------------------------------
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


# ---- hard gates ------------------------------------------------------------------------------------------------------
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


def _ocr_project(root):
    (parade,) = _dirs(1, seed=9)
    specs = [(f"take{k}.mp4", 60.0, {"emb": _emb(np.random.default_rng(30 + k), int(60 * FPS), (parade,),
                                                 [(0, 60, parade)])}) for k in (0, 1)]
    make_project(root, specs)
    for d in sorted((root / ".vhe" / "files").iterdir()):
        if d.name.startswith("take1"):
            items = [[s, "Executed by IRGC on Apr 3", 0.95, [0.1, 0.2, 0.4, 0.3]] for s in range(60, 81, 3)]
            (d / "ocr.json").write_text(json.dumps({"version": 1, "engine": "fake", "every": 3, "min_conf": 0.7,
                                                    "items": items}))
    return root, parade


def test_engine_drops_every_clip_that_shows_a_forbidden_word(tmp_path):
    root, parade = _ocr_project(tmp_path / "ocr")
    vec = {"the parade": parade, "parade": parade}
    eng = _engine(root, vec)
    sign = (eng.corpus.offsets[1] + 20.0 - OCR_HOLD_S, eng.corpus.offsets[1] + 26.67 + OCR_HOLD_S)

    def shows(c):
        return c["file"] == "take1.mp4" and c["start"] < sign[1] and c["end"] > sign[0]
    plain = eng.clips("12 clips of the parade, 4-6 s each", llm=None)
    assert any(shows(c) for c in plain["clips"])  # the sign's stretch is good footage otherwise
    res = eng.clips("12 clips of the parade, 4-6 s each. Nothing with 'executed' on it.", llm=None)
    assert res["clips"] and not any(shows(c) for c in res["clips"])


def test_speech_free_brief_gates_out_every_clip_with_speech(tmp_path):
    root = _speech_index(tmp_path / "talk.mp4.vhe")
    eng = _engine(root)
    speech = eng.corpus.grid["speech"]
    assert NO_SPEECH_MAX_FRAC <= 0.05
    res = eng.clips("8 clips 3-5 s each, the music bed goes on later", llm=None)
    assert res["clips"]
    for c in res["clips"]:
        i0, i1 = int(c["start"] * FPS), int(np.ceil(c["end"] * FPS))
        assert np.mean(speech[i0:i1] > 0.5) <= NO_SPEECH_MAX_FRAC


# ---- budgets ---------------------------------------------------------------------------------------------------------
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


# ---- highlight prior and talking tracks ------------------------------------------------------------------------------
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


def test_engine_talk_track_edges_are_word_clear(tmp_path):
    root = _speech_index(tmp_path / "talk.mp4.vhe")
    eng = _engine(root)
    words = eng.corpus.words
    res = eng.clips("every moment someone speaks", llm=None)
    assert res["clips"]
    assert any(c["source"] == "speech" for c in res["clips"])
    for c in res["clips"]:
        assert word_clear(words, c["start"]) and word_clear(words, c["end"])
        if c["source"] == "speech":  # whole sentences, in the long pauses between them: full lead and tail
            inside = [w for w in words if c["start"] <= w["start"] and w["end"] <= c["end"]]
            assert inside[0]["start"] - c["start"] == pytest.approx(SPEECH_LEAD, abs=1e-3)
            assert c["end"] - inside[-1]["end"] == pytest.approx(SPEECH_TAIL, abs=1e-3)
