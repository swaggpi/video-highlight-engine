"""vtl: fast tests, no network, GPU or video files (the LLM is a fake)."""
import json

import numpy as np
import pytest

from vtl import find as FD
from vtl import frames as F
from vtl import llm as L
from vtl import timeline as T
from vtl.cli import parser
from vtl.common import extract_json, hms, parse_time, read_jsonl, write_json, write_jsonl


def test_extract_json_fenced_prose_and_failure():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise {"a": {"b": [1, 2]}} trailing } text') == {"a": {"b": [1, 2]}}
    assert extract_json('[1] then {"ok": true}') == {"ok": True}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_llm_json_retries_once_then_fails():
    replies = iter(["not json", '{"events": []}'])
    assert L.llm_json("s", "u", "fake", call=lambda *a: next(replies)) == {"events": []}
    with pytest.raises(RuntimeError):
        L.llm_json("s", "u", "fake", call=lambda *a: "nope")
    bad = iter(['{"events": 1}', '{"events": 2}'])
    with pytest.raises(RuntimeError):
        L.llm_json("s", "u", "fake", check=T._check, call=lambda *a: next(bad))


def test_model_selection_default_is_local(monkeypatch):
    for k in ("VTL_LLM", "VTL_LLM_TIMELINE", "VTL_LLM_DECISION"):
        monkeypatch.delenv(k, raising=False)
    assert L.model_for("timeline") == "ollama:qwen3:4b" and L.is_local(L.model_for("timeline"))
    monkeypatch.setenv("VTL_LLM", "haiku")
    monkeypatch.setenv("VTL_LLM_DECISION", "ollama:qwen3:1.7b")
    assert L.model_for("timeline") == "haiku" and L.model_for("decision") == "ollama:qwen3:1.7b"
    with pytest.raises(ValueError):
        L.llm("s", "u", "gpt-4")


def test_times():
    assert hms(3539.4) == "0:58:59" and hms(59.96) == "0:01:00" and hms(676.33, 1) == "0:11:16.3"
    assert parse_time("11:16.3") == pytest.approx(676.3) and parse_time("1:02:03") == 3723
    assert parse_time(12) == 12.0 and parse_time("x") is None and parse_time(True) is None
    assert T.mmss(676.33) == "11:16.3" and T.mmss(3725) == "62:05.0"


def test_chunks_cover_duration_without_gaps():
    spans = T.chunks(185.0, 60.0)
    assert len(spans) == 3 and spans[0][0] == 0 and spans[-1][1] == 185.0
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert T.chunks(20.0) == [(0.0, 20.0)]


def test_chunk_lines_interleave_and_collapse_duplicates():
    frames = [{"t": 0.0, "text": "The image shows a dark hallway."}, {"t": 1.0, "text": "The image shows a dark hallway."},
              {"t": 2.0, "text": "A monster jumps out."}, {"t": 61.0, "text": "later"}]
    segs = [{"start": 1.5, "end": 2.5, "text": "oh no"}]
    assert T.chunk_lines(frames, segs, 0, 60) == ["[0:00.0] SEE: A dark hallway.", '[0:01.5] SAY: "oh no"',
                                                  "[0:02.0] SEE: A monster jumps out."]


def test_cache_key_changes_with_inputs_and_model():
    k = T.cache_key("ollama:qwen3:4b", "user")
    assert k == T.cache_key("ollama:qwen3:4b", "user")
    assert k != T.cache_key("haiku", "user") and k != T.cache_key("ollama:qwen3:4b", "user2")


def test_normalize_events_clamps_and_drops_bad():
    ev = [{"t": "1:05.0", "type": "Dies", "who": "player", "text": "caught", "confidence": 2},
          {"t": "bad", "type": "x", "text": "y"}, {"t": "0:10", "type": "", "text": "y"},
          {"t": "9:00", "type": "laughs", "text": "haha", "confidence": "high"}]
    out = T.normalize(ev, 60, 120, 60.0)
    assert [e["type"] for e in out] == ["dies", "laughs"]
    assert out[0]["frame"] == 3900 and out[0]["confidence"] == 1.0
    assert out[1]["t"] == 120 and out[1]["confidence"] == 0.5


def test_strict_drops_caption_copies_repeats_and_caps_count(monkeypatch):
    caps = ["A dark room with a wooden door and red dots on the walls."]
    ev = [{"t": 1.0, "type": "enters room", "text": "A dark room with a wooden door", "confidence": 0.9},
          {"t": 2.0, "type": "monster appears", "text": "A dark room with a wooden door", "confidence": 0.5},
          {"t": 3.0, "type": "screams", "text": "'get away'", "confidence": 0.9},
          {"t": 4.0, "type": "screams", "text": "'get away'", "confidence": 0.9}]
    assert [e["t"] for e in T.strict(ev, caps)] == [2.0, 3.0]  # an action type may restate a caption
    monkeypatch.setattr(T, "MAX_EVENTS", 1)
    assert [e["t"] for e in T.strict(ev, caps)] == [3.0]


def test_timeline_line_format():
    e = {"t": 3539.4, "frame": 212364, "who": "", "type": "dies", "text": "'I FED YOU MEAT'", "confidence": 0.9}
    assert T.line("VOD.mp4", e) == "VOD.mp4 0:58:59 f212364 dies — 'I FED YOU MEAT' (conf .9)"
    assert T.line("a.mp4", dict(e, who="Granny", confidence=1.0)).endswith("Granny: dies — 'I FED YOU MEAT' (conf 1.0)")


def test_snap_moves_edges_out_of_words():
    words = [{"start": 9.5, "end": 10.4, "word": "hello"}, {"start": 19.8, "end": 20.6, "word": "world"}]
    assert FD.snap(10.0, 20.0, words) == pytest.approx((9.45, 20.65))
    assert FD.snap(11.0, 15.0, words) == (11.0, 15.0)
    assert FD.snap(0.02, 5, [{"start": 0.0, "end": 0.5}]) == (0.0, 5)


def test_fit_length():
    assert FD.fit_length(10, 14, 10, 11, 100) == (7, 17)
    assert FD.fit_length(1, 3, 10, 0, 100) == (0.0, 10)
    assert FD.fit_length(10, 40, 0, 20, 100) == (10, 30)
    assert FD.fit_length(95, 99, 10, 0, 100) == (90, 100)


def test_select_budget_overlap_and_order():
    files = {"a.mp4": {"duration": 100.0, "segments": []}}
    scored = [{"file": "a.mp4", "start": s, "end": s + 20, "score": sc, "label": "", "reason": ""}
              for s, sc in [(50, 9), (52, 8), (10, 7), (80, 6), (30, 0)]]
    pick = FD.select(scored, {"total_s": 45, "order": "chronological"}, None, files)
    assert [c["start"] for c in pick] == [10, 50]  # 52 overlaps 50, the budget stops after two, chronological
    pick = FD.select(scored, {"total_s": 8, "order": "rank"}, 3, files)  # an implausible 8 s runtime is ignored
    assert [c["start"] for c in pick] == [50, 10, 80]


def test_adaptive_selection_and_spikes():
    kinds = F.select(30, 3, [10], 3)
    assert [i for i, k in kinds.items() if k == "base"] == list(range(0, 30, 3))
    assert [i for i, k in kinds.items() if k == "dense"] == [7, 8, 10, 11, 13]
    vis = np.zeros(300)
    vis[150] = 0.3
    db = np.full(300, -40.0)
    db[200] = -10.0
    sp = F.spikes(vis, db, [50.0], 3.0)
    assert sp == {"visual": [150], "audio": [200], "speech": [150]}
    segs = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 3.0}, {"start": 6.0, "end": 7.0}]
    assert F.speech_onsets(segs) == [0.0, 6.0]
    assert F.trim_caption("A man runs. He is chased by") == "A man runs."


def test_cli_args():
    a = parser().parse_args(["index", "x.mp4", "--fps", "2"])
    assert a.cmd == "index" and a.fps == 2.0 and a.max_fps == 3.0 and not a.force
    a = parser().parse_args(["find", "dir", "3 tiktok clips", "--count", "3", "--json", "o.json"])
    assert (a.cmd, a.brief, a.count, str(a.json_out)) == ("find", "3 tiktok clips", 3, "o.json")
    with pytest.raises(SystemExit):
        parser().parse_args(["find", "dir"])


class FakeLLM:
    """Answers each stage from the prompt alone, like a (very literal) model would."""

    def __init__(self):
        self.calls = []

    def __call__(self, system, user, model, schema=None):
        self.calls.append(system[:20])
        if system == T.SYSTEM:
            ev = [{"t": ln[1:ln.index("]")], "type": "dies", "who": "player", "text": "caught by monster",
                   "confidence": 0.9} for ln in user.splitlines() if "monster" in ln]
            return json.dumps({"events": ev, "summary": "stuff happened"})
        if system == FD.CAND_SYSTEM:
            cands = [{"file": "clip.mp4", "start": "0:00:40", "end": "0:00:46", "why": "death"},
                     {"file": "clip.mp4", "start": "0:01:00", "end": "0:01:04", "why": "dull"},
                     {"file": "nope.mp4", "start": "0:00:01", "end": "0:00:05", "why": "bad file"}]
            return "Here you go: " + json.dumps({"plan": {"count": 0, "min_len": 10, "max_len": 12, "total_s": 0,
                                                          "order": "rank"}, "candidates": cands})
        picks = [{"id": 0, "score": 9, "start": "0:00:39.0", "end": "0:00:47.0", "label": "scare/death",
                  "reason": "monster catches player"}, {"id": 1, "score": 0, "start": "x", "end": "y", "label": "",
                                                        "reason": "nothing"}]
        return json.dumps({"picks": picks})


def test_end_to_end_fake_llm(tmp_path):
    out = tmp_path / ".vtl" / "clip.mp4"
    frames = [{"frame": int(t * 30), "t": float(t), "kind": "base",
               "text": "A monster grabs the player." if t == 42 else "A dark corridor."} for t in range(90)]
    segs = [{"start": 46.6, "end": 48.3, "text": "no way", "words": [{"start": 46.6, "end": 47.2, "word": "no"},
                                                                    {"start": 47.3, "end": 48.3, "word": "way"}]}]
    fake = FakeLLM()
    events = T.build("clip.mp4", frames, segs, 90.0, 30.0, "fake", out / "cache", call=fake, log=lambda *a: None)
    assert [(e["t"], e["type"], e["frame"]) for e in events] == [(42.0, "dies", 1260)]
    n = len(fake.calls)
    T.build("clip.mp4", frames, segs, 90.0, 30.0, "fake", out / "cache", call=fake, log=lambda *a: None)
    assert len(fake.calls) == n  # second build is served from the chunk cache
    write_jsonl(out / "frames.jsonl", frames)
    write_jsonl(out / "timeline.jsonl", events)
    write_json(out / "transcript.json", {"segments": segs})
    write_json(out / "meta.json", {"file": "clip.mp4", "duration": 90.0, "timeline_key": "x"})
    res = FD.find(tmp_path, "a 10 s clip of each death", call=fake, log=lambda *a: None)
    assert res["n_candidates"] == 3 and len(res["clips"]) == 1
    c = res["clips"][0]
    assert c["source"] == "vtl" and c["file"] == "clip.mp4" and c["label"] == "scare/death"
    assert c["start"] == 38.0 and c["end"] == pytest.approx(48.35)  # 39-47 grown to 10 s, end snapped past "way"
    assert {io["stage"] for io in res["_io"]} == {"candidates", "decision"}
    assert "0:00:42 f1260 player: dies" in res["_io"][0]["user"]
    assert read_jsonl(out / "timeline.jsonl")[0]["chunk"] == 0
