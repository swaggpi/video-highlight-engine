"""vtl: fast tests, no network, GPU or video files (the LLM is a fake; Claude is never called)."""
import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from vtl import find as FD
from vtl import frames as F
from vtl import gpu as G
from vtl import index as IX
from vtl import llm as L
from vtl import ocr as O
from vtl import overlay as OV
from vtl import timeline as T
from vtl.cli import parser
from vtl.common import extract_json, hms, parse_time, read_json, read_jsonl, write_json, write_jsonl


@pytest.fixture(autouse=True)
def _no_claude(monkeypatch):
    """Candidates and decision default to Haiku: a test that forgets its fake LLM fails instead of calling Claude."""
    def refuse(*a, **k):
        raise AssertionError("a test tried to call the Claude CLI")

    monkeypatch.setattr(L, "_claude", refuse)


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


def test_model_selection_defaults_local_timeline_haiku_rest(monkeypatch):
    for k in ("VTL_LLM", "VTL_LLM_TIMELINE", "VTL_LLM_DECISION", "VTL_LLM_CANDIDATES"):
        monkeypatch.delenv(k, raising=False)
    assert L.model_for("timeline") == "ollama:qwen3:4b" and L.is_local(L.model_for("timeline"))
    assert L.model_for("candidates") == "haiku" and L.model_for("decision") == "haiku"
    monkeypatch.setenv("VTL_LLM_CANDIDATES", "ollama:qwen3:4b")
    assert L.model_for("candidates") == "ollama:qwen3:4b" and L.model_for("decision") == "haiku"
    monkeypatch.delenv("VTL_LLM_CANDIDATES")
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
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert T.chunks(20.0) == [(0.0, 20.0)]


def test_chunk_lines_interleave_and_collapse_duplicates():
    frames = [{"t": 0.0, "text": "The image shows a dark hallway."},
              {"t": 1.0, "text": "The image shows a dark hallway."}, {"t": 2.0, "text": "A monster jumps out."},
              {"t": 61.0, "text": "later"}]
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
    assert a.streamer is None and a.crop is None and not a.exclusive_gpu  # None: VTL_STREAMER decides
    a = parser().parse_args(["index", "x.mp4", "--streamer", "--crop", "373,0,907,720", "--exclusive-gpu"])
    assert a.streamer and a.crop == [373, 0, 907, 720] and a.facecam is None and a.exclusive_gpu
    with pytest.raises(SystemExit):
        parser().parse_args(["index", "x.mp4", "--crop", "1,2,3"])
    assert parser().parse_args(["find", "dir", "b", "--streamer"]).streamer
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


# ---- streamer mode -------------------------------------------------------------------------------------------------

def _stream_frames(k=40, h=108, w=192, seed=0, overlays=True):
    """Synthetic stream: moving blurry blobs and a moving bright bar (the game), a static striped panel at
    x 128-184 / y 8-40 (chat), and a static textured facecam at x 0-64 / y 64-108 with border lines."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cam_tex = rng.random((44, 64)).astype(np.float32) * 0.6 + 0.2
    frames = []
    for i in range(k):
        cx, cy = (i * 37) % w, (i * 23) % h
        g = 0.6 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12.0 ** 2))
        bx = (i * 29) % (w - 20)
        g[(yy > 50) & (yy < 58) & (xx > bx) & (xx < bx + 20)] = 0.9
        if overlays:
            g[8:40, 128:184] = 0.1
            g[10:40:6, 130:182] = 0.9  # text rows
            g[64:108, 0:64] = cam_tex
            g[64, 0:65] = 1.0
            g[64:108, 64] = 1.0
        frames.append(g.astype(np.float32))
    faces = [[[20, 76, 16, 18]] if i % 5 else [] for i in range(k)]  # the streamer, in 80% of the frames
    faces[3].append([150, 70, 20, 22])  # a game character's face, once
    return np.stack(frames), faces


def test_overlay_detection_finds_static_panel_facecam_and_free_crop():
    grey, faces = _stream_frames()
    lay = OV.layout(grey, faces)
    cam = lay["facecam"]
    assert abs(cam[0]) <= 2 and abs(cam[1] - 64) <= 2 and abs(cam[0] + cam[2] - 64) <= 2 and cam[1] + cam[3] == 108
    assert len(lay["panels"]) == 1
    px, py, pw, ph = lay["panels"][0]
    assert px <= 130 and py <= 10 and px + pw >= 182 and py + ph >= 38  # covers the striped panel ...
    assert pw * ph < 2 * 56 * 32  # ... and not much more
    crop = lay["crop"]
    assert not any(OV._overlap(crop, b) for b in [cam, *lay["panels"]])
    assert crop == OV.largest_free_rect(192, 108, [cam, *lay["panels"]]) and lay["crop_share"] >= 0.3
    # the moving game content alone is no overlay; a full-size "facecam" is a camera, not an overlay
    plain = OV.layout(_stream_frames(overlays=False)[0], [[] for _ in grey])
    assert plain["facecam"] is None and plain["panels"] == [] and plain["crop"] == [0, 0, 192, 108]
    big = OV.layout(grey, [[[60, 20, 70, 60]]] * len(grey))
    assert big["facecam"] is None or big["facecam"][2] * big["facecam"][3] <= OV.FACECAM_MAX * 192 * 108


def test_overlay_manual_override_and_crop_math():
    grey, faces = _stream_frames(k=10)
    lay = OV.layout(grey, faces, facecam=[0, 60, 70, 48], crop=[70, 0, 122, 108])
    assert lay["facecam"] == [0, 60, 70, 48] and lay["crop"] == [70, 0, 122, 108]
    assert OV.largest_free_rect(100, 50, []) == [0, 0, 100, 50]
    assert OV.largest_free_rect(100, 50, [[0, 30, 20, 20]]) == [20, 0, 80, 50]
    assert OV.largest_free_rect(100, 50, [[0, 30, 20, 20], [80, 0, 20, 10]]) == [20, 10, 80, 40]
    assert OV.scale_box([373, 0, 907, 720], 796 / 1280) == [232, 0, 564, 448]
    assert OV.parse_box("10, 20,30,40") == [10, 20, 30, 40]
    with pytest.raises(ValueError):
        OV.parse_box("10,20,0,40")
    assert OV.face_cluster([[]] * 5, 100, 50) is None
    assert OV.sample_paths(list(range(10)), 4) == [1, 3, 6, 8]


def test_ocr_cards_merge_flicker_drop_alerts_and_flag_chrome():
    rows = [{"t": 0.0, "lines": [["BIRDSEED", 0.97]]},
            {"t": 1.0, "lines": [["BIROSE", 0.85], ["(Space) to drop", 0.99]]},
            {"t": 2.0, "lines": [["BIRD SEED", 0.9], ["(Space)to drop", 0.97], ["xx", 0.99], ["blurry", 0.5]]},
            {"t": 10.0, "lines": [["DAY5", 0.98], ["(THE LASTDAY)", 0.92]]},
            {"t": 11.0, "lines": [["DAY5", 0.96], ["(THE LASTDAY)", 0.83]]},
            {"t": 12.0, "lines": [["@ox2005 donated $5", 0.95], ["through superchat!", 0.97], ["PSCARE", 0.95]]},
            {"t": 13.0, "lines": [["@ox2005 donated $5", 0.95], ["through superchat!", 0.97],
                                  ["bellathegoat32313:Yess the best", 0.95]]},
            {"t": 14.0, "lines": [["bellathegoat32313:Yess the best", 0.95]]}]  # alerts, chat, a one-read fragment
    cards = O.cards(rows)
    assert [(c["t"], c["end"], c["text"]) for c in cards] == [
        (0.0, 2.0, "BIRD SEED"), (1.0, 2.0, "(Space) to drop"), (10.0, 11.0, "DAY5 | (THE LASTDAY)")]
    assert not any(c["chrome"] for c in cards)
    many = [{"t": 10.0 * i, "lines": [["(Space) to drop", 0.99]]} for i in range(O.CHROME_N + 1)]
    assert all(c["chrome"] for c in O.cards(many))
    assert O.card_text([f"line {i}" for i in range(6)]) == "line 0 | line 1 | line 2 | line 3 | ..."


def test_ocr_cards_and_facecam_become_timeline_inputs():
    cards = [{"t": 3556.0, "end": 3558.0, "text": "DAY5 | (THE LASTDAY)", "conf": 0.98, "chrome": False},
             {"t": 3540.0, "end": 3545.0, "text": "(Space) to drop", "conf": 0.99, "chrome": True},
             {"t": 3557.0, "end": 3557.0, "text": "DAY5 | (THE LASTDAY)", "conf": 0.9, "chrome": False}]
    cams = [{"t": 3546.7, "text": "a man standing with his hands on his head", "trigger": "loud"},
            {"t": 3550.0, "text": "", "trigger": "speech"}]
    frames = [{"t": 3545.0, "text": "The image shows a dark room."}]
    segs = [{"start": 3547.0, "end": 3548.0, "text": "NO"}]
    assert T.card_line(cards[0]) == '[59:16.0] TEXT: "DAY5 | (THE LASTDAY)" (on screen 3 s)'
    assert T.chunk_lines(frames, segs, 3540, 3600, cards=cards, cams=cams) == [
        "[59:05.0] SEE: A dark room.",
        "[59:06.7] CAM: streamer (loud) a man standing with his hands on his head",
        '[59:07.0] SAY: "NO"',
        "[59:10.0] CAM: streamer (starts talking) (no visible reaction)",
        '[59:16.0] TEXT: "DAY5 | (THE LASTDAY)" (on screen 3 s)']  # chrome hidden, repeat not shown twice
    assert T.chunk_lines(frames, segs, 3540, 3600) == ["[59:05.0] SEE: A dark room.", '[59:07.0] SAY: "NO"']
    ev = T.card_events(cards + [{"t": 5.0, "end": 6.0, "text": "Game Over", "chrome": False}], 60.0)
    assert [(e["t"], e["type"], e["source"]) for e in ev] == [(3556.0, "new day", "ocr"), (3557.0, "new day", "ocr"),
                                                            (5.0, "game over", "ocr")]
    llm_ev = [{"t": 4.0, "type": "game over", "text": "screen says game over", "confidence": 0.8},
              {"t": 3556.0, "type": "text appears", "text": "'DAY5'", "confidence": 0.9},
              {"t": 3548.0, "type": "dies", "text": "caught by Granny", "confidence": 0.8}]
    merged = T.merge_cards(llm_ev, ev)
    assert [(e["t"], e["type"]) for e in merged] == [(4.0, "game over"), (3548.0, "dies"), (3556.0, "new day")]


def test_streamer_timeline_uses_its_own_prompt_and_cache_key(tmp_path):
    assert T.cache_key("m", "u") == T.cache_key("m", "u", T.SYSTEM) != T.cache_key("m", "u", T.SYSTEM_STREAMER)
    seen = []

    def fake(system, user, model, schema=None):
        seen.append((system, user))
        return json.dumps({"summary": "", "events": [{"t": "0:01.0", "type": "text appears", "who": "",
                                                       "text": "'GAME OVER'", "confidence": 0.9}]})

    frames = [{"t": 0.0, "text": "A dark room."}]
    cards = [{"t": 1.0, "end": 3.0, "text": "GAME OVER", "conf": 0.99, "chrome": False}]
    ev = T.build("a.mp4", frames, [], 30.0, 30.0, "fake", tmp_path / "c", call=fake, log=lambda *a: None,
                 streamer={"cards": cards, "cams": []})
    assert seen[0][0] == T.SYSTEM_STREAMER and 'TEXT: "GAME OVER"' in seen[0][1]
    assert [(e["type"], e.get("source")) for e in ev] == [("game over", "ocr")]
    T.build("a.mp4", frames, [], 30.0, 30.0, "fake", tmp_path / "c", call=fake, log=lambda *a: None)
    assert seen[1][0] == T.SYSTEM and "TEXT" not in seen[1][1] and len(list((tmp_path / "c").glob("*.json"))) == 2


def test_failed_chunk_has_no_events_is_not_cached_and_is_reported(tmp_path):
    answers = iter(["garbage", "still garbage", json.dumps({"summary": "s", "events": []})])
    frames = [{"t": float(t), "text": "A room."} for t in range(0, 120, 5)]
    failed = []
    ev = T.build("a.mp4", frames, [], 120.0, 30.0, "fake", tmp_path, call=lambda *a: next(answers),
                 log=lambda *a: None, failed=failed)
    assert ev == [] and failed == [(0.0, 60.0)] and len(list(tmp_path.glob("*.json"))) == 1


def test_reaction_points_and_text():
    db = np.full(60, -40.0)
    db[[10, 11, 14, 40]] = -10.0
    pts = F.reaction_points(db, [3.4, 13.4, 19.0], 3.0)  # speech at dense 10 (same as a loud spike), 40, 57
    assert [(i, k) for i, k, _ in pts] == [(10, "loud"), (40, "loud"), (57, "speech")]
    assert pts[0][2] == 30.0
    assert F.reaction_text("The image shows a man standing in front of a green screen with his hands on his head. "
                           "He is wearing a brown t-shirt and there is a piano on the left side of the image.") == \
        "a man standing, with his hands on his head"
    assert F.reaction_text("A piano and a wall with stickers in the background.") == ""


def test_stage_keys_isolate_modes_and_keep_plain_keys():
    assert IX.index_dir(OV.Path("/r")) == OV.Path("/r/.vtl")
    assert IX.index_dir(OV.Path("/r"), True) == OV.Path("/r/.vtl/streamer")
    # the plain frames key is exactly the pre-streamer formula (this value is in the stream VOD's plain meta.json)
    assert IX.stage_keys({}, 1.0, 3.0, False) == {"frames_key": "1ad1f30b26e91beb"}
    assert all(v is None for k, v in IX.stage_keys({}, 1.0, 3.0, True, "L").items() if k != "layout_key")
    meta = {"layout_key": "L", "layout": {"crop_dense": [232, 0, 564, 448], "facecam_dense": [0, 304, 232, 144]}}
    k = IX.stage_keys(meta, 1.0, 3.0, True, "L")
    assert k["frames_key"] not in (None, "1ad1f30b26e91beb") and k["react_key"] and k["ocr_key"]
    assert IX.stage_keys(dict(meta, layout_key="other"), 1.0, 3.0, True, "L")["frames_key"] is None
    meta2 = {"layout_key": "L", "layout": {"crop_dense": [0, 0, 564, 448], "facecam_dense": None}}
    k2 = IX.stage_keys(meta2, 1.0, 3.0, True, "L")
    assert k2["frames_key"] != k["frames_key"] and k2["react_key"] == "none"


def test_index_modes_do_not_share_files_or_caches(tmp_path, monkeypatch):
    import cv2

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    calls = {"decode": 0, "caption": 0, "react": 0, "ocr": 0, "llm": 0, "layout": 0}

    def decode(v, out, max_fps):
        calls["decode"] += 1
        (out / "dense").mkdir(parents=True, exist_ok=True)
        for i in range(12):
            img = np.full((36, 64, 3), 20 * i % 255, np.uint8)
            cv2.imwrite(str(out / "dense" / f"{i:06d}.jpg"), img)
        return {"duration": 4.0, "src_fps": 30.0, "width": 128, "height": 72, "has_audio": False,
                "max_fps": max_fps, "n_dense": 12}

    class Cap:
        def caption(self, images, task=F.CAPTION_TASK):
            calls["react" if task == F.REACT_TASK else "caption"] += len(images)
            return [f"a frame {im.size}" for im in images]

        def close(self):
            pass

    def layout(paths, facecam=None, crop=None):
        calls["layout"] += 1
        return {"size": [64, 36], "samples": len(paths), "facecam": [0, 20, 20, 16], "panels": [],
                "crop": crop or [20, 0, 44, 36], "crop_share": 0.69, "notes": [], "face_frames": len(paths)}

    crops = []

    def ocr_read(paths, times, crop, log=print):
        calls["ocr"] += 1
        crops.append(crop)
        return [{"t": t, "lines": [["GAME OVER", 0.99]] if t >= 2.0 else []} for t in times]

    def llm(system, user, model, schema=None):
        calls["llm"] += 1
        return json.dumps({"summary": "", "events": []})

    monkeypatch.setattr(F, "decode", decode)
    monkeypatch.setattr(F, "Captioner", Cap)
    monkeypatch.setattr(OV, "detect", layout)
    monkeypatch.setattr(O, "read", ocr_read)
    monkeypatch.setattr(IX.S, "transcribe", lambda wav: {"segments": [{"start": 1.0, "end": 2.0, "text": "hi",
                                                                       "words": []}]})
    monkeypatch.setattr(L, "llm", llm)
    monkeypatch.setattr(L, "unload", lambda model: None)
    monkeypatch.setenv("VTL_LLM_TIMELINE", "ollama:fake")
    quiet = {"log": lambda *a: None}

    m = IX.index(video, streamer=True, **quiet)[0]
    sdir = tmp_path / ".vtl" / "streamer" / "clip.mp4"
    assert m["mode"] == "streamer" and m["layout"]["crop"] == [40, 0, 88, 72]  # dense px scaled to source px
    assert (sdir / "layout.jpg").exists() and (sdir / "cards.jsonl").exists() and (sdir / "facecam.jsonl").exists()
    assert [e["type"] for e in read_jsonl(sdir / "timeline.jsonl")] == ["game over"]
    assert not (tmp_path / ".vtl" / "clip.mp4").exists() and not (sdir / "dense").exists()
    assert calls["react"] == 1 and calls["caption"] > 0  # one speech start -> one facecam reaction
    first = dict(calls)
    IX.index(video, streamer=True, **quiet)  # everything current: nothing recomputed
    assert calls == first

    p = IX.index(video, **quiet)[0]  # plain mode: its own folder, own captions and timeline, no streamer stages
    pdir = tmp_path / ".vtl" / "clip.mp4"
    assert "mode" not in p and "layout" not in p and not (pdir / "cards.jsonl").exists()
    assert p["frames_key"] == "1ad1f30b26e91beb" and p["frames_key"] != m["frames_key"]
    assert calls["decode"] == first["decode"] + 1 and calls["caption"] > first["caption"]
    assert calls["layout"] == first["layout"] and calls["ocr"] == first["ocr"] and calls["react"] == first["react"]
    assert read_json(sdir / "meta.json")["timeline_key"] == m["timeline_key"]  # streamer index untouched

    calls_before = dict(calls)  # a manual crop is a different layout: re-decode, recaption, re-OCR
    m2 = IX.index(video, streamer=True, crop=[48, 0, 80, 72], **quiet)[0]
    assert m2["layout"]["manual"]["crop"] == [48, 0, 80, 72] and m2["layout"]["crop"] == [48, 0, 80, 72]
    assert calls["decode"] == calls_before["decode"] + 1 and calls["caption"] > calls_before["caption"]
    assert calls["ocr"] == calls_before["ocr"] + 1 and crops == [[20, 0, 44, 36], [24, 0, 40, 36]]
    assert calls["react"] == calls_before["react"]  # same facecam: reactions stay
    with pytest.raises(SystemExit):
        IX.index(video, crop=[0, 0, 10, 10], **quiet)

    write_jsonl(sdir / "timeline.jsonl", [{"t": 2.0, "frame": 60, "type": "game over", "who": "", "text": "x",
                                           "confidence": 0.9}])
    assert FD.load_index(tmp_path)[0]["timeline"] != FD.load_index(tmp_path, True)[0]["timeline"]
    assert FD.load_index(tmp_path, True)[0]["cards"][0]["text"] == "GAME OVER"
    assert "cards" not in FD.load_index(tmp_path)[0]
    assert FD.default_out(tmp_path, "b", True).parent == tmp_path / ".vtl" / "streamer" / "queries"


def test_exclusive_gpu_restarts_in_finally(monkeypatch):
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd[2:])
        active = cmd[2] == "is-active" and cmd[-1] == "a.service"
        return SimpleNamespace(returncode=0 if active or cmd[2] != "is-active" else 3, stderr="")

    old = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError):
        with G.exclusive(("a.service", "b.service"), log=lambda *a: None, run=run) as stopped:
            assert stopped == ["a.service"] and ["stop", "a.service"] in cmds
            assert signal.getsignal(signal.SIGTERM) is not old
            raise RuntimeError("indexing failed")
    assert cmds == [["is-active", "--quiet", "a.service"], ["stop", "a.service"],
                    ["is-active", "--quiet", "b.service"], ["start", "a.service"]]
    assert signal.getsignal(signal.SIGTERM) is old
    cmds.clear()
    with pytest.raises(KeyboardInterrupt):
        with G.exclusive(("a.service",), log=lambda *a: None, run=run):
            raise KeyboardInterrupt
    assert cmds[-1] == ["start", "a.service"]


def test_ollama_alone_unloads_others_and_offloads_all_layers(monkeypatch):
    unloaded = []
    monkeypatch.setattr(L, "running", lambda: ["bge-m3:latest", "qwen3:4b"])
    monkeypatch.setattr(L, "unload", lambda m: unloaded.append(m))
    monkeypatch.setattr(L, "OLLAMA_OPTIONS", {})
    G.ollama_alone("haiku", log=lambda *a: None)
    assert unloaded == [] and L.OLLAMA_OPTIONS == {}
    G.ollama_alone("ollama:qwen3:4b", log=lambda *a: None)
    assert unloaded == ["ollama:bge-m3:latest"] and L.OLLAMA_OPTIONS == {"num_gpu": 999}


def test_truncated_reply_is_salvaged_and_interrupts_stop_the_timeline(tmp_path):
    cut = ('{"summary": "He \\"runs\\".", "events": [{"t": "0:01.0", "type": "dies", "who": "", "text": "caught", '
           '"confidence": 0.9}, {"t": "0:02.0", "type": "sees room", "who": "", "text": "Dark room", "confid')
    assert json.loads(T.salvage(cut)) == {"summary": 'He "runs".', "events": [
        {"t": "0:01.0", "type": "dies", "who": "", "text": "caught", "confidence": 0.9}]}
    assert T.salvage('{"summary": "", "events": []}') is None and T.salvage("no json") is None
    frames = [{"t": float(t), "text": "A room."} for t in range(0, 180, 5)]
    ev = T.build("a.mp4", frames, [], 60.0, 30.0, "fake", tmp_path / "s", call=lambda *a: cut, log=lambda *a: None)
    assert [e["type"] for e in ev] == ["dies"]
    assert json.loads(next((tmp_path / "s").glob("*.json")).read_text())["salvaged_truncated_reply"]

    calls = []

    def interrupted(*a):
        calls.append(1)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):  # one group (local model): no worker thread keeps going
        T.build("a.mp4", frames, [], 180.0, 30.0, "fake", tmp_path / "k", call=interrupted, log=lambda *a: None)
    assert len(calls) == 1
