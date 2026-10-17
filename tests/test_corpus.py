"""Folder-of-footage query side: corpus concatenation, absolute relevance, pooled signals, usable gate, edit order,
per-file cap, CLI JSON and EDL/CSV export."""
import csv
import itertools
import json
import shutil
import subprocess

import numpy as np
import pytest

from vhe.corpus import Corpus, open_corpus
from vhe.engine import Engine
from vhe.index.store import Index, save_index
from vhe.project import PROJECT_DIR, slug, write_manifest
from vhe.query.candidates import cuts_inside
from vhe.query.parser import ParsedQuery
from vhe.rank.signals import relevance, robust_z, timestep_signals
from vhe.rank.select import per_file_cap, select_for

FPS = 3.0
DIM = 64
SCALE, BIAS = 112.9, -16.8  # SigLIP2-base logit parameters as stored in meta.json


def _unit(rng, n, avoid=None):
    v = rng.normal(size=(n, DIM))
    if avoid is not None:  # orthogonal to `avoid`: similarity exactly 0
        v -= np.outer(v @ avoid, avoid)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_index(root, dur, seed=0, has_audio=True, emb=None, **overrides):
    n = int(dur * FPS)
    rng = np.random.default_rng(seed)
    grid = {"t": np.arange(n) / FPS, "emb": (emb if emb is not None else _unit(rng, n)).astype(np.float16),
            "sharpness": rng.random(n), "blur_ratio": 4 + rng.random(n), "blown": np.zeros(n),
            "motion": rng.random(n), "loudness": rng.random(n) if has_audio else np.zeros(n),
            "onset": rng.random(n) if has_audio else np.zeros(n), "cam_speed": rng.random(n),
            "shake": 0.05 * rng.random(n), "speech": np.zeros(n), "usable": np.ones(n, bool),
            "tags": np.zeros((n, 2), np.float16)}
    grid.update(overrides)
    meta = {"video": str(root / "v.mp4"), "fps": FPS, "duration": dur,
            "cuts": [float(c) for c in range(20, int(dur), 20)],
            "vad": [], "encoder": "none", "labels": ["a", "b"], "has_audio": has_audio, "logit_scale": SCALE,
            "logit_bias": BIAS}
    save_index(root, meta, grid, None)
    return Index(root)


def make_project(root, specs):
    """specs: [(rel_path, dur, kwargs)] -> a project folder with a manifest and per-file indexes."""
    files = []
    for k, (rel, dur, kw) in enumerate(specs):
        index_dir = f"{PROJECT_DIR}/files/{slug(rel)}"
        make_index(root / index_dir, dur, seed=k, **kw)
        files.append({"rel_path": rel, "status": "ok", "index_dir": index_dir, "duration": dur, "recorded_at": None})
    write_manifest(root, {"version": 1, "language": None, "files": files})
    return root


def _no_overlap(clips):
    spans = sorted((c["start"], c["end"]) for c in clips)
    return all(a[1] <= b[0] + 1e-6 for a, b in itertools.pairwise(spans))


def test_corpus_offsets_and_no_cross_file_clips(tmp_path):
    root = make_project(tmp_path, [("a.mp4", 61.0, {}), ("b/c.mp4", 45.0, {"has_audio": False}), ("d.mp4", 50.0, {})])
    c = open_corpus(root)
    assert c.files == ["a.mp4", "b/c.mp4", "d.mp4"] and list(c.bounds) == [0, 183, 318, 468]
    assert list(c.offsets) == [0.0, 61.0, 106.0]
    assert 61.0 in c.cuts and 106.0 in c.cuts  # file boundaries are hard cuts
    assert c.grid["emb"].shape == (468, DIM) and c.file_of(61.0) == 1 and c.local(1, 70.0) == 9.0
    assert list(c.audio_steps[[0, 200, 400]]) == [True, False, True]
    assert c.sources[1] == root.resolve() / "b/c.mp4"
    res = Engine(c, device="cpu", use_nc=False).clips("60-90 s energetic recap")
    assert res["clips"] and _no_overlap(res["clips"])
    for clip in res["clips"]:
        k = c.files.index(clip["file"])
        assert 0.0 <= clip["file_start"] < clip["file_end"] <= c.durations[k] + 1e-6
        assert c.offsets[k] - 1e-6 <= clip["start"] and clip["end"] <= c.offsets[k] + c.durations[k] + 1e-6
    # Edit order: manifest file order, then time.
    assert [(c.files.index(x["file"]), x["file_start"]) for x in res["clips"]] == \
        sorted((c.files.index(x["file"]), x["file_start"]) for x in res["clips"])
    assert list(cuts_inside(np.array([0.0, 50.0]), np.array([30.0, 70.0]), np.array(c.cuts))) == [1, 2]


def test_single_video_is_a_corpus_of_one(tmp_path):
    ix = make_index(tmp_path / "v.mp4.vhe", 30.0)
    c = open_corpus(tmp_path / "v.mp4.vhe")
    assert c.n == ix.n and list(c.offsets) == [0.0] and c.cuts == ix.cuts


def test_sigmoid_relevance_is_absolute_across_files():
    rng = np.random.default_rng(1)
    text = _unit(rng, 1)[0]
    match = 0.2 * text + np.sqrt(1 - 0.04) * _unit(rng, 90, avoid=text)  # cosine 0.2: a clear SigLIP match
    other = _unit(rng, 90, avoid=text)  # cosine ~0: unrelated
    sim = np.concatenate([match, other]) @ text + rng.normal(0, 0.01, 180)
    fid = np.repeat([0, 1], 90)
    rel = relevance(sim, FPS, (SCALE, BIAS), file_id=fid)
    assert rel[:90].mean() > 2.0 and rel[90:].mean() < -2.0 and rel[90:].max() < -1.0
    # The per-file z-score this replaces could not tell the files apart: both put their best tenth well above 0.
    assert np.percentile(robust_z(sim[:90]), 90) > 1 and np.percentile(robust_z(sim[90:]), 90) > 1


def test_relevant_file_wins_and_irrelevant_file_scores_low(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    text = _unit(rng, 1)[0]
    on = (0.2 * text + 0.98 * _unit(rng, 120, avoid=text))
    root = make_project(tmp_path, [("noise.mp4", 40.0, {"emb": _unit(rng, 120, avoid=text)}),
                                   ("dogs.mp4", 40.0, {"emb": on})])
    monkeypatch.setattr(Engine, "text_vector", lambda self, t: text.astype(np.float32))
    eng = Engine.open(root, device="cpu", use_nc=False)
    res = eng.clips("3 clips of dogs")
    assert [c["file"] for c in res["clips"]] == ["dogs.mp4"] * 3
    assert all(c["relevance"] > 2 for c in res["clips"])
    hits = eng.search("dogs", k=4)
    assert hits[0]["file"] == "dogs.mp4" and hits[0]["score"] > 2
    assert all(h["score"] < -2 for h in hits if h["file"] == "noise.mp4")


def test_usable_gate_drops_defects_even_when_energetic(tmp_path):
    n = int(90 * FPS)
    loud, usable = np.full(n, 0.1), np.ones(n, bool)
    loud[90:150] = 5.0  # 30-50 s: by far the loudest, most moving part ...
    usable[90:150] = False  # ... but the lens is covered / camera swinging
    motion = np.full(n, 0.1)
    motion[90:150] = 5.0
    root = make_project(tmp_path, [("a.mp4", 90.0, {"loudness": loud, "motion": motion, "usable": usable})])
    res = Engine.open(root, device="cpu", use_nc=False).clips("5 energetic clips")
    assert res["clips"] and res["unusable_dropped"] > 0
    for c in res["clips"]:
        inside = max(0.0, min(c["end"], 50.0) - max(c["start"], 30.0))
        assert inside <= 0.2 * (c["end"] - c["start"]) + 1e-6


def test_audioless_file_does_not_dominate_energy(tmp_path):
    root = make_project(tmp_path, [("loud.mp4", 120.0, {}), ("silent.mp4", 120.0, {"has_audio": False})])
    c = open_corpus(root)
    sig = timestep_signals(c.grid, FPS, None, c.audio_steps, c.file_id)
    a, b = sig["energy"][c.file_id == 0], sig["energy"][c.file_id == 1]
    assert np.percentile(np.abs(b), 90) <= 1.15 * np.percentile(np.abs(a), 90)
    res = Engine(c, device="cpu", use_nc=False).clips("8 energetic clips")
    assert {x["file"] for x in res["clips"]} == {"loud.mp4", "silent.mp4"}


def test_loudness_centred_per_file_before_pooling(tmp_path):
    rng = np.random.default_rng(3)
    base, flat = rng.random(360), np.ones(360)  # energy from loudness alone
    same = {"motion": flat, "onset": flat, "cam_speed": flat}
    root = make_project(tmp_path, [("phone.mp4", 120.0, {"loudness": base, **same}),
                                   ("cam.mp4", 120.0, {"loudness": base + 20.0, **same})])  # same scene, +20 dB gain
    c = open_corpus(root)
    e = timestep_signals(c.grid, FPS, None, c.audio_steps, c.file_id)["energy"]
    np.testing.assert_allclose(e[:360], e[360:], atol=1e-9)


def test_singleton_length_bucket_keeps_its_score(tmp_path):
    n = int(60 * FPS)
    energy = np.zeros(n)
    energy[120:150] = 1.0  # 40-50 s
    c = Corpus([make_index(tmp_path / "x.vhe", 60.0)])
    eng = Engine(c, device="cpu", use_nc=False)
    sig = {k: np.zeros(n) for k in ("relevance", "quality", "steady", "speech")} | {"energy": energy}
    S = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 40.0])
    E = S + np.array([3.0] * 6 + [10.0])
    p = ParsedQuery(raw="", text="", min_len=3.0, max_len=10.0)
    pool = eng._pool(sig, {"energy": 1.0}, (S, E, np.zeros(7, int), ["window"] * 7), p, 7)
    single = next(o for o in pool if o["end"] - o["start"] > 5)
    assert single["total"] > 0.5  # median-centring a one-member bucket used to make this exactly 0 (minus priors)


def test_opener_first_closer_last_and_budget_held(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    wide, crowd = _unit(rng, 2)
    crowd = crowd - (crowd @ wide) * wide
    crowd /= np.linalg.norm(crowd)

    def emb(seed, shot=None, at=(30, 60)):
        e = _unit(np.random.default_rng(seed), 180, avoid=wide)
        e -= np.outer(e @ crowd, crowd)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        if shot is not None:
            e[at[0]:at[1]] = shot  # 10-20 s
        return e

    shots = {2: wide, 0: crowd}  # take2 has the wide shot, take0 the crowd
    root = make_project(tmp_path, [(f"take{k}.mp4", 60.0, {"emb": emb(k, shots.get(k))}) for k in range(4)])

    def text_vector(self, t):
        return (wide if "wide" in t else crowd if "crowd" in t else wide * 0).astype(np.float32)

    monkeypatch.setattr(Engine, "text_vector", text_vector)
    brief = "45-60 s recap, open with a wide shot, end with the crowd"
    res = Engine.open(root, device="cpu", use_nc=False).clips(brief)
    clips = res["clips"]
    assert clips[0]["role"] == "opener" and clips[0]["file"] == "take2.mp4" and 9.0 <= clips[0]["file_start"] < 20.0
    assert clips[-1]["role"] == "closer" and clips[-1]["file"] == "take0.mp4"
    assert 45.0 <= sum(c["duration"] for c in clips) <= 60.0
    body = clips[1:-1]
    assert [(c["file"], c["file_start"]) for c in body] == sorted((c["file"], c["file_start"]) for c in body)
    assert _no_overlap(clips)


def test_per_file_cap_defers_but_does_not_forbid():
    cands, groups = [], []
    for k in range(10):  # file 0: ten great candidates
        cands.append({"start": k * 5.0, "end": k * 5.0 + 3, "total": 5.0 - 0.01 * k})
        groups.append(0)
    for g in (1, 2, 3):  # one modest candidate in each other file
        cands.append({"start": 100.0 * g, "end": 100.0 * g + 3, "total": 1.0})
        groups.append(g)
    groups = np.array(groups)
    p = ParsedQuery(raw="", text="", count=6, count_explicit=True)
    picked = select_for(cands, np.eye(len(cands)), p, groups=groups)
    from_0 = sum(c["start"] < 100 for c in picked)
    assert len(picked) == 6 and from_0 == 3  # capped at 2, then the fallback adds one more when nothing else is left
    assert per_file_cap(6, 13) == 2 and per_file_cap(30, 20) == 10 and per_file_cap(30, 3) is None
    p = ParsedQuery(raw="", text="", count=4, count_explicit=True)
    assert sum(c["start"] < 100 for c in select_for(cands, np.eye(len(cands)), p, groups=groups)) == 2


def test_cli_json_edit_list(tmp_path):
    from vhe.cli import main

    root = make_project(tmp_path / "proj", [("a.mp4", 60.0, {}), ("b.mp4", 60.0, {"has_audio": False}),
                                            ("c.mp4", 60.0, {})])
    out = tmp_path / "edit.json"
    main(["clips", str(root), "30-40 s energetic recap", "--json", str(out), "--device", "cpu", "--no-nc"])
    doc = json.loads(out.read_text())
    assert doc["query"] == "30-40 s energetic recap" and tuple(doc["parsed"]["budget"]) == (30.0, 40.0)
    assert doc["parser"] == "rules"  # the test session turns the LLM off (conftest)
    assert doc["clips"] and all(set(c) == {"file", "start", "end", "duration", "score", "reason", "source", "role",
                                           "item"} for c in doc["clips"])
    assert all(0 <= c["start"] < c["end"] <= 60.0 and abs(c["end"] - c["start"] - c["duration"]) < 1e-3
               for c in doc["clips"])  # file-local seconds
    order = ["a.mp4", "b.mp4", "c.mp4"]
    keys = [(order.index(c["file"]), c["start"]) for c in doc["clips"]]
    assert keys == sorted(keys) and 30.0 <= sum(c["duration"] for c in doc["clips"]) <= 40.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_export_edl_csv_rough_cut_from_two_files(tmp_path):
    from vhe.export.clips import export_clips
    from vhe.ingest.decode import probe

    a, b = tmp_path / "A001.mp4", tmp_path / "B002.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest", "-pix_fmt", "yuv420p",
                    "-metadata", "timecode=10:00:00:00", str(a)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=240x320:rate=25:duration=5",
                    "-pix_fmt", "yuv420p", str(b)], check=True)  # portrait, no audio
    clips = [{"file": "B002.mov", "file_start": 1.0, "file_end": 3.0},
             {"file": "A001.mp4", "file_start": 2.0, "file_end": 4.5}]
    out = export_clips(clips, tmp_path / "out", {"A001.mp4": a, "B002.mov": b}, with_rough_cut=True)
    assert len(out["clips"]) == 2 and out["clips"][0].name.startswith("01_B002")
    rows = list(csv.DictReader(out["csv"].open()))
    assert [(r["order"], r["file"], r["start"], r["end"], r["duration"]) for r in rows] == \
        [("1", "B002.mov", "1.000", "3.000", "2.000"), ("2", "A001.mp4", "2.000", "4.500", "2.500")]
    edl = out["edl"].read_text().splitlines()
    events = [line.split() for line in edl if line[:3].isdigit()]
    assert edl[0].startswith("TITLE:") and len(events) == 2
    assert events[0][2] == "V" and events[0][4:] == ["00:00:01:00", "00:00:03:00", "01:00:00:00", "01:00:02:00"]
    # Source times are offset by the file's embedded start timecode; the record side continues.
    assert events[1][2] == "AA/V" and events[1][4:] == ["10:00:02:00", "10:00:04:12", "01:00:02:00", "01:00:04:12"]
    assert f"* SOURCE FILE: {a}" in edl
    rc = probe(out["rough_cut"])
    assert (rc.width, rc.height) == (1280, 720) and rc.has_audio and abs(rc.duration - 4.5) < 0.3
