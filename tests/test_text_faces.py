"""Index-time text, face, shot-scale and approach signals (index version 3), on synthetic data."""
import json

import cv2
import numpy as np
import pytest

from vhe import models
from vhe.config import Config
from vhe.corpus import Corpus
from vhe.index.store import INDEX_VERSION, Index, save_index
from vhe.signals.faces import approach, face_stats
from vhe.signals.ocr import confident, ocr_frames, text_area
from vhe.signals.tags import CLOSE_PROMPTS, WIDE_PROMPTS, scale_scores, scale_texts


def _frames(tmp_path, n, h=90, w=160):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i + 1:06d}.jpg"
        cv2.imwrite(str(p), np.full((h, w, 3), 128, np.uint8))
        paths.append(p)
    return paths


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class FakeEngine:
    """Returns a scripted detection list per call, records how often it was called."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def __call__(self, img):
        out = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return out


def test_ocr_reads_every_third_step_and_carries_to_neighbours(tmp_path):
    paths = _frames(tmp_path, 7)  # read steps 0, 3, 6
    eng = FakeEngine([
        [(_box(0, 0, 80, 45), "NOWRUZ", 0.95), (_box(0, 0, 10, 10), "x", 0.99), (_box(0, 50, 20, 60), "ab", 0.5)],
        [],
        [(_box(0, 0, 160, 45), "GATE OF THE NATIONS", 0.9), (_box(0, 0, 160, 45), "GATE", 0.8)],
    ])
    sig, items = ocr_frames(paths, eng, every=3)
    assert eng.calls == 3
    # Only "NOWRUZ" is confident and long enough at step 0: a quarter of the frame.
    np.testing.assert_allclose(sig["text_area"], [0.25, 0.25, 0, 0, 0, 0.5, 0.5], atol=0.02)
    np.testing.assert_allclose(sig["text_conf"], [0.95, 0.95, 0, 0, 0, 0.85, 0.85], atol=1e-6)
    assert sig["text_area"].dtype == np.float32
    assert items[0] == [0, "NOWRUZ", 0.95, [0.0, 0.0, 0.5, 0.5]]
    assert [it[0] for it in items] == [0, 6, 6] and items[1][1] == "GATE OF THE NATIONS"


def test_confident_filter_and_union_area():
    dets = [(_box(0, 0, 50, 50), " AB ", 0.7), (_box(0, 0, 50, 50), "A B", 0.9), (_box(0, 0, 9, 9), "A", 0.99),
            (_box(0, 0, 9, 9), "LONG", 0.69)]
    kept = confident(dets)
    assert [t for _, t, _ in kept] == ["AB", "A B"]  # >= 0.7, >= 2 non-space characters, stripped
    assert text_area(kept, 100, 100) == pytest.approx(0.2601, abs=0.01)  # overlapping boxes count once
    assert text_area([], 100, 100) == 0.0


def _meta(video, n):
    return {"video": str(video), "fps": 3.0, "duration": n / 3.0, "cuts": [], "encoder": "e"}


def _grid(n, seed=0):
    rng = np.random.default_rng(seed)
    return {"t": np.arange(n) / 3.0, "text_area": rng.random(n).astype(np.float32),
            "text_conf": rng.random(n).astype(np.float32), "face_n": rng.integers(0, 4, n).astype(np.int16),
            "face_area_max": rng.random(n).astype(np.float32), "face_center": rng.random(n).astype(np.float32),
            "scale_close": rng.normal(size=n).astype(np.float32), "scale_wide": rng.normal(size=n).astype(np.float32),
            "approach": rng.normal(size=n).astype(np.float32)}


def test_text_store_round_trip_and_corpus(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    ocr = {"engine": "fake", "every": 3, "min_conf": 0.7,
           "items": [[0, "NOWRUZ", 0.95, [0.0, 0.0, 0.5, 0.5]], [6, "Ünïcode SHIRAZ", 0.8, [0.1, 0.2, 0.3, 0.4]]]}
    a, b = tmp_path / "a", tmp_path / "b"
    save_index(a, _meta(video, 9), _grid(9), None, ocr)
    save_index(b, _meta(video, 6), _grid(6, 1), None, {**ocr, "items": [[3, "PASARGAD", 0.9, [0, 0, 1, 1]]]})
    stored = json.loads((a / "ocr.json").read_text())
    assert stored["version"] == 1 and stored["every"] == 3
    ia = Index(a)
    assert "ocr" not in ia.__dict__  # lazy: not read until used
    assert ia.text_items[1] == {"step": 6, "t": 2.0, "text": "Ünïcode SHIRAZ", "conf": 0.8, "box": [0.1, 0.2, 0.3, 0.4]}
    for key in ("text_area", "text_conf", "face_n", "face_area_max", "face_center", "scale_close", "scale_wide",
                "approach"):
        assert key in ia.grid
    c = Corpus([ia, Index(b)])
    assert c.text_items[-1] == {"step": 12, "t": 4.0, "text": "PASARGAD", "conf": 0.9, "box": [0, 0, 1, 1], "file": 1}
    assert c.grid["face_n"].dtype == np.int16 and len(c.grid["approach"]) == 15
    np.testing.assert_array_equal(c.grid["text_area"][9:], Index(b).grid["text_area"])
    # Re-saving without OCR removes a stale text store.
    save_index(a, _meta(video, 9), _grid(9), None)
    assert not (a / "ocr.json").exists() and Index(a).text_items == []


def test_corpus_fills_missing_new_keys(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    save_index(tmp_path / "a", _meta(video, 6), _grid(6), None)
    save_index(tmp_path / "b", _meta(video, 3), {"t": np.arange(3) / 3.0}, None)  # OCR and faces skipped
    c = Corpus([Index(tmp_path / "a"), Index(tmp_path / "b")])
    assert np.isnan(c.grid["text_area"][6:]).all() and (c.grid["face_n"][6:] == 0).all()
    assert c.text_items == []


def test_old_index_version_refused(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    save_index(tmp_path, _meta(video, 3), {"t": np.arange(3) / 3.0}, None)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["version"] == INDEX_VERSION == 3
    (tmp_path / "meta.json").write_text(json.dumps({**meta, "version": 2}))
    with pytest.raises(ValueError, match="re-index"):
        Index(tmp_path)


def test_scale_margins_with_mocked_embeddings():
    texts = scale_texts()
    d = len(texts) + 2
    text_emb = np.eye(len(texts), d, dtype=np.float32)  # each prompt its own axis
    close_i, wide_i = texts.index(CLOSE_PROMPTS[0]), texts.index(WIDE_PROMPTS[0])
    emb = np.zeros((3, d), np.float32)
    emb[0, close_i] = 1.0  # a close-up
    emb[1, wide_i] = 1.0  # a wide establishing view
    emb[2, d - 1] = 1.0  # neither
    s = scale_scores(emb, text_emb, scale=10.0, bias=-5.0)
    assert s["scale_close"].dtype == np.float32
    assert s["scale_close"][0] > 5 and s["scale_wide"][0] < 1
    assert s["scale_wide"][1] > 5 and s["scale_close"][1] < 1
    assert s["scale_close"][2] < 1 and s["scale_wide"][2] < 1
    # Bias cancels in the margin.
    s2 = scale_scores(emb, text_emb, scale=10.0, bias=3.0)
    np.testing.assert_allclose(s["scale_close"], s2["scale_close"], atol=1e-5)


def test_face_stats_area_and_centre():
    rows = np.zeros((3, 15), np.float32)
    rows[0, :4], rows[0, 14] = (70, 40, 20, 20), 0.9  # centred, 400 px of 16000
    rows[1, :4], rows[1, 14] = (0, 0, 10, 10), 0.95  # corner
    rows[2, :4], rows[2, 14] = (0, 0, 100, 80), 0.5  # big but below the score threshold
    n, area, centre = face_stats(rows, 100, 160)
    assert n == 2 and area == pytest.approx(400 / 16000) and centre == pytest.approx(1.0)
    n, area, centre = face_stats(rows[1:2], 100, 160)
    assert centre < 0.35
    assert face_stats(np.zeros((0, 15), np.float32), 100, 160) == (0, 0.0, 0.0)


def test_yunet_on_synthetic_image():
    try:
        det = models.yunet(0.7)
    except (FileNotFoundError, OSError) as exc:  # not cached and downloads are off in tests
        pytest.skip(f"YuNet model not available: {exc}")
    from vhe.signals.faces import detect

    blank = np.full((360, 640, 3), 127, np.uint8)
    assert detect(det, blank).shape == (0, 15)
    big = np.full((720, 1280, 3), 127, np.uint8)  # downscaled to 640 px internally, coordinates mapped back
    assert detect(det, big).shape == (0, 15)


def test_approach_on_growing_and_shrinking_series():
    fps = 3.0
    t = np.arange(30) / fps
    growing = (0.05 * np.exp(0.5 * t)).astype(np.float32)  # face area doubles every ~1.4 s
    ap = approach(growing, None, fps)
    assert ap.dtype == np.float32
    inner = ap[3:-3]
    assert ((inner > 0.45) & (inner < 0.501)).all()  # log area per second (eps is small against the area)
    assert ap[0] == 0 and ap[-1] == 0  # file edges
    assert (approach(growing[::-1].copy(), None, fps)[3:-3] < -0.45).all()
    # A cut at 5 s: windows across it read 0.
    cut = approach(growing, None, fps, cuts=[5.0])
    assert (cut[13:17] == 0).all() and cut[10] > 0.45
    # Tiny faces (below APPROACH_FACE_MIN) are not growth.
    assert (approach(np.linspace(0.0001, 0.002, 30).astype(np.float32), None, fps) == 0).all()
    # No face: the close-up margin rising into close-up territory counts, its jitter in wide shots does not.
    rising = np.linspace(-2, 4, 30).astype(np.float32)  # 6/29 logits per step
    ap = approach(np.zeros(30, np.float32), rising, fps)
    assert ap[25] == pytest.approx(0.35 * 6 / 29 * fps, rel=0.02) and ap[5] == 0  # still below the close gate at 5
    jitter = (-5 + np.random.default_rng(0).normal(0, 1, 30)).astype(np.float32)
    assert (approach(np.zeros(30, np.float32), jitter, fps) == 0).all()


def test_pipeline_skips_missing_ocr_and_faces(tmp_path, monkeypatch, caplog):
    from vhe import pipeline
    from vhe.signals import faces, ocr
    from vhe.timing import Timings

    def no_ocr(*a, **k):
        raise ImportError("No module named 'rapidocr_onnxruntime'")

    def no_model(*a, **k):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(ocr, "load_engine", no_ocr)
    monkeypatch.setattr(faces, "load_detector", no_model)
    grid, said = {}, []
    with caplog.at_level("WARNING", logger="vhe"):
        assert pipeline._ocr(_frames(tmp_path, 3), grid, Config(), said.append, Timings()) is None
        assert pipeline._faces(_frames(tmp_path, 3), grid, Config(), said.append, Timings()) is None
    assert grid == {} and len(caplog.records) == 2 and "skipping on-screen text" in said[0]
    # Switched off: not even attempted.
    assert pipeline._ocr([], grid, Config(use_ocr=False), said.append, Timings()) is None


def test_pipeline_ocr_stage_with_fake_engine(tmp_path, monkeypatch):
    from vhe import pipeline
    from vhe.signals import ocr
    from vhe.timing import Timings

    monkeypatch.setattr(ocr, "load_engine", lambda h: FakeEngine([[(_box(0, 0, 16, 9), "SHIRAZ", 0.99)]]))
    grid = {}
    store = pipeline._ocr(_frames(tmp_path, 4), grid, Config(), lambda m: None, Timings())
    assert store["every"] == 3 and store["items"] == [[0, "SHIRAZ", 0.99, [0.0, 0.0, 0.1, 0.1]]]
    assert grid["text_area"][1] == pytest.approx(0.01, abs=0.005) and grid["text_area"][2] == 0
