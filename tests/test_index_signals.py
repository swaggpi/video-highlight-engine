
import cv2
import numpy as np

from vhe.index.store import Index, save_index
from vhe.ingest.decode import recorded_at, source_fps, tiny_rate
from vhe.signals.audio import speech_mask
from vhe.signals.quality import blur_ratio, frame_quality, reason_names, usable_mask
from vhe.signals.speech import keep_segment, majority_language, whisper_spans


def test_tiny_fps_follows_source_rate():
    assert tiny_rate(source_fps({"avg_frame_rate": "25/1", "r_frame_rate": "25/1"})) == 25.0
    assert tiny_rate(source_fps({"avg_frame_rate": "30000/1001"})) == 29.97
    assert tiny_rate(source_fps({"avg_frame_rate": "120/1"})) == 60.0  # capped
    assert tiny_rate(source_fps({"avg_frame_rate": "5/1"})) == 10.0  # floored
    # VFR phone footage: unusable average -> r_frame_rate -> 30.
    assert tiny_rate(source_fps({"avg_frame_rate": "0/0", "r_frame_rate": "30/1"})) == 30.0
    assert tiny_rate(source_fps({"avg_frame_rate": "90000/1", "r_frame_rate": "24/1"})) == 24.0
    assert tiny_rate(source_fps({"avg_frame_rate": "0/0", "r_frame_rate": "0/0"})) == 30.0
    assert tiny_rate(25.0, override=30.0) == 30.0


def test_recorded_at_tags():
    apple = {"tags": {"com.apple.quicktime.creationdate": "2024-06-01T18:30:12+0200",
                      "creation_time": "2024-06-01T16:31:00.000000Z"}}
    assert recorded_at(apple, []) == "2024-06-01T16:30:12Z"
    assert recorded_at({"tags": {}}, [{"tags": {"creation_time": "2024-06-01T16:31:00.000000Z"}}]) \
        == "2024-06-01T16:31:00Z"
    assert recorded_at({"tags": {"creation_time": "1970-01-01T00:00:00.000000Z"}}, []) is None
    assert recorded_at({}, [{}]) is None


def _meta(video):
    return {"video": str(video), "fps": 3.0, "duration": 1.0, "cuts": []}


def test_reindex_removes_stale_speech_and_records_source(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"12345")
    root = tmp_path / "idx"
    grid = {"t": np.arange(3) / 3.0}
    save_index(root, _meta(video), grid, {"language": "en", "segments": []})
    assert (root / "speech.json").exists()
    save_index(root, _meta(video), grid, None)
    assert not (root / "speech.json").exists()
    idx = Index(root)
    assert idx.transcript is None
    assert idx.meta["source_size"] == 5 and idx.meta["source_mtime_ns"] == video.stat().st_mtime_ns
    assert idx.usable.all() and not idx.unusable_reasons.any()  # older indexes without a mask


def _frame(kind: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if kind == "texture":
        g = rng.normal(120, 40, (90, 160))
        return np.clip(cv2.resize(g, (640, 360), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
    if kind == "blurred":
        return cv2.GaussianBlur(_frame("texture"), (0, 0), 6)
    if kind == "sky":  # smooth gradient, in focus, sensor noise only: not blurry
        g = np.tile(np.linspace(150, 210, 360)[:, None], (1, 640)) + rng.normal(0, 1.5, (360, 640))
        return np.clip(g, 0, 255).astype(np.uint8)
    if kind == "wall":  # plain wall right in front of the lens: near-uniform, but still not blurry
        return np.clip(rng.normal(140, 3, (360, 640)), 0, 255).astype(np.uint8)
    if kind == "black":
        return np.full((360, 640), 3, np.uint8)
    if kind == "concert":  # dark but well exposed: not a defect
        g = _frame("texture").astype(np.float32) * 0.25
        return g.astype(np.uint8)
    if kind == "blown":
        g = _frame("texture").copy()
        g[:, :400] = 255
        return g
    raise ValueError(kind)


def test_usable_mask_on_synthetic_frames(tmp_path):
    kinds = ["texture", "blurred", "sky", "black", "concert", "blown", "texture", "wall"]
    paths = []
    for i, k in enumerate(kinds):
        paths.append(tmp_path / f"{i:06d}.jpg")
        cv2.imwrite(str(paths[-1]), _frame(k), [cv2.IMWRITE_JPEG_QUALITY, 90])
    q = frame_quality(paths)
    shake = np.array([0, 0, 0, 0, 0, 0, 0.8, 0])
    usable, bits = usable_mask(q["luma"], q["contrast"], q["blown"], q["blur_ratio"], shake)
    reasons = [reason_names(b) for b in bits]
    assert reasons[0] == [] and usable[0]
    assert "blurry" in reasons[1]
    assert reasons[2] == []  # sharp textureless sky
    assert "covered" in reasons[3] and "uniform" in reasons[3] and "blurry" not in reasons[3]
    assert reasons[4] == [] and q["luma"][4] < 0.15  # dark concert frame is usable
    assert reasons[5] == ["blown_out"]
    assert reasons[6] == ["shaky"]
    assert reasons[7] == ["uniform"]
    assert usable.tolist() == [True, False, True, False, True, False, False, False]
    assert blur_ratio(np.full((50, 50), 7, np.uint8)) == float("inf")


def test_low_confidence_segments_dropped():
    assert keep_segment({"no_speech_prob": 0.1, "avg_logprob": -0.3})
    assert keep_segment({"no_speech_prob": 0.9, "avg_logprob": -0.4})  # sure of the text: keep
    assert keep_segment({"no_speech_prob": 0.3, "avg_logprob": -1.5})  # probably speech: keep
    assert not keep_segment({"no_speech_prob": 0.8, "avg_logprob": -1.4})


def test_majority_language():
    dets = [("de", 0.9), ("de", 0.75), ("en", 0.99), ("fr", 0.5), (None, 0.0), ("fr", 0.6)]
    assert majority_language(dets) == "de"
    assert majority_language([("en", 0.9), ("de", 0.8)]) == "en"  # tie: higher total probability
    assert majority_language([("en", 0.5), (None, 0.0)]) is None


def test_whisper_spans_merge_pad_cap():
    spans = whisper_spans([(1.0, 2.0), (2.5, 3.0), (10.0, 11.0), (20.0, 85.0)], duration=86.0)
    assert spans[0] == (0.7, 3.3)  # merged across the 0.5 s gap, padded
    assert spans[1] == (9.7, 11.3)
    rest = spans[2:]
    assert len(rest) == 3 and rest[0][0] == 19.7 and rest[-1][1] == 85.3
    assert all(e - s <= 30.0 + 1e-6 for s, e in rest) and all(a[1] == b[0] for a, b in zip(rest, rest[1:]))
    assert whisper_spans([(0.1, 0.5)], duration=0.6) == [(0.0, 0.6)]
    # 1.2 s apart: not merged, and padding never makes spans overlap
    a, b = whisper_spans([(1.0, 2.0), (3.2, 4.0)], duration=10.0)
    assert a[1] <= b[0]


def test_speech_mask_stops_at_segment_end():
    m = speech_mask([(1.0, 2.05)], n_grid=12, grid_fps=3.0)
    assert np.flatnonzero(m).tolist() == [3, 4, 5, 6]  # t = 1.0 .. 2.0, not 2.33
