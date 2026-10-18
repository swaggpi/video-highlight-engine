"""Usable-mask defects on synthetic input: camera shake vs subject motion (motion.py), and the zero-shot obstruction
and framing margins (tags.defect_scores) with mocked embeddings."""
import cv2
import numpy as np

from vhe.config import BAD_FRAMING_MARGIN, OBSTRUCTED_MARGIN, SHAKY
from vhe.rank.signals import timestep_signals
from vhe.signals.motion import motion_signals, phase_shift
from vhe.signals.quality import reason_names, usable_mask
from vhe.signals.tags import (
    BAD_FRAMING_PROMPTS,
    OBSTRUCTED_PROMPTS,
    SCENE_PROMPTS,
    defect_scores,
    defect_texts,
)

FPS, N = 24.0, 96  # 4 s of tiny stream


def _texture(h=300, w=400, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random((h, w)).astype(np.float32) * 255, (0, 0), 1.5)


def _stream(offsets, subject=None):
    """64x36 crops of a textured scene at per-frame camera offsets (x, y); subject: per-frame (x, y) of a separately
    textured 18x14 patch pasted on top (a foreground object moving independently of the camera)."""
    scene, obj = _texture(), _texture(14, 18, seed=1)
    out = np.empty((len(offsets), 36, 64), np.uint8)
    for k, (ox, oy) in enumerate(offsets):
        f = scene[100 + oy:136 + oy, 100 + ox:164 + ox].copy()
        if subject is not None:
            sx, sy = subject[k]
            f[sy:sy + 14, sx:sx + 18] = obj
        out[k] = np.clip(f, 0, 255)
    return out


def _shake(tiny):
    n_grid = int(len(tiny) / FPS * 3)
    return motion_signals(tiny, FPS, n_grid, 3.0, [])


def test_phase_shift_is_exact_and_deterministic():
    scene = _texture()
    a, b = scene[50:86, 50:114], scene[51:87, 48:112]  # content moves +2 px in x, -1 px in y
    first = phase_shift(a, b)
    d, resp = first
    assert np.allclose(d, [2, -1], atol=0.15) and resp > 0.7
    for _ in range(3):  # cv2.phaseCorrelate (OpenCV 4.11) drifts on repeated calls; this must not
        again = phase_shift(a, b)
        assert np.array_equal(again[0], first[0]) and again[1] == first[1]


def test_steady_camera_with_moving_subject_is_not_shake():
    """A foreground object jumping around in one tile (a passer-by close to the lens) does not move the camera
    estimate: it is the median of the 3x3 tiles. (A subject filling most tiles with one consistent translation
    would still read as camera motion; that is the limit of a translation-only model.)"""
    rng = np.random.default_rng(3)
    subject = [(int(rng.integers(0, 5)), int(rng.integers(0, 5))) for _ in range(N)]
    m = _shake(_stream([(0, 0)] * N, subject))
    assert np.nanmax(m["shake"]) < 0.1 * SHAKY
    assert np.nanmax(m["cam_speed"]) < 0.05
    still = _shake(_stream([(0, 0)] * N))
    assert np.nanmean(m["motion"]) > 5 * np.nanmean(still["motion"]) + 1e-3  # still measured as subject motion


def test_global_jitter_is_shake():
    rng = np.random.default_rng(4)
    offsets = [(int(x), int(y)) for x, y in rng.integers(-3, 4, (N, 2))]
    m = _shake(_stream(offsets))
    assert np.median(m["shake"]) > SHAKY


def test_smooth_pan_is_not_shake():
    m = _shake(_stream([(k, 0) for k in range(N)]))  # 1 px per frame = 0.375 frame-widths per second
    assert np.max(m["shake"][1:-1]) < 0.2 * SHAKY
    assert np.allclose(np.median(m["cam_speed"]), 24 / 64, atol=0.03)


def test_featureless_frames_are_not_shake():
    """A covered lens (flat frame plus sensor noise) has no reliable shift; it must not read as violent shake."""
    rng = np.random.default_rng(5)
    tiny = np.clip(rng.normal(120, 2, (N, 36, 64)), 0, 255).astype(np.uint8)
    m = _shake(tiny)
    assert np.nanmax(m["shake"]) < SHAKY


def test_defect_scores_with_mocked_embeddings():
    # Orthonormal mock text embeddings, one per defect_texts() prompt, plus a direction orthogonal to all of them.
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(64, 64)))
    te, other = q.T[:len(defect_texts())].astype(np.float32), q.T[-1]
    no, nf = len(OBSTRUCTED_PROMPTS), len(BAD_FRAMING_PROMPTS)
    assert len(te) == no + nf + len(SCENE_PROMPTS)

    def frame(i, w=0.3):  # unit image embedding with cosine w to prompt i and 0 to every other prompt
        return w * te[i] + np.sqrt(1 - w * w) * other

    emb = np.stack([frame(0), frame(no + 1), frame(no + nf + 2), frame(no + nf)]).astype(np.float16)
    s = defect_scores(emb, te, scale=112.9, bias=-16.8)
    # covered lens / pocket shot / normal scene / normal scene
    assert s["obstruction"][0] > OBSTRUCTED_MARGIN and s["framing"][0] < BAD_FRAMING_MARGIN
    assert s["framing"][1] > BAD_FRAMING_MARGIN and s["obstruction"][1] < OBSTRUCTED_MARGIN
    for i in (2, 3):
        assert s["obstruction"][i] < OBSTRUCTED_MARGIN and s["framing"][i] < BAD_FRAMING_MARGIN
    # the bias cancels in the margin
    s2 = defect_scores(emb, te, scale=112.9, bias=5.0)
    assert np.allclose(s["obstruction"], s2["obstruction"], atol=1e-3)


def test_obstructed_is_hard_and_bad_framing_soft():
    n = 4
    ok = (np.full(n, 0.4), np.full(n, 0.2), np.zeros(n), np.full(n, 18.0), np.zeros(n))  # luma..shake: no defect
    obstruction = np.array([OBSTRUCTED_MARGIN + 1, -5, -5, np.nan])
    framing = np.array([-5, BAD_FRAMING_MARGIN + 1, -5, np.nan])
    usable, bits = usable_mask(*ok, obstruction, framing)
    assert [reason_names(b) for b in bits] == [["obstructed"], ["bad_framing"], [], []]
    assert usable.tolist() == [False, True, True, True]
    # older call signature (no defect scores) still works and never sets the new bits
    usable5, bits5 = usable_mask(*ok)
    assert usable5.all() and not bits5.any()


def test_bad_framing_lowers_quality_only():
    n = 40
    rng = np.random.default_rng(6)
    grid = {"t": np.arange(n) / 3.0, "blur_ratio": 15 + rng.random(n), "blown": np.zeros(n),
            "shake": 0.05 * rng.random(n), "motion": rng.random(n), "cam_speed": rng.random(n),
            "loudness": rng.random(n), "onset": rng.random(n), "speech": np.zeros(n), "usable": np.ones(n, bool),
            "unusable": np.zeros(n, np.uint8)}
    base = timestep_signals(grid, 3.0)
    grid["unusable"][10:20] = 1 << 6  # bad_framing
    framed = timestep_signals(grid, 3.0)
    assert np.all(framed["quality"][10:20] < base["quality"][10:20] - 0.5)
    assert np.array_equal(framed["quality"][:10], base["quality"][:10])
    assert np.array_equal(framed["energy"], base["energy"])
