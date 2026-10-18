import itertools

import numpy as np

from vhe.signals import fill_nan, to_grid
from vhe.signals.changes import change_points
from vhe.signals.highlight_nc import window_starts
from vhe.signals.shots import detect_cuts, shots_from_cuts


def _tiny(levels, frames_each=30):
    return np.concatenate([np.full((frames_each, 36, 64), lv, np.uint8) for lv in levels])


def test_detect_cuts_hard_cuts():
    rng = np.random.default_rng(0)
    tiny = _tiny([20, 200, 60]) + rng.integers(0, 3, (90, 36, 64), dtype=np.uint8)
    cuts = detect_cuts(tiny, src_fps=30.0)
    assert cuts == [1.0, 2.0]


def test_detect_cuts_ignores_flash():
    tiny = _tiny([50], 60)
    tiny[30] = 250  # one-frame flash
    assert detect_cuts(tiny, src_fps=30.0) == []


def test_shots_from_cuts():
    assert shots_from_cuts([5.0, 12.0], 20.0) == [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)]


def test_to_grid_and_fill():
    times = np.array([0.0, 0.1, 1.0])
    vals = np.array([1.0, 3.0, 5.0])
    g = to_grid(times, vals, n=4, fps=3.0)
    assert g[0] == 2.0 and g[3] == 5.0 and np.isfinite(g).all()
    assert np.all(fill_nan(np.array([np.nan, np.nan])) == 0)


def test_window_starts_cover_video():
    starts = window_starts(400.0, 148.0, 30.0)
    assert starts[0] == 0.0 and abs(starts[-1] + 148.0 - 400.0) < 1e-6
    assert all(b - a <= 118.0 + 1e-6 for a, b in itertools.pairwise(starts))
    assert window_starts(100.0, 148.0, 30.0) == [0.0]


def test_window_starts_slightly_longer_than_one_window():
    starts = window_starts(150.0, 148.0, 30.0)  # used to be [2.0]: the first 2 s were never scored
    assert starts == [0.0, 2.0]
    assert window_starts(270.0, 148.0, 30.0) == [0.0, 122.0]  # near-duplicate last window still merged


def test_change_points_finds_content_switch():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=16), rng.normal(size=16)
    emb = np.concatenate([np.tile(a, (30, 1)), np.tile(b, (30, 1))]) + rng.normal(scale=0.01, size=(60, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cps = change_points(emb.astype(np.float16), fps=3.0, cuts=[])
    assert len(cps) == 1 and abs(cps[0] - 10.0) < 0.5
