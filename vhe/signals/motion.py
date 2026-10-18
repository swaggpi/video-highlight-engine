"""Camera motion, shake and subject motion from the tiny full-frame-rate stream.

Camera shift per frame pair comes from phase correlation (a translation-only stand-in for a homography at this
resolution), run on the whole frame and on a 3x3 grid of overlapping tiles. The camera shift is the median of the
tiles with a confident correlation peak, if enough of them agree with it, so a large moving subject (a spinning
umbrella, a passer-by) in a few tiles does not read as camera motion. Without such a majority it falls back to the
whole-frame shift when that peak is confident, else it is unknown: a covered lens or a featureless frame has no
reliable shift, and a guessed one is what made such frames read as violent shake. Shake is the high-frequency part
of the camera path (jitter, not pans); subject motion is what remains after compensating the camera shift. All three
are computed at the source rate, then averaged onto the 3 fps grid.

Phase correlation is done here in numpy rather than with cv2.phaseCorrelate: in OpenCV 4.11 repeated calls on the
same frame pair return different sub-pixel shifts (and, on weak peaks, shifts of tens of pixels), so the old shake
signal was partly noise.
"""
from __future__ import annotations

import cv2
import numpy as np

from vhe.signals import to_grid

TILES = 3  # tiles per side; each tile is half the frame wide and high (on 64x36: 32x18 px, stride 16x9)
MIN_RESPONSE = 0.5  # correlation peak (3x3 sum, 1 = identical shifted content) below this: no reliable shift
AGREE_PX = 1.0  # a confident tile agrees with the tile median within this many pixels (both axes)
MIN_AGREE = 4  # agreeing tiles needed (of TILES**2) to call the median the camera shift
TREND_S, SHAKE_S = 0.5, 1.0  # jitter = velocity minus its TREND_S moving mean; shake = its RMS over SHAKE_S
_CHUNK = 512  # frame pairs per FFT batch (bounds memory on long files)


def _hann(h: int, w: int) -> np.ndarray:
    return (np.hanning(h)[:, None] * np.hanning(w)[None, :]).astype(np.float32)


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translation of b relative to a (b(x) ~ a(x - d)) by phase correlation, over the last two axes of any batch.
    Returns (d[..., 2] as (dx, dy) in pixels with parabolic sub-pixel refinement, peak response[...]). The response
    is the correlation mass in the 3x3 cells around the peak: ~1 for the same content shifted, ~0.1-0.3 for
    unrelated or featureless frames."""
    h, w = a.shape[-2:]
    win = _hann(h, w)
    fa = np.fft.rfft2((a - a.mean(axis=(-2, -1), keepdims=True)) * win)
    fb = np.fft.rfft2((b - b.mean(axis=(-2, -1), keepdims=True)) * win)
    cross = fb * np.conj(fa)
    mag = np.abs(cross)
    # Whiten, with a small floor so near-empty frequency bins do not turn sensor noise into a peak.
    cross /= mag + 1e-3 * mag.max(axis=(-2, -1), keepdims=True) + 1e-12
    corr = np.fft.irfft2(cross, s=(h, w))
    py, px = np.unravel_index(corr.reshape(*corr.shape[:-2], -1).argmax(-1), (h, w))
    lead = tuple(np.indices(corr.shape[:-2]))

    def at(dy: int, dx: int) -> np.ndarray:
        return corr[lead + ((py + dy) % h, (px + dx) % w)]

    c0 = at(0, 0)

    def sub(m: np.ndarray, p: np.ndarray) -> np.ndarray:
        den = m - 2 * c0 + p
        safe = np.where(np.abs(den) > 1e-9, den, 1.0)
        return np.clip(np.where(np.abs(den) > 1e-9, 0.5 * (m - p) / safe, 0.0), -0.5, 0.5)

    dx = np.where(px > w // 2, px - w, px) + sub(at(0, -1), at(0, 1))
    dy = np.where(py > h // 2, py - h, py) + sub(at(-1, 0), at(1, 0))
    resp = sum(np.clip(at(j, i), 0, None) for j in (-1, 0, 1) for i in (-1, 0, 1))
    return np.stack([dx, dy], -1).astype(np.float32), np.asarray(resp, dtype=np.float32)


def _tiles(x: np.ndarray, n: int = TILES) -> np.ndarray:
    """(k, h, w) -> (k, n*n, h/2, w/2): overlapping tiles on a regular n x n grid."""
    h, w = x.shape[-2:]
    th, tw = h // 2, w // 2
    ys = np.linspace(0, h - th, n).round().astype(int)
    xs = np.linspace(0, w - tw, n).round().astype(int)
    return np.stack([x[:, y:y + th, c:c + tw] for y in ys for c in xs], 1)


def camera_shift(tiny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame k, the camera shift from frame k-1 in pixels (NaN where unknown; row 0 is NaN), and the
    whole-frame shift (always defined, for motion compensation)."""
    n = len(tiny)
    cam = np.full((n, 2), np.nan, dtype=np.float32)
    whole = np.zeros((n, 2), dtype=np.float32)
    f = tiny.astype(np.float32)
    for s in range(1, n, _CHUNK):
        e = min(n, s + _CHUNK)
        a, b = f[s - 1:e - 1], f[s:e]
        g_shift, g_resp = phase_shift(a, b)
        t_shift, t_resp = phase_shift(_tiles(a), _tiles(b))
        conf = t_resp >= MIN_RESPONSE
        # Median of the confident tiles; rows without any stay NaN (no warning: nanmedian over all-NaN is masked).
        masked = np.where(conf[..., None], t_shift, np.nan)
        med = np.full((e - s, 2), np.nan, dtype=np.float32)
        any_conf = conf.any(1)
        med[any_conf] = np.nanmedian(masked[any_conf], axis=1)
        agree = conf & (np.abs(t_shift - med[:, None, :]).max(-1) <= AGREE_PX)
        ok_tiles = agree.sum(1) >= MIN_AGREE
        ok_whole = g_resp >= MIN_RESPONSE
        cam[s:e] = np.where(ok_tiles[:, None], med, np.where(ok_whole[:, None], g_shift, np.nan))
        whole[s:e] = g_shift
    return cam, whole


def nan_smooth(x: np.ndarray, width: float) -> np.ndarray:
    """Centred moving mean over `width` samples ignoring NaN; NaN where fewer than half of the window is finite.
    The edges repeat the end values, like signals.smooth."""
    x = np.asarray(x, dtype=float)
    width = max(1, min(int(width), len(x)))
    ok = np.isfinite(x)
    pad, k = width // 2, np.ones(width)
    s = np.convolve(np.pad(np.where(ok, x, 0.0), pad, mode="edge"), k, mode="valid")[:len(x)]
    c = np.convolve(np.pad(ok.astype(float), pad, mode="edge"), k, mode="valid")[:len(x)]
    return np.where(c >= 0.5 * width, s / np.maximum(c, 1.0), np.nan)


def motion_signals(tiny: np.ndarray, src_fps: float, n_grid: int, grid_fps: float,
                   cuts: list[float]) -> dict[str, np.ndarray]:
    n, h, w = tiny.shape
    cam, whole = camera_shift(tiny)

    # Frames at a cut compare two different shots; drop them.
    bad = np.zeros(n, dtype=bool)
    bad[0] = True
    for c in cuts:
        k = int(round(c * src_fps))
        bad[max(0, k - 1):min(n, k + 2)] = True
    cam[bad] = np.nan

    resid = np.full(n, np.nan, dtype=np.float32)
    comp = np.where(np.isfinite(cam), cam, whole)  # best available compensation for the subject-motion residual
    for k in range(1, n):
        if bad[k]:
            continue
        m = np.float32([[1, 0, comp[k, 0]], [0, 1, comp[k, 1]]])
        warped = cv2.warpAffine(tiny[k - 1].astype(np.float32), m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        resid[k] = np.abs(tiny[k].astype(np.float32) - warped).mean() / 255.0

    rel = cam / np.array([w, h], dtype=np.float32)  # fraction of the frame per source frame
    cam_speed = np.hypot(rel[:, 0], rel[:, 1]) * src_fps  # frame-widths per second; NaN where unknown
    trend = np.stack([nan_smooth(rel[:, i], src_fps * TREND_S) for i in range(2)], axis=1)
    jitter = np.hypot(*(rel - trend).T) * src_fps
    shake = np.sqrt(nan_smooth(jitter ** 2, src_fps * SHAKE_S))

    times = np.arange(n) / src_fps
    return {
        "cam_speed": to_grid(times, cam_speed, n_grid, grid_fps),
        "shake": to_grid(times, shake, n_grid, grid_fps),
        "motion": to_grid(times, resid, n_grid, grid_fps),
    }
