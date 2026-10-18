"""Per-video signals. Everything is resampled onto the shared grid t_i = i / fps."""
from __future__ import annotations

import numpy as np


def to_grid(times: np.ndarray, values: np.ndarray, n: int, fps: float) -> np.ndarray:
    """Mean of `values` within each grid bin [t_i - 0.5/fps, t_i + 0.5/fps); empty bins are interpolated."""
    idx = np.round(times * fps).astype(int)
    ok = np.isfinite(values) & (idx >= 0) & (idx < n)  # samples past the last frame are dropped, not folded in
    sums = np.bincount(idx[ok], weights=values[ok], minlength=n)
    counts = np.bincount(idx[ok], minlength=n)
    out = np.full(n, np.nan)
    has = counts > 0
    out[has] = sums[has] / counts[has]
    return fill_nan(out)


def smooth(x: np.ndarray, width: float) -> np.ndarray:
    """Centred moving average over `width` samples; the edges repeat the end values instead of fading to zero."""
    x = np.asarray(x, dtype=float)
    width = max(1, min(int(width), len(x)))
    if width <= 1:
        return x
    pad = width // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(width) / width, mode="valid")[:len(x)]


def robust_z(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    med = np.median(x[finite])
    mad = np.median(np.abs(x[finite] - med)) * 1.4826
    if mad < 1e-9:
        mad = np.std(x[finite])
    if mad < 1e-9:
        return np.zeros_like(x)
    z = (x - med) / mad
    z[~finite] = 0.0
    return np.clip(z, -clip, clip)


def fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        good = np.flatnonzero(~bad)
        x[bad] = np.interp(np.flatnonzero(bad), good, x[good])
    return x
