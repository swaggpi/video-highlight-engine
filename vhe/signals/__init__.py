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


def fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        good = np.flatnonzero(~bad)
        x[bad] = np.interp(np.flatnonzero(bad), good, x[good])
    return x
