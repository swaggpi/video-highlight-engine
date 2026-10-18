"""Content changes inside shots: raw footage is often one long take, so cuts alone do not segment it."""
from __future__ import annotations

import numpy as np

from vhe.signals import robust_z


def change_points(emb: np.ndarray, fps: float, cuts: list[float], half_window_s: float = 1.0,
                  z_thresh: float = 2.5, min_gap_s: float = 2.0, min_dist: float = 0.05) -> list[float]:
    """Peaks of the embedding distance between the second before and the second after each step."""
    k = max(1, int(round(half_window_s * fps)))
    e = emb.astype(np.float32)
    n = len(e)
    if n < 2 * k + 2:
        return []
    c = np.concatenate([np.zeros((1, e.shape[1]), np.float32), np.cumsum(e, axis=0)])
    i = np.arange(k, n - k)
    before = c[i] - c[i - k]
    after = c[i + k] - c[i]
    cos = (before * after).sum(1) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) + 1e-9)
    dist = 1.0 - cos
    z = robust_z(dist, clip=np.inf)
    cut_arr = np.asarray(cuts)
    out: list[float] = []
    for j in np.argsort(-z):
        if z[j] < z_thresh:
            break
        if dist[j] < min_dist:  # relative outliers in a static take are noise, not content changes
            continue
        t = float(i[j] / fps)
        if cut_arr.size and np.min(np.abs(cut_arr - t)) < 1.0:
            continue
        if any(abs(t - o) < min_gap_s for o in out):
            continue
        out.append(round(t, 3))
    return sorted(out)
