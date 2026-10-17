"""Camera motion, shake and subject motion from the tiny full-frame-rate stream.

Global shift per frame pair comes from phase correlation (a translation-only stand-in for a homography at
this resolution). Shake is the high-frequency part of that shift; subject motion is what remains after
compensating for it. All three are computed at the source rate, then averaged onto the 3 fps grid.
"""
from __future__ import annotations

import cv2
import numpy as np

from vhe.signals import to_grid


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(int(width), len(x)))  # mode="same" returns max(len, width) samples otherwise
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def motion_signals(tiny: np.ndarray, src_fps: float, n_grid: int, grid_fps: float,
                   cuts: list[float]) -> dict[str, np.ndarray]:
    n, h, w = tiny.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift = np.zeros((n, 2), dtype=np.float32)
    resid = np.zeros(n, dtype=np.float32)
    prev = tiny[0].astype(np.float32)
    for k in range(1, n):
        cur = tiny[k].astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
        shift[k] = (dx, dy)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(prev, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        resid[k] = np.abs(cur - warped).mean() / 255.0
        prev = cur

    # Frames at a cut compare two different shots; drop them.
    bad = np.zeros(n, dtype=bool)
    bad[0] = True
    for c in cuts:
        k = int(round(c * src_fps))
        bad[max(0, k - 1):min(n, k + 2)] = True
    shift[bad] = np.nan
    resid[bad] = np.nan

    rel = shift / np.array([w, h], dtype=np.float32)  # fraction of the frame per source frame
    rel = np.nan_to_num(rel)
    cam_speed = np.hypot(rel[:, 0], rel[:, 1]) * src_fps  # frame-widths per second
    smooth = np.stack([_moving_average(rel[:, i], src_fps * 0.5) for i in range(2)], axis=1)
    jitter = np.hypot(*(rel - smooth).T) * src_fps
    shake = np.sqrt(_moving_average(jitter ** 2, src_fps))

    times = np.arange(n) / src_fps
    resid_f = np.where(bad, np.nan, resid)
    return {
        "cam_speed": to_grid(times, np.where(bad, np.nan, cam_speed), n_grid, grid_fps),
        "shake": to_grid(times, np.where(bad, np.nan, shake), n_grid, grid_fps),
        "motion": to_grid(times, resid_f, n_grid, grid_fps),
    }
