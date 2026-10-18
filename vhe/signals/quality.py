"""Cheap per-frame technical quality: sharpness, exposure and clipping, plus the absolute "usable" mask for raw
footage. A trained aesthetic head on the cached embeddings replaces this in v2 (Q-ReAlign needs legal review;
Apple Vision covers macOS)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vhe.config import (
    BAD_FRAMING_MARGIN,
    BLOWN_FRAC,
    BLUR_RATIO,
    BLUR_SIDE,
    COVERED_LUMA,
    OBSTRUCTED_MARGIN,
    SHAKY,
    UNIFORM_STD,
)

# Bit i of grid["unusable"] set = reason REASONS[i] applies at that step. New reasons are appended, so older indexes
# decode the same. SOFT reasons are recorded but do not make a step unusable (vhe.rank.signals lowers quality instead).
REASONS = ("covered", "uniform", "blown_out", "blurry", "shaky", "obstructed", "bad_framing")
SOFT = ("bad_framing",)
SOFT_BITS = sum(1 << REASONS.index(r) for r in SOFT)


def blur_ratio(g: np.ndarray, side: int = BLUR_SIDE) -> float:
    """var(Lap(g)) / var(Lap(blur(g))) at a fixed short side of `side` px. Re-blurring removes most of the fine
    detail a sharp frame has and little of a blurry one's, whatever the amount of texture, so a sharp plain wall
    is not "blurry". The downscale makes it resolution-independent and averages away JPEG block and sensor noise,
    which otherwise reads as detail in heavily blurred frames. No detail at all is inf (the uniform test covers
    it). Sharp real footage reads ~13-26, defocused ~3-11; see BLUR_RATIO in config.py for the calibration."""
    g = g.astype(np.float32)
    f = side / min(g.shape)
    if f < 1:
        g = cv2.resize(g, (max(1, round(g.shape[1] * f)), max(1, round(g.shape[0] * f))), interpolation=cv2.INTER_AREA)
    v = cv2.Laplacian(g, cv2.CV_32F).var()
    vb = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F).var()
    return float(v / vb) if vb > 1e-6 else float("inf")


def frame_quality(paths: list[Path]) -> dict[str, np.ndarray]:
    n = len(paths)
    sharp, exposure, clipped = (np.zeros(n, dtype=np.float32) for _ in range(3))
    luma, contrast, blown = (np.zeros(n, dtype=np.float32) for _ in range(3))
    ratio = np.full(n, np.inf, dtype=np.float32)
    for i, p in enumerate(paths):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        sharp[i] = np.log1p(cv2.Laplacian(g, cv2.CV_32F).var())
        mean = g.mean() / 255.0
        exposure[i] = 1.0 - min(1.0, abs(mean - 0.45) / 0.45)  # 1 = well exposed, 0 = black or white
        clipped[i] = ((g < 5) | (g > 250)).mean()
        luma[i], contrast[i], blown[i] = mean, g.std() / 255.0, (g >= 250).mean()
        ratio[i] = blur_ratio(g)
    return {"sharpness": sharp, "exposure": exposure, "clipped": clipped,
            "luma": luma, "contrast": contrast, "blown": blown, "blur_ratio": ratio}


def usable_mask(luma: np.ndarray, contrast: np.ndarray, blown: np.ndarray, ratio: np.ndarray,
                shake: np.ndarray, obstruction: np.ndarray | None = None,
                framing: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(usable bool[n], reason bits uint8[n]) from absolute thresholds in config.py. obstruction / framing: the
    SigLIP margins from tags.defect_scores (None = not tested). Usable = no hard reason (SOFT ones are allowed)."""
    n = len(luma)

    def above(x: np.ndarray | None, thr: float) -> np.ndarray:  # NaN / missing: not flagged
        return np.zeros(n, dtype=bool) if x is None else np.nan_to_num(np.asarray(x, dtype=float), nan=-np.inf) > thr

    tests = (luma < COVERED_LUMA, contrast < UNIFORM_STD, blown > BLOWN_FRAC, ratio < BLUR_RATIO,
             above(shake, SHAKY), above(obstruction, OBSTRUCTED_MARGIN), above(framing, BAD_FRAMING_MARGIN))
    bits = np.zeros(n, dtype=np.uint8)
    for i, t in enumerate(tests):
        bits |= np.asarray(t, dtype=np.uint8) << i
    return (bits & ~np.uint8(SOFT_BITS)) == 0, bits


def reason_names(bits: int) -> list[str]:
    """The REASONS set in one step's bits."""
    return [r for i, r in enumerate(REASONS) if int(bits) >> i & 1]


def reason_mask(bits: np.ndarray, *names: str) -> np.ndarray:
    """bool per step: any of the named REASONS set in `bits` (grid["unusable"])."""
    return (np.asarray(bits, dtype=np.uint8) & np.uint8(sum(1 << REASONS.index(r) for r in names))) > 0
