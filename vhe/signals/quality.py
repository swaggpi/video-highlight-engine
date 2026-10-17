"""Cheap per-frame technical quality: sharpness, exposure and clipping. A trained aesthetic head on the
cached embeddings replaces this in v2 (Q-ReAlign needs legal review; Apple Vision covers macOS)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def frame_quality(paths: list[Path]) -> dict[str, np.ndarray]:
    sharp = np.zeros(len(paths), dtype=np.float32)
    exposure = np.zeros(len(paths), dtype=np.float32)
    clipped = np.zeros(len(paths), dtype=np.float32)
    for i, p in enumerate(paths):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        sharp[i] = np.log1p(cv2.Laplacian(g, cv2.CV_32F).var())
        mean = g.mean() / 255.0
        exposure[i] = 1.0 - min(1.0, abs(mean - 0.45) / 0.45)  # 1 = well exposed, 0 = black or white
        clipped[i] = ((g < 5) | (g > 250)).mean()
    return {"sharpness": sharp, "exposure": exposure, "clipped": clipped}
