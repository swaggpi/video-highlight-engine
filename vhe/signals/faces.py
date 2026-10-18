"""Faces (OpenCV Zoo YuNet, MIT) on every cached analysis frame, and the approach / looming signal.

Per step: `face_n` (faces at or above FACE_MIN_SCORE), `face_area_max` (largest face box as a fraction of the frame)
and `face_center` (how centred that largest face is: 1 = frame centre, 0 = a corner; 0 when there is no face).

`approach`: positive while the dominant subject grows (a face, or whatever SigLIP calls a close-up, coming at the
lens), negative while it recedes, 0 across a cut. See approach() for the definition.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vhe.config import (
    APPROACH_CLOSE_MIN,
    APPROACH_CLOSE_WEIGHT,
    APPROACH_FACE_MIN,
    APPROACH_WINDOW_S,
    FACE_AREA_EPS,
    FACE_MIN_SCORE,
    FACE_SIDE,
)

MODEL_NAME = "opencv-zoo face_detection_yunet_2023mar"


def load_detector(score_threshold: float = FACE_MIN_SCORE):
    """cv2.FaceDetectorYN; fetches the model into the models cache once (checksum-verified)."""
    from vhe import models

    return models.yunet(score_threshold)


def detect(detector, img: np.ndarray, side: int = FACE_SIDE) -> np.ndarray:
    """(k, 15) YuNet rows (x, y, w, h, 5 landmarks, score) in the coordinates of `img`."""
    h, w = img.shape[:2]
    f = min(1.0, side / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return np.zeros((0, 15), np.float32)
    faces = np.asarray(faces, np.float32).copy()
    faces[:, :14] /= f
    return faces


def face_stats(faces: np.ndarray, h: int, w: int, min_score: float = FACE_MIN_SCORE) -> tuple[int, float, float]:
    """(count, largest face area fraction, centredness of the largest face) for one frame's YuNet rows."""
    faces = faces[faces[:, 14] >= min_score] if len(faces) else faces
    if not len(faces):
        return 0, 0.0, 0.0
    x0 = np.clip(faces[:, 0], 0, w)
    y0 = np.clip(faces[:, 1], 0, h)
    x1 = np.clip(faces[:, 0] + faces[:, 2], 0, w)
    y1 = np.clip(faces[:, 1] + faces[:, 3], 0, h)
    areas = (x1 - x0) * (y1 - y0) / float(h * w)
    j = int(np.argmax(areas))
    dx = ((x0[j] + x1[j]) / 2 - w / 2) / (w / 2)
    dy = ((y0[j] + y1[j]) / 2 - h / 2) / (h / 2)
    centre = 1.0 - min(1.0, float(np.hypot(dx, dy)) / np.sqrt(2))
    return len(faces), float(areas[j]), centre


def face_signals(paths: list[Path], detector, min_score: float = FACE_MIN_SCORE,
                 progress=None) -> dict[str, np.ndarray]:
    n = len(paths)
    count = np.zeros(n, np.int16)
    area = np.zeros(n, np.float32)
    centre = np.zeros(n, np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is not None:
            count[i], area[i], centre[i] = face_stats(detect(detector, img), *img.shape[:2], min_score=min_score)
        if progress:
            progress(i + 1, n)
    return {"face_n": count, "face_area_max": area, "face_center": centre}


def _median3(x: np.ndarray) -> np.ndarray:
    """Running median over 3 steps (edges repeated): the largest face switching identity for one frame is not growth."""
    if len(x) < 3:
        return x.copy()
    p = np.pad(x, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def approach(face_area: np.ndarray, close: np.ndarray | None, fps: float, cuts: list[float] | None = None,
             window_s: float = APPROACH_WINDOW_S, close_weight: float = APPROACH_CLOSE_WEIGHT,
             eps: float = FACE_AREA_EPS, face_min: float = APPROACH_FACE_MIN,
             close_min: float = APPROACH_CLOSE_MIN) -> np.ndarray:
    """Looming rate per step (float32). With k = round(window_s * fps / 2) steps either side, on 3-step medians:

    face rate  = (log(a[i+k] + eps) - log(a[i-k] + eps)) / (2k / fps), a = face_area_max: log area per second
                 (doubling per second = 0.69). Counts only where the face at the end is >= face_min of the frame.
    close rate = close_weight * the same difference of scale_close (logits per second). Counts only where the
                 end of the window looks like a close-up (scale_close >= close_min): a flag or a body filling the
                 frame has no face, and the margin's frame-to-frame jitter in wide shots is not growth.
    approach   = the larger of the counting rates (0 where neither counts), clipped to [-3, 3]; 0 where the window
                 crosses a cut or the file edge. Negative = the subject recedes (while still large/close).
    """
    n = len(face_area)
    out = np.zeros(n, np.float32)
    k = max(1, int(round(window_s * fps / 2)))
    if n <= 2 * k:
        return out
    i = np.arange(k, n - k)
    dt = 2 * k / fps
    a = _median3(np.asarray(face_area, np.float64))
    la = np.log(a + eps)
    rate = (la[i + k] - la[i - k]) / dt
    rate[a[i + k] < face_min] = np.nan
    if close is not None:
        c = _median3(np.nan_to_num(np.asarray(close, np.float64), nan=-np.inf))
        with np.errstate(invalid="ignore"):
            close_rate = close_weight * (c[i + k] - c[i - k]) / dt
        close_rate[~np.isfinite(close_rate) | (c[i + k] < close_min)] = np.nan
        rate = np.fmax(rate, close_rate)
    out[i] = np.clip(np.nan_to_num(rate, nan=0.0), -3.0, 3.0)
    for c_s in cuts or []:  # a step whose window spans a cut compares two different shots
        j = int(np.ceil(c_s * fps - 1e-9))  # first step of the new shot
        out[max(0, j - k):min(n, j + k)] = 0.0
    return out
