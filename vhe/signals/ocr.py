"""On-screen text: RapidOCR (Apache-2.0; PP-OCRv4 detector + recogniser, Apache-2.0) on the cached analysis frames.

Briefs ask for readable text ("every float with its name clearly readable", "no signs with paragraphs", "nothing with
'executed' on it"), which SigLIP cannot read. Every `every`-th grid step (1 fps at the 3 fps grid) is read on the CPU,
one frame at a time, and its values are carried to the neighbouring steps.

Per step: `text_area` (fraction of the frame covered by confident text boxes, their union) and `text_conf` (mean
recognition confidence of those boxes, 0 = none). The confident detections themselves go to the text store
(ocr.json, see store.save_index): [step, text, conf, [x0, y0, x1, y1]] with the box normalised to 0..1 of the frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from vhe.config import OCR_EVERY, OCR_MIN_CONF, OCR_MIN_LEN, OCR_THREADS

# One detection: (box as 4 [x, y] corner points in pixels, text, confidence 0..1), as RapidOCR returns them.
Detection = tuple[list, str, float]
Engine = Callable[[np.ndarray], Iterable[Detection]]
ENGINE_NAME = "rapidocr-onnxruntime PP-OCRv4 (det at frame height, no angle classifier)"


def load_engine(frame_height: int) -> Engine:
    """RapidOCR as an Engine. Raises ImportError when rapidocr-onnxruntime is not installed."""
    from vhe import models

    ocr = models.rapidocr(frame_height, OCR_THREADS)

    def run(img: np.ndarray) -> list[Detection]:
        # use_cls=False: the 180-degree classifier only matters for upside-down text, and costs a pass per box.
        res, _ = ocr(img, use_cls=False)
        return res or []

    return run


def confident(dets: Iterable[Detection], min_conf: float = OCR_MIN_CONF, min_len: int = OCR_MIN_LEN) -> list[Detection]:
    return [(box, str(text).strip(), float(conf)) for box, text, conf in dets
            if float(conf) >= min_conf and len("".join(str(text).split())) >= min_len]


def text_area(dets: list[Detection], h: int, w: int) -> float:
    """Union of the detection polygons as a fraction of the frame (overlapping boxes count once)."""
    if not dets:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    for box, _, _ in dets:
        cv2.fillPoly(mask, [np.round(np.asarray(box, dtype=np.float64)).astype(np.int32).reshape(-1, 2)], 1)
    return float(mask.mean())


def norm_box(box, h: int, w: int) -> list[float]:
    pts = np.asarray(box, dtype=np.float64).reshape(-1, 2)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return [round(float(np.clip(v / s, 0, 1)), 4) for v, s in ((x0, w), (y0, h), (x1, w), (y1, h))]


def ocr_frames(paths: list[Path], engine: Engine, every: int = OCR_EVERY, min_conf: float = OCR_MIN_CONF,
               min_len: int = OCR_MIN_LEN, progress=None) -> tuple[dict[str, np.ndarray], list[list]]:
    """({'text_area', 'text_conf'} float32[n], text store items [step, text, conf, box]) for the grid frames `paths`.
    Step i takes the values of the nearest read step (read steps: 0, every, 2*every, ...)."""
    n = len(paths)
    every = max(1, int(every))
    read = np.arange(0, n, every)
    area_r = np.zeros(len(read), np.float32)
    conf_r = np.zeros(len(read), np.float32)
    items: list[list] = []
    for k, i in enumerate(read):
        img = cv2.imread(str(paths[i]))
        if img is not None:
            h, w = img.shape[:2]
            dets = confident(engine(img), min_conf, min_len)
            area_r[k] = text_area(dets, h, w)
            conf_r[k] = np.mean([c for _, _, c in dets]) if dets else 0.0
            items += [[int(i), text, round(conf, 3), norm_box(box, h, w)] for box, text, conf in dets]
        if progress:
            progress(k + 1, len(read))
    # Nearest read step for every step (ties go to the earlier one).
    near = np.clip(np.round((np.arange(n) - 1e-6) / every).astype(int), 0, max(len(read) - 1, 0))
    if n == 0:
        return {"text_area": np.zeros(0, np.float32), "text_conf": np.zeros(0, np.float32)}, items
    return {"text_area": area_r[near], "text_conf": conf_r[near]}, items
