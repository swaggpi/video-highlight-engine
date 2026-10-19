"""Streamer layout: find the facecam box and static overlay panels (chat, alerts) per file, and the game-area crop.

Measured on the 85-min stream VOD (120 frames sampled over the whole file): what separates overlays from gameplay is
not temporal variance (a dark horror game is as still as a webcam background) but EDGE PERSISTENCE, the share of
frames in which a pixel lies on an edge. Gameplay edges move with the camera (persistence 0.05-0.2); the facecam's
room background, its border lines and the chat's fixed text rows stay put (0.3-0.6, borders 0.65-0.76).
  facecam: the YuNet face that sits in the same small box in many sampled frames; the box grows from that face to the
           strongest persistent border line on each side (or the frame edge when there is none).
  panels:  connected regions of persistent edges outside the facecam (chat text rows, logos, fixed frames).
  crop:    the largest axis-aligned rectangle that overlaps neither (all in pixels of the frames passed in).
Boxes are [x, y, w, h]."""
from __future__ import annotations

from pathlib import Path

import numpy as np

VERSION = "overlay-1"
SAMPLES = 120          # frames sampled evenly over the file
EDGE_T = 0.25          # |Sobel x| + |Sobel y| on 0..1 grey above this is an edge
CELL = 8               # structure is judged on CELL x CELL pixel cells
STRUCT_T = 0.3         # a cell whose mean edge persistence is above this is overlay-like structure
BORDER_T = 0.3         # a facecam border: mean persistence along the line this much above the band outside it
FACE_SHARE = 0.25      # a facecam: faces in the same place in at least this share of the sampled frames ...
FACE_MAX_H = 0.4       # ... each smaller than this share of the frame height (a full-screen camera is no overlay)
FACECAM_MAX = 0.4      # a "facecam" box larger than this share of the frame is a full-screen camera: ignored
PANEL_MIN = 0.01       # panels smaller than this share of the frame are ignored (crosshairs, small HUD icons)
CROP_MIN = 0.3         # a game crop smaller than this share of the frame is not trusted: the full frame is used


def edge_persistence(grey: np.ndarray) -> np.ndarray:
    """grey: [K, H, W] float 0..1 -> [H, W] share of the K frames in which each pixel is on an edge."""
    import cv2

    hits = np.zeros(grey.shape[1:], np.float32)
    for g in grey:
        mag = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        hits += mag > EDGE_T
    return hits / max(1, len(grey))


def face_cluster(faces: list[list], w: int, h: int, share: float = FACE_SHARE) -> list[int] | None:
    """faces: per sampled frame a list of [x, y, w, h] face boxes. The box (10th-90th percentile extent) of the faces
    whose centre falls in the most-visited cell of a coarse 8x8 grid, if at least `share` of the frames have one."""
    small = [[b for b in fs if b[3] < FACE_MAX_H * h] for fs in faces]
    counts: dict[tuple[int, int], int] = {}
    for fs in small:
        for cell in {(int((b[0] + b[2] / 2) * 8 // w), int((b[1] + b[3] / 2) * 8 // h)) for b in fs}:
            counts[cell] = counts.get(cell, 0) + 1
    if not counts:
        return None
    cell, n = max(counts.items(), key=lambda kv: kv[1])
    if n < share * max(1, len(faces)):
        return None
    mine = np.array([b for fs in small for b in fs
                     if (int((b[0] + b[2] / 2) * 8 // w), int((b[1] + b[3] / 2) * 8 // h)) == cell], np.float64)
    x0, y0 = np.percentile(mine[:, 0], 10), np.percentile(mine[:, 1], 10)
    x1, y1 = np.percentile(mine[:, 0] + mine[:, 2], 90), np.percentile(mine[:, 1] + mine[:, 3], 90)
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    return [x0, y0, x1 - x0, y1 - y0]


def _line(p: np.ndarray, axis: int, pos: int, a: int, b: int) -> float:
    """Mean persistence along row (axis 0) or column (axis 1) `pos`, from a to b; the better of pos and pos-1."""
    lines = [p[q, a:b] if axis == 0 else p[a:b, q] for q in (pos, pos - 1) if 0 <= q < p.shape[axis]]
    return max((float(v.mean()) for v in lines if v.size), default=0.0)


def _band(p: np.ndarray, axis: int, lo: int, hi: int, a: int, b: int) -> float | None:
    """Mean persistence of rows/columns lo..hi (over a..b); None if the band leaves the frame."""
    if lo < 0 or hi > p.shape[axis]:
        return None
    return float((p[lo:hi, a:b] if axis == 0 else p[a:b, lo:hi]).mean())


def grow_facecam(face: list[int], p: np.ndarray, border: float = BORDER_T) -> list[int]:
    """From the face box out to the best border line on each side, else to the frame edge. A border is a line of
    persistent edges with less persistence just outside it: score = line - mean of the band 3-6 px further out, so a
    line inside a textured, static webcam picture (static on both sides) does not count. Two passes: the second
    measures each line over the extent found by the first."""
    h, w = p.shape
    x0, y0, x1, y1 = face[0], face[1], face[0] + face[2], face[1] + face[3]
    fx0, fy0, fx1, fy1 = x0, y0, x1, y1

    def best(axis, positions, a, b, edge, out):  # out: -1 = outside is towards index 0, +1 = away from it
        def score(q):  # a line with no room for an outside band is the frame edge itself: no border
            band = _band(p, axis, q - 6, q - 2, a, b) if out < 0 else _band(p, axis, q + 3, q + 7, a, b)
            return -1.0 if band is None else _line(p, axis, q, a, b) - band

        s, q = max(((score(q), q) for q in positions), default=(0.0, edge))
        return q if s >= border else edge

    for _ in range(2):
        top = best(0, range(1, fy0), x0, x1, 0, -1)
        bottom = best(0, range(fy1 + 1, h), x0, x1, h, 1)
        left = best(1, range(1, fx0), y0, y1, 0, -1)
        right = best(1, range(fx1 + 1, w), y0, y1, w, 1)
        x0, y0, x1, y1 = left, top, right, bottom
    return [x0, y0, x1 - x0, y1 - y0]


def panels(p: np.ndarray, exclude: list[list[int]], cell: int = CELL) -> list[list[int]]:
    """Boxes of connected overlay-like structure (cells above STRUCT_T) outside the `exclude` boxes."""
    import cv2

    h, w = p.shape
    gh, gw = h // cell, w // cell
    cells = p[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    mask = (cells > STRUCT_T).astype(np.uint8)
    for x, y, bw, bh in exclude:
        mask[max(0, y // cell - 1):(y + bh) // cell + 2, max(0, x // cell - 1):(x + bw) // cell + 2] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for x, y, bw, bh, area in stats[1:n]:
        if area >= PANEL_MIN * gh * gw:
            out.append([int(x * cell), int(y * cell), int(bw * cell), int(bh * cell)])
    return out


def _overlap(a: list[int], b: list[int]) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def largest_free_rect(w: int, h: int, boxes: list[list[int]]) -> list[int]:
    """The largest axis-aligned [x, y, w, h] inside the w x h frame that overlaps none of `boxes`. Optimal edges lie
    on the frame edges or box edges, so those coordinates are tried exhaustively (a handful of boxes)."""
    xs = sorted({0, w, *(c for b in boxes for c in (b[0], b[0] + b[2]) if 0 < c < w)})
    ys = sorted({0, h, *(c for b in boxes for c in (b[1], b[1] + b[3]) if 0 < c < h)})
    best, best_area = [0, 0, 0, 0], 0
    for i, x0 in enumerate(xs):
        for x1 in xs[i + 1:]:
            for j, y0 in enumerate(ys):
                for y1 in ys[j + 1:]:
                    area = (x1 - x0) * (y1 - y0)
                    if area > best_area and not any(_overlap([x0, y0, x1 - x0, y1 - y0], b) for b in boxes):
                        best, best_area = [x0, y0, x1 - x0, y1 - y0], area
    return best


def layout(grey: np.ndarray, faces: list[list], facecam: list[int] | None = None,
           crop: list[int] | None = None) -> dict:
    """{size, facecam, panels, crop, crop_share, notes} for sampled frames `grey` [K, H, W] and their face boxes.
    `facecam` / `crop` are manual overrides (skip that part of the detection)."""
    k, h, w = grey.shape
    p = edge_persistence(grey)
    notes = []
    cam = facecam
    if cam is None:
        face = face_cluster(faces, w, h)
        if face is None:
            notes.append("no facecam: no face in the same place often enough")
        else:
            cam = grow_facecam(face, p)
            if cam[2] * cam[3] > FACECAM_MAX * w * h:
                notes.append(f"face box {face} grows to {cam}: a full-screen camera, not a facecam overlay")
                cam = None
    found = panels(p, [cam] if cam else [])
    if crop is None:
        crop = largest_free_rect(w, h, ([cam] if cam else []) + found)
        if crop[2] * crop[3] < CROP_MIN * w * h:
            notes.append(f"largest free area {crop} is under {CROP_MIN:.0%} of the frame: using the full frame")
            crop = [0, 0, w, h]
    return {"size": [w, h], "samples": k, "facecam": cam, "panels": found, "crop": crop,
            "crop_share": round(crop[2] * crop[3] / (w * h), 3), "notes": notes}


def sample_paths(paths: list[Path], n: int = SAMPLES) -> list[Path]:
    """n paths spread evenly over the file (centres of n equal parts)."""
    if len(paths) <= n:
        return list(paths)
    return [paths[int((i + 0.5) * len(paths) / n)] for i in range(n)]


def detect(paths: list[Path], facecam: list[int] | None = None, crop: list[int] | None = None) -> dict:
    """layout() on SAMPLES of the decoded frames `paths`, with YuNet faces (the vhe face model)."""
    import cv2

    from vhe import models
    from vhe.config import FACE_MIN_SCORE

    pick = sample_paths(paths)
    imgs = [cv2.imread(str(p)) for p in pick]
    grey = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]).astype(np.float32) / 255.0
    det = models.yunet(FACE_MIN_SCORE)
    faces = []
    for img in imgs:
        det.setInputSize((img.shape[1], img.shape[0]))
        _, f = det.detect(img)
        faces.append([[int(v) for v in row[:4]] for row in (f if f is not None else [])])
    out = layout(grey, faces, facecam, crop)
    out["face_frames"] = sum(bool(f) for f in faces)
    return out


def scale_box(box: list[int] | None, s: float) -> list[int] | None:
    return None if box is None else [int(round(v * s)) for v in box]


def parse_box(text: str) -> list[int]:
    """'x,y,w,h' -> [x, y, w, h] (argparse type)."""
    parts = [int(float(v)) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0 or min(parts[:2]) < 0:
        raise ValueError(f"expected x,y,w,h with w, h > 0, got {text!r}")
    return parts


def draw(frame: Path, lay: dict, out: Path) -> None:
    """Debug image: the frame with the crop (green), facecam (red) and panels (yellow) drawn on it."""
    import cv2

    img = cv2.imread(str(frame))
    boxes = [(lay.get("crop"), (0, 220, 0), "game crop"), (lay.get("facecam"), (0, 0, 255), "facecam")]
    boxes += [(b, (0, 220, 255), "panel") for b in lay.get("panels") or []]
    for b, color, name in boxes:
        if b:
            x, y, w, h = b
            cv2.rectangle(img, (x + 1, y + 1), (x + w - 2, y + h - 2), color, 2)
            cv2.putText(img, name, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
