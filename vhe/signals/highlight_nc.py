"""NON-COMMERCIAL COMPONENT. Query-conditioned highlight scores from Lighthouse CG-DETR.

The code is Apache-2.0 but the checkpoint is trained on QVHighlights (CC BY-NC-SA 4.0), so this module
must not ship in the product. It is isolated behind Config.use_nc_highlight and recorded in the index
metadata. Replacement: the same architecture retrained on owned data (see README).

The model only accepts 150 s of video, so long videos are split into overlapping windows whose features are
cached once; per query only the small DETR head runs. Window scores are stitched with weights that favour
each window's centre.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

NC_COMPONENT = "lighthouse-cg-detr-qvhighlights (CC BY-NC-SA 4.0 training data)"


def window_starts(duration: float, window: float, overlap: float) -> list[float]:
    if duration <= window:
        return [0.0]
    step = window - overlap
    starts = list(np.arange(0.0, duration - window, step))
    last = max(0.0, duration - window)  # last window ends exactly at the end
    if starts and last - starts[-1] < 5.0:
        starts[-1] = last  # avoid a near-duplicate final window
    else:
        starts.append(last)
    return [round(float(s), 3) for s in starts]


class NCHighlighter:
    def __init__(self, ckpt: str, device: str) -> None:
        from lighthouse.models import CGDETRPredictor

        if not Path(ckpt).exists():
            raise FileNotFoundError(f"missing checkpoint {ckpt}; see README 'Non-commercial highlight model'")
        self.device = "cuda" if device == "cuda" else "cpu"  # Lighthouse does not support MPS
        self.predictor = CGDETRPredictor(ckpt, device=self.device, feature_name="clip")
        self.clip_len = float(self.predictor._clip_len)
        self._feats: dict[str, dict] = {}

    def encode(self, video: Path, duration: float, window: float, overlap: float, cache: Path,
               progress=None) -> list[dict]:
        cache.mkdir(parents=True, exist_ok=True)
        window = min(window, 74 * self.clip_len)  # positional embedding limit: 75 clips
        chunks = []
        with tempfile.TemporaryDirectory() as tmp:
            starts = window_starts(duration, window, overlap)
            for k, start in enumerate(starts):
                if progress:
                    progress(k, len(starts))
                length = min(window, duration - start)
                if length < 2 * self.clip_len:
                    continue
                part = Path(tmp) / f"part_{k}.avi"
                # Lighthouse samples one frame per clip_len seconds at 224 px, so a 1 fps proxy is enough.
                # MJPEG is in every ffmpeg build and cheap to write at 1 fps.
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-i", str(video), "-t", f"{length}",
                                "-an", "-vf", "fps=1,scale=-2:224", "-c:v", "mjpeg", "-q:v", "3", str(part)],
                               check=True)
                feats = self.predictor.encode_video(str(part))
                path = cache / f"chunk_{k:03d}.pt"
                torch.save({k2: (v.cpu() if isinstance(v, torch.Tensor) else v) for k2, v in feats.items()}, path)
                chunks.append({"start": start, "length": length, "path": path.name})
        if progress:
            progress(len(starts), len(starts))
        return chunks

    def predict(self, query: str, chunks: list[dict], cache: Path, n_grid: int, grid_fps: float):
        """Return (saliency on the grid, proposed windows [(start, end, score)] in absolute seconds)."""
        acc = np.zeros(n_grid)
        wsum = np.zeros(n_grid)
        proposals = []
        grid_t = np.arange(n_grid) / grid_fps
        for ch in chunks:
            key = str(cache / ch["path"])
            if key not in self._feats:
                # weights_only: index folders may be shared; never unpickle arbitrary objects from them.
                feats = torch.load(key, map_location=self.device, weights_only=True)
                self._feats[key] = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                                    for k, v in feats.items()}
            pred = self.predictor.predict(query, self._feats[key])
            if not pred:
                continue
            sal = np.asarray(pred["pred_saliency_scores"], dtype=float)
            centres = ch["start"] + (np.arange(len(sal)) + 0.5) * self.clip_len
            inside = (grid_t >= ch["start"]) & (grid_t <= ch["start"] + ch["length"])
            values = np.interp(grid_t[inside], centres, sal)
            # Triangular weight: windows overlap, trust each one most near its middle.
            rel = (grid_t[inside] - ch["start"]) / max(ch["length"], 1e-6)
            weight = 1e-3 + 1.0 - np.abs(2 * rel - 1)
            acc[inside] += values * weight
            wsum[inside] += weight
            for s, e, score in pred["pred_relevant_windows"]:
                proposals.append((ch["start"] + s, ch["start"] + e, float(score)))
        sal_grid = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), np.nan)
        return sal_grid, proposals
