"""Per-video normalised signals and segment scores. Relevance (does it match the query) and quality (is it
good footage) are kept separate and weighted per query."""
from __future__ import annotations

import numpy as np


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


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    pad = width // 2
    return np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")[:len(x)]


def timestep_signals(grid: dict, fps: float, text_emb: np.ndarray | None,
                     nc_saliency: np.ndarray | None) -> dict[str, np.ndarray]:
    n = len(grid["t"])
    if text_emb is not None:
        sim = grid["emb"].astype(np.float32) @ text_emb.astype(np.float32)
        rel = robust_z(smooth(sim, int(fps)))
        if nc_saliency is not None and np.isfinite(nc_saliency).any():
            rel = 0.5 * rel + 0.5 * robust_z(nc_saliency)
    else:
        rel = np.zeros(n)
    quality = (0.6 * robust_z(grid["sharpness"]) + 0.4 * robust_z(grid["exposure"])
               - 0.5 * np.clip(grid["clipped"] * 4, 0, 1))
    energy = (robust_z(grid["motion"]) + robust_z(grid["loudness"]) + robust_z(grid["onset"])
              + 0.5 * robust_z(grid["cam_speed"])) / 3.5
    steady = -robust_z(grid["shake"])
    return {"relevance": rel, "quality": quality, "energy": energy, "steady": steady,
            "speech": np.asarray(grid["speech"], dtype=float)}


def segment_scores(sig: dict[str, np.ndarray], i0: int, i1: int, weights: dict) -> dict[str, float]:
    i1 = max(i1, i0 + 1)
    rel = sig["relevance"][i0:i1]
    # A segment is as relevant as its best part, but a single matching frame should not carry it.
    top = np.sort(rel)[-max(1, int(np.ceil(0.6 * len(rel)))):]
    parts = {
        "relevance": float(top.mean()),
        "quality": float(sig["quality"][i0:i1].mean()),
        "energy": float(sig["energy"][i0:i1].mean()),
        "steady": float(sig["steady"][i0:i1].mean()),
        "speech": float(sig["speech"][i0:i1].mean()) * 2 - 1,
    }
    parts["total"] = float(sum(weights.get(k, 0.0) * v for k, v in parts.items()))
    return parts
