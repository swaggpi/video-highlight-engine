"""Per-timestep signals over a corpus grid (one file, or many files concatenated) and segment scores.
Relevance (does it match the query) and quality (is it good footage) are kept separate and weighted per query.

Relevance is absolute: the SigLIP sigmoid probability on a log-odds scale, so a file with nothing matching scores
low instead of having its best frames pushed to +3 by a per-file z-score. Every other signal is z-scored over the
pooled grid of all files, so a calm take stays calm next to a loud one; loudness is first centred per file
(camera and phone gain differ)."""
from __future__ import annotations

import numpy as np

from vhe.config import BLOWN_FRAC
from vhe.signals import robust_z, smooth  # noqa: F401  (re-exported: engine and tests import them from here)
from vhe.signals.quality import REASONS

# log-odds -> relevance: odds of e^-8 (p ~ 3e-4) is 0, every REL_WIDTH nats one unit, clipped to +-3 like the
# z-scored signals. SigLIP2 logits: unrelated frames ~ -15..-10, partial matches ~ -6..-2, clear matches > 0.
REL_MID, REL_WIDTH = -8.0, 2.5
FRAMING_PENALTY = 1.0  # quality units off a step with the soft bad_framing reason (detail is 0.6 per z unit)


def per_file(fn, x: np.ndarray, file_id: np.ndarray | None) -> np.ndarray:
    """fn applied to each file's block of x separately (file_id is non-decreasing)."""
    x = np.asarray(x, dtype=float)
    if file_id is None:
        return fn(x)
    out = np.empty_like(x)
    edges = np.flatnonzero(np.diff(file_id)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(x)]):
        out[a:b] = fn(x[a:b])
    return out


def relevance(sim: np.ndarray, fps: float, logit: tuple[float, float] | None = None,
              nc_saliency: np.ndarray | None = None, file_id: np.ndarray | None = None) -> np.ndarray:
    """sim: cosine similarity of every step's frame embedding with the text. logit: SigLIP (scale, bias) from the
    index meta; without it (older indexes) relevance falls back to a pooled robust z. The highlight model's
    saliency (NaN where a file has none) is z-scored over the pool and averaged in where present."""
    sim = per_file(lambda s: smooth(s, fps), sim, file_id)
    rel = robust_z(sim) if logit is None else np.clip((logit[0] * sim + logit[1] - REL_MID) / REL_WIDTH, -3, 3)
    if nc_saliency is not None and np.isfinite(nc_saliency).any():
        has = np.isfinite(nc_saliency)
        rel = np.where(has, 0.5 * rel + 0.5 * robust_z(nc_saliency), rel)
    return rel


def _quality(grid: dict, usable: np.ndarray) -> np.ndarray:
    """Technical quality from defects only: detail (content-invariant re-blur ratio), blown highlights, shake, and
    the soft bad-framing reason (camera at the ground or straight up). Mean brightness is not used: a dark concert is
    not bad footage. Unusable steps get the floor."""
    if "blur_ratio" in grid:
        r = np.asarray(grid["blur_ratio"], dtype=float)
        detail = robust_z(np.log(np.where(np.isfinite(r) & (r > 0), r, np.nan)))
    else:  # indexes before the usable mask
        detail = robust_z(grid["sharpness"])
    blown = np.clip(np.nan_to_num(np.asarray(grid.get("blown", 0.0), dtype=float)) / BLOWN_FRAC, 0, 1)
    shaky = np.clip(robust_z(grid["shake"]), 0, None) / 3
    bits = np.asarray(grid.get("unusable", 0), dtype=np.uint8)
    framing = (bits & np.uint8(1 << REASONS.index("bad_framing"))) > 0
    q = 0.6 * detail - 1.5 * blown - 0.6 * shaky - FRAMING_PENALTY * framing
    return np.where(usable, np.clip(q, -3, 3), -3.0)


def _energy(grid: dict, audio: np.ndarray, file_id: np.ndarray | None) -> np.ndarray:
    mz, cz = robust_z(grid["motion"]), robust_z(grid["cam_speed"])
    loud = per_file(lambda x: x - np.median(x), grid["loudness"], file_id)  # device gain
    lz = robust_z(np.where(audio, loud, np.nan))
    oz = robust_z(np.where(audio, grid["onset"], np.nan))
    composite = np.where(audio, (mz + lz + oz + 0.5 * cz) / 3.5, (mz + 0.5 * cz) / 1.5)
    # The audio-less average has fewer terms, hence more variance; normalise each kind on its own so silent
    # files do not own the extremes.
    energy = np.zeros(len(composite))
    for kind in (audio, ~audio):
        if kind.any():
            energy[kind] = robust_z(composite[kind])
    return energy


def timestep_signals(grid: dict, fps: float, rel: np.ndarray | None = None, has_audio: bool | np.ndarray = True,
                     file_id: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Signals per grid step. rel: from relevance() (zeros without a content query). has_audio: per step or one
    flag. file_id: per step, for per-file loudness centring and smoothing; None = one file."""
    n = len(grid["t"])
    audio = np.broadcast_to(np.asarray(has_audio, dtype=bool), (n,))
    usable = np.asarray(grid["usable"], dtype=bool) if "usable" in grid else np.ones(n, dtype=bool)
    return {"relevance": np.zeros(n) if rel is None else rel, "quality": _quality(grid, usable),
            "energy": _energy(grid, audio, file_id), "steady": -robust_z(grid["shake"]),
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
