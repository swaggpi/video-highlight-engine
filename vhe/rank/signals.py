"""Per-timestep signals over a corpus grid (one file, or many files concatenated), span statistics over them, and
segment scores. Relevance (does it match the query) and quality (is it good footage) are kept separate and weighted
per query.

Relevance is absolute: the SigLIP sigmoid probability on a log-odds scale, so a file with nothing matching scores
low instead of having its best frames pushed to +3 by a per-file z-score. It is SigLIP only: the non-commercial
highlight model's saliency is its own signal (nc_signal), the slot an owned highlight head will fill; the hand-weighted
scorer blends the two per step (blend_nc), as it always has. Every other signal is z-scored over the pooled grid of
all files, so a calm take stays calm next to a loud one; loudness is first centred per file (camera and phone gain
differ)."""
from __future__ import annotations

import numpy as np

from vhe.config import BLOWN_FRAC
from vhe.signals import per_file, robust_z, smooth
from vhe.signals.quality import reason_active, reason_mask
from vhe.signals.reaction import reaction_steps

# log-odds -> relevance: odds of e^-8 (p ~ 3e-4) is 0, every REL_WIDTH nats one unit, clipped to +-3 like the
# z-scored signals. SigLIP2 logits: unrelated frames ~ -15..-10, partial matches ~ -6..-2, clear matches > 0.
REL_MID, REL_WIDTH = -8.0, 2.5
FRAMING_PENALTY = 1.0  # quality units off a step with the soft bad_framing reason (detail is 0.6 per z unit)
TOP_FRAC = 0.6  # a span is as relevant as the mean of its best 60 % of steps
LIVE_MIN_DB = -60.0  # loudness (dBFS) below this is muted: phone mutes are digital silence at -120 dB


def relevance(sim: np.ndarray, fps: float, logit: tuple[float, float],
              file_id: np.ndarray | None = None) -> np.ndarray:
    """SigLIP relevance per step. sim: cosine similarity of every step's frame embedding with the text, smoothed per
    file. logit: SigLIP (scale, bias) from the index meta."""
    sim = per_file(lambda s: smooth(s, fps), sim, file_id)
    return np.clip((logit[0] * sim + logit[1] - REL_MID) / REL_WIDTH, -3, 3)


def nc_signal(saliency: np.ndarray | None) -> np.ndarray | None:
    """The highlight model's saliency z-scored over the pool, NaN where a file has none; None without any."""
    if saliency is None or not np.isfinite(saliency).any():
        return None
    return np.where(np.isfinite(saliency), robust_z(saliency), np.nan)


def blend_nc(rel: np.ndarray, nc: np.ndarray | None) -> np.ndarray:
    """The hand-weighted relevance: SigLIP relevance (one row per text, or one curve) averaged with the saliency
    where a file has it. HeuristicScorer's relevance input; learned scorers see the two as separate features."""
    return rel if nc is None else np.where(np.isfinite(nc), 0.5 * rel + 0.5 * nc, rel)


# ---- span statistics: [i0, i1) grid-step ranges, vectorised over spans ------------------------------------------
def span_mean(x: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    """Mean of x over each span, from prefix sums (the hand-weighted scorer's segment means)."""
    cum = np.concatenate([[0.0], np.cumsum(x)])
    return (cum[i1] - cum[i0]) / (i1 - i0)


def _span_blocks(i0: np.ndarray, i1: np.ndarray):
    """(span indices, step index matrix) per span length: one gather per length instead of a loop per span."""
    i0, i1 = np.asarray(i0, int), np.maximum(np.asarray(i1, int), np.asarray(i0, int) + 1)
    lengths = i1 - i0
    for n in np.unique(lengths):
        rows = np.flatnonzero(lengths == n)
        yield rows, i0[rows, None] + np.arange(n)


def span_top_mean(x: np.ndarray, i0: np.ndarray, i1: np.ndarray, frac: float = TOP_FRAC) -> np.ndarray:
    """Mean of the best `frac` of each span's steps (at least one): as relevant as its best part, but a single
    matching frame does not carry it. Same numbers as segment_scores' relevance."""
    out = np.zeros(len(i0))
    for rows, idx in _span_blocks(i0, i1):
        block = np.sort(x[idx], axis=1)
        out[rows] = block[:, -max(1, int(np.ceil(frac * idx.shape[1]))):].mean(1)
    return out


def span_min(x: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    out = np.zeros(len(i0))
    for rows, idx in _span_blocks(i0, i1):
        out[rows] = x[idx].min(1)
    return out


def span_max(x: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    out = np.zeros(len(i0))
    for rows, idx in _span_blocks(i0, i1):
        out[rows] = x[idx].max(1)
    return out


def muted_steps(loudness: np.ndarray, has_audio: np.ndarray) -> np.ndarray:
    """True where there is no usable live sound: no audio track, or digital silence (a streamer's mute)."""
    loud = np.nan_to_num(np.asarray(loudness, dtype=float), nan=-120.0)
    return ~np.asarray(has_audio, dtype=bool) | (loud < LIVE_MIN_DB)


def _quality_parts(grid: dict) -> dict[str, np.ndarray]:
    """The quality components per step: detail (pooled robust z of the log re-blur ratio, NaN where unmeasured),
    blown highlights (0..1 of BLOWN_FRAC), shake above the pool's median (0..1 per 3 z; 0 while the shaky reason is
    inactive, config.ACTIVE_REASONS) and the soft bad-framing reason (0/1; 0 while inactive)."""
    r = np.asarray(grid["blur_ratio"], dtype=float)
    return {"detail": robust_z(np.log(np.where(np.isfinite(r) & (r > 0), r, np.nan))),
            "blown": np.clip(np.nan_to_num(np.asarray(grid.get("blown", 0.0), dtype=float)) / BLOWN_FRAC, 0, 1),
            "shaky": np.clip(robust_z(grid["shake"]), 0, None) / 3 if reason_active("shaky") else np.zeros(len(r)),
            "framing": reason_mask(grid["unusable"], "bad_framing").astype(float)}


def _quality(parts: dict[str, np.ndarray], usable: np.ndarray) -> np.ndarray:
    """Technical quality from defects only: detail (content-invariant re-blur ratio), blown highlights, shake, and
    the soft bad-framing reason (camera at the ground or straight up). Mean brightness is not used: a dark concert is
    not bad footage. Unusable steps get the floor."""
    q = 0.6 * parts["detail"] - 1.5 * parts["blown"] - 0.6 * parts["shaky"] - FRAMING_PENALTY * parts["framing"]
    return np.where(usable, np.clip(q, -3, 3), -3.0)


def _energy_parts(grid: dict, audio: np.ndarray, file_id: np.ndarray | None) -> dict[str, np.ndarray]:
    """Pooled z-scores of the energy components (loudness and onsets 0 where a file has no audio)."""
    loud = per_file(lambda x: x - np.median(x), grid["loudness"], file_id)  # device gain
    return {"motion_z": robust_z(grid["motion"]), "cam_speed_z": robust_z(grid["cam_speed"]),
            "loud_z": robust_z(np.where(audio, loud, np.nan)),
            "onset_z": robust_z(np.where(audio, grid["onset"], np.nan))}


def _energy(parts: dict[str, np.ndarray], audio: np.ndarray) -> np.ndarray:
    mz, cz, lz, oz = parts["motion_z"], parts["cam_speed_z"], parts["loud_z"], parts["onset_z"]
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
    flag. file_id: per step, for per-file loudness centring and smoothing; None = one file. Besides the five
    weighted signals: the energy components, the quality components detail_z (0 where unmeasured) and framing (the
    soft bad-framing reason), the audio reaction curve (vhe.signals.reaction: loudness and onset bursts over each
    file's running baseline), and per-step audio, unusable and blown-pixel fractions (features)."""
    n = len(grid["t"])
    audio = np.broadcast_to(np.asarray(has_audio, dtype=bool), (n,))
    usable = np.asarray(grid["usable"], dtype=bool)
    parts = _energy_parts(grid, audio, file_id)
    qparts = _quality_parts(grid)
    return {"relevance": np.zeros(n) if rel is None else rel, "quality": _quality(qparts, usable),
            "detail_z": np.nan_to_num(qparts["detail"]), "framing": qparts["framing"],
            "energy": _energy(parts, audio), "steady": -robust_z(grid["shake"]),
            "speech": np.asarray(grid["speech"], dtype=float), **parts,
            "reaction": reaction_steps(grid["loudness"], grid["onset"], audio, fps, file_id),
            "audio": audio.astype(float), "unusable": (~usable).astype(float),
            "blown": np.nan_to_num(np.asarray(grid.get("blown", np.zeros(n)), dtype=float))}


def segment_scores(sig: dict[str, np.ndarray], i0: int, i1: int, weights: dict) -> dict[str, float]:
    i1 = max(i1, i0 + 1)
    rel = sig["relevance"][i0:i1]
    # A segment is as relevant as its best part, but a single matching frame should not carry it.
    top = np.sort(rel)[-max(1, int(np.ceil(TOP_FRAC * len(rel)))):]
    parts = {
        "relevance": float(top.mean()),
        "quality": float(sig["quality"][i0:i1].mean()),
        "energy": float(sig["energy"][i0:i1].mean()),
        "steady": float(sig["steady"][i0:i1].mean()),
        "speech": float(sig["speech"][i0:i1].mean()) * 2 - 1,
    }
    parts["total"] = float(sum(weights.get(k, 0.0) * v for k, v in parts.items()))
    return parts
