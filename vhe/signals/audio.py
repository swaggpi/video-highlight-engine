"""Loudness, onset rate and voice activity on the 3 fps grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from vhe.signals import to_grid

SR = 16000


def load_audio(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"expected {SR} Hz audio, got {sr}")
    return y


def audio_signals(y: np.ndarray, n_grid: int, grid_fps: float, chunk_s: float = 300.0) -> dict[str, np.ndarray]:
    """Processed in 5-minute chunks: librosa's frame/STFT matrices for a full hour would need ~2 GB."""
    import librosa

    hop = 512
    if len(y) < 2048:
        return {"loudness": np.zeros(n_grid), "onset": np.zeros(n_grid)}
    step = int(chunk_s * SR) // hop * hop
    times, loud, onset = [], [], []
    for start in range(0, len(y), step):
        part = y[start:start + step + 2048]  # overlap one window so frames at chunk edges are complete
        rms = librosa.feature.rms(y=part, frame_length=2048, hop_length=hop, center=False)[0]
        ons = librosa.onset.onset_strength(y=part, sr=SR, hop_length=hop, center=False)
        k = min(len(rms), len(ons), step // hop)
        times.append(start / SR + (np.arange(k) * hop + 1024) / SR)
        loud.append(20 * np.log10(rms[:k] + 1e-6))
        onset.append(ons[:k])
    t = np.concatenate(times)
    return {
        "loudness": to_grid(t, np.concatenate(loud), n_grid, grid_fps),
        "onset": to_grid(t, np.concatenate(onset), n_grid, grid_fps),
    }


def voice_activity(y: np.ndarray) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps
    import torch

    from vhe import models

    ts = get_speech_timestamps(torch.from_numpy(y), models.silero_vad(), sampling_rate=SR, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in ts]


def speech_mask(segments: list[tuple[float, float]], n_grid: int, grid_fps: float) -> np.ndarray:
    """1 from the grid step at or before each segment start to the last step before its end."""
    mask = np.zeros(n_grid, dtype=np.float32)
    for s, e in segments:
        mask[int(np.floor(s * grid_fps)):int(np.ceil(e * grid_fps))] = 1.0
    return mask[:n_grid]
