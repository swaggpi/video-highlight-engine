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


def audio_signals(y: np.ndarray, n_grid: int, grid_fps: float) -> dict[str, np.ndarray]:
    import librosa

    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=hop)
    loud_db = 20 * np.log10(rms + 1e-6)
    m = min(len(times), len(onset))
    return {
        "loudness": to_grid(times, loud_db, n_grid, grid_fps),
        "onset": to_grid(times[:m], onset[:m], n_grid, grid_fps),
    }


def voice_activity(y: np.ndarray) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps, load_silero_vad
    import torch

    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(y), model, sampling_rate=SR, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in ts]


def speech_mask(segments: list[tuple[float, float]], n_grid: int, grid_fps: float) -> np.ndarray:
    mask = np.zeros(n_grid, dtype=np.float32)
    for s, e in segments:
        mask[int(np.floor(s * grid_fps)):int(np.ceil(e * grid_fps)) + 1] = 1.0
    return mask[:n_grid]
