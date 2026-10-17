"""Process-wide model cache: each model is loaded once per (name, device) and shared by indexing and queries.
Indexing a folder of 40 files would otherwise reload SigLIP2, Whisper and VAD 40 times."""
from __future__ import annotations

import threading
from typing import Any, Callable

_CACHE: dict[tuple, Any] = {}
_LOCK = threading.RLock()


def cached(key: tuple, load: Callable[[], Any]) -> Any:
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = load()
        return _CACHE[key]


def clear() -> None:
    with _LOCK:
        _CACHE.clear()


def siglip(name: str, device: str):
    """(model, processor, dtype); fp16 on GPUs."""
    def load():
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        return model, AutoProcessor.from_pretrained(name), dtype

    return cached(("siglip", name, device), load)


def whisper(size: str, device: str):
    def load():
        from faster_whisper import WhisperModel

        cuda = device == "cuda"
        return WhisperModel(size, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")

    return cached(("whisper", size, "cuda" if device == "cuda" else "cpu"), load)


def silero_vad():
    def load():
        from silero_vad import load_silero_vad

        return load_silero_vad()

    return cached(("silero_vad",), load)


def cg_detr(ckpt: str, device: str):
    """Lighthouse CG-DETR predictor (NON-COMMERCIAL weights); CPU unless CUDA (no MPS support)."""
    device = "cuda" if device == "cuda" else "cpu"

    def load():
        from lighthouse.models import CGDETRPredictor

        return CGDETRPredictor(ckpt, device=device, feature_name="clip")

    return cached(("cg_detr", ckpt, device), load)
