"""Transcript with word times: faster-whisper (MIT) with its VAD filter."""
from __future__ import annotations

import ctypes
import gc
import glob
import os
import site
from pathlib import Path

VERSION = "speech-1"
MODEL = "small"


def _preload_cublas() -> None:
    """CTranslate2 dlopens libcublas.so.12, which only exists inside the nvidia-cublas wheel here."""
    for lib in ("libcublasLt.so.12", "libcublas.so.12"):
        for d in site.getsitepackages():
            for p in glob.glob(os.path.join(d, "nvidia", "*", "lib", lib)):
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)


def compute_type(cuda: bool) -> str:
    """float32 on GTX 16xx (fp16 GEMMs are ~9x slower there: 18.6 s vs 7.4 s per 2 min of audio), else float16."""
    if os.environ.get("VTL_WHISPER_TYPE"):
        return os.environ["VTL_WHISPER_TYPE"]
    if not cuda:
        return "int8"
    import torch

    return "float32" if "GTX 16" in torch.cuda.get_device_name() else "float16"


def transcribe(wav: Path) -> dict:
    """{version, model, language, segments: [{start, end, text, words: [{start, end, word}]}]}."""
    import torch

    if not wav.exists():
        return {"version": VERSION, "model": MODEL, "language": None, "segments": []}
    cuda = torch.cuda.is_available()
    if cuda:
        _preload_cublas()
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL, device="cuda" if cuda else "cpu", compute_type=compute_type(cuda))
    segs, info = model.transcribe(str(wav), vad_filter=True, word_timestamps=True, condition_on_previous_text=False)
    segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                 "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                           for w in (s.words or [])]} for s in segs]
    del model
    gc.collect()
    return {"version": VERSION, "model": MODEL, "language": info.language, "segments": segments}
