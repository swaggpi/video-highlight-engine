"""Transcript with word timestamps. v1 uses faster-whisper (MIT), restricted to the speech regions our Silero VAD
already found (no second VAD pass, no second read of the audio). Parakeet-TDT (CC-BY-4.0) is the planned
replacement for speed and German accuracy."""
from __future__ import annotations

import numpy as np


def transcribe(y: np.ndarray, vad: list[tuple[float, float]], model_size: str, language: str | None,
               device: str, progress=None) -> dict:
    from faster_whisper import WhisperModel

    cuda = device == "cuda"
    model = WhisperModel(model_size, device="cuda" if cuda else "cpu",
                         compute_type="float16" if cuda else "int8")
    clips = [t for s, e in vad for t in (round(s, 2), round(e, 2))]
    segments, info = model.transcribe(y, language=language, vad_filter=False, clip_timestamps=clips,
                                      word_timestamps=True, condition_on_previous_text=False)
    out = []
    total = vad[-1][1] if vad else 0.0
    for seg in segments:  # a generator: decoding happens here
        if progress:
            progress(min(seg.end, total), total)
        out.append({
            "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip(),
            "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
                      for w in (seg.words or [])],
        })
    if progress:
        progress(total, total)
    return {"language": info.language, "segments": out}
