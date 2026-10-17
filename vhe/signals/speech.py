"""Transcript with word timestamps. v1 uses faster-whisper (MIT) gated by its Silero VAD; Parakeet-TDT
(CC-BY-4.0) is the planned replacement for speed and German accuracy."""
from __future__ import annotations

from pathlib import Path


def transcribe(wav: Path, model_size: str, language: str | None, device: str) -> dict:
    from faster_whisper import WhisperModel

    cuda = device == "cuda"
    model = WhisperModel(model_size, device="cuda" if cuda else "cpu",
                         compute_type="float16" if cuda else "int8")
    segments, info = model.transcribe(str(wav), language=language, vad_filter=True, word_timestamps=True,
                                      condition_on_previous_text=False)
    out = []
    for seg in segments:
        out.append({
            "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip(),
            "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
                      for w in (seg.words or [])],
        })
    return {"language": info.language, "segments": out}
