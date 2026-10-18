"""Transcript with word timestamps. v1 uses faster-whisper (MIT), restricted to the speech regions our Silero VAD
already found (no second VAD pass, no second read of the audio). Parakeet-TDT (CC-BY-4.0) is the planned
replacement for speed and German accuracy."""
from __future__ import annotations

from collections import Counter

import numpy as np

from vhe.config import (ASR_DROP_AVG_LOGPROB, ASR_DROP_NO_SPEECH_PROB, ASR_MAX_SPAN_S, ASR_MERGE_GAP_S, ASR_PAD_S,
                        LANG_DETECT_S, LANG_MIN_PROB)

from vhe.signals.audio import SR


def whisper_spans(vad: list[tuple[float, float]], duration: float, gap: float = ASR_MERGE_GAP_S,
                  pad: float = ASR_PAD_S, cap: float = ASR_MAX_SPAN_S) -> list[tuple[float, float]]:
    """VAD spans merged across gaps < `gap`, padded by `pad`, split into pieces of at most `cap` seconds."""
    merged: list[list[float]] = []
    for s, e in sorted(vad):
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[tuple[float, float]] = []
    for s, e in merged:
        s = max(0.0, s - pad, out[-1][1] if out else 0.0)  # padding never overlaps the previous span
        e = min(duration, e + pad)
        if e <= s:
            continue
        k = int(np.ceil((e - s) / cap - 1e-9))
        out += [(round(s + i * (e - s) / k, 2), round(s + (i + 1) * (e - s) / k, 2)) for i in range(k)]
    return out


def keep_segment(seg: dict) -> bool:
    return not (seg["no_speech_prob"] > ASR_DROP_NO_SPEECH_PROB and seg["avg_logprob"] < ASR_DROP_AVG_LOGPROB)


def detect_language(y: np.ndarray, vad: list[tuple[float, float]], model_size: str, device: str):
    """(language, probability) from up to LANG_DETECT_S seconds of speech, or (None, 0.0) without speech."""
    from vhe import models

    parts, total = [], 0
    for s, e in vad:
        part = y[int(s * SR):int(e * SR)]
        parts.append(part)
        total += len(part)
        if total >= LANG_DETECT_S * SR:
            break
    if not total:
        return None, 0.0
    audio = np.concatenate(parts)[:int(LANG_DETECT_S * SR)]
    # transcribe() detects the language eagerly; the segment generator is never consumed.
    _, info = models.whisper(model_size, device).transcribe(audio, vad_filter=False)
    return info.language, float(info.language_probability)


def majority_language(detections: list[tuple[str | None, float]], min_prob: float = LANG_MIN_PROB) -> str | None:
    """Most common language among confident detections (ties: the higher summed probability); None if none."""
    votes, weight = Counter(), Counter()
    for lang, prob in detections:
        if lang and prob >= min_prob:
            votes[lang] += 1
            weight[lang] += prob
    if not votes:
        return None
    return max(votes, key=lambda k: (votes[k], weight[k]))


def transcribe(y: np.ndarray, vad: list[tuple[float, float]], model_size: str, language: str | None,
               device: str, progress=None) -> dict:
    from vhe import models

    model = models.whisper(model_size, device)
    spans = whisper_spans(vad, len(y) / SR)
    clips = [t for s, e in spans for t in (s, e)]
    segments, info = model.transcribe(y, language=language, vad_filter=False, clip_timestamps=clips,
                                      word_timestamps=True, condition_on_previous_text=False)
    out, dropped = [], 0
    total = spans[-1][1] if spans else 0.0
    for seg in segments:  # a generator: decoding happens here
        if progress:
            progress(min(seg.end, total), total)
        rec = {
            "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip(),
            "avg_logprob": round(float(seg.avg_logprob), 3), "no_speech_prob": round(float(seg.no_speech_prob), 3),
            "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
                      for w in (seg.words or [])],
        }
        if keep_segment(rec):
            out.append(rec)
        else:
            dropped += 1
    if progress:
        progress(total, total)
    return {"language": info.language, "language_probability": round(float(info.language_probability), 3),
            "segments": out, "dropped_low_confidence": dropped}
