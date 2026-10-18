"""Browser previews of result clips: exactly the clip's span of its source, transcoded on demand to WebM (VP9 + Opus,
at most PREVIEW_WIDTH wide) and cached on disk.

Camera and phone sources are H.264/HEVC + AAC MP4s; a browser without an H.264 decoder (Firefox builds without the
system codecs, e.g. the Zen browser snap: "the file is corrupt") cannot play them, while VP9/Opus WebM plays in every
modern browser. Measured 2026-10-18 (16 cores, 720p30 H.264 source, 7 s span): VP9 realtime cpu-used 8 in 0.75 s,
VP8 in 0.78 s, so VP9. Seeking before the input (-ss before -i) is frame-accurate when transcoding and halves the time
of a seek after it (0.75 vs 1.62 s at 60 s into the file). HDR sources are tone-mapped as for export."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

from vhe.ingest.decode import probe, tonemap_filter

PREVIEW_VERSION = 1  # part of the cache key: bump it when the settings below change
PREVIEW_WIDTH = 640
MAX_PREVIEW_S = 600.0  # a result clip is seconds to a few minutes; a longer span is not a preview request
VIDEO_ARGS = ("-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-b:v", "1M")
AUDIO_ARGS = ("-c:a", "libopus", "-b:a", "96k", "-ac", "2")
ENCODE_TIMEOUT_S = 300.0


def span(start: str | float, end: str | float) -> tuple[float, float]:
    """(start, end) in seconds, rounded to milliseconds; ValueError unless 0 <= start < end and the span is at most
    MAX_PREVIEW_S (non-numbers, NaN and infinities included)."""
    s, e = float(start), float(end)
    if not (math.isfinite(s) and math.isfinite(e)):
        raise ValueError("start and end must be finite numbers")
    s, e = round(s, 3), round(e, 3)
    if s < 0 or e <= s or e - s > MAX_PREVIEW_S:
        raise ValueError(f"need 0 <= start < end and end - start <= {MAX_PREVIEW_S:.0f} s")
    return s, e


def preview_key(source: Path, start: float, end: float) -> str:
    """Cache key of a preview: the source (resolved path, size, modification time), the span in milliseconds and the
    encoding settings."""
    source = Path(source).resolve()
    st = source.stat()
    raw = json.dumps([PREVIEW_VERSION, str(source), st.st_size, st.st_mtime_ns, round(start * 1000),
                      round(end * 1000), PREVIEW_WIDTH, VIDEO_ARGS, AUDIO_ARGS])
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def preview_path(cache_dir: Path, source: Path, start: float, end: float) -> Path:
    return Path(cache_dir) / f"{preview_key(source, start, end)}.webm"


def encoders_available(ffmpeg: str = "ffmpeg") -> bool:
    """Whether ffmpeg has the preview encoders (libvpx-vp9, libopus)."""
    try:
        listed = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return " libvpx-vp9 " in listed and " libopus " in listed


def make_preview(source: Path, start: float, end: float, out: Path, ffmpeg: str = "ffmpeg") -> Path:
    """Transcode source[start:end] to `out` (WebM, VP9 + Opus); written to a temporary name first, so a preview that
    exists is complete. Raises subprocess.CalledProcessError (stderr attached) when ffmpeg fails."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = probe(Path(source))
    vf = f"scale='trunc(min({PREVIEW_WIDTH},iw)/2)*2':-2,{tonemap_filter(info)}format=yuv420p"
    tmp = out.with_name(out.name + ".part")
    cmd = [ffmpeg, "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
           "-map", "0:v:0", *(["-map", "0:a:0"] if info.has_audio else []), "-vf", vf, *VIDEO_ARGS,
           *(AUDIO_ARGS if info.has_audio else ()), "-f", "webm", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=ENCODE_TIMEOUT_S)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out
