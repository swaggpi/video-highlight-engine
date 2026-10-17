from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "vhe" / "models"

# Tiny grey stream rate: the source's average rate (resampling 25 fps to 30 duplicates every 5th frame, which
# reads as shake), clamped; VFR files without a usable average fall back to r_frame_rate, then to 30.
TINY_FPS_MIN, TINY_FPS_MAX, TINY_FPS_FALLBACK = 10.0, 60.0, 30.0

# Projects (`vhe index DIR`).
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".mpg", ".mpeg", ".webm", ".3gp",
                    ".mxf", ".wmv"}
MIN_DURATION_S = 2.0  # shorter files (accidental taps) are recorded as too_short and not indexed

# Speech. Whisper gets VAD spans merged across short gaps, padded and capped, so edge words are not cut off.
ASR_MERGE_GAP_S, ASR_PAD_S, ASR_MAX_SPAN_S = 1.0, 0.3, 30.0
# A segment is dropped from the saved transcript when Whisper thinks it is probably not speech AND is unsure of
# the text (crowd noise and music produce such hallucinations).
ASR_DROP_NO_SPEECH_PROB, ASR_DROP_AVG_LOGPROB = 0.6, -1.0
LANG_MIN_PROB = 0.7  # only confident language detections vote for the project language
LANG_DETECT_S = 30.0  # seconds of speech per file used for language detection

# Usable mask: ABSOLUTE thresholds per analysis step. BLUR_RATIO is calibrated on real footage (below); the others
# are still initial guesses.
# Darkness alone is not a defect (a well exposed concert can average 0.1), so only these count.
COVERED_LUMA = 0.03  # mean luma (0..1) below this: lens covered / in a pocket
UNIFORM_STD = 0.02  # luma std (0..1) below this: near-uniform frame (fabric, lens cap, wall right in front)
BLOWN_FRAC = 0.5  # fraction of pixels >= 250 above this: blown out
# Blurry: var(Lap(g)) / var(Lap(blur(g))) at BLUR_SIDE px below BLUR_RATIO (re-blur test, independent of texture).
# Calibrated 2026-10-17 on 200 frames of real phone footage (vhe-field-test festival-commons, 20 clips, analysis
# JPEGs from the pipeline's own ffmpeg path) plus synthetic defocus/motion blur applied at source resolution:
# 197 visually sharp frames read 12.6..26 (also with noise sigma 3 at 360 px, and from 360-1080p sources);
# defocus gaussian sigma 1.3/2.7 px and disk r 2.7/5.3 px at 360 px read <= 11.8/5.4 and 10.9/4.4. At 10: 0/591
# sharp flagged, 0/1200 dark, deep-dark, haze, strong-haze, sky-filled and flat-gradient frames flagged, 5/200
# textureless real-region crops (4 visibly soft: motion or out-of-focus foreground, 1 borderline); recall 96%
# sigma 1.3, 98% r 2.7, 100% stronger, 94% heavy motion blur. Strong defocus is still caught with noise sigma 10
# at 360 px (the 120 px downscale averages sensor/JPEG noise away). The old 2.5 missed even disk r 5.3 (reads
# ~3.1). Alternatives on the same set: Crete-Roffet 2007 and a noise-floored re-blur lost defocused frames at
# noise sigma >= 6; edge width flagged 26% of dark frames; an FFT high-band ratio flagged more textureless crops.
BLUR_RATIO = 10.0
BLUR_SIDE = 120  # px short side the re-blur test runs at (resolution-independent; averages away JPEG/sensor noise)
SHAKY = 0.3  # camera jitter (motion.py shake, frame-widths per second) above this: very shaky


@dataclass
class Config:
    # Sampling grid: every per-timestep signal lives on t_i = i / fps.
    fps: float = 3.0
    frame_height: int = 360  # cached analysis frames; encoders resize further
    tiny_fps: float | None = None  # grey stream for cuts, shake and motion; None = source rate (see decode)
    tiny_w: int = 64
    tiny_h: int = 36
    hwaccel: str = "auto"  # auto | none | videotoolbox | cuda

    encoder: str = "google/siglip2-base-patch16-256"
    batch_size: int = 32
    device: str = "auto"  # auto | cpu | cuda | mps

    asr_model: str = "small"  # faster-whisper size; Parakeet planned for v2
    asr_language: str | None = None  # None = detect

    # Lighthouse CG-DETR trained on QVHighlights: NON-COMMERCIAL, prototype only.
    use_nc_highlight: bool = True
    nc_checkpoint: str = str(MODEL_DIR / "clip_cg_detr_qvhighlight.ckpt")
    nc_window_s: float = 150.0
    nc_overlap_s: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
