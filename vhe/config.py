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

# Usable mask: ABSOLUTE thresholds per analysis step. Initial guesses, to be calibrated on real event footage.
# Darkness alone is not a defect (a well exposed concert can average 0.1), so only these count.
COVERED_LUMA = 0.03  # mean luma (0..1) below this: lens covered / in a pocket
UNIFORM_STD = 0.02  # luma std (0..1) below this: near-uniform frame (fabric, lens cap, wall right in front)
BLOWN_FRAC = 0.5  # fraction of pixels >= 250 above this: blown out
BLUR_RATIO = 2.5  # var(Lap(g)) / var(Lap(blur(g))) below this: blurry (re-blur test, independent of texture)
BLUR_SIDE = 120  # px short side the re-blur test runs at (resolution-independent; averages away JPEG noise)
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
