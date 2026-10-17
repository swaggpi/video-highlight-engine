from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "vhe" / "models"


@dataclass
class Config:
    # Sampling grid: every per-timestep signal lives on t_i = i / fps.
    fps: float = 3.0
    frame_height: int = 360  # cached analysis frames; encoders resize further
    tiny_fps: float = 30.0  # grey stream for cuts, shake and motion (fixed rate: VFR-safe)
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
