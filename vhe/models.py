"""Process-wide model cache: each model is loaded once per (name, device) and shared by indexing and queries.
Indexing a folder of 40 files would otherwise reload SigLIP2, Whisper and VAD 40 times."""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import urllib.request
from pathlib import Path
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


def fetch(name: str, url: str, sha256: str, model_dir: Path | None = None) -> Path:
    """A model file in the models cache (~/.cache/vhe/models), downloaded once and checked against `sha256`.
    A mismatch (truncated download, changed upstream file) is deleted and raises. $VHE_NO_DOWNLOAD=1 (tests,
    offline machines) raises FileNotFoundError instead of downloading."""
    from vhe.config import MODEL_DIR

    path = Path(model_dir or MODEL_DIR) / name
    if path.exists():
        return path
    if os.environ.get("VHE_NO_DOWNLOAD"):
        raise FileNotFoundError(f"{path} is missing and downloads are disabled ($VHE_NO_DOWNLOAD)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {name}: got {digest}, expected {sha256}")
    tmp.replace(path)  # atomic: a half-written file never looks like a cached model
    return path


# OpenCV Zoo YuNet face detector (MIT). Same file on GitHub (git LFS) and the opencv/face_detection_yunet HF mirror.
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def yunet(score_threshold: float):
    """cv2.FaceDetectorYN on the cached YuNet model; callers set the input size per frame (setInputSize).
    Not thread-safe: indexing runs one file at a time."""
    def load():
        import cv2

        path = fetch(YUNET_FILE, YUNET_URL, YUNET_SHA256)
        return cv2.FaceDetectorYN.create(str(path), "", (640, 360), score_threshold, 0.3, 5000)

    return cached(("yunet", score_threshold), load)


def rapidocr(det_side: int, threads: int):
    """RapidOCR (PP-OCRv4 det + rec, ONNX Runtime on the CPU). det_side: the detector's short side; the analysis
    frames' height, so they are not upscaled (RapidOCR's default 736 doubles the cost for no recall on 360 p)."""
    def load():
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR(det_limit_side_len=det_side, det_limit_type="min", intra_op_num_threads=threads,
                        inter_op_num_threads=1)

    return cached(("rapidocr", det_side, threads), load)
