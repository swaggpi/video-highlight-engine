"""Process-wide model cache: each model is loaded once per (name, device) and shared by indexing and queries.
Indexing a folder of 40 files would otherwise reload SigLIP2, Whisper and VAD 40 times. Also the device choice, the
one lock that serialises inference across UI requests and indexing, and the CPU fallback after a GPU out-of-memory
error (on_device)."""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable

_CACHE: dict[tuple, Any] = {}
_LOCK = threading.RLock()
# One model lock per process: UI requests and indexing must not run inference concurrently (MPS/CUDA memory).
MODEL_LOCK = threading.RLock()


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda', 'mps' or 'cpu' (whatever torch finds); anything else as given."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_oom(exc: BaseException) -> bool:
    """A CUDA (or MPS) out-of-memory error."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # torch missing or without CUDA support
        pass
    return "out of memory" in str(exc).lower()


def on_device(device: str, what: str, fn: Callable[[], Any], to_cpu: Callable[[str], None]) -> Any:
    """fn() on `device`. On a GPU out-of-memory error (another process holds the memory, e.g. a resident LLM): warn,
    call to_cpu(message) (the caller switches to the CPU and drops the models it holds), clear the cache and run fn()
    again, so a query never fails for memory. fn must fetch its models through the caller, not hold them."""
    try:
        return fn()
    except Exception as exc:
        if device == "cpu" or not is_oom(exc):
            raise
        msg = f"{what}: out of GPU memory on {device}; running on the CPU instead (slower)"
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        to_cpu(msg)
        try:
            clear()  # the cached GPU copies would keep holding the memory
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return fn()


def cached(key: tuple, load: Callable[[], Any]) -> Any:
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = load()
        return _CACHE[key]


def clear() -> None:
    with _LOCK:
        _CACHE.clear()


# ---- lending the GPU to another process (the parser LLM) -------------------------------------------------------------
GPU_LEND_LOCK_WAIT_S = 2.0  # a running inference (or indexing) keeps the models: the parse then runs beside them
GPU_RETURN_WAIT_S = 10.0  # how long the models wait for the borrower to free the memory before they move back
GPU_RETURN_MARGIN = 256 << 20  # bytes free beyond the models' own size before they move back
_LOST: list = []  # weak callbacks: models could not go back to the GPU (each owner switches to the CPU)


def on_gpu_lost(callback: Callable[[str], None]) -> None:
    """Register an owner's switch-to-CPU (a bound method, held weakly): called with a message when models lent to
    another process (gpu_released) cannot return to the GPU. The owner must drop its models, as on_device's to_cpu."""
    import weakref

    _LOST[:] = [ref for ref in _LOST if ref() is not None]
    _LOST.append(weakref.WeakMethod(callback) if hasattr(callback, "__self__") else (lambda cb=callback: cb))


def _modules(obj, depth: int = 6, seen: set | None = None):
    """torch modules in a cached model entry: the entry itself, members of short lists and tuples, or attributes of
    wrapper objects (Lighthouse's predictor holds its CLIP encoders a few attributes deep). Dicts and long lists are
    not searched (a tokenizer's vocabulary holds no model)."""
    import torch

    seen = set() if seen is None else seen
    if id(obj) in seen or depth < 0:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        yield obj
    elif isinstance(obj, (list, tuple)) and len(obj) <= 16:
        for x in obj:
            yield from _modules(x, depth - 1, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for x in vars(obj).values():
            yield from _modules(x, depth - 1, seen)


def _cuda_modules() -> list:
    out, seen = [], set()
    with _LOCK:
        entries = list(_CACHE.values())
    for entry in entries:
        for m in _modules(entry):
            if id(m) not in seen and any(t.is_cuda for t in (*m.parameters(), *m.buffers())):
                seen.add(id(m))
                out.append(m)
    return out


def _nbytes(m) -> int:
    return sum(t.numel() * t.element_size() for t in (*m.parameters(), *m.buffers()))


@contextlib.contextmanager
def gpu_released():
    """While the block runs, the cached models' weights wait in host memory and the GPU memory they held is free for
    another process: the parser LLM (Ollama, unloaded when it answers) otherwise gets only part of a 6 GB card next
    to SigLIP2 and CG-DETR (1.5 GB) and runs mostly on the CPU (2026-10-18, GTX 1660 Super: 30-34 s per parse with
    0.8 of qwen3:4b's 2.7 GB on the GPU). The weights move back afterwards, bit for bit (same dtype), once the
    borrower has freed the memory. If they cannot, every registered owner switches to the CPU (on_gpu_lost) and the
    cache is cleared, as after an out-of-memory error (on_device). A no-op unless this process has models on a CUDA
    GPU and the model lock is free within GPU_LEND_LOCK_WAIT_S (inference and indexing keep their models)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        yield
        return
    if not MODEL_LOCK.acquire(timeout=GPU_LEND_LOCK_WAIT_S):
        yield
        return
    try:
        moved = _cuda_modules()
        devices = [next(t for t in (*m.parameters(), *m.buffers()) if t.is_cuda).device for m in moved]
        for m in moved:
            m.to("cpu")
        if moved:
            torch.cuda.empty_cache()
        try:
            yield
        finally:
            if moved:
                _return_to_gpu(torch, moved, devices)
    finally:
        MODEL_LOCK.release()


def _return_to_gpu(torch, moved: list, devices: list) -> None:
    need = sum(_nbytes(m) for m in moved) + GPU_RETURN_MARGIN
    deadline = time.monotonic() + GPU_RETURN_WAIT_S
    while time.monotonic() < deadline and torch.cuda.mem_get_info()[0] < need:
        time.sleep(0.1)  # Ollama's runner releases the memory a moment after it answers
    try:
        for m, dev in zip(moved, devices, strict=True):
            m.to(dev)
    except Exception as exc:
        if not is_oom(exc):
            raise
        for m in moved:  # one consistent place for every weight; the owners reload on the CPU
            m.to("cpu")
        clear()
        torch.cuda.empty_cache()
        msg = "models could not return to the GPU after the brief parse (out of memory); running on the CPU (slower)"
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        for ref in list(_LOST):
            cb = ref()
            if cb is not None:
                cb(msg)
        _LOST[:] = [ref for ref in _LOST if ref() is not None]


def siglip(name: str, device: str):
    """(model, processor, dtype); fp16 on GPUs. Loaded from the local Hugging Face snapshot when there is one: asking
    the Hub whether each file changed cost 12-15 s of a 17-20 s load (2026-10-18, over a VPN), and the processor
    still asks with local_files_only; the Hub only when the model was never downloaded."""
    def load():
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        try:
            source = snapshot_download(name, local_files_only=True)
        except Exception:  # not in the local cache (or `name` is a local folder): from_pretrained resolves it
            source = name
        model = AutoModel.from_pretrained(source, torch_dtype=dtype).to(device).eval()
        return model, AutoProcessor.from_pretrained(source), dtype

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
