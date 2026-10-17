"""decode -> index -> project manifest on tiny lavfi clips. Skipped without ffmpeg or a cached SigLIP2."""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from vhe.config import Config


def _cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
needs_models = pytest.mark.skipif(not shutil.which("ffmpeg") or not _cached(Config.encoder),
                                  reason="needs ffmpeg and a cached SigLIP2 (no downloads in tests)")


@needs_ffmpeg
def test_nc_windows_built_from_cached_frames(tmp_path):
    """The highlight model's 1 fps input is every third cached frame, aligned to the window start."""
    from vhe.signals.highlight_nc import NCHighlighter

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(60):  # 20 s at 3 fps; brightness encodes the frame index
        cv2.imwrite(str(frames / f"{i + 1:06d}.jpg"), np.full((36, 64), 4 * i, np.uint8))
    seen = []

    class Predictor:
        def encode_video(self, path):
            cap, levels = cv2.VideoCapture(path), []
            while (f := cap.read())[0]:
                levels.append(int(round(f[1].mean() / 4)))
            seen.append(levels)
            return {}

    hl = NCHighlighter.__new__(NCHighlighter)
    hl.predictor, hl.clip_len, hl.device, hl._feats = Predictor(), 2.0, "cpu", {}
    chunks = hl.encode(frames, 3.0, 20.0, window=10.0, overlap=2.0, cache=tmp_path / "nc")
    assert [(c["start"], c["length"]) for c in chunks] == [(0.0, 10.0), (10.0, 10.0)]
    assert seen == [[3 * k for k in range(10)], [30 + 3 * k for k in range(10)]]


def _clip(path, seconds: float, rate: int) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=320x180:r={rate}:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=f=440:d={seconds}", "-shortest", "-c:v", "mpeg4", "-q:v", "4",
                    "-c:a", "pcm_s16le", str(path)], check=True)


@needs_models
def test_index_folder_end_to_end(tmp_path):
    from vhe.project import index_project, load_project

    _clip(tmp_path / "A001.mov", 4.0, 25)
    _clip(tmp_path / "A002.mov", 1.0, 25)
    cfg = Config(use_nc_highlight=False, asr_language="en")
    m = index_project(tmp_path, cfg)
    assert [(e["rel_path"], e["status"]) for e in m["files"]] == [("A001.mov", "ok"), ("A002.mov", "too_short")]
    manifest, (idx,) = load_project(tmp_path)
    assert idx.meta["tiny_fps"] == 25.0 and idx.meta["tiny_frames"] == 100
    assert idx.has_audio and "has_audio" not in idx.grid
    assert idx.n == 12 and idx.usable.shape == (12,) and idx.unusable_reasons.dtype == np.uint8
    assert manifest["files"][0]["usable_fraction"] == idx.meta["usable_fraction"]
    assert idx.meta["source_size"] == (tmp_path / "A001.mov").stat().st_size
    assert not (tmp_path / manifest["files"][0]["index_dir"] / "frames").exists()  # cache removed

    calls = []
    index_project(tmp_path, cfg, indexer=lambda *a, **k: calls.append(a))
    assert calls == []  # unchanged: nothing re-indexed
