"""Single decode pass: cached 3 fps frames, a tiny full-frame-rate grey stream, and 16 kHz mono audio.

Compressed video is decoded frame by frame at source resolution regardless of the sample rate, so
everything that needs pixels comes out of one ffmpeg run. Camera proxy files (GoPro .LRV etc.) are
analysed instead of the original when present; timestamps match, so clips are still cut from the original.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vhe.config import Config

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
PROXY_SUFFIXES = (".LRV", ".lrv", ".LRF", ".lrf")


@dataclass
class Probe:
    duration: float
    fps: float
    width: int
    height: int
    color_transfer: str | None
    has_audio: bool

    @property
    def is_hdr(self) -> bool:
        return (self.color_transfer or "") in HDR_TRANSFERS


def probe(path: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(info["format"].get("duration") or video.get("duration") or 0.0)
    return Probe(duration=duration, fps=fps, width=int(video["width"]), height=int(video["height"]),
                 color_transfer=video.get("color_transfer"),
                 has_audio=any(s["codec_type"] == "audio" for s in info["streams"]))


def find_proxy(path: Path) -> Path | None:
    for suffix in PROXY_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def hwaccel_args(mode: str) -> list[str]:
    if mode == "none":
        return []
    if mode == "auto":
        if platform.system() == "Darwin":
            mode = "videotoolbox"
        elif shutil.which("nvidia-smi"):
            mode = "cuda"
        else:
            return []
    return ["-hwaccel", mode]


def _has_zscale() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " zscale " in out


def _tonemap_chain(p: Probe) -> str:
    # HDR (PQ/HLG) footage looks washed out to SDR-trained encoders; map it to BT.709 first.
    if not p.is_hdr or not _has_zscale():
        return ""
    return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,")


def decode(video: Path, out_dir: Path, cfg: Config) -> dict:
    """Write frames/%06d.jpg (3 fps), tiny.gray (full fps) and audio.wav into out_dir."""
    source = find_proxy(video) or video
    p = probe(source)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tone = _tonemap_chain(p)
    graph = (f"[0:v]{tone}split=2[a][b];"
             f"[a]fps={cfg.fps},scale=-2:{cfg.frame_height}[f];"
             f"[b]scale={cfg.tiny_w}:{cfg.tiny_h},format=gray[t]")
    tail = ["-i", str(source), "-filter_complex", graph,
            "-map", "[f]", "-q:v", "3", str(frames_dir / "%06d.jpg"),
            "-map", "[t]", "-f", "rawvideo", str(out_dir / "tiny.gray")]
    accel = hwaccel_args(cfg.hwaccel)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *accel, *tail], check=True)
    except subprocess.CalledProcessError:
        if not accel:
            raise
        # Hardware decode fails on some codecs; software decode always works.
        subprocess.run(["ffmpeg", "-v", "error", "-y", *tail], check=True)

    if p.has_audio:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                        str(out_dir / "audio.wav")], check=True)

    n_frames = len(list(frames_dir.glob("*.jpg")))
    tiny_frames = (out_dir / "tiny.gray").stat().st_size // (cfg.tiny_w * cfg.tiny_h)
    return {"video": str(video), "source": str(source), "used_proxy": source != video,
            "duration": p.duration, "src_fps": p.fps, "width": p.width, "height": p.height,
            "hdr_tonemapped": bool(tone), "has_audio": p.has_audio,
            "n_frames": n_frames, "tiny_frames": int(tiny_frames)}


def load_tiny(out_dir: Path, cfg: Config) -> np.ndarray:
    raw = np.fromfile(out_dir / "tiny.gray", dtype=np.uint8)
    return raw.reshape(-1, cfg.tiny_h, cfg.tiny_w)


def frame_paths(out_dir: Path) -> list[Path]:
    return sorted((out_dir / "frames").glob("*.jpg"))
