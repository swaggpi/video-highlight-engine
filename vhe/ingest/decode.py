"""Single decode pass: cached 3 fps frames, a tiny full-frame-rate grey stream, and 16 kHz mono audio.

Compressed video is decoded frame by frame at source resolution regardless of the sample rate, so
everything that needs pixels comes out of one ffmpeg run. Camera proxy files (GoPro .LRV etc.) are
analysed instead of the original when present; timestamps match, so clips are still cut from the original.
"""
from __future__ import annotations

import functools
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
    stems = [path.stem]
    # GoPro: GH010001.MP4 / GX010001.MP4 pair with GL010001.LRV.
    if len(path.stem) > 2 and path.stem[:2].upper() in ("GH", "GX"):
        stems.append("GL" + path.stem[2:])
    for stem in stems:
        for suffix in PROXY_SUFFIXES:
            candidate = path.with_name(stem + suffix)
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


_ENCODER_PREFERENCE = [
    ("libx264", ["-preset", "veryfast", "-crf", "20"]),
    ("h264_videotoolbox", ["-q:v", "65"]),
    ("h264_nvenc", ["-preset", "p4", "-cq", "21"]),
    ("libopenh264", ["-b:v", "8M"]),
    ("mpeg4", ["-q:v", "3"]),  # always present, even in patent-free ffmpeg builds
]


@functools.lru_cache(maxsize=1)
def video_encoder_args() -> tuple[str, ...]:
    """Best encoder that actually works here. Distro builds (e.g. Fedora ffmpeg-free) ship without libx264, and
    hardware encoders can be compiled in without the hardware present, so each candidate gets a test encode."""
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    for name, args in _ENCODER_PREFERENCE:
        if f" {name} " not in listed:
            continue
        probe_cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                     "-c:v", name, *args, "-f", "null", "-"]
        if subprocess.run(probe_cmd, capture_output=True).returncode == 0:
            return ("-c:v", name, *args)
    return ()


def _has_zscale() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " zscale " in out


def tonemap_filter(p: Probe) -> str:
    """HDR (PQ/HLG, iPhone default) looks washed out to SDR-trained encoders and players; map it to BT.709.
    Returns a filter chain ending in ',' or ''. Apply after downscaling: it runs in float32."""
    if not p.is_hdr or not _has_zscale():
        return ""
    return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,")


def _short_side(h: int) -> str:
    # Short side = h for landscape and portrait alike.
    return f"scale='if(gt(iw,ih),-2,{h})':'if(gt(iw,ih),{h},-2)'"


def decode(video: Path, out_dir: Path, cfg: Config) -> dict:
    """One ffmpeg pass: frames/%06d.jpg (3 fps), tiny.gray (fixed rate) and audio.wav (16 kHz mono)."""
    source = find_proxy(video) or video
    p = probe(source)
    frames_dir = out_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)  # stale frames from an earlier run would lengthen the grid
    frames_dir.mkdir(parents=True, exist_ok=True)
    tone = tonemap_filter(p)
    # round=up makes frame i the first source frame at or after i/fps, matching the audio/motion grid.
    # The tiny stream is resampled to a fixed rate so variable-frame-rate phone footage keeps true timing.
    graph = (f"[0:v]split=2[a][b];"
             f"[a]fps={cfg.fps}:round=up,{_short_side(cfg.frame_height)},{tone}format=yuvj420p[f];"
             f"[b]fps={cfg.tiny_fps},scale={cfg.tiny_w}:{cfg.tiny_h}:flags=area,format=gray[t]")
    tail = ["-i", str(source), "-filter_complex", graph,
            "-map", "[f]", "-q:v", "3", str(frames_dir / "%06d.jpg"),
            "-map", "[t]", "-f", "rawvideo", str(out_dir / "tiny.gray")]
    if p.has_audio:
        tail += ["-map", "0:a:0", "-ac", "1", "-ar", "16000", str(out_dir / "audio.wav")]
    accel = hwaccel_args(cfg.hwaccel)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *accel, *tail], check=True)
    except subprocess.CalledProcessError:
        if not accel:
            raise
        # Hardware decode fails on some codecs; software decode always works.
        subprocess.run(["ffmpeg", "-v", "error", "-y", *tail], check=True)

    n_frames = len(list(frames_dir.glob("*.jpg")))
    tiny_frames = (out_dir / "tiny.gray").stat().st_size // (cfg.tiny_w * cfg.tiny_h)
    duration = p.duration if p.duration > 0 else n_frames / cfg.fps
    return {"video": str(video), "source": str(source), "used_proxy": source != video,
            "duration": duration, "src_fps": p.fps, "tiny_fps": float(cfg.tiny_fps), "width": p.width,
            "height": p.height, "hdr_tonemapped": bool(tone), "has_audio": p.has_audio,
            "n_frames": n_frames, "tiny_frames": int(tiny_frames)}


def load_tiny(out_dir: Path, cfg: Config) -> np.ndarray:
    raw = np.fromfile(out_dir / "tiny.gray", dtype=np.uint8)
    return raw.reshape(-1, cfg.tiny_h, cfg.tiny_w)


def frame_paths(out_dir: Path) -> list[Path]:
    return sorted((out_dir / "frames").glob("*.jpg"))
