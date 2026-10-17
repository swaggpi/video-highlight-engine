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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vhe.config import TINY_FPS_FALLBACK, TINY_FPS_MAX, TINY_FPS_MIN, Config

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
    recorded_at: str | None = None  # UTC ISO 8601 from the container's creation tags

    @property
    def is_hdr(self) -> bool:
        return (self.color_transfer or "") in HDR_TRANSFERS


def _rate(s: str | None) -> float | None:
    """'30000/1001' -> 29.97; None for missing, 0/0 and implausible rates (VFR files report odd values)."""
    try:
        num, _, den = (s or "").partition("/")
        r = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return r if 1.0 <= r <= 240.0 else None


def source_fps(stream: dict) -> float:
    return _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")) or TINY_FPS_FALLBACK


def tiny_rate(src_fps: float, override: float | None = None) -> float:
    """Tiny stream rate: the source rate (to 0.001, so 29.97 is not resampled to 30 either), clamped."""
    if override:
        return float(override)
    return float(min(TINY_FPS_MAX, max(TINY_FPS_MIN, round(src_fps, 3))))


def recorded_at(fmt: dict, streams: list[dict]) -> str | None:
    """Capture time as UTC ISO 8601: Apple's creationdate (has the local offset) first, then creation_time."""
    tags = [{k.lower(): v for k, v in (d.get("tags") or {}).items()} for d in (fmt, *streams)]
    values = [t.get("com.apple.quicktime.creationdate") for t in tags[:1]] + [t.get("creation_time") for t in tags]
    for v in values:
        if not v:
            continue
        try:
            d = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if d.year < 1990:  # unset clocks write 1904 or 1970
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def probe(path: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    duration = float(info["format"].get("duration") or video.get("duration") or 0.0)
    return Probe(duration=duration, fps=source_fps(video), width=int(video["width"]), height=int(video["height"]),
                 color_transfer=video.get("color_transfer"),
                 has_audio=any(s["codec_type"] == "audio" for s in info["streams"]),
                 recorded_at=recorded_at(info["format"], info["streams"]))


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


@functools.lru_cache(maxsize=1)
def ffmpeg_hwaccels() -> frozenset[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
    return frozenset(line.strip() for line in out.splitlines()[1:] if line.strip())


def hwaccel_choice(mode: str) -> tuple[str, str]:
    """(hwaccel or 'none', why). 'auto' needs the hardware AND an ffmpeg build that supports it (static builds
    often lack cuda even when nvidia-smi is present)."""
    if mode == "none":
        return "none", "disabled"
    if mode != "auto":
        return mode, "requested"
    want = "videotoolbox" if platform.system() == "Darwin" else "cuda" if shutil.which("nvidia-smi") else None
    if want is None:
        return "none", "no supported GPU found"
    if want not in ffmpeg_hwaccels():
        return "none", f"ffmpeg build lacks {want}"
    return want, "auto"


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
    """One ffmpeg pass: frames/%06d.jpg (3 fps), tiny.gray (source rate) and audio.wav (16 kHz mono)."""
    source = find_proxy(video) or video
    p = probe(source)
    frames_dir = out_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)  # stale frames from an earlier run would lengthen the grid
    frames_dir.mkdir(parents=True, exist_ok=True)
    tone = tonemap_filter(p)
    tiny_fps = tiny_rate(p.fps, cfg.tiny_fps)
    # round=up makes frame i the first source frame at or after i/fps, matching the audio/motion grid.
    # The tiny stream is resampled to a constant rate (VFR phone footage keeps true timing) equal to the source's
    # average rate, so constant-rate footage gets no duplicated frames.
    graph = (f"[0:v]split=2[a][b];"
             f"[a]fps={cfg.fps}:round=up,{_short_side(cfg.frame_height)},{tone}format=yuvj420p[f];"
             f"[b]fps={tiny_fps},scale={cfg.tiny_w}:{cfg.tiny_h}:flags=area,format=gray[t]")
    tail = ["-i", str(source), "-filter_complex", graph,
            "-map", "[f]", "-q:v", "3", str(frames_dir / "%06d.jpg"),
            "-map", "[t]", "-f", "rawvideo", str(out_dir / "tiny.gray")]
    if p.has_audio:
        tail += ["-map", "0:a:0", "-ac", "1", "-ar", "16000", str(out_dir / "audio.wav")]
    accel, why = hwaccel_choice(cfg.hwaccel)
    args = [] if accel == "none" else ["-hwaccel", accel]
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args, *tail], check=True, capture_output=bool(args))
    except subprocess.CalledProcessError:
        if not args:
            raise
        # Hardware decode fails on some codecs; software decode always works.
        accel, why = "none", f"{accel} decode failed, software fallback"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *tail], check=True)

    n_frames = len(list(frames_dir.glob("*.jpg")))
    tiny_frames = (out_dir / "tiny.gray").stat().st_size // (cfg.tiny_w * cfg.tiny_h)
    duration = p.duration if p.duration > 0 else n_frames / cfg.fps
    return {"video": str(video), "source": str(source), "used_proxy": source != video,
            "duration": duration, "src_fps": p.fps, "tiny_fps": tiny_fps, "width": p.width,
            "height": p.height, "hdr_tonemapped": bool(tone), "has_audio": p.has_audio,
            "recorded_at": p.recorded_at, "hwaccel": f"{accel} ({why})",
            "n_frames": n_frames, "tiny_frames": int(tiny_frames)}


def load_tiny(out_dir: Path, cfg: Config) -> np.ndarray:
    raw = np.fromfile(out_dir / "tiny.gray", dtype=np.uint8)
    return raw.reshape(-1, cfg.tiny_h, cfg.tiny_w)


def frame_paths(out_dir: Path) -> list[Path]:
    return sorted((out_dir / "frames").glob("*.jpg"))
