"""Cut clips from the original video. Re-encodes, because stream copy can only cut on keyframes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vhe.ingest.decode import probe, tonemap_filter, video_encoder_args


def _stamp(t: float) -> str:
    return f"{int(t // 60):02d}m{t % 60:04.1f}s"


def export_clips(video: Path, clips: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    encoder = video_encoder_args()
    tone = tonemap_filter(probe(video)).rstrip(",")  # HDR phone footage stays playable in SDR players
    vf = ["-vf", tone] if tone else []
    for i, c in enumerate(clips, 1):
        path = out_dir / f"{i:02d}_{_stamp(c['start'])}-{_stamp(c['end'])}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{c['start']:.3f}", "-i", str(video),
                        "-t", f"{c['end'] - c['start']:.3f}", *vf, *encoder, "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(path)], check=True)
        paths.append(path)
    (out_dir / "clips.json").write_text(json.dumps(clips, indent=1, ensure_ascii=False))
    return paths
