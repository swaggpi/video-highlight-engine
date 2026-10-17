"""Export an edit: one re-encoded file per clip (stream copy can only cut on keyframes), clips.csv, a CMX3600 EDL
that references the ORIGINAL files (relink in Resolve/Premiere for the full-quality conform), and optionally a
rough-cut preview: every clip re-encoded to one uniform format, then joined with the concat demuxer.

Clips carry 'file' (label), 'file_start' and 'file_end' (seconds in that file); `sources` maps label -> path."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from vhe.ingest.decode import probe, tonemap_filter, video_encoder_args

# Per-clip files are previews (the EDL points at the originals): long side at most 1920, scaled BEFORE the HDR
# tonemap, which runs in float32 and is slow at 4K.
CLIP_SCALE = "scale='min(iw,1920)':'min(ih,1920)':force_original_aspect_ratio=decrease:force_divisible_by=2"
ROUGH_W, ROUGH_H, ROUGH_FPS = 1280, 720, 30
EDL_START_TC = 3600.0  # record timeline starts at 01:00:00:00


def _stamp(t: float) -> str:
    return f"{int(t // 60):02d}m{t % 60:04.1f}s"


def _tc(f: int, fps: int) -> str:
    """Frame count -> HH:MM:SS:FF."""
    return f"{f // (3600 * fps):02d}:{f // (60 * fps) % 60:02d}:{f // fps % 60:02d}:{f % fps:02d}"


def _source_tc(path: Path, fps: int) -> float:
    """Embedded start timecode in seconds (camera files), else 0; NLEs match EDL source times against it."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=timecode:stream_tags=timecode",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True).stdout.split()
    for tc in out:
        parts = tc.replace(";", ":").split(":")
        if len(parts) == 4 and all(x.isdigit() for x in parts):
            h, m, s, f = map(int, parts)
            return h * 3600 + m * 60 + s + f / fps
    return 0.0


def write_edl(clips: list[dict], sources: dict[str, Path], path: Path, title: str = "vhe edit") -> Path:
    """CMX3600, non-drop frame at the first source's rounded frame rate; one event per clip, in edit order."""
    probes = {f: probe(sources[f]) for f in dict.fromkeys(c["file"] for c in clips)}
    fps = max(1, int(round(next(iter(probes.values())).fps))) if probes else 25
    starts = {f: _source_tc(sources[f], fps) for f in probes}
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    rec = int(EDL_START_TC * fps)
    for i, c in enumerate(clips, 1):
        # Whole frames throughout, so record durations equal source durations and never drift.
        src_in, src_out = (int(round((starts[c["file"]] + c[k]) * fps)) for k in ("file_start", "file_end"))
        track = "AA/V" if probes[c["file"]].has_audio else "V"
        lines.append(f"{i:03d}  AX       {track:<5} C        {_tc(src_in, fps)} {_tc(src_out, fps)} "
                     f"{_tc(rec, fps)} {_tc(rec + src_out - src_in, fps)}")
        lines.append(f"* FROM CLIP NAME: {Path(c['file']).name}")
        lines.append(f"* SOURCE FILE: {sources[c['file']]}")
        lines.append("")
        rec += src_out - src_in
    path.write_text("\n".join(lines))
    return path


def write_csv(clips: list[dict], path: Path) -> Path:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "file", "start", "end", "duration"])
        for i, c in enumerate(clips, 1):
            w.writerow([i, c["file"], f"{c['file_start']:.3f}", f"{c['file_end']:.3f}",
                        f"{c['file_end'] - c['file_start']:.3f}"])
    return path


def rough_cut(parts: list[Path], out: Path) -> Path:
    """Every part re-encoded to one format (size, rate, pixel format, stereo AAC; silence where a part has no
    audio) so the concat demuxer can join them without another encode."""
    encoder = video_encoder_args()
    vf = (f"scale={ROUGH_W}:{ROUGH_H}:force_original_aspect_ratio=decrease,"
          f"pad={ROUGH_W}:{ROUGH_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={ROUGH_FPS},format=yuv420p")
    tmp = out.parent / (out.stem + "_parts")
    tmp.mkdir(parents=True, exist_ok=True)
    uniform = []
    for i, part in enumerate(parts):
        u = tmp / f"{i:03d}.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(part)]
        if probe(part).has_audio:
            cmd += ["-map", "0:v:0", "-map", "0:a:0"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        cmd += ["-vf", vf, *encoder, "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", str(u)]
        subprocess.run(cmd, check=True)
        uniform.append(u)
    listing = tmp / "list.txt"
    listing.write_text("".join(f"file '{u.name}'\n" for u in uniform))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    for u in uniform:
        u.unlink()
    listing.unlink()
    tmp.rmdir()
    return out


def export_clips(clips: list[dict], out_dir: Path, sources: dict[str, Path], with_rough_cut: bool = False) -> dict:
    """Write the clip files, clips.csv, edit.edl and clips.json (and rough_cut.mp4) to out_dir, in the given order.
    Returns {"clips": [paths], "csv", "edl", "rough_cut"}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = video_encoder_args()
    tones = {f: tonemap_filter(probe(sources[f])) for f in dict.fromkeys(c["file"] for c in clips)}
    paths = []
    for i, c in enumerate(clips, 1):
        src = Path(sources[c["file"]])
        path = out_dir / f"{i:02d}_{src.stem}_{_stamp(c['file_start'])}-{_stamp(c['file_end'])}.mp4"
        # HDR phone footage stays playable in SDR players; tonemap after the downscale.
        vf = CLIP_SCALE + ("," + tones[c["file"]].rstrip(",") if tones[c["file"]] else "")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{c['file_start']:.3f}", "-i", str(src),
                        "-t", f"{c['file_end'] - c['file_start']:.3f}", "-vf", vf, *encoder, "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(path)], check=True)
        paths.append(path)
    (out_dir / "clips.json").write_text(json.dumps(clips, indent=1, ensure_ascii=False))
    out = {"clips": paths, "csv": write_csv(clips, out_dir / "clips.csv"),
           "edl": write_edl(clips, sources, out_dir / "edit.edl"), "rough_cut": None}
    if with_rough_cut and paths:
        out["rough_cut"] = rough_cut(paths, out_dir / "rough_cut.mp4")
    return out
