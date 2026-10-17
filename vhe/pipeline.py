"""Index a video once: decode, signals, embeddings, tags, transcript, non-commercial highlight features."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np

from vhe.config import Config, resolve_device
from vhe.index.store import default_index_dir, save_index
from vhe.ingest.decode import decode, frame_paths, load_tiny
from vhe.signals.audio import audio_signals, load_audio, speech_mask, voice_activity
from vhe.signals.frames import Encoder
from vhe.signals.motion import motion_signals
from vhe.signals.quality import frame_quality
from vhe.signals.shots import detect_cuts, frame_diffs
from vhe.signals.tags import LABELS, label_texts, tag_probs
from vhe.timing import Counter, Timings


def change_points(emb: np.ndarray, fps: float, cuts: list[float], half_window_s: float = 1.0,
                  z_thresh: float = 2.5, min_gap_s: float = 2.0, min_dist: float = 0.05) -> list[float]:
    """Content changes inside shots (raw footage is often one long take): peaks of the embedding distance
    between the second before and the second after each step."""
    k = max(1, int(round(half_window_s * fps)))
    e = emb.astype(np.float32)
    n = len(e)
    if n < 2 * k + 2:
        return []
    c = np.concatenate([np.zeros((1, e.shape[1]), np.float32), np.cumsum(e, axis=0)])
    i = np.arange(k, n - k)
    before = c[i] - c[i - k]
    after = c[i + k] - c[i]
    cos = (before * after).sum(1) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) + 1e-9)
    dist = 1.0 - cos
    med = np.median(dist)
    mad = np.median(np.abs(dist - med)) * 1.4826 + 1e-9
    z = (dist - med) / mad
    cut_arr = np.asarray(cuts)
    out: list[float] = []
    for j in np.argsort(-z):
        if z[j] < z_thresh:
            break
        if dist[j] < min_dist:  # relative outliers in a static take are noise, not content changes
            continue
        t = float(i[j] / fps)
        if cut_arr.size and np.min(np.abs(cut_arr - t)) < 1.0:
            continue
        if any(abs(t - o) < min_gap_s for o in out):
            continue
        out.append(round(t, 3))
    return sorted(out)


def build_index(video: Path, out_dir: Path | None = None, cfg: Config | None = None,
                progress: Callable[[str], None] | None = None, keep_cache: bool = False) -> Path:
    cfg = cfg or Config()
    video = Path(video).resolve()
    out = Path(out_dir) if out_dir else default_index_dir(video)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda msg: None)
    device = resolve_device(cfg.device)
    timings = Timings()
    fps = cfg.fps

    try:
        return _build(video, out, cfg, say, device, timings)
    finally:
        if not keep_cache:
            shutil.rmtree(out / "frames", ignore_errors=True)
            (out / "tiny.gray").unlink(missing_ok=True)
            (out / "audio.wav").unlink(missing_ok=True)


def _build(video: Path, out: Path, cfg: Config, say, device: str, timings: Timings) -> Path:
    fps = cfg.fps
    say("decoding")
    with timings.stage("decode") as r:
        info = decode(video, out, cfg)
        r["items"] = info["n_frames"]
    n = info["n_frames"]
    if n == 0:
        raise RuntimeError("no frames decoded")
    duration = info["duration"]
    tiny = load_tiny(out, cfg)

    say("detecting cuts")
    with timings.stage("cuts", items=len(tiny)):
        cuts = detect_cuts(tiny, info["tiny_fps"], d=frame_diffs(tiny))

    say("motion and shake")
    with timings.stage("motion", items=len(tiny)):
        grid: dict[str, np.ndarray] = {"t": np.arange(n) / fps}
        grid.update(motion_signals(tiny, info["tiny_fps"], n, fps, cuts))
    del tiny

    vad: list[tuple[float, float]] = []
    transcript = None
    if info["has_audio"]:
        y = load_audio(out / "audio.wav")
        say("audio energy")
        with timings.stage("audio"):
            grid.update(audio_signals(y, n, fps))
        say("voice activity")
        with timings.stage("vad"):
            vad = voice_activity(y)
        grid["speech"] = speech_mask(vad, n, fps)
        if vad and cfg.asr_model != "none":
            say("transcribing speech")
            from vhe.signals.speech import transcribe

            with timings.stage("speech"):
                transcript = transcribe(y, vad, cfg.asr_model, cfg.asr_language, device,
                                        progress=Counter(say, "transcribing", unit="s"))
        del y
    else:
        for key in ("loudness", "onset", "speech"):
            grid[key] = np.zeros(n, dtype=np.float32)
    grid["has_audio"] = np.array(info["has_audio"])

    paths = frame_paths(out)
    say(f"embedding {len(paths)} frames")
    with timings.stage("load_encoder"):
        encoder = Encoder(cfg.encoder, device)
    with timings.stage("frame_embed", items=len(paths)):
        emb = encoder.images(paths, cfg.batch_size,
                             progress=Counter(say, "embedding frames"))
    with timings.stage("tags"):
        scale, bias = encoder.logit_params()
        grid["tags"] = tag_probs(emb, encoder.texts(label_texts()), scale, bias)
    say("frame quality")
    with timings.stage("quality", items=len(paths)):
        grid.update(frame_quality(paths))
    grid["emb"] = emb
    cps = change_points(emb, fps, cuts)
    del encoder  # free memory before the optional model loads

    meta = {
        **info, "fps": fps, "config": cfg.to_dict(), "device": device, "encoder": cfg.encoder,
        "logit_scale": scale, "logit_bias": bias, "labels": LABELS,
        "cuts": cuts, "change_points": cps, "vad": vad, "nc_highlight": None,
        "noncommercial_components": [],
        "timings": timings.stages, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Save before the optional stage so a failure there never loses the expensive work above.
    save_index(out, meta, grid, transcript)

    if cfg.use_nc_highlight:
        say("non-commercial highlight features")
        try:
            from vhe.signals.highlight_nc import NC_COMPONENT, NCHighlighter

            with timings.stage("load_nc_model"):
                hl = NCHighlighter(cfg.nc_checkpoint, device)
            with timings.stage("nc_highlight_encode"):
                chunks = hl.encode(Path(info["source"]), duration, cfg.nc_window_s, cfg.nc_overlap_s, out / "nc",
                                   progress=Counter(say, "highlight windows", every_s=0))
            if chunks:
                meta.update(nc_highlight={"component": NC_COMPONENT, "chunks": chunks},
                            noncommercial_components=[NC_COMPONENT])
        except Exception as exc:  # optional stage: keep the index without it
            say(f"skipping non-commercial highlight model: {exc}")
        meta["timings"] = timings.stages
        save_index(out, meta, grid, transcript)
    say("done")
    return out
