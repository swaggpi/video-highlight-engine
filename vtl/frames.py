"""Per-frame descriptions: decode a dense frame grid, pick which frames to caption (adaptive), caption them.

Sampling: a baseline of `fps` (default 1) captions per second, densified up to `max_fps` (default 3) only within
+-1 s of a sudden change: a visual jump between consecutive dense frames, an audio loudness onset, or speech starting
after a pause. Baseline frames that look the same as the last captioned frame reuse its caption ("dup_of").
Captioner prototype: Florence-2-base (MIT). Its remote code imports flash_attn and timm; neither is installed or
needed at inference, so the import check is patched and timm's two helpers are shimmed.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np

from vtl.common import read_jsonl

VERSION = "frames-1"
CAPTION_MODEL = "microsoft/Florence-2-base"
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
FRAME_SHORT_SIDE = 448
# Spike thresholds, tuned on the 85-min stream VOD (+40% frames; catches the 11:16 jump scare at 678.3 s).
CONTEXT_S = 10.0      # spikes are measured against the median of the previous 10 s
VIS_FLOOR = 0.06      # minimum mean abs grey change (0..1) of 32x18 thumbs for a visual spike ...
VIS_RATIO = 5.0       # ... that is also 5x the recent median change
AUDIO_JUMP_DB = 18.0  # loudness this far above the recent median level is an onset
SPEECH_GAP_S = 1.5    # speech after at least this much silence counts as an onset
DENSE_PAD_S = 1.0
DUP_DIFF = 0.012      # thumbs closer than this to the last captioned frame reuse its caption


def decode(video: Path, out: Path, max_fps: float) -> dict:
    """One ffmpeg pass: dense/%06d.jpg at max_fps (frame i is at t = i / max_fps) and audio.wav (16 kHz mono)."""
    from vhe.ingest.decode import hwaccel_choice, probe

    p = probe(video)
    dense = out / "dense"
    if dense.exists():
        for f in dense.glob("*.jpg"):
            f.unlink()
    dense.mkdir(parents=True, exist_ok=True)
    (out / "frames.partial.jsonl").unlink(missing_ok=True)  # resumable captions belong to the old frame grid
    h = FRAME_SHORT_SIDE
    vf = f"fps={max_fps}:round=up,scale='if(gt(iw,ih),-2,{h})':'if(gt(iw,ih),{h},-2)',format=yuvj420p"
    tail = ["-i", str(video), "-map", "0:v:0", "-vf", vf, "-q:v", "4", "-start_number", "0", str(dense / "%06d.jpg")]
    if p.has_audio:
        tail += ["-map", "0:a:0", "-ac", "1", "-ar", "16000", str(out / "audio.wav")]
    accel, _ = hwaccel_choice("auto")
    cmd = ["ffmpeg", "-v", "error", "-y", *([] if accel == "none" else ["-hwaccel", accel]), *tail]
    if subprocess.run(cmd).returncode != 0:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *tail], check=True)
    return {"duration": p.duration, "src_fps": p.fps, "width": p.width, "height": p.height,
            "has_audio": p.has_audio, "max_fps": max_fps, "n_dense": len(list(dense.glob("*.jpg")))}


def visual_change(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """(32x18 grey thumbnails, mean abs difference (0..1) of each to the previous one, [0] = 0)."""
    import cv2

    thumbs = [cv2.resize(cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8), (32, 18), interpolation=cv2.INTER_AREA)
              for p in paths]
    g = np.stack(thumbs).astype(np.float32) / 255.0
    return g, np.concatenate([[0.0], np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))])


def loudness_db(wav: Path, n: int, rate: float) -> np.ndarray:
    """RMS level (dB) per dense step (n steps of 1/rate s)."""
    if not wav.exists():
        return np.full(n, -90.0)
    with wave.open(str(wav)) as w:
        y = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        sr = w.getframerate()
    hop = int(sr / rate)
    y = np.pad(y, (0, max(0, n * hop - len(y))))[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((y ** 2).mean(axis=1)) + 1e-6)


def _prev_median(x: np.ndarray, w: int) -> np.ndarray:
    """Median of the w values before each position (edge-padded)."""
    p = np.pad(x, (w, 0), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, w)[: len(x)], axis=1)


def spikes(vis: np.ndarray, db: np.ndarray, speech_starts: list[float], rate: float) -> dict[str, list[int]]:
    """Dense indices of sudden changes, by kind, each relative to the previous CONTEXT_S seconds."""
    w = max(1, int(CONTEXT_S * rate))
    jump = (vis > VIS_FLOOR) & (vis > VIS_RATIO * _prev_median(vis, w))
    loud = db - _prev_median(db, w) > AUDIO_JUMP_DB
    return {"visual": [int(i) for i in np.flatnonzero(jump)], "audio": [int(i) for i in np.flatnonzero(loud)],
            "speech": [int(round(t * rate)) for t in speech_starts]}


def speech_onsets(segments: list[dict], gap: float = SPEECH_GAP_S) -> list[float]:
    out, last_end = [], -1e9
    for s in segments:
        if s["start"] - last_end >= gap:
            out.append(s["start"])
        last_end = s["end"]
    return out


def select(n: int, step: int, spike_idx: list[int], pad: int) -> dict[int, str]:
    """{dense index: 'base' | 'dense'}: every step-th frame, plus all frames within +-pad of a spike."""
    kinds = {i: "base" for i in range(0, n, step)}
    for s in spike_idx:
        for i in range(max(0, s - pad), min(n, s + pad + 1)):
            kinds.setdefault(i, "dense")
    return dict(sorted(kinds.items()))


def trim_caption(text: str) -> str:
    """Florence stops mid-sentence at the token limit: drop the unfinished tail."""
    text = text.strip()
    cut = text.rfind(". ")
    return text[: cut + 1] if cut > 0 and not text.endswith(".") else text


class Captioner:
    """caption(list of PIL images) -> list of str. fp32 by default on GTX 16xx (no tensor cores: fp16 matmuls ran
    ~9x slower than fp32 there), fp16 elsewhere; override with VTL_CAPTION_DTYPE=float16|float32."""

    def __init__(self, device: str = "cuda"):
        import torch
        from unittest import mock
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        if not torch.cuda.is_available():
            device = "cpu"
        want = os.environ.get("VTL_CAPTION_DTYPE", "auto")
        if want == "auto":
            slow16 = device == "cpu" or "GTX 16" in torch.cuda.get_device_name()
            want = "float32" if slow16 else "float16"
        self.dtype, self.device, self.torch = getattr(torch, want), device, torch
        if "timm.layers" not in sys.modules:
            timm, layers = types.ModuleType("timm"), types.ModuleType("timm.layers")
            layers.DropPath = lambda *a, **k: torch.nn.Identity()  # stochastic depth: identity at inference
            layers.trunc_normal_ = torch.nn.init.trunc_normal_
            timm.layers = layers
            sys.modules.update({"timm": timm, "timm.layers": layers})

        def imports(f):
            return [i for i in get_imports(f) if i not in ("flash_attn", "timm")]

        with mock.patch("transformers.dynamic_module_utils.get_imports", imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                CAPTION_MODEL, trust_remote_code=True, torch_dtype=self.dtype,
                attn_implementation="eager").to(device).eval()
            self.proc = AutoProcessor.from_pretrained(CAPTION_MODEL, trust_remote_code=True)

    def caption(self, images) -> list[str]:
        inp = self.proc(text=[CAPTION_TASK] * len(images), images=images, return_tensors="pt")
        inp = inp.to(self.device, self.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                      max_new_tokens=64, num_beams=1, do_sample=False)
        return [trim_caption(t) for t in self.proc.batch_decode(out, skip_special_tokens=True)]

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def describe(out: Path, meta: dict, fps: float, segments: list[dict], captioner, batch: int = 4,
             log=print) -> tuple[list[dict], dict]:
    """frames.jsonl rows {frame, t, text, kind[, dup_of]} and sampling stats. Resumable via frames.partial.jsonl."""
    from PIL import Image

    rate, src_fps = meta["max_fps"], meta["src_fps"]
    paths = sorted((out / "dense").glob("*.jpg"))
    thumbs, vis = visual_change(paths)
    db = loudness_db(out / "audio.wav", len(paths), rate)
    sp = spikes(vis, db, speech_onsets(segments), rate)
    step = max(1, round(rate / fps))
    kinds = select(len(paths), step, sorted({i for v in sp.values() for i in v}), round(DENSE_PAD_S * rate))
    partial = out / "frames.partial.jsonl"
    done = {r["i"]: r for r in read_jsonl(partial)}
    rows, last, todo = [], None, []
    for i, kind in kinds.items():  # duplicates are decided up front so batches stay full
        if kind == "base" and last is not None and np.abs(thumbs[i] - thumbs[last]).mean() < DUP_DIFF:
            rows.append({"i": i, "kind": kind, "dup_of_i": last})
            continue
        rows.append({"i": i, "kind": kind})
        last = i
        if i not in done:
            todo.append(i)
    t0, shown = time.time(), time.time()
    with partial.open("a") as fh:
        for b in range(0, len(todo), batch):
            idx = todo[b:b + batch]
            texts = captioner.caption([Image.open(paths[i]).convert("RGB") for i in idx])
            for i, text in zip(idx, texts):
                done[i] = {"i": i, "text": text}
                fh.write(json.dumps(done[i]) + "\n")
            if time.time() - shown > 60:
                shown = time.time()
                log(f"  captions {b + len(idx)}/{len(todo)} ({(b + len(idx)) / (time.time() - t0):.1f}/s)")
    result = []
    for r in rows:
        i = r["i"]
        t = round(i / rate, 3)
        rec = {"frame": int(round(t * src_fps)), "t": t, "kind": r["kind"]}
        if "dup_of_i" in r:
            rec["text"] = done[r["dup_of_i"]]["text"]
            rec["dup_of"] = int(round(r["dup_of_i"] / rate * src_fps))
        else:
            rec["text"] = done[i]["text"]
        result.append(rec)
    n_dup = sum("dup_of" in r for r in result)
    stats = {"dense_frames": len(paths), "baseline": sum(r["kind"] == "base" for r in result),
             "densified": sum(r["kind"] == "dense" for r in result), "dups_reused": n_dup,
             "captioned": len(result) - n_dup, "spikes": {k: len(v) for k, v in sp.items()},
             "caption_s": round(time.time() - t0, 1)}
    return result, stats
