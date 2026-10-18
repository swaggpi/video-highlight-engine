"""index: per file under <root>/.vtl/<file-slug>/: frames.jsonl, transcript.json, timeline.jsonl, meta.json.
Stage-major over files (all decodes+transcripts, then all captions, then all timelines) so each GPU model is loaded
once and released before the next. A stage reruns only when its inputs or version changed."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from vtl import frames as F
from vtl import llm as L
from vtl import speech as S
from vtl import timeline as T
from vtl.common import read_json, read_jsonl, sha, slug, video_files, write_json, write_jsonl


def _sig(video: Path) -> dict:
    st = video.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def index(path: Path, fps: float = 1.0, max_fps: float = 3.0, force: bool = False, log=print) -> list[dict]:
    root, videos = video_files(path)
    if not videos:
        raise SystemExit(f"no videos under {path}")
    jobs = []
    for v in videos:
        out = root / ".vtl" / slug(str(v.relative_to(root)))
        meta = read_json(out / "meta.json", {})
        if force or meta.get("source") != _sig(v):
            shutil.rmtree(out, ignore_errors=True)
            meta = {"file": str(v.relative_to(root)), "source": _sig(v), "timings": {}}
        out.mkdir(parents=True, exist_ok=True)
        jobs.append({"video": v, "out": out, "meta": meta})
    fkey = sha(F.VERSION, F.CAPTION_MODEL, F.CAPTION_TASK, fps, max_fps)

    for j in jobs:  # decode + transcript
        out, meta = j["out"], j["meta"]
        need_frames, need_tx = meta.get("frames_key") != fkey, meta.get("transcript_version") != S.VERSION
        if (need_frames and not (out / "dense").exists()) or (need_tx and not (out / "audio.wav").exists()):
            t0 = time.time()
            meta.update(F.decode(j["video"], out, max_fps))
            meta["timings"]["decode_s"] = round(time.time() - t0, 1)
            log(f"{meta['file']}: decoded {meta['n_dense']} frames in {meta['timings']['decode_s']} s")
        if need_tx:
            t0 = time.time()
            write_json(out / "transcript.json", S.transcribe(out / "audio.wav"))
            meta["transcript_version"] = S.VERSION
            meta["timings"]["whisper_s"] = round(time.time() - t0, 1)
            log(f"{meta['file']}: transcript in {meta['timings']['whisper_s']} s")
        write_json(out / "meta.json", meta)

    todo = [j for j in jobs if j["meta"].get("frames_key") != fkey]
    if todo:
        cap = F.Captioner()
        try:
            for j in todo:
                out, meta = j["out"], j["meta"]
                segs = read_json(out / "transcript.json", {}).get("segments", [])
                rows, stats = F.describe(out, meta, fps, segs, cap, log=log)
                write_jsonl(out / "frames.jsonl", rows)
                (out / "frames.partial.jsonl").unlink(missing_ok=True)
                meta.update(frames_key=fkey, frames_version=F.VERSION, sampling=stats)
                meta["timings"]["captions_s"] = stats["caption_s"]
                write_json(out / "meta.json", meta)
                log(f"{meta['file']}: {stats['captioned']} captions in {stats['caption_s']} s ({stats})")
        finally:
            cap.close()
    for j in jobs:  # the dense JPEGs and audio are only inputs of the stages above
        if j["meta"].get("frames_key") == fkey and j["meta"].get("transcript_version") == S.VERSION:
            shutil.rmtree(j["out"] / "dense", ignore_errors=True)
            (j["out"] / "audio.wav").unlink(missing_ok=True)

    model = L.model_for("timeline")
    for j in jobs:
        out, meta = j["out"], j["meta"]
        tkey = sha(T.VERSION, model, fkey, S.VERSION)
        if meta.get("timeline_key") == tkey and (out / "timeline.jsonl").exists():
            continue
        t0, n0 = time.time(), len(L.STATS)
        events = T.build(meta["file"], read_jsonl(out / "frames.jsonl"),
                         read_json(out / "transcript.json", {}).get("segments", []), meta["duration"],
                         meta["src_fps"], model, out / "cache" / "timeline", par=1 if L.is_local(model) else 4, log=log)
        write_jsonl(out / "timeline.jsonl", events)
        meta.update(timeline_key=tkey, timeline_version=T.VERSION, timeline_model=model)
        if len(L.STATS) > n0:  # all-cached rebuilds keep the cost of the run that made the cache
            meta["timings"]["timeline_s"] = round(time.time() - t0, 1)
            meta["timeline_llm"] = L.summary(n0)
        write_json(out / "meta.json", meta)
        log(f"{meta['file']}: {len(events)} events in {time.time() - t0:.0f} s ({L.summary(n0)})")
    L.unload(model)
    return [j["meta"] for j in jobs]
