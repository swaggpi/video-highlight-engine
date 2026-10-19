"""index: per file under <root>/.vtl/<file-slug>/: frames.jsonl, transcript.json, timeline.jsonl, meta.json.
Stage-major over files (all decodes+transcripts, then all captions, then all timelines) so each GPU model is loaded
once and released before the next. A stage reruns only when its inputs or version changed.

Streamer mode (--streamer / VTL_STREAMER=1) indexes into <root>/.vtl/streamer/<file-slug>/ instead, so the two
modes never share files or caches, and every stage key includes the mode. It adds, per file:
  layout   facecam box, overlay panels and the game-area crop (vtl.overlay; meta["layout"], layout.jpg to check it)
  frames   captions of the game-area crop only (same adaptive sampling, judged on the crop)
  facecam  short facecam reaction captions around loudness onsets and speech starts (facecam.jsonl)
  ocr      on-screen game text at 1 fps on the crop (ocr.jsonl) folded into text cards (cards.jsonl)
and the timeline prompt gets the cards (TEXT) and reactions (CAM) as extra input lines."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from vtl import frames as F
from vtl import llm as L
from vtl import ocr as O
from vtl import overlay as OV
from vtl import speech as S
from vtl import timeline as T
from vtl.common import read_json, read_jsonl, sha, slug, video_files, write_json, write_jsonl


def _sig(video: Path) -> dict:
    st = video.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def streamer_default() -> bool:
    return os.environ.get("VTL_STREAMER", "").strip().lower() not in ("", "0", "false", "no", "off")


def index_dir(root: Path, streamer: bool = False) -> Path:
    """Where a mode's per-file folders live: <root>/.vtl (plain) or <root>/.vtl/streamer."""
    return root / ".vtl" / "streamer" if streamer else root / ".vtl"


def stage_keys(meta: dict, fps: float, max_fps: float, streamer: bool, lkey: str | None = None) -> dict:
    """The key every dense-frame stage must have in meta to be current. Plain mode: only frames_key (unchanged from
    before streamer mode existed). Streamer mode: layout_key, and keys that depend on the detected layout (None while
    the layout is not known yet, i.e. stale)."""
    plain = sha(F.VERSION, F.CAPTION_MODEL, F.CAPTION_TASK, fps, max_fps)
    if not streamer:
        return {"frames_key": plain}
    lay = meta.get("layout") if meta.get("layout_key") == lkey else None
    if lay is None:
        return {"layout_key": lkey, "frames_key": None, "react_key": None, "ocr_key": None}
    crop, cam = lay["crop_dense"], lay["facecam_dense"]
    return {"layout_key": lkey,
            "frames_key": sha(F.VERSION, F.CAPTION_MODEL, F.CAPTION_TASK, fps, max_fps, "streamer", crop),
            "react_key": sha(F.REACT_VERSION, F.REACT_TASK, F.REACT_GAP_S, cam, S.VERSION, max_fps) if cam else "none",
            "ocr_key": sha(O.VERSION, O.MIN_CONF, crop, fps, max_fps)}


def _stale(meta: dict, keys: dict) -> list[str]:
    return [k for k, v in keys.items() if v is None or meta.get(k) != v]


def _layout(j: dict, crop, facecam, log) -> dict:
    """Detect (or take the manual) overlay layout on the dense frames; boxes in source pixels and dense pixels."""
    import cv2

    out, meta = j["out"], j["meta"]
    paths = sorted((out / "dense").glob("*.jpg"))
    dw = cv2.imread(str(paths[0])).shape[1]
    s = dw / meta["width"]  # source pixels -> dense-frame pixels
    lay = OV.detect(paths, OV.scale_box(facecam, s), OV.scale_box(crop, s))
    rec = {"facecam": OV.scale_box(lay["facecam"], 1 / s), "panels": [OV.scale_box(b, 1 / s) for b in lay["panels"]],
           "crop": OV.scale_box(lay["crop"], 1 / s), "crop_share": lay["crop_share"],
           "facecam_dense": lay["facecam"], "crop_dense": lay["crop"], "dense_scale": round(s, 5),
           "manual": {"crop": crop, "facecam": facecam}, "samples": lay["samples"],
           "face_frames": lay["face_frames"], "notes": lay["notes"], "debug_image": str(out / "layout.jpg")}
    OV.draw(paths[len(paths) // 2], lay, out / "layout.jpg")
    log(f"{meta['file']}: layout (source px x,y,w,h) facecam {rec['facecam']}, panels {rec['panels']}, game crop "
        f"{rec['crop']} ({rec['crop_share']:.0%} of the frame){'; ' + '; '.join(lay['notes']) if lay['notes'] else ''}"
        f" -> {out / 'layout.jpg'}")
    return rec


def index(path: Path, fps: float = 1.0, max_fps: float = 3.0, force: bool = False, log=print,
          streamer: bool = False, crop: list[int] | None = None, facecam: list[int] | None = None,
          exclusive_gpu: bool = False) -> list[dict]:
    """crop / facecam: manual [x, y, w, h] overrides in source-video pixels (streamer mode only).
    exclusive_gpu: the caller has stopped other GPU users (vtl.gpu.exclusive); the local timeline LLM gets the whole
    GPU (other Ollama models unloaded, all layers offloaded)."""
    if (crop or facecam) and not streamer:
        raise SystemExit("--crop / --facecam need streamer mode (--streamer or VTL_STREAMER=1)")
    root, videos = video_files(path)
    if not videos:
        raise SystemExit(f"no videos under {path}")
    base = index_dir(root, streamer)
    lkey = sha(OV.VERSION, crop, facecam) if streamer else None
    jobs = []
    for v in videos:
        out = base / slug(str(v.relative_to(root)))
        meta = read_json(out / "meta.json", {})
        if force or meta.get("source") != _sig(v):
            shutil.rmtree(out, ignore_errors=True)
            meta = {"file": str(v.relative_to(root)), "source": _sig(v), "timings": {}}
            if streamer:
                meta["mode"] = "streamer"
        out.mkdir(parents=True, exist_ok=True)
        jobs.append({"video": v, "out": out, "meta": meta})

    def keys(meta):
        return stage_keys(meta, fps, max_fps, streamer, lkey)

    for j in jobs:  # decode + transcript
        out, meta = j["out"], j["meta"]
        need_frames, need_tx = bool(_stale(meta, keys(meta))), meta.get("transcript_version") != S.VERSION
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

    if streamer:  # layout (CPU, seconds)
        for j in jobs:
            meta = j["meta"]
            if meta.get("layout_key") != lkey:
                t0 = time.time()
                meta["layout"] = _layout(j, crop, facecam, log)
                meta["layout_key"] = lkey
                meta["timings"]["layout_s"] = round(time.time() - t0, 1)
                write_json(j["out"] / "meta.json", meta)

    todo = [j for j in jobs if {"frames_key", "react_key"} & set(_stale(j["meta"], keys(j["meta"])))]
    if todo:
        cap = F.Captioner()
        try:
            for j in todo:
                out, meta = j["out"], j["meta"]
                k = keys(meta)
                segs = read_json(out / "transcript.json", {}).get("segments", [])
                if meta.get("frames_key") != k["frames_key"]:
                    box = meta["layout"]["crop_dense"] if streamer else None
                    rows, stats = F.describe(out, meta, fps, segs, cap, log=log, crop=box)
                    write_jsonl(out / "frames.jsonl", rows)
                    (out / "frames.partial.jsonl").unlink(missing_ok=True)
                    meta.update(frames_key=k["frames_key"], frames_version=F.VERSION, sampling=stats)
                    meta["timings"]["captions_s"] = stats["caption_s"]
                    write_json(out / "meta.json", meta)
                    log(f"{meta['file']}: {stats['captioned']} captions in {stats['caption_s']} s ({stats})")
                if streamer and meta.get("react_key") != k["react_key"]:
                    cam = meta["layout"]["facecam_dense"]
                    rows, stats = F.reactions(out, meta, segs, cap, cam, log=log) if cam else ([], {"reactions": 0})
                    write_jsonl(out / "facecam.jsonl", rows)
                    meta.update(react_key=k["react_key"], reactions=stats)
                    meta["timings"]["reactions_s"] = stats.get("react_s", 0.0)
                    write_json(out / "meta.json", meta)
                    log(f"{meta['file']}: facecam reactions {stats}")
        finally:
            cap.close()

    if streamer:  # on-screen text (CPU)
        for j in jobs:
            out, meta = j["out"], j["meta"]
            k = keys(meta)
            if meta.get("ocr_key") == k["ocr_key"] and meta.get("cards_version") == O.CARDS_VERSION:
                continue
            if meta.get("ocr_key") != k["ocr_key"]:
                t0 = time.time()
                paths = sorted((out / "dense").glob("*.jpg"))
                step = max(1, round(meta["max_fps"] / fps))
                rows = O.read(paths[::step], [i * step / meta["max_fps"] for i in range(len(paths[::step]))],
                              meta["layout"]["crop_dense"], log=log)
                write_jsonl(out / "ocr.jsonl", rows)
                meta["timings"]["ocr_s"] = round(time.time() - t0, 1)
                log(f"{meta['file']}: OCR of {len(rows)} frames in {meta['timings']['ocr_s']} s")
            rows = read_jsonl(out / "ocr.jsonl")  # cards alone are rebuilt from the saved reads
            cards = O.cards(rows)
            write_jsonl(out / "cards.jsonl", cards)
            meta.update(ocr_key=k["ocr_key"], cards_version=O.CARDS_VERSION,
                        ocr={"frames": len(rows), "with_text": sum(bool(r["lines"]) for r in rows),
                             "cards": len(cards), "chrome": sum(c["chrome"] for c in cards)})
            write_json(out / "meta.json", meta)
            log(f"{meta['file']}: {len(cards)} text cards ({meta['ocr']['chrome']} interface chrome)")

    for j in jobs:  # the dense JPEGs and audio are only inputs of the stages above
        if not _stale(j["meta"], keys(j["meta"])) and j["meta"].get("transcript_version") == S.VERSION:
            shutil.rmtree(j["out"] / "dense", ignore_errors=True)
            (j["out"] / "audio.wav").unlink(missing_ok=True)

    model = L.model_for("timeline")
    if exclusive_gpu:
        from vtl.gpu import ollama_alone

        ollama_alone(model, log)
    try:
        for j in jobs:
            out, meta = j["out"], j["meta"]
            k = keys(meta)
            tkey = sha(T.VERSION, model, k["frames_key"], S.VERSION)
            extra = None
            if streamer:
                tkey = sha(T.VERSION, T.STREAMER_VERSION, model, k, O.CARDS_VERSION, S.VERSION)
                extra = {"cards": read_jsonl(out / "cards.jsonl"), "cams": read_jsonl(out / "facecam.jsonl")}
            if meta.get("timeline_key") == tkey and (out / "timeline.jsonl").exists():
                continue
            t0, n0, failed = time.time(), len(L.STATS), []
            events = T.build(meta["file"], read_jsonl(out / "frames.jsonl"),
                             read_json(out / "transcript.json", {}).get("segments", []), meta["duration"],
                             meta["src_fps"], model, out / "cache" / "timeline", par=1 if L.is_local(model) else 4,
                             log=log, streamer=extra, failed=failed)
            write_jsonl(out / "timeline.jsonl", events)
            # a timeline with failed chunks stays stale, so the next `vtl index` retries just those (the rest is cached)
            meta.update(timeline_key=None if failed else tkey, timeline_version=T.VERSION, timeline_model=model,
                        timeline_failed_chunks=failed)
            if failed:
                log(f"{meta['file']}: {len(failed)} timeline chunk(s) failed and have no events; run vtl index again "
                    f"to retry them")
            if len(L.STATS) > n0:  # all-cached rebuilds keep the cost of the run that made the cache
                meta["timings"]["timeline_s"] = round(time.time() - t0, 1)
                meta["timeline_llm"] = L.summary(n0)
            write_json(out / "meta.json", meta)
            log(f"{meta['file']}: {len(events)} events in {time.time() - t0:.0f} s ({L.summary(n0)})")
    finally:
        L.OLLAMA_OPTIONS.pop("num_gpu", None)
    L.unload(model)
    return [j["meta"] for j in jobs]
