"""vtl index PATH [--streamer [--crop x,y,w,h] [--facecam x,y,w,h]] [--exclusive-gpu]
vtl find PATH "brief" [--json OUT] [--count N] [--streamer]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vtl.common import hms, video_files, write_json


def _box(text: str) -> list[int]:
    from vtl.overlay import parse_box

    try:
        return parse_box(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vtl", description="Minimal timeline-based clip finder.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="captions + transcript + event timeline for a file or folder")
    ix.add_argument("path", type=Path)
    ix.add_argument("--fps", type=float, default=1.0, help="baseline captions per second (default 1)")
    ix.add_argument("--max-fps", type=float, default=3.0, help="caption rate around sudden changes (default 3)")
    ix.add_argument("--force", action="store_true", help="recompute every stage")
    ix.add_argument("--streamer", action="store_true", default=None,
                    help="streamer mode (also VTL_STREAMER=1): detect facecam/overlays, caption only the game area, "
                         "facecam reactions, on-screen game text; indexes into .vtl/streamer/")
    ix.add_argument("--crop", type=_box, help="streamer mode: game area x,y,w,h in video pixels (skips detection)")
    ix.add_argument("--facecam", type=_box, help="streamer mode: facecam x,y,w,h in video pixels (skips detection)")
    ix.add_argument("--exclusive-gpu", action="store_true",
                    help="stop semsearch.service for the run (restarted afterwards) and give the local timeline LLM "
                         "the whole GPU")
    fd = sub.add_parser("find", help="clips for an editing brief")
    fd.add_argument("path", type=Path)
    fd.add_argument("brief")
    fd.add_argument("--json", type=Path, dest="json_out", help="write the clips JSON here")
    fd.add_argument("--count", type=int, help="number of clips (overrides the brief)")
    fd.add_argument("--streamer", action="store_true", default=None,
                    help="use the streamer-mode index (also VTL_STREAMER=1)")
    return ap


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress lines show up promptly in logs
    from vtl.index import streamer_default

    streamer = streamer_default() if a.streamer is None else a.streamer
    if a.cmd == "index":
        from contextlib import nullcontext

        from vtl.gpu import exclusive
        from vtl.index import index

        with exclusive() if a.exclusive_gpu else nullcontext():
            metas = index(a.path, fps=a.fps, max_fps=a.max_fps, force=a.force, streamer=streamer, crop=a.crop,
                          facecam=a.facecam, exclusive_gpu=a.exclusive_gpu)
        keys = ("file", "duration", "timings", "sampling", "timeline_model", "timeline_llm")
        keys += ("mode", "layout", "reactions", "ocr") if streamer else ()
        for m in metas:
            print(json.dumps({k: m.get(k) for k in keys}))
        return
    from vtl.find import default_out, find

    root, _ = video_files(a.path)
    res = find(root, a.brief, count=a.count, streamer=streamer)
    out = a.json_out or default_out(root, a.brief, res["mode"] == "streamer")
    io = res.pop("_io")
    write_json(out, res)
    write_json(out.with_name(out.stem + ".llm.json"), io)
    for i, c in enumerate(res["clips"], 1):
        print(f"{i:2d}. {c['file']} {hms(c['start'], 1)}-{hms(c['end'], 1)} ({c['duration']:.1f} s) "
              f"[{c['label']}, {c['score']:g}] {c['reason']}")
    t, llm = res["timing"], res["llm"]
    print(f"{len(res['clips'])} clips from {res['n_candidates']} candidates in {t['total_s']} s "
          f"(candidates {t['candidates_s']} s, decision {t['decision_s']} s, ${llm['cost_usd']}) -> {out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
