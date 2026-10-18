"""vtl index PATH | vtl find PATH "brief" [--json OUT] [--count N]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vtl.common import hms, video_files, write_json


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vtl", description="Minimal timeline-based clip finder.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="captions + transcript + event timeline for a file or folder")
    ix.add_argument("path", type=Path)
    ix.add_argument("--fps", type=float, default=1.0, help="baseline captions per second (default 1)")
    ix.add_argument("--max-fps", type=float, default=3.0, help="caption rate around sudden changes (default 3)")
    ix.add_argument("--force", action="store_true", help="recompute every stage")
    fd = sub.add_parser("find", help="clips for an editing brief")
    fd.add_argument("path", type=Path)
    fd.add_argument("brief")
    fd.add_argument("--json", type=Path, dest="json_out", help="write the clips JSON here")
    fd.add_argument("--count", type=int, help="number of clips (overrides the brief)")
    return ap


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress lines show up promptly in logs
    if a.cmd == "index":
        from vtl.index import index

        for m in index(a.path, fps=a.fps, max_fps=a.max_fps, force=a.force):
            print(json.dumps({k: m.get(k) for k in ("file", "duration", "timings", "sampling", "timeline_model",
                                                    "timeline_llm")}))
        return
    from vtl.find import default_out, find

    root, _ = video_files(a.path)
    res = find(root, a.brief, count=a.count)
    out = a.json_out or default_out(root, a.brief)
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
