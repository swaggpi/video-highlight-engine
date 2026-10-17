from __future__ import annotations

import argparse
import json
from pathlib import Path

from vhe.config import Config


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def cmd_index(a: argparse.Namespace) -> None:
    import time

    from vhe.pipeline import build_index
    from vhe.timing import format_report, hms

    cfg = Config(device=a.device, use_nc_highlight=not a.no_nc, asr_model=a.asr, asr_language=a.lang,
                 encoder=a.encoder, frame_height=a.frame_height, hwaccel=a.hwaccel)
    t0 = time.perf_counter()
    # flush: stdout is block-buffered when redirected to a log file
    out = build_index(Path(a.video), Path(a.out) if a.out else None, cfg, keep_cache=a.keep_cache,
                      progress=lambda m: print(f"  [{hms(time.perf_counter() - t0)}] {m}", flush=True))
    meta = json.loads((out / "meta.json").read_text())
    print(f"index: {out}")
    print(format_report(meta["timings"], meta["duration"]))
    if meta["noncommercial_components"]:
        print("NON-COMMERCIAL components used:", ", ".join(meta["noncommercial_components"]))


def cmd_search(a: argparse.Namespace) -> None:
    from vhe.engine import Engine

    for r in Engine(Path(a.index), a.device).search(a.text, k=a.k):
        extra = f"  \"{r['text']}\"" if "text" in r else ""
        print(f"{_fmt(r['start'])}-{_fmt(r['end'])}  {r['score']:.3f}  {r['source']}{extra}")


def cmd_clips(a: argparse.Namespace) -> None:
    from vhe.engine import Engine

    eng = Engine(Path(a.index), a.device, use_nc=not a.no_nc)
    res = eng.clips(a.query, count=a.n, min_len=a.min_len, max_len=a.max_len, llm=a.llm, budget_s=a.budget)
    q = res["query"]
    print(f"parsed: text={q['text']!r} mood={q['mood']} n={q['count']} len={q['min_len']:.0f}-{q['max_len']:.0f}s "
          f"weights={q['weights']}  ({res.get('candidates', 0)} candidates, ranking {res['seconds']:.2f}s, "
          f"model load {res['load_seconds']:.1f}s)")
    for c in res["clips"]:
        print(f"{_fmt(c['start'])}-{_fmt(c['end'])}  score {c['score']:+.2f}  rel {c['relevance']:+.2f}  "
              f"qual {c['quality']:+.2f}  energy {c['energy']:+.2f}  {', '.join(c['labels'])}")
    if res.get("noncommercial"):
        print("NON-COMMERCIAL components used:", ", ".join(res["noncommercial"]))
    if a.export:
        from vhe.export.clips import export_clips

        paths = export_clips(eng.index.video, res["clips"], Path(a.export))
        print(f"exported {len(paths)} clips to {a.export}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))


def cmd_info(a: argparse.Namespace) -> None:
    from vhe.timing import format_report

    meta = json.loads((Path(a.index) / "meta.json").read_text())
    print(f"video: {meta['video']}  duration {meta['duration']:.1f}s  frames {meta['n_frames']}  "
          f"cuts {len(meta['cuts'])}  change points {len(meta['change_points'])}  encoder {meta['encoder']}")
    print(format_report(meta["timings"], meta["duration"]))
    print("non-commercial components:", meta["noncommercial_components"] or "none")


def cmd_ui(a: argparse.Namespace) -> None:
    from vhe.ui.server import serve

    serve(a.port, a.device)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vhe", description="Local video highlight and search engine")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("index", help="index a video once")
    p.add_argument("video")
    p.add_argument("--out")
    p.add_argument("--no-nc", action="store_true", help="skip the non-commercial highlight model")
    p.add_argument("--asr", default="small", help="faster-whisper size, or 'none'")
    p.add_argument("--lang", default=None, help="speech language, e.g. en or de (default: detect)")
    p.add_argument("--encoder", default=Config.encoder)
    p.add_argument("--frame-height", type=int, default=Config.frame_height)
    p.add_argument("--hwaccel", default="auto")
    p.add_argument("--keep-cache", action="store_true", help="keep decoded frames and audio")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="free-text search")
    p.add_argument("index")
    p.add_argument("text")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("clips", help="ranked highlight clips for a query")
    p.add_argument("index")
    p.add_argument("query")
    p.add_argument("-n", type=int, default=None)
    p.add_argument("--min-len", type=float, default=None)
    p.add_argument("--max-len", type=float, default=None)
    p.add_argument("--llm", default=None, help="Ollama model for query parsing (default: rules)")
    p.add_argument("--budget", type=float, default=None, help="total seconds for all clips together")
    p.add_argument("--no-nc", action="store_true")
    p.add_argument("--export", default=None, help="write clip files to this folder")
    p.add_argument("--json", default=None, help="write the full result as JSON")
    p.set_defaults(func=cmd_clips)

    p = sub.add_parser("info", help="index summary and timings")
    p.add_argument("index")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("ui", help="minimal local web UI")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_ui)

    for sp in sub.choices.values():
        sp.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
