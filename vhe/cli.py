from __future__ import annotations

import argparse
import json
from pathlib import Path

from vhe.config import Config
from vhe.query.parser import DASHES


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def cmd_index(a: argparse.Namespace) -> None:
    import time

    from vhe.pipeline import build_index
    from vhe.timing import format_report, hms

    cfg = Config(device=a.device, use_nc_highlight=not a.no_nc, asr_model=a.asr, asr_language=a.lang,
                 encoder=a.encoder, frame_height=a.frame_height, hwaccel=a.hwaccel)
    t0 = time.perf_counter()

    def say(m: str) -> None:
        print(f"  [{hms(time.perf_counter() - t0)}] {m}", flush=True)  # flush: block-buffered when redirected

    if Path(a.video).is_dir():
        if a.out:
            raise SystemExit("--out applies to single files; a folder is indexed into <DIR>/.vhe/")
        from vhe.project import index_project

        manifest = index_project(Path(a.video), cfg, progress=say, keep_cache=a.keep_cache)
        for e in manifest["files"]:
            extra = (f"  usable {100 * e['usable_fraction']:.0f}%" if e.get("usable_fraction") is not None else "")
            extra += f"  {e['language']}" if e.get("language") else ""
            extra += f"  {e['error']}" if e.get("error") else ""
            print(f"{e['status']:<10}{e['rel_path']}  {e.get('recorded_at') or '-'}{extra}")
        counts = {s: sum(e["status"] == s for e in manifest["files"]) for s in ("ok", "too_short", "failed")}
        print(f"project: {Path(a.video) / '.vhe'}  {counts}  language {manifest['language'] or '-'}  "
              f"{hms(time.perf_counter() - t0)}")
        return
    out = build_index(Path(a.video), Path(a.out) if a.out else None, cfg, keep_cache=a.keep_cache, progress=say)
    meta = json.loads((out / "meta.json").read_text())
    print(f"index: {out}")
    print(format_report(meta["timings"], meta["duration"]))
    if meta["noncommercial_components"]:
        print("NON-COMMERCIAL components used:", ", ".join(meta["noncommercial_components"]))


def _budget(v: str | None) -> float | tuple[float, float] | None:
    """--budget 75 or --budget 60-90 (seconds, all clips together)."""
    if not v:
        return None
    lo, _, hi = DASHES.sub("-", v).partition("-")
    return (float(lo), float(hi)) if hi else float(lo)


def edit_json(res: dict) -> dict:
    """The edit list in edit order; start/end are seconds in each clip's own file."""
    return {"query": res["query"]["raw"], "parser": res["query"].get("parser"), "parsed": res["query"],
            "clips": [{"file": c["file"], "start": c["file_start"], "end": c["file_end"], "duration": c["duration"],
                       "score": c["score"], "reason": c["reason"], "source": c["source"], "role": c.get("role"),
                       "item": c.get("item")} for c in res["clips"]]}


def cmd_search(a: argparse.Namespace) -> None:
    from vhe.engine import Engine

    for r in Engine.open(Path(a.path), a.device).search(a.text, k=a.k):
        extra = f"  \"{r['text']}\"" if "text" in r else ""
        print(f"{r['file']}  {_fmt(r['file_start'])}-{_fmt(r['file_end'])}  {r['score']:+.3f}  {r['source']}{extra}")


def cmd_clips(a: argparse.Namespace) -> None:
    from vhe.engine import Engine

    eng = Engine.open(Path(a.path), a.device, use_nc=not a.no_nc, scorer=a.scorer)
    res = eng.clips(a.query, count=a.count, min_len=a.min_len, max_len=a.max_len, llm=None if a.no_llm else a.llm,
                    budget_s=_budget(a.budget), persona=a.persona, dump=a.dump_candidates, dump_all=a.dump_all)
    q = res["query"]
    budget = f" budget={q['budget'][0]:.0f}-{q['budget'][1]:.0f}s" if q.get("budget") else ""
    edges = "".join(f" {k}={q[k]!r}" for k in ("opener", "closer") if q.get(k))
    note = f" ({q['parser_note']})" if q.get("parser_note") else ""
    print(f"parsed by {q['parser']}{note}: text={q['text']!r} mood={q['mood']} n={q['count']} "
          f"len={q['min_len']:.0f}-{q['max_len']:.0f}s{budget}{edges} weights={q['weights']}")
    for key in ("shots", "avoid", "topics"):
        if q.get(key):
            print(f"  {key}: " + "; ".join(q[key]))
    if q.get("need_live_audio") or q.get("soundbite_len"):
        bite = f" soundbite={q['soundbite_len'][0]:.0f}-{q['soundbite_len'][1]:.0f}s" if q.get("soundbite_len") else ""
        print(f"  live audio: {'required' if q.get('need_live_audio') else 'any'}{bite}")
    print(f"{res['files']} files, {res.get('candidates', 0)} candidates ({res.get('unusable_dropped', 0)} unusable), "
          f"ranking {res['seconds']:.2f}s, model load {res['load_seconds']:.1f}s")
    for i, c in enumerate(res["clips"], 1):
        said = f'  "{c["transcript"]}"' if c["transcript"] else ""
        item = f"  [{c['item']}]" if c.get("item") else ""
        print(f"{i:2d}. {c['file']}  {_fmt(c['file_start'])}-{_fmt(c['file_end'])} ({c['duration']:.1f}s)  "
              f"score {c['score']:+.2f}{item}  {c['reason']}{said}")
    print(f"total {sum(c['duration'] for c in res['clips']):.1f}s in {len(res['clips'])} clips")
    if res.get("noncommercial"):
        print("NON-COMMERCIAL components used:", ", ".join(res["noncommercial"]))
    if a.json:
        Path(a.json).write_text(json.dumps(edit_json(res), indent=1, ensure_ascii=False))
    if a.dump_candidates:
        print(f"scorer {eng.scorer.name}: candidates dumped to {a.dump_candidates}")
    if a.export:
        from vhe.export.clips import export_clips

        out = export_clips(res["clips"], Path(a.export), eng.corpus.source_map,
                           with_rough_cut=a.rough_cut)
        print(f"exported {len(out['clips'])} clips, {out['csv'].name}, {out['edl'].name}"
              f"{', ' + out['rough_cut'].name if out['rough_cut'] else ''} to {a.export}")


def cmd_features(a: argparse.Namespace) -> None:
    from vhe.engine import Engine
    from vhe.rank.dump import load_spans, write

    eng = Engine.open(Path(a.path), a.device, use_nc=not a.no_nc, scorer=a.scorer)
    doc = eng.features_for_spans(a.brief, load_spans(a.spans), persona=a.persona, llm=None if a.no_llm else a.llm)
    write(a.out, doc)
    print(f"{len(doc['candidates'])} spans, {len(doc['feature_names'])} features ({doc['feature_version']}) "
          f"-> {a.out}")


def cmd_info(a: argparse.Namespace) -> None:
    from vhe.project import read_manifest
    from vhe.timing import format_report

    manifest = read_manifest(Path(a.index))
    if manifest is not None:
        for e in manifest["files"]:
            print(f"{e['status']:<10}{e['rel_path']}  {e.get('duration') or 0:.1f}s  {e.get('recorded_at') or '-'}")
        return
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

    p = sub.add_parser("index", help="index a video, or a folder of footage (incremental)")
    p.add_argument("video", help="video file, or a folder indexed into <DIR>/.vhe/")
    p.add_argument("--out")
    p.add_argument("--no-nc", action="store_true", help="skip the non-commercial highlight model")
    p.add_argument("--asr", default="small", help="faster-whisper size, or 'none'")
    p.add_argument("--lang", default=None, help="speech language, e.g. en or de (default: detect)")
    p.add_argument("--encoder", default=Config.encoder)
    p.add_argument("--frame-height", type=int, default=Config.frame_height)
    p.add_argument("--hwaccel", default="auto")
    p.add_argument("--keep-cache", action="store_true", help="keep decoded frames and audio")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="free-text search over a video or a project folder")
    p.add_argument("path", help="project folder, video file, or per-video index dir")
    p.add_argument("text")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("clips", help="best clips for an editor's brief, in edit order")
    p.add_argument("path", help="project folder, video file, or per-video index dir")
    p.add_argument("query", help='e.g. "60-90 s energetic recap, open with an establishing shot"')
    p.add_argument("--count", "-n", type=int, default=None, help="number of clips (default: from the brief)")
    p.add_argument("--min-len", type=float, default=None)
    p.add_argument("--max-len", type=float, default=None)
    p.add_argument("--llm", default="auto",
                   help="Ollama model that parses the brief (default: $VHE_LLM or qwen3:4b when reachable, "
                        "else the rules)")
    p.add_argument("--no-llm", action="store_true", help="parse the brief with the rules only")
    p.add_argument("--budget", default=None, help="total seconds for all clips together: 75 or 60-90")
    p.add_argument("--no-nc", action="store_true")
    p.add_argument("--export", default=None, help="write clips, clips.csv and edit.edl to this folder")
    p.add_argument("--rough-cut", action="store_true", help="with --export: also a joined preview rough_cut.mp4")
    p.add_argument("--json", default=None, help="write the edit list (file-local times) as JSON")
    p.add_argument("--scorer", default=None, help="ranking model file (vhe-linear-rank/1); default: heuristic")
    p.add_argument("--persona", default=None, help="who the edit is for (an id, e.g. p2-tiktok)")
    p.add_argument("--dump-candidates", default=None, metavar="OUT.json",
                   help="write every scored candidate with its features (vhe-candidates/1)")
    p.add_argument("--dump-all", action="store_true",
                   help="with --dump-candidates: also every gated candidate before refinement (large)")
    p.set_defaults(func=cmd_clips)

    p = sub.add_parser("features", help="feature vectors for given spans under a brief (vhe-candidates/1)")
    p.add_argument("path", help="project folder, video file, or per-video index dir")
    p.add_argument("--brief", required=True)
    p.add_argument("--spans", required=True,
                   help="JSON: [{file, start, end}] in file-local seconds, or a `clips --json` edit list")
    p.add_argument("--out", required=True)
    p.add_argument("--persona", default=None)
    p.add_argument("--scorer", default=None, help="model whose score is recorded; default: heuristic")
    p.add_argument("--llm", default="auto", help="Ollama model that parses the brief (as for clips)")
    p.add_argument("--no-llm", action="store_true", help="parse the brief with the rules only")
    p.add_argument("--no-nc", action="store_true")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("info", help="index summary and timings (or a project's files)")
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
