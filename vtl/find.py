"""find: brief + timeline -> candidate clips (LLM) -> decision pass (LLM) -> word-safe edges -> clips JSON."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vtl import llm as L
from vtl.common import hms, parse_time, read_json, read_jsonl, slug
from vtl.index import index_dir
from vtl.timeline import compact_caption, line

VERSION = "find-1"
WINDOW_CHARS = 24000  # one candidate call holds ~7k tokens of timeline; longer footage is map-reduced
DEC_BATCH = 8
PAD = 3.0

CAND_SYSTEM = """You are a video editor's assistant. You get an editor's BRIEF, a TIMELINE of events found in the
footage (one line per event: file, time H:MM:SS, frame, who: type — text (confidence)) and a coarse TRANSCRIPT
index (file, time, what is said in the next 30 s).
1. "plan": what the brief asks for. count = number of clips wanted (0 if it asks for a runtime instead or does not
   say), min_len/max_len = allowed clip length in seconds (0 = not specified; "a 10 s clip" means about 9-11),
   total_s = total runtime asked for in seconds (0 if none), order = "chronological" if clips should follow the
   footage order (highlight videos, montages), else "rank" (best first).
2. "candidates": about 3x as many candidate clips as needed (fewer if the footage has no more), each
   {file, start, end, why}. Choose start/end (H:MM:SS) from the event times so the setup AND the payoff are inside:
   e.g. monster appears 0:14:40, player dies 0:14:45 -> 0:14:35 to 0:14:48. Respect the requested length.
   why = at most 12 words naming the events used. At most 30 candidates. Do not invent moments that are not
   in the timeline or transcript."""
CAND_SCHEMA = {"type": "object", "required": ["plan", "candidates"], "properties": {
    "plan": {"type": "object", "required": ["count", "min_len", "max_len", "total_s", "order"], "properties": {
        "count": {"type": "integer"}, "min_len": {"type": "number"}, "max_len": {"type": "number"},
        "total_s": {"type": "number"}, "order": {"type": "string", "enum": ["chronological", "rank"]}}},
    "candidates": {"type": "array", "items": {"type": "object", "required": ["file", "start", "end", "why"],
                                              "properties": {"file": {"type": "string"}, "start": {"type": "string"},
                                                             "end": {"type": "string"}, "why": {"type": "string"}}}}}}
DEC_SYSTEM = """You are the final editor. You get an editor's BRIEF and a few candidate clips. For each candidate you
see the events, speech (SAY) and frame captions (SEE, machine-made, noisy) from 3 s before to 3 s after it.
For EVERY candidate return {id, score, start, end, label, reason}:
  score: 0-10, how well this clip serves the brief (0 = reject: dull, off-brief, nothing happens, or it repeats
         another candidate)
  start/end: H:MM:SS.s, adjusted if the boundary cuts an action or a sentence; keep setup and payoff inside;
         respect the requested clip length
  label: a 1-3 word tag (e.g. "scare/death", "clutch/win", "funny")
  reason: at most 20 words on what happens and why it fits the brief"""
DEC_STREAMER = """
This is a game stream: SEE captions show the game area only; TEXT lines are on-screen game text read by OCR (GAME
OVER, DAY 2, prompts); CAM lines describe the streamer's facecam at loud moments."""
DEC_SCHEMA = {"type": "object", "required": ["picks"], "properties": {"picks": {"type": "array", "items": {
    "type": "object", "required": ["id", "score", "start", "end", "label", "reason"], "properties": {
        "id": {"type": "integer"}, "score": {"type": "number"}, "start": {"type": "string"},
        "end": {"type": "string"}, "label": {"type": "string"}, "reason": {"type": "string"}}}}}}


def load_index(root: Path, streamer: bool = False) -> list[dict]:
    files = []
    for meta_path in sorted(index_dir(root, streamer).glob("*/meta.json")):
        meta, out = read_json(meta_path), meta_path.parent
        if not meta or not (out / "timeline.jsonl").exists():
            continue
        f = {"file": meta["file"], "duration": meta["duration"], "timeline": read_jsonl(out / "timeline.jsonl"),
             "frames": read_jsonl(out / "frames.jsonl"),
             "segments": (read_json(out / "transcript.json") or {}).get("segments", [])}
        if streamer:
            f.update(cards=read_jsonl(out / "cards.jsonl"), cams=read_jsonl(out / "facecam.jsonl"))
        files.append(f)
    return files


def transcript_index(file: str, segments: list[dict], bucket: float = 30.0, width: int = 110) -> list[tuple]:
    rows: dict[int, list[str]] = {}
    for s in segments:
        rows.setdefault(int(s["start"] // bucket), []).append(s["text"])
    return [(k * bucket, f"{file} {hms(k * bucket)} | {' '.join(v)[:width]}") for k, v in sorted(rows.items())]


def windows(files: list[dict], max_chars: int = WINDOW_CHARS) -> list[tuple[list[str], list[str]]]:
    """(timeline lines, transcript lines) per candidate call, split in time order when too long."""
    items = []
    for f in files:
        items += [(f["file"], e["t"], 0, line(f["file"], e)) for e in f["timeline"]]
        items += [(f["file"], t, 1, s) for t, s in transcript_index(f["file"], f["segments"])]
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    out, cur, size = [], ([], []), 0
    for _, _, kind, text in items:
        if size + len(text) > max_chars and (cur[0] or cur[1]):
            out.append(cur)
            cur, size = ([], []), 0
        cur[kind].append(text)
        size += len(text) + 1
    return out + [cur] if (cur[0] or cur[1]) else out


def cand_prompt(brief: str, tl: list[str], tx: list[str], part: int, parts: int, count: int | None) -> str:
    head = f"BRIEF: {brief}\n"
    if count:
        head += f"(The editor wants exactly {count} clips.)\n"
    if parts > 1:
        head += f"(This is part {part + 1} of {parts} of the footage. The plan describes the WHOLE brief; propose " \
                f"candidates from this part only, about 3x its share of the needed clips, at most 15.)\n"
    return head + "\nTIMELINE:\n" + "\n".join(tl or ["(none)"]) + "\n\nTRANSCRIPT:\n" + "\n".join(tx or ["(none)"])


def clean_candidates(raw: list, files: dict[str, dict]) -> list[dict]:
    out = []
    for c in raw or []:
        name = str(c.get("file", ""))
        f = files.get(name) or (next(iter(files.values())) if len(files) == 1 else None)
        s, e = parse_time(c.get("start")), parse_time(c.get("end"))
        if f is None or s is None or e is None:
            continue
        s, e = max(0.0, min(s, e)), min(f["duration"], max(s, e))
        if e - s >= 1.0:
            out.append({"file": f["file"], "start": round(s, 2), "end": round(e, 2), "why": str(c.get("why", ""))})
    return out


def context(f: dict, start: float, end: float, max_see: int = 12) -> list[str]:
    a, b = start - PAD, end + PAD
    rows = [(e["t"], 0, f"[{hms(e['t'], 1)}] EVENT {e.get('who') or ''} {e['type']}: {e['text']}".replace("  ", " "))
            for e in f["timeline"] if a <= e["t"] <= b]
    rows += [(s["start"], 1, f"[{hms(s['start'], 1)}] SAY: \"{s['text']}\"")
             for s in f["segments"] if s["end"] >= a and s["start"] <= b]
    rows += [(c["t"], 1, f"[{hms(c['t'], 1)}] TEXT: \"{c['text']}\"") for c in f.get("cards", [])
             if c["end"] >= a and c["t"] <= b]
    rows += [(r["t"], 1, f"[{hms(r['t'], 1)}] CAM: {r['text']}") for r in f.get("cams", [])
             if a <= r["t"] <= b and r["text"]]
    see, prev = [], None
    for fr in f["frames"]:
        if a <= fr["t"] <= b:
            text = compact_caption(fr["text"])[:110]
            if text != prev:
                see.append((fr["t"], 2, f"[{hms(fr['t'], 1)}] SEE: {text}"))
            prev = text
    step = max(1, -(-len(see) // max_see))
    return [r for _, _, r in sorted(rows + see[::step])]


def dec_prompt(brief: str, plan: dict, batch: list[tuple[int, dict]], files: dict[str, dict]) -> str:
    want = []
    if plan.get("min_len") or plan.get("max_len"):
        want.append(f"clip length {plan.get('min_len') or '?'}-{plan.get('max_len') or '?'} s")
    if plan.get("count"):
        want.append(f"{plan['count']} clips in total")
    parts = [f"BRIEF: {brief}" + (f"\n(Wanted: {', '.join(want)}.)" if want else "")]
    for i, c in batch:
        ctx = "\n".join(context(files[c["file"]], c["start"], c["end"]))
        parts.append(f"--- candidate id={i}: {c['file']} {hms(c['start'], 1)} - {hms(c['end'], 1)} "
                     f"({c['end'] - c['start']:.0f} s). Proposed because: {c['why']}\n{ctx}")
    return "\n\n".join(parts)


def snap(start: float, end: float, words: list[dict], pad: float = 0.05) -> tuple[float, float]:
    """Move edges outward so no word is cut: an edge inside a word goes to that word's boundary."""
    for w in words:
        if w["start"] < start < w["end"]:
            start = w["start"] - pad
        if w["start"] < end < w["end"]:
            end = w["end"] + pad
    return max(0.0, start), end


def fit_length(s: float, e: float, lo: float, hi: float, duration: float) -> tuple[float, float]:
    """Grow (around the centre) to lo or shrink (keeping the start, where the setup is) to hi; 0 = no limit."""
    if lo and e - s < lo:
        grow = (lo - (e - s)) / 2
        s, e = s - grow, e + grow
        if s < 0:
            s, e = 0.0, e - s
        if e > duration:
            s, e = max(0.0, s - (e - duration)), duration
    if hi and e - s > hi:
        e = s + hi
    return s, e


def overlaps(a: dict, b: dict, frac: float = 0.3) -> bool:
    if a["file"] != b["file"]:
        return False
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return inter > frac * min(a["end"] - a["start"], b["end"] - b["start"])


def select(scored: list[dict], plan: dict, count: int | None, files: dict[str, dict]) -> list[dict]:
    """Best-scored first, no overlaps, length-fitted and word-snapped, until count or total_s is reached."""
    count = count or plan.get("count") or 0
    total, picked = plan.get("total_s") or 0, []
    if total < 30:  # a runtime under 30 s is a mis-parse (qwen3:4b read "8-12 min" as 8); lengths are min/max_len
        total = 0
    lo, hi = plan.get("min_len") or 0, plan.get("max_len") or 0
    if lo and hi and lo > hi:
        lo, hi = hi, lo
    for c in sorted(scored, key=lambda c: -c["score"]):
        if c["score"] <= 0 or (not count and not total and c["score"] < 5):
            continue
        f = files[c["file"]]
        words = [w for s in f["segments"] for w in s["words"]]
        s, e = fit_length(c["start"], c["end"], lo, hi, f["duration"])
        s, e = snap(s, e, words)
        clip = dict(c, start=round(s, 2), end=round(min(e, f["duration"]), 2))
        clip["duration"] = round(clip["end"] - clip["start"], 2)
        if any(overlaps(clip, p) for p in picked):
            continue
        if total and sum(p["duration"] for p in picked) + clip["duration"] > total * 1.1:
            continue
        picked.append(clip)
        if count and len(picked) >= count:
            break
    if plan.get("order") == "chronological":
        picked.sort(key=lambda c: (c["file"], c["start"]))
    return picked


def find(root: Path, brief: str, count: int | None = None, call=None, log=print, streamer: bool = False) -> dict:
    files = {f["file"]: f for f in load_index(root, streamer)}
    if not files and not streamer and load_index(root, True):
        log("  no plain index; using the streamer-mode index")
        streamer, files = True, {f["file"]: f for f in load_index(root, True)}
    if not files:
        raise SystemExit(f"no vtl {'streamer ' if streamer else ''}index under {root} "
                         f"(run: vtl index {'--streamer ' if streamer else ''}{root})")
    dec_system = DEC_SYSTEM + (DEC_STREAMER if streamer else "")
    io: list[dict] = []
    base = call or L.llm

    def recorded(stage):
        def c(system, user, model, schema=None):
            t0 = time.time()
            out = base(system, user, model, schema)
            io.append({"stage": stage, "model": model, "wall_s": round(time.time() - t0, 2), "system": system,
                       "user": user, "response": out})
            return out
        return c

    t0, n0 = time.time(), len(L.STATS)
    m_cand, m_dec = L.model_for("candidates"), L.model_for("decision")
    wins = windows(list(files.values()))
    plans, cands = [], []

    def one_window(k):
        tl, tx = wins[k]
        return L.llm_json(CAND_SYSTEM, cand_prompt(brief, tl, tx, k, len(wins), count), m_cand, CAND_SCHEMA,
                          call=recorded("candidates"))

    par = 1 if L.is_local(m_cand) else 4
    with ThreadPoolExecutor(par) as ex:
        for d in ex.map(one_window, range(len(wins))):
            plans += [d["plan"]] if isinstance(d.get("plan"), dict) else []
            cands += clean_candidates(d.get("candidates"), files)
    # Windows sometimes size the plan to their own part: keep the first plan but the largest count.
    plan = dict(plans[0]) if plans else {}
    plan["count"] = count or max((int(p["count"]) for p in plans if str(p.get("count")).isdigit()), default=0)
    t_cand = time.time() - t0
    log(f"  {len(cands)} candidates from {len(wins)} call(s) in {t_cand:.0f} s; plan {plan}")

    batches = [list(enumerate(cands))[i:i + DEC_BATCH] for i in range(0, len(cands), DEC_BATCH)]

    def one_batch(batch):
        return L.llm_json(dec_system, dec_prompt(brief, plan, batch, files), m_dec, DEC_SCHEMA,
                          call=recorded("decision")).get("picks") or []

    scored = []
    with ThreadPoolExecutor(1 if L.is_local(m_dec) else 4) as ex:
        for picks in ex.map(one_batch, batches):
            for p in picks:
                try:
                    c = cands[int(p.get("id"))]
                    score = float(p.get("score", 0))
                except (TypeError, ValueError, IndexError):
                    continue
                s, e = parse_time(p.get("start")), parse_time(p.get("end"))
                if s is None or e is None or e - s < 1.0 or abs(s - c["start"]) > 30 or abs(e - c["end"]) > 30:
                    s, e = c["start"], c["end"]  # an unusable adjustment keeps the candidate's own edges
                scored.append({"file": c["file"], "start": s, "end": e, "score": score,
                               "label": str(p.get("label", "")), "reason": str(p.get("reason") or c["why"])})
    clips = select(scored, plan, count, files)
    stats = L.summary(n0)
    return {"query": brief, "version": VERSION, "mode": "streamer" if streamer else "plain", "plan": plan,
            "clips": [{"file": c["file"], "start": c["start"], "end": c["end"], "duration": c["duration"],
                       "reason": c["reason"], "label": c["label"], "score": c["score"], "source": "vtl"}
                      for c in clips],
            "models": {"candidates": m_cand, "decision": m_dec}, "n_candidates": len(cands),
            "timing": {"candidates_s": round(t_cand, 1), "decision_s": round(time.time() - t0 - t_cand, 1),
                       "total_s": round(time.time() - t0, 1)}, "llm": stats, "_io": io}


def default_out(root: Path, brief: str, streamer: bool = False) -> Path:
    return index_dir(root, streamer) / "queries" / f"{slug(brief)[:60]}.json"
