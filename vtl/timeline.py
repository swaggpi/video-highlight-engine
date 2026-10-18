"""Timeline of strict events: an LLM reads the stream in ~60 s chunks (captions + speech + the previous chunk's
summary) and lists only what HAPPENS. Chunks are cached by a hash of their inputs, prompt version and model."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vtl.common import hms, parse_time, read_json, sha, write_json
from vtl.llm import llm_json

PROMPT_VERSION = "timeline-4"  # part of the chunk cache key (prompt and inputs)
VERSION = "timeline-5"         # the output: prompt version + post-filters below
MAX_EVENTS = 15                # per chunk; more means the model is copying captions
_ACTION = re.compile(r"appear|attack|hit|die|death|dead|kill|caught|catch|scare|game over|chase|escape|fight|"
                     r"shoot|trap|fall|win|lose|grab", re.I)
SYSTEM = """You turn one minute of a video into a STRICT EVENT LOG.
Input lines, in time order:
  [M:SS.s] SEE: <automatic caption of that frame>  (machine captions: noisy, often just describe the layout)
  [M:SS.s] SAY: "<speech transcript>"  (what the person on camera / the video says)
First write "summary": 2-3 short lines on what happened this minute. Then "events": only things that HAPPEN:
attacks, chases, deaths, getting caught, jump scares, game over, wins, escapes, someone appearing or arriving,
entering a new place, starting a new activity, and at most 3 NOTABLE utterances per minute (screams, jokes,
punchlines, strong reactions). Ordinary talk is NOT an event.
SEE lines are never events on their own: no "sees ...", no descriptions of rooms, people, screens or layouts. Use a
SEE line only when it shows a change that matters (a monster appears, a game-over screen, a new person on camera).
Each event:
  t: the timestamp of the line where it starts, copied exactly (e.g. "11:16.3")
  type: short verb phrase, e.g. "meets other player", "fights", "dies", "jump scare", "laughs", "screams",
        "game over", "enters room", "gets caught", "escapes", "says <short quote>"
  who: who does it ("streamer", "player", "monster", a name if known; "" if unclear)
  text: one concrete line, at most 15 words; quote speech in 'single quotes'
  confidence: 0-1 (lower when only the noisy captions support it)
Usually 0-8 events per minute.
Example output:
{"summary": "He searches the kitchen. Granny appears behind him and hits him; day 2 starts.",
 "events": [{"t": "14:40.3", "type": "monster appears", "who": "Granny", "text": "Granny walks in behind him", "confidence": 0.6},
  {"t": "14:42.0", "type": "screams", "who": "streamer", "text": "'NO NO NO get away!'", "confidence": 0.9},
  {"t": "14:45.0", "type": "dies", "who": "streamer", "text": "Granny hits him with the bat, screen goes black", "confidence": 0.8}]}"""
SCHEMA = {"type": "object", "required": ["summary", "events"], "properties": {
    "summary": {"type": "string"},
    "events": {"type": "array", "items": {"type": "object", "required": ["t", "type", "who", "text", "confidence"],
                                          "properties": {"t": {"type": "string"}, "type": {"type": "string"},
                                                         "who": {"type": "string"}, "text": {"type": "string"},
                                                         "confidence": {"type": "number"}}}}}}
_BOILER = re.compile(r"^(the image (is|shows) (a )?(screenshot|still|close-up)[^.]*\.\s*)?"
                     r"(it shows\s+|the image shows\s+|the image is\s+)?", re.I)
CAPTION_CHARS = 160   # captions are cut at a sentence end within this length
SAME_CAPTION = 0.5    # word-set Jaccard >= this vs the last shown caption: a near-duplicate, not shown


def mmss(t: float) -> str:
    """Chunk-local display time: 676.33 -> '11:16.3' (hours folded into minutes)."""
    m, s = divmod(round(t, 1), 60)
    return f"{int(m)}:{s:04.1f}"


def compact_caption(text: str, n: int = CAPTION_CHARS) -> str:
    """Drop Florence's boilerplate opener and cut at a sentence end within n chars."""
    text = _BOILER.sub("", text.strip())
    text = text[:1].upper() + text[1:]
    if len(text) > n:
        cut = text.rfind(". ", 0, n)
        text = text[:cut + 1] if cut > 40 else text[:n]
    return text


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def chunk_lines(frames: list[dict], segments: list[dict], start: float, end: float) -> list[str]:
    """Time-ordered SEE/SAY lines in [start, end). A caption that is a near-duplicate of the last shown one
    (same words, Jaccard >= SAME_CAPTION) is left out: on the stream VOD this cuts caption text by 60%."""
    items, prev = [], None
    for f in frames:
        if not start <= f["t"] < end:
            continue
        text = compact_caption(f["text"])
        w = _words(text)
        if prev is not None and len(w & prev) / max(1, len(w | prev)) >= SAME_CAPTION:
            continue
        items.append((f["t"], 0, f"[{mmss(f['t'])}] SEE: {text}"))
        prev = w
    for s in segments:
        if start <= s["start"] < end:
            items.append((s["start"], 1, f'[{mmss(s["start"])}] SAY: "{s["text"]}"'))
    return [line for _, _, line in sorted(items)]


def chunks(duration: float, size: float = 60.0) -> list[tuple[float, float]]:
    n = max(1, round(duration / size))
    step = duration / n
    return [(round(i * step, 3), round(duration if i == n - 1 else (i + 1) * step, 3)) for i in range(n)]


def chunk_prompt(name: str, start: float, end: float, lines: list[str], prev_summary: str) -> str:
    head = f"Video: {name}, {mmss(start)}-{mmss(end)}.\n"
    head += f"Previous minute: {prev_summary.strip() or '(start of video)'}\n\n"
    return head + "\n".join(lines or ["(no captions or speech)"])


def cache_key(model: str, user: str) -> str:
    return sha(PROMPT_VERSION, model, SYSTEM, user)


def normalize(events: list, start: float, end: float, src_fps: float) -> list[dict]:
    out = []
    for e in events or []:
        if not isinstance(e, dict) or not str(e.get("type", "")).strip() or not str(e.get("text", "")).strip():
            continue
        t = parse_time(e.get("t"))
        if t is None:
            continue
        t = min(max(t, start), end)
        try:
            conf = min(1.0, max(0.0, float(e.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        out.append({"frame": int(round(t * src_fps)), "t": round(t, 2), "type": str(e["type"]).strip().lower()[:60],
                    "who": str(e.get("who") or "").strip()[:40], "text": str(e["text"]).strip()[:300],
                    "confidence": round(conf, 2)})
    return sorted(out, key=lambda e: e["t"])


def strict(events: list[dict], captions: list[str], same: float = 0.5) -> list[dict]:
    """Enforce "strict events only" on small models: drop events that merely restate a frame caption (word
    Jaccard >= same) unless their type is an action, drop repeats, then keep the MAX_EVENTS most confident."""
    caps = [_words(c) for c in captions]
    out, seen = [], set()
    for e in events:
        w = _words(e["text"])
        copy = not _ACTION.search(e["type"]) and any(len(w & c) / max(1, len(w | c)) >= same for c in caps)
        if copy or (e["type"], e["text"]) in seen:
            continue
        seen.add((e["type"], e["text"]))
        out.append(e)
    keep = sorted(out, key=lambda e: -e["confidence"])[:MAX_EVENTS]
    return sorted(keep, key=lambda e: e["t"])


def _check(d: dict) -> None:
    if not isinstance(d.get("events"), list):
        raise ValueError("'events' must be a list")


def run_chunk(name: str, span: tuple[float, float], frames, segments, prev_summary: str, model: str,
              cache: Path, src_fps: float, call=None) -> dict:
    start, end = span
    user = chunk_prompt(name, start, end, chunk_lines(frames, segments, start, end), prev_summary)
    path = cache / f"{cache_key(model, user)}.json"
    rec = read_json(path)
    if rec is None:
        data = llm_json(SYSTEM, user, model, SCHEMA, check=_check, call=call)
        rec = {"version": PROMPT_VERSION, "model": model, "span": span, "system": SYSTEM, "user": user, "response": data}
        write_json(path, rec)
    captions = [compact_caption(f["text"]) for f in frames if start <= f["t"] < end]
    events = strict(normalize(rec["response"].get("events"), start, end, src_fps), captions)
    return {"span": span, "events": events,
            "summary": str(rec["response"].get("summary") or "")[:400]}


def build(name: str, frames: list[dict], segments: list[dict], duration: float, src_fps: float, model: str,
          cache: Path, par: int = 1, chunk_s: float = 60.0, call=None, log=print) -> list[dict]:
    """All events, in time order. `par` contiguous groups of chunks run in parallel; within a group each chunk
    sees the previous chunk's summary (so only group boundaries lose continuity)."""
    spans = chunks(duration, chunk_s)
    par = max(1, min(par, len(spans)))
    size = -(-len(spans) // par)
    groups = [spans[i:i + size] for i in range(0, len(spans), size)]

    def run_group(group):
        out, summary = [], ""
        for span in group:
            r = run_chunk(name, span, frames, segments, summary, model, cache, src_fps, call)
            out.append(r)
            summary = r["summary"]
            log(f"  timeline {name} {hms(span[0])}: {len(r['events'])} events")
        return out

    with ThreadPoolExecutor(len(groups)) as ex:
        results = [r for g in ex.map(run_group, groups) for r in g]
    return [dict(e, chunk=i) for i, r in enumerate(results) for e in r["events"]]


def line(file: str, e: dict) -> str:
    """Compact timeline line: "VOD.mp4 0:58:59 f212340 dies — 'I FED YOU MEAT' (conf .9)"."""
    who = f"{e['who']}: " if e.get("who") else ""
    conf = f"{e['confidence']:.1f}".lstrip("0") if e["confidence"] < 1 else "1.0"
    return f"{file} {hms(e['t'])} f{e['frame']} {who}{e['type']} — {e['text']} (conf {conf})"
