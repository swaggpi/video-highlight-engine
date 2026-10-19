"""Timeline of strict events: an LLM reads the stream in ~60 s chunks (captions + speech + the previous chunk's
summary) and lists only what HAPPENS. Chunks are cached by a hash of their inputs, prompt version and model."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vtl import llm as L
from vtl.common import extract_json, hms, parse_time, read_json, sha, write_json
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
  {"t": "14:45.0", "type": "dies", "who": "streamer", "text": "Granny hits him with the bat, screen goes black", "confidence": 0.8}]}"""  # noqa: E501 - prompt text is part of the cache key
# Streamer mode (vtl index --streamer): SEE describes the game-area crop only, and two more line kinds come in.
STREAMER_VERSION = "timeline-streamer-1"
SYSTEM_STREAMER = """You turn one minute of a game stream into a STRICT EVENT LOG.
Input lines, in time order:
  [M:SS.s] SEE: <automatic caption of the GAME screen>  (machine captions of the gameplay only, the stream overlays
           are cut away; noisy)
  [M:SS.s] TEXT: "<text shown on the game screen>" (on screen N s)  (read by OCR: GAME OVER, DAY 2, YOU DIED, item
           names, prompts; lines read together are joined with |, OCR may drop spaces)
  [M:SS.s] CAM: <the streamer's facecam, at a loud moment or when he starts talking>  (machine caption, noisy)
  [M:SS.s] SAY: "<speech transcript>"  (what the streamer says)
First write "summary": 2-3 short lines on what happened this minute. Then "events": only things that HAPPEN:
attacks, chases, deaths, getting caught, jump scares, game over, wins, escapes, someone appearing or arriving,
entering a new place, starting a new activity, and at most 3 NOTABLE utterances per minute (screams, jokes,
punchlines, strong reactions). Ordinary talk is NOT an event.
TEXT lines are the game's own words and reliable: a GAME OVER / YOU DIED / WASTED screen is a "game over" event; a
new DAY / LEVEL / ROUND card is a "new day" event (with its text), and when it follows a black screen or panic the
player was caught or died just before it, so also log that "dies" event at the moment it happened. Other TEXT
(item names, "I need a hammer", key prompts, menus) is an event only when it shows something that matters.
SEE and CAM lines are never events on their own: no "sees ...", no descriptions of rooms, people, screens or layouts.
Use them only when they show a change that matters (a monster appears, the streamer jumps up or covers his face).
Each event:
  t: the timestamp of the line where it starts, copied exactly (e.g. "11:16.3")
  type: short verb phrase, e.g. "meets other player", "fights", "dies", "jump scare", "laughs", "screams",
        "game over", "new day", "enters room", "gets caught", "escapes", "says <short quote>"
  who: who does it ("streamer", "player", "monster", a name if known; "" if unclear)
  text: one concrete line, at most 15 words; quote speech and on-screen text in 'single quotes'
  confidence: 0-1 (lower when only the noisy captions support it)
Usually 0-8 events per minute.
Example output:
{"summary": "He searches the kitchen. Granny appears behind him and hits him; day 2 starts.",
 "events": [{"t": "14:40.3", "type": "monster appears", "who": "Granny", "text": "Granny walks in behind him", "confidence": 0.6},
  {"t": "14:42.0", "type": "screams", "who": "streamer", "text": "'NO NO NO get away!'", "confidence": 0.9},
  {"t": "14:45.0", "type": "dies", "who": "streamer", "text": "Granny hits him with the bat, screen goes black", "confidence": 0.8},
  {"t": "14:52.0", "type": "new day", "who": "", "text": "'DAY 2' card after the death", "confidence": 0.9}]}"""  # noqa: E501
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


def card_line(c: dict) -> str:
    """An OCR text card as a timeline input: '[59:16.0] TEXT: "DAY5 | (THE LASTDAY)" (on screen 3 s)'."""
    return f'[{mmss(c["t"])}] TEXT: "{c["text"]}" (on screen {round(c["end"] - c["t"]) + 1} s)'


def cam_line(r: dict) -> str:
    """A facecam reaction as a timeline input: '[11:16.7] CAM: streamer (loud) a man standing with his hands ...'."""
    cue = "loud" if r.get("trigger") == "loud" else "starts talking"
    return f"[{mmss(r['t'])}] CAM: streamer ({cue}) {r['text'] or '(no visible reaction)'}"


def chunk_lines(frames: list[dict], segments: list[dict], start: float, end: float,
                cards: list[dict] | None = None, cams: list[dict] | None = None) -> list[str]:
    """Time-ordered SEE/SAY lines in [start, end). A caption that is a near-duplicate of the last shown one
    (same words, Jaccard >= SAME_CAPTION) is left out: on the stream VOD this cuts caption text by 60%.
    Streamer mode adds TEXT lines (OCR cards; a text already shown in this chunk is not repeated) and CAM lines."""
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
    shown = set()
    for c in cards or []:
        if start <= c["t"] < end and c["text"] not in shown and not c.get("chrome"):
            shown.add(c["text"])
            items.append((c["t"], 2, card_line(c)))
    items += [(r["t"], 3, cam_line(r)) for r in cams or [] if start <= r["t"] < end]
    return [line for _, _, line in sorted(items)]


def chunks(duration: float, size: float = 60.0) -> list[tuple[float, float]]:
    n = max(1, round(duration / size))
    step = duration / n
    return [(round(i * step, 3), round(duration if i == n - 1 else (i + 1) * step, 3)) for i in range(n)]


def chunk_prompt(name: str, start: float, end: float, lines: list[str], prev_summary: str) -> str:
    head = f"Video: {name}, {mmss(start)}-{mmss(end)}.\n"
    head += f"Previous minute: {prev_summary.strip() or '(start of video)'}\n\n"
    return head + "\n".join(lines or ["(no captions or speech)"])


def cache_key(model: str, user: str, system: str = SYSTEM) -> str:
    return sha(PROMPT_VERSION, model, system, user)


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


# Streamer mode post-filters. A small model turns every TEXT line into a "text appears" event and does not reliably
# turn a GAME OVER / DAY N card into one, so: events that only report text are dropped, and cards with a known
# round/outcome wording become events directly (source "ocr") unless the model already logged that type nearby.
_TEXT_EVENT = re.compile(r"\btext\b|\bprompt\b|on.?screen|\bsign\b|\breads?\b|\bmessage\b", re.I)
_CARD_TYPES = [
    ("game over", re.compile(r"\bGAME ?OVER\b|\bYOU ?(DIED|ARE ?DEAD|LOST|LOSE)\b|\bWASTED\b|\bMISSION ?FAILED\b|"
                             r"\bDEFEAT(ED)?\b", re.I)),
    ("wins", re.compile(r"\bYOU ?(WIN|WON|ESCAPED|SURVIVED)\b|\bVICTORY\b|\b(MISSION|LEVEL) ?COMPLETE", re.I)),
    ("new {0}", re.compile(r"\b(DAY|NIGHT|LEVEL|CHAPTER|ROUND|WAVE) ?\d+\b", re.I)),
]
CARD_NEAR_S = 5.0     # an LLM event of the same type this close to a card makes the card event redundant


def card_events(cards: list[dict], src_fps: float) -> list[dict]:
    """Timeline events straight from on-screen text cards: game over, wins, new day/level/round."""
    out = []
    for c in cards:
        if c.get("chrome"):
            continue
        for kind, rx in _CARD_TYPES:
            m = rx.search(c["text"])
            if m:
                out.append({"frame": int(round(c["t"] * src_fps)), "t": round(c["t"], 2),
                            "type": kind.format((m.group(1) or "").lower()), "who": "",
                            "text": f"'{c['text']}' on screen", "confidence": 0.95, "source": "ocr"})
                break
    return out


def merge_cards(events: list[dict], extra: list[dict]) -> list[dict]:
    """LLM events minus "text appears" reports, plus card events the LLM did not already log nearby."""
    keep = [e for e in events if not _TEXT_EVENT.search(e["type"])]
    for c in extra:
        if not any(e["type"] == c["type"] and abs(e["t"] - c["t"]) <= CARD_NEAR_S for e in keep):
            keep.append(c)
    return sorted(keep, key=lambda e: e["t"])


def _check(d: dict) -> None:
    if not isinstance(d.get("events"), list):
        raise ValueError("'events' must be a list")


def salvage(reply: str) -> str | None:
    """A reply cut off mid-list (qwen3:4b sometimes loops on "sees room" events until num_predict) -> the summary and
    the events that were complete, as JSON; None if the reply is fine or has no events list to salvage."""
    try:
        if isinstance(extract_json(reply).get("events"), list):
            return None
    except ValueError:
        pass
    i = reply.find('"events"')
    j = reply.find("[", i)
    if i < 0 or j < 0:
        return None
    dec, events, pos = json.JSONDecoder(), [], j + 1
    while True:
        pos = re.compile(r"[\s,]*").match(reply, pos).end()
        try:
            obj, pos = dec.raw_decode(reply, pos)
        except ValueError:
            break
        if isinstance(obj, dict):
            events.append(obj)
    m = re.search(r'"summary"\s*:\s*("(?:[^"\\]|\\.)*")', reply)
    return json.dumps({"summary": json.loads(m.group(1)) if m else "", "events": events})


def run_chunk(name: str, span: tuple[float, float], frames, segments, prev_summary: str, model: str,
              cache: Path, src_fps: float, call=None, streamer: dict | None = None) -> dict:
    """streamer: {"cards": OCR text cards, "cams": facecam reactions} switches to the streamer prompt."""
    start, end = span
    system = SYSTEM if streamer is None else SYSTEM_STREAMER
    extra = {} if streamer is None else {"cards": streamer.get("cards"), "cams": streamer.get("cams")}
    user = chunk_prompt(name, start, end, chunk_lines(frames, segments, start, end, **extra), prev_summary)
    path = cache / f"{cache_key(model, user, system)}.json"
    rec = read_json(path)
    if rec is None:
        salvaged = []

        def repaired(*a):
            reply = (call or L.llm)(*a)
            fixed = salvage(reply)
            salvaged.append(fixed is not None)
            return reply if fixed is None else fixed

        try:
            data = llm_json(system, user, model, SCHEMA, check=_check, call=repaired)
        except RuntimeError as e:  # one bad minute must not sink an hour-long index; not cached, so a rerun retries
            return {"span": span, "events": [], "summary": prev_summary, "failed": str(e)[:300]}
        rec = {"version": PROMPT_VERSION, "model": model, "span": span, "system": system, "user": user,
               "response": data, **({"salvaged_truncated_reply": True} if salvaged[-1] else {})}
        write_json(path, rec)
    captions = [compact_caption(f["text"]) for f in frames if start <= f["t"] < end]
    events = strict(normalize(rec["response"].get("events"), start, end, src_fps), captions)
    if streamer is not None:
        cards = [c for c in streamer.get("cards") or [] if start <= c["t"] < end]
        events = merge_cards(events, card_events(cards, src_fps))
    return {"span": span, "events": events,
            "summary": str(rec["response"].get("summary") or "")[:400]}


def build(name: str, frames: list[dict], segments: list[dict], duration: float, src_fps: float, model: str,
          cache: Path, par: int = 1, chunk_s: float = 60.0, call=None, log=print,
          streamer: dict | None = None, failed: list | None = None) -> list[dict]:
    """All events, in time order. `par` contiguous groups of chunks run in parallel; within a group each chunk
    sees the previous chunk's summary (so only group boundaries lose continuity)."""
    spans = chunks(duration, chunk_s)
    par = max(1, min(par, len(spans)))
    size = -(-len(spans) // par)
    groups = [spans[i:i + size] for i in range(0, len(spans), size)]

    stop = threading.Event()  # set on an error or Ctrl-C/SIGTERM in the caller: workers stop after their chunk

    def run_group(group):
        out, summary = [], ""
        for span in group:
            if stop.is_set():
                break
            r = run_chunk(name, span, frames, segments, summary, model, cache, src_fps, call, streamer)
            out.append(r)
            summary = r["summary"]
            log(f"  timeline {name} {hms(span[0])}: " + (f"FAILED, no events ({r['failed']})" if r.get("failed")
                                                         else f"{len(r['events'])} events"))
        return out

    if len(groups) == 1:  # a local model runs one group: in this thread, so Ctrl-C/SIGTERM interrupt it right away
        results = run_group(groups[0])
    else:
        ex = ThreadPoolExecutor(len(groups))
        try:
            results = [r for g in ex.map(run_group, groups) for r in g]
        finally:
            stop.set()
            ex.shutdown(wait=False, cancel_futures=True)
    if failed is not None:  # chunks whose LLM answer stayed invalid after the retry (they have no events)
        failed += [r["span"] for r in results if r.get("failed")]
    return [dict(e, chunk=i) for i, r in enumerate(results) for e in r["events"]]


def line(file: str, e: dict) -> str:
    """Compact timeline line: "VOD.mp4 0:58:59 f212340 dies — 'I FED YOU MEAT' (conf .9)"."""
    who = f"{e['who']}: " if e.get("who") else ""
    conf = f"{e['confidence']:.1f}".lstrip("0") if e["confidence"] < 1 else "1.0"
    return f"{file} {hms(e['t'])} f{e['frame']} {who}{e['type']} — {e['text']} (conf {conf})"
