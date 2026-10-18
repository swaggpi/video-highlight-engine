"""A local LLM as the brief parser (Ollama, JSON schema enforced, answers cached on disk per prompt version, model and
brief). The rules ground it: they run first, numbers the brief does not contain are dropped, avoid/topic items the
rules find are added, and a visual cue the model adds counts only where the brief has a matching stem (CUE_GROUND).
Both paths end in vhe.query.rules.finalise; any failure (server unreachable, unknown model, timeout, malformed answer)
falls back to the rules."""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from vhe.models import gpu_released
from vhe.query.lexicon import (
    ALL_MATCHES,
    CUE_FILLER,
    CUE_GROUND,
    EDGE_STOP,
    GENERIC_HEAD,
    HARD_CUE,
    ITEM_EDGE,
    NUM,
    SPEECH_ITEM,
    STYLE,
)
from vhe.query.model import BUDGET_TOL, ParsedQuery
from vhe.query.rules import finalise, normalise, parse_rules, set_mood
from vhe.query.text_match import word_tokens

DEFAULT_LLM = "qwen3:4b"
OLLAMA_HOST = "http://127.0.0.1:11434"
LLM_TIMEOUT = 90.0  # a cold load of a 4B model takes seconds; a hung server must not hang the query
# Unload the model as soon as the answer is back: the query then loads SigLIP and CG-DETR on the same (6 GB) GPU, and
# a resident qwen3:4b (~2.8 GB) made 3 of 17 dev briefs fail with CUDA OOM. Measured 2026-10-18 (16 cores, GTX 1660
# Super, two dev briefs, cold): GPU 12-22 s per parse, CPU-only (num_gpu 0) 44-57 s. So the parse runs on the GPU and
# unloads at once (keep_alive 0: Ollama's /api/ps is empty when the response returns); the engine's CPU fallback
# covers any other GPU tenant. In a warm process (the UI) the loaded SigLIP2 and CG-DETR lend their 1.4 GB to the parse
# (vhe.models.gpu_released) and return afterwards, which a resident LLM would prevent: keep_alive stays 0. Measured
# 2026-10-18 next to another 1.3 GB GPU tenant and a second UI process: 45-47 s per uncached parse with the models
# resident, 31-34 s with them lent (then 7 of 37 layers fit: Ollama keeps ~0.8 GB free; all of them need ~3.6 GB
# free); a warm, resident model would save the 4-12 s load and most of the 8.5 s prompt evaluation, at the cost of the
# models' GPU memory.
LLM_KEEP_ALIVE = 0
LLM_OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096}  # the prompt with its examples is ~1.5k tokens
PARSE_CACHE = Path.home() / ".cache" / "vhe" / "parse"  # one JSON per (prompt version, model, brief)
PROMPT_VERSION = 4  # 4: framing, want_faces, hook, legible_text, centered

_TEXT = {"type": "string"}
_NUM = {"type": "number"}
_LIST = {"type": "array", "items": {"type": "string"}}
LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "content": _TEXT, "shots": _LIST, "avoid": _LIST, "exclude": _LIST, "topics": _LIST,
        "need_live_audio": {"type": "boolean"},
        "mood": {"type": "string", "enum": ["calm", "energetic", "neutral"]},
        "speech": {"type": "string", "enum": ["wanted", "unwanted", "any"]},
        "min_seconds": _NUM, "max_seconds": _NUM, "count": {"type": "integer"},
        "budget_min_seconds": _NUM, "budget_max_seconds": _NUM,
        "soundbite_min_seconds": _NUM, "soundbite_max_seconds": _NUM,
        "exhaustive": {"type": "boolean"}, "opener": _TEXT, "closer": _TEXT,
        "framing": {"type": "string", "enum": ["none", "close", "wide"]}, "want_faces": {"type": "boolean"},
        "hook": {"type": "boolean"}, "legible_text": {"type": "boolean"}, "centered": {"type": "boolean"},
    },
}
_NO_CUES = {"framing": "none", "want_faces": False, "hook": False, "legible_text": False, "centered": False}
LLM_SCHEMA["required"] = list(LLM_SCHEMA["properties"])
LLM_PROMPT = """You turn a video editor's brief into a JSON edit plan for searching raw footage with an image-text \
model and a transcript. Only extract what the brief says; never invent items or numbers.
- Write content, shots, avoid, exclude, topics, opener and closer in English. If the brief is in another language, \
translate them.
- content: the main subject of the edit in at most 10 words.
- shots: one entry per visual item the brief asks for, each a short phrase of 2-6 words as a photo caption would say \
it. Keep names and give them their category (a named float -> "<name> float"). Do not repeat the opening or closing \
shot. Leave out mood, lengths, camera style and things to avoid. [] when the brief only asks for speech or names no \
items (e.g. just "the best moments").
- exclude: visual things the brief forbids outright ("no ...", "never ...", "absolutely no ...", "nothing with ...", \
"leave out ...", "exclude ..."), as short phrases. avoid: visual things it only prefers to leave out ("avoid ...", \
"preferably no ...", "try not to ..."). [] when there are none. Unwanted talking or voices go in speech, not here.
- topics: only when the brief asks for moments where someone talks about something: one short description per \
topic, with any quoted words. [] otherwise.
- need_live_audio: true only if the brief explicitly asks for live, natural or unmuted sound.
- opener / closer: what the first / last shot shows, "" if not given.
- min_seconds / max_seconds: length of each clip. budget_min_seconds / budget_max_seconds: total runtime of the \
whole edit. count: number of clips. soundbite_min_seconds / soundbite_max_seconds: length of a requested spoken \
soundbite. 0 for anything the brief does not state as a number.
- exhaustive: true for "every / all / each ..." lists.
- mood: "neutral" unless the brief asks for an energetic or calm feel. speech: "wanted" only if the clips must be \
someone talking, "unwanted" if they must not contain talking, else "any".
- framing: the shot size the whole edit asks for: "close" for close-ups, faces, subjects filling the frame or coming \
right at the camera, vertical / 9:16 or POV edits; "wide" for establishing or wide views, the place itself as the \
subject; else "none". The opener's or closer's own shot size and shot sizes the brief rules out do not count.
- want_faces: true if the brief asks for faces. hook: true if the first clip must grab attention in its first \
second ("hook", "scroll-stopper"). legible_text: true if names, signs, banners or text must be readable. centered: \
true if the subject must sit in the middle of the frame (centred, a 9:16 or square crop)."""
# Worked examples from other domains (never from an evaluation set): a visual list, a speech brief, a casual German
# brief (translated items), a vertical hook reel (the visual cue fields).
LLM_EXAMPLES = [
    ("Make a 2 minute highlight video of the Riverside Food Festival. Start with a drone view of the market tents. "
     "Show the chefs at the grill, kids eating ice cream, the Golden Ladle stall and the brass band. Finish with the "
     "lantern release. No shots of the toilets or of people's backs blocking the view.",
     {"content": "Riverside Food Festival highlights",
      "shots": ["chefs cooking at the grill", "kids eating ice cream", "Golden Ladle food stall", "brass band playing"],
      "avoid": [], "exclude": ["portable toilets", "person's back blocking the view"], "topics": [],
      "need_live_audio": False, "mood": "neutral", "speech": "any", "min_seconds": 0, "max_seconds": 0, "count": 0,
      "budget_min_seconds": 102, "budget_max_seconds": 138, "soundbite_min_seconds": 0, "soundbite_max_seconds": 0,
      "exhaustive": False, "opener": "drone view of the market tents", "closer": "lantern release", **_NO_CUES}),
    ("Pull all the bits where the coach explains the new training plan or talks about the injury (\"out for six "
     "weeks\"). 10-20 s each.",
     {"content": "coach talking", "shots": [], "avoid": [], "exclude": [],
      "topics": ["the new training plan", "the injury, \"out for six weeks\""], "need_live_audio": False,
      "mood": "neutral", "speech": "wanted", "min_seconds": 10, "max_seconds": 20, "count": 0,
      "budget_min_seconds": 0, "budget_max_seconds": 0, "soundbite_min_seconds": 0, "soundbite_max_seconds": 0,
      "exhaustive": True, "opener": "", "closer": "", **_NO_CUES}),
    ("kannst du mir die schönsten momente vom geburtstag raussuchen, die torte, das tanzen und die kinder beim "
     "spielen, so 45 sek. keine wackler und bitte kein essen",
     {"content": "birthday party highlights", "shots": ["birthday cake", "people dancing", "children playing"],
      "avoid": [], "exclude": ["shaky footage", "food"], "topics": [], "need_live_audio": False, "mood": "neutral",
      "speech": "any", "min_seconds": 0, "max_seconds": 0, "count": 0, "budget_min_seconds": 38,
      "budget_max_seconds": 52, "soundbite_min_seconds": 0, "soundbite_max_seconds": 0, "exhaustive": False,
      "opener": "", "closer": "", **_NO_CUES}),
    ("30 sec vertical reel of the skate contest. first shot has to hook: a rider flying straight at the lens. then "
     "tight shots of faces and tricks, keep the riders centred for the 9:16 crop. sponsor banners must be readable",
     {"content": "skate contest reel", "shots": ["skater faces", "skateboard tricks", "sponsor banners"], "avoid": [],
      "exclude": [], "topics": [], "need_live_audio": False, "mood": "neutral", "speech": "any", "min_seconds": 0,
      "max_seconds": 0, "count": 0, "budget_min_seconds": 25.5, "budget_max_seconds": 34.5,
      "soundbite_min_seconds": 0, "soundbite_max_seconds": 0, "exhaustive": False,
      "opener": "skater flying straight at the camera", "closer": "", "framing": "close", "want_faces": True,
      "hook": True, "legible_text": True, "centered": True}),
]
LIVE_CUE = r"\b(?:sound|audio|noise|ton|muted)\b"
EXHAUSTIVE_CUE = r"\b(?:every|each|all|any|whenever|jede[nrs]?|alle)\b"


def default_llm() -> str | None:
    """The parser model: VHE_LLM overrides DEFAULT_LLM; 'none', 'off' or '' means rules only."""
    v = os.environ.get("VHE_LLM", DEFAULT_LLM).strip()
    return None if v.lower() in ("", "none", "off", "0", "rules") else v


def _ask_llm(query: str, model: str, host: str) -> tuple[dict, bool]:
    """(answer, from_cache). Raises on any failure. Answers are cached on disk, so a brief parses the same way on
    every run and repeated runs do not wait for the model."""
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, model, query]).encode()).hexdigest()[:32]
    path = Path(PARSE_CACHE) / f"{key}.json"
    try:
        hit = json.loads(path.read_text())
        if hit.get("model") == model and hit.get("brief") == query and isinstance(hit.get("data"), dict):
            return hit["data"], True
    except (OSError, ValueError):
        pass
    messages = [{"role": "system", "content": LLM_PROMPT}]
    for user, answer in LLM_EXAMPLES:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": json.dumps(answer)}]
    messages.append({"role": "user", "content": query})
    body = {"model": model, "stream": False, "format": LLM_SCHEMA, "think": False, "keep_alive": LLM_KEEP_ALIVE,
            "options": dict(LLM_OPTIONS), "messages": messages}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # The loaded query models lend their GPU memory to the parse (vhe.models.gpu_released), so the model fits on it.
    with gpu_released(), urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        data = json.loads(json.loads(resp.read())["message"]["content"])
    if not isinstance(data, dict):
        raise ValueError("LLM answer is not a JSON object")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": model, "brief": query, "prompt_version": PROMPT_VERSION, "data": data},
                                  ensure_ascii=False, indent=1))
        tmp.replace(path)
    except OSError:
        pass
    return data, False


def _numbers(raw: str) -> set[float]:
    """Every number in the brief, also as minutes -> seconds ('a 2 minute recap' grounds 120)."""
    nums = {float(x) for x in re.findall(NUM, normalise(raw))}
    return nums | {60.0 * x for x in nums}


EN_WORDS = {"the", "and", "with", "of", "to", "is", "in", "for", "on", "that", "it", "this", "from", "but", "or",
            "me", "my", "you", "where", "every", "all", "no", "not", "shot", "shots", "clip", "clips", "need", "want"}
DE_WORDS = {"der", "die", "das", "und", "mit", "von", "vom", "zu", "ist", "ein", "eine", "nicht", "nich", "für", "auf",
            "ich", "du", "bitte", "wo", "sind", "kannst", "mir", "den", "dem", "auch", "keine", "kein", "ohne", "oder",
            "welche", "momente", "szenen", "aus", "noch", "mal", "weg"}


def is_english(text: str) -> bool:
    """More English than German function words (the rules and grounding are English-first)."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return sum(w in EN_WORDS for w in words) >= sum(w in DE_WORDS for w in words)


# Words of an instruction to the editor or the tool, not of a shot ("dont cut anything yet", "tell me the clip and
# the exact seconds", "mein handy ist voll", "welche sind die 5 die ich behalten soll").
INSTRUCTION_WORDS = {"i", "i'm", "im", "me", "my", "we", "we're", "our", "us", "you", "your", "you're", "dont", "don't",
                     "do", "does", "give", "tell", "rank", "ranked", "ranking", "list", "mark", "send", "need", "want",
                     "bonus", "cut", "let", "let's", "please", "lol", "ich", "mein", "meine", "meinen", "mir", "mich",
                     "du", "dein", "deine", "welche", "welcher", "welches", "soll", "sollte", "kann", "kannst", "sind",
                     "ist", "bitte", "behalten", "löschen", "loeschen"}
NOT_SHOT = {"max", "min", "yet", "exact", "exactly", "seconds", "second", "sec", "secs", "ranked", "context",
            "possible", "actually", "really", "right", "there", "here"}


def _instruction(item: str) -> bool:
    """A clause the rules took for a shot that is an instruction (a pronoun or instruction verb), or that names
    nothing visible (only style, cue and filler words: "ranked", "max")."""
    words = word_tokens(item)
    if any(w in INSTRUCTION_WORDS for w in words):
        return True
    return not any(len(w) > 2 and w not in STYLE | ITEM_EDGE | CUE_FILLER | GENERIC_HEAD | NOT_SHOT for w in words)


def _shot_fallback(rules_shots: list[str]) -> list[str]:
    """The rules' shot list when the LLM's items are gone (none given, or every one invented: grounding dropped
    them): its clauses that name something visible, but no shot list at all when instruction clauses are the
    majority (p2-tiktok-2: "dont cut anything yet; ...; tell me the clip and the exact seconds ..."), so relevance
    falls back to the content text and the brief's framing / hook cues instead of clauses read as pictures."""
    keep = [x for x in rules_shots if not _instruction(x)]
    return keep if len(keep) >= 2 and 2 * len(keep) > len(rules_shots) else []


def _grounded(item: str, vocab: set[str]) -> bool:
    """More than half of the item's content words (4-letter stems) occur in the brief: drops invented items."""
    stems = [w[:4] for w in re.findall(r"[^\W\d_]+", item.lower()) if len(w) > 2 and w not in EDGE_STOP]
    return bool(stems) and sum(s in vocab for s in stems) / len(stems) > 0.5


def _from_llm(data: dict, base: ParsedQuery, explicit_len: bool) -> tuple[ParsedQuery, bool]:
    """The model's plan, grounded in the brief: numbers it gives that the brief does not contain are ignored, and
    numbers, live sound, speech and exhaustiveness the rules found always stand."""
    nums, vocab = _numbers(base.raw), {w[:4] for w in re.findall(r"[^\W\d_]+", base.raw.lower())}
    # A brief in another language gets its items translated (the prompt asks for English), so they cannot be grounded
    # word by word in it; the model is still told never to invent items.
    english = is_english(base.raw)
    low = normalise(base.raw)

    def num(key: str) -> float:
        try:
            v = float(data.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0
        return v if v > 0 and any(abs(v - n) < 1e-6 for n in nums) else 0.0

    def items(key: str) -> list[str]:
        v = data.get(key)
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip() and (not english or _grounded(str(x), vocab))]

    p = ParsedQuery(raw=base.raw, text=" ".join(str(data.get("content") or "").split()[:12]) or base.text)
    p.weights = dict(base.weights)
    set_mood(p, base.mood or data.get("mood"))
    p.weights["speech"] = base.weights["speech"]
    p.min_len, p.max_len = base.min_len, base.max_len
    if not explicit_len and (num("min_seconds") or num("max_seconds")):
        lo, hi = num("min_seconds"), num("max_seconds")
        p.min_len, p.max_len, explicit_len = lo or max(1.0, 0.7 * hi), hi or 1.3 * lo, True
    p.count, p.count_explicit = base.count, base.count_explicit
    if not p.count_explicit and num("count"):
        p.count, p.count_explicit = int(num("count")), True
    p.budget = base.budget
    lo, hi = num("budget_min_seconds"), num("budget_max_seconds")
    # "4-8 s per select" is a clip length the rules already read: the same numbers are not also a runtime budget.
    clip_numbers = {round(base.min_len, 2), round(base.max_len, 2)} if explicit_len else set()
    if p.budget is None and (lo or hi) and not {round(x, 2) for x in (lo, hi) if x} <= clip_numbers:
        p.budget = (lo or round(hi * (1 - BUDGET_TOL), 2), hi or round(lo * (1 + BUDGET_TOL), 2))
    p.soundbite_len = base.soundbite_len
    if p.soundbite_len is None and num("soundbite_max_seconds"):
        p.soundbite_len = (num("soundbite_min_seconds") or 0.7 * num("soundbite_max_seconds"),
                           num("soundbite_max_seconds"))
    p.exhaustive = base.exhaustive or (bool(data.get("exhaustive")) and re.search(EXHAUSTIVE_CUE, p.raw.lower())
                                       is not None)
    p.need_live_audio = base.need_live_audio or (bool(data.get("need_live_audio")) and
                                                 re.search(LIVE_CUE, p.raw.lower()) is not None)
    p.opener = " ".join(str(data.get("opener") or "").split()) or base.opener
    p.closer = " ".join(str(data.get("closer") or "").split()) or base.closer
    p.shots = items("shots") or _shot_fallback(base.shots)
    # Forbidden items stay hard only when the brief has a word that forbids ("no", "never", "keine", ...).
    ex = items("exclude")
    hard = re.search(HARD_CUE, low) is not None
    p.avoid = items("avoid") + ex + base.avoid
    p.exclude = (ex if hard else []) + base.exclude
    p.no_speech = base.no_speech or (data.get("speech") == "unwanted" and hard and
                                     re.search(SPEECH_ITEM, low) is not None)
    if p.no_speech:
        p.weights["speech"] = min(p.weights["speech"], -1.0)
    p.per_item = re.search(ALL_MATCHES, low) is None
    p.highlight = base.highlight
    p.topics = items("topics") or base.topics
    if p.topics and not p.shots and p.text == base.text:
        p.text = ""
    # Visual cues: the rules' always stand; the model's count only where the brief (without its opener/closer and
    # avoid clauses; the whole brief for the hook) has a matching stem, so "open on an establishing wide" does not
    # make the whole edit wide.
    scope = base.cue_text
    p.cue_text, p.hook_list = scope, base.hook_list
    fr = data.get("framing")
    p.framing = base.framing if base.framing != "none" else \
        fr if fr in ("close", "wide") and re.search(CUE_GROUND[fr], scope) else "none"
    for key in ("want_faces", "hook", "legible_text", "centered"):
        ground = re.search(CUE_GROUND[key], low if key == "hook" else scope) is not None
        setattr(p, key, getattr(base, key) or (data.get(key) is True and ground))
    return p, explicit_len


def parse_with_llm(query: str, model: str, host: str = OLLAMA_HOST, **overrides) -> ParsedQuery:
    """Ask a local Ollama model for the plan, grounded by the rules; the rules alone on any failure."""
    base, explicit_len, hint = parse_rules(query)
    ov = (overrides.get("count"), overrides.get("min_len"), overrides.get("max_len"), overrides.get("budget_s"))
    try:
        data, cached = _ask_llm(base.raw, model, host)
        p, explicit_len = _from_llm(data, base, explicit_len)
    except Exception as exc:  # unreachable server, unknown model, timeout, malformed answer
        base.parser_note = f"{model} unavailable ({type(exc).__name__}): rules used"
        return finalise(base, explicit_len, hint, *ov)
    p.parser, p.parser_note = f"llm:{model}", "cached" if cached else None
    return finalise(p, explicit_len, hint, *ov)
