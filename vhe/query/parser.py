"""Turn an editor's brief into signal weights, length limits, a runtime budget and the content text used for
relevance.

"chill scenic view" and "short high-energy scenes" differ mostly in motion, camera steadiness and loudness,
which image-text similarity barely measures. So mood words become weights on those signals instead of being
matched as text. Durations are either a total runtime ("60-90 s energetic recap", "a 1 minute recap") or a
per-clip length ("3-6 s each", "clips of 2-4 seconds"); every parsed phrase is removed from the content text so
only visual words reach the image-text model. English and German. An optional local LLM (Ollama, JSON schema
enforced) can replace the rules; the rules are always the fallback.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field

DEFAULT_MIN_LEN = 3.0
DEFAULT_MAX_LEN = 30.0
DEFAULT_COUNT = 5
MAX_COUNT = 50       # explicit counts
MAX_CLIPS = 150      # counts derived from a runtime budget
EXHAUSTIVE_CAP = 50  # "every moment ...": all matches, up to this many
BUDGET_TOL = 0.15    # a single budget value means +-15 %
BARE_BUDGET_S = 30.0  # a duration with no clip/recap noun anywhere: at least this long reads as total runtime

DEFAULT_WEIGHTS = {"relevance": 1.0, "quality": 0.4, "energy": 0.0, "steady": 0.3, "speech": 0.0}
# Steadiness stays >= 0.3 for every mood: otherwise a pocket shot (swinging camera, loud muffled audio) wins
# energetic queries.
MOODS = {"energetic": {"energy": 1.2, "steady": 0.3, "quality": 0.3},
         "calm": {"energy": -0.9, "steady": 0.8, "quality": 0.6}}
# Typical clip length (s) when a runtime budget is given but no per-clip length.
TYPICAL_LEN = {"energetic": (2.0, 4.0), "calm": (4.0, 8.0), None: (3.0, 6.0), "short": (2.0, 4.0),
               "long": (8.0, 15.0)}

CALM = ["chill", "calm", "relaxed", "relaxing", "peaceful", "quiet", "slow", "serene", "tranquil", "cozy", "scenic",
        "ruhig", "ruhige", "entspannt", "entspannte", "gemütlich", "friedlich", "langsam", "chillig"]
ENERGY = ["high-energy", "high energy", "energetic", "action", "action-packed", "exciting", "intense", "fast",
          "hype", "wild", "crazy", "epic", "energiegeladen", "energiegeladene", "actiongeladen", "spannend",
          "spannende", "schnell", "schnelle", "aufregend", "wilde"]
# Speech intent. Verbs carry no visual meaning and are removed from the content text; the nouns in SPEECH_VISUAL
# ("toast", "interview", "to camera") also describe what is visible and stay in it.
SPEECH_VERBS = ["speaks", "speaking", "speak", "talks", "talking", "talk", "says", "saying", "addresses",
                "addressing", "speech", "speeches", "announcement", "announcements", "spricht", "sprechen", "redet",
                "reden", "rede", "sagt"]
SPEECH_VISUAL = ["to camera", "to the camera", "interview", "interviews", "toast", "toasts", "presenter",
                 "conversation", "gespräch", "in die kamera"]
SPEECH_ON = SPEECH_VERBS + SPEECH_VISUAL
SHORT = ["short", "quick", "kurz", "kurze", "kurzen"]
LONG = ["long", "longer", "lang", "lange", "langen"]
CLIP_NOUNS = r"(?:clips?|scenes?|shots?|moments?|highlights|parts|szenen?|momente|stellen|aufnahmen)"
BUDGET_NOUNS = (r"(?:recaps?|edit|video|reel|montage|highlights?|cut|film|trailer|teaser|sizzle|summary|supercut|"
                r"compilation|rückblick|zusammenfassung|zusammenschnitt|schnitt)")
FORMAT = ["recap", "recaps", "edit", "reel", "montage", "highlight", "highlights", "supercut", "compilation",
          "summary", "trailer", "teaser", "sizzle", "video", "rückblick", "zusammenfassung", "zusammenschnitt"]
FILLER = ["show me", "find me", "find", "give me", "i want", "i need", "make", "create", "zeig mir", "finde",
          "ich will", "ich brauche", "scenes", "scene", "clips", "clip", "shots", "shot", "moments", "moment",
          "parts", "szenen", "szene", "momente", "stellen", "aufnahmen",
          "with", "mit", "and", "und", "the", "die", "der", "das", "some", "ein paar", "einige"]
EDGE_STOP = {"a", "an", "the", "of", "with", "for", "from", "in", "on", "at", "and", "or", "to", "some", "ein", "eine",
             "einer", "einem", "der", "die", "das", "den", "dem", "des", "von", "vom", "im", "am", "mit", "und"}
UNIT = r"(?:s|sec|secs|second|seconds|sek|sekunde|sekunden|min|mins|minute|minutes|minuten)\b"
NUM = r"\d+(?:\.\d+)?"
VALUE = rf"(?P<a>{NUM})\s*(?P<ua>{UNIT})?(?:\s*(?:-|to|bis)\s*(?P<b>{NUM}))?[\s-]*(?P<u>{UNIT})"
WORDS = r"(?:[^\W\d][\w-]*\s+)"  # one word and its trailing space; never crosses digits or punctuation
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
                "fifty": 50, "sixty": 60, "ninety": 90, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "zehn": 10,
                "dreißig": 30}
MODIFIERS = (r"(?:(?P<total>in total|total|overall|altogether|insgesamt|gesamt)|"
             r"(?P<le>under|less than|below|at most|max(?:imum)?|no (?:longer|more) than|up to|within|unter|"
             r"höchstens|maximal|bis zu)|"
             r"(?P<ge>over|more than|at least|min(?:imum)?|über|mindestens)|"
             r"(?P<about>about|around|roughly|approx(?:imately)?|circa|ca\.?|ungefähr|etwa|rund))\s*$")
EDGE_OF = (r"\s+(?:(?:it|the\s+(?:edit|video|recap|reel|film|montage))\s+)?(?:with|on|mit|auf)\s+"
           r"(?:(?:a|an|the|some|ein|eine|einem|einer|dem|der)\s+)?(?P<x>[^,;.]+?)\s*"
           r"(?=[,;.]|\b(?:and|then|und|dann|end|ends|close|closes|finish|start|open|begin)\b|$)")
OPENER = (r"\b(?:open|opens|opening|start|starts|starting|begin|begins|beginning|kick off|lead|beginne|beginnen|"
          r"starte|starten|fange an)" + EDGE_OF)
CLOSER = (r"\b(?:end|ends|ending|close|closes|closing|finish|finishes|finishing|wrap up|wrap it up|ende|enden|"
          r"schließe|schließen|höre auf)" + EDGE_OF)
# "every toast" is exhaustive on its own; "all"/"each" only with a generic noun ("all smiles" is not).
GENERIC = r"(?:moments?|times?|clips?|shots?|scenes?|instances?|parts?|bits?|occurrences?|momente|szenen|stellen|male?)"
EXHAUSTIVE = (rf"\bevery(?:\s+single)?(?:\s+{GENERIC})?\b|\b(?:all(?: the)?|any|each|jede[nrs]?|alle)\s+{GENERIC}\b|"
              r"\bwhenever\b|\bimmer wenn\b")


@dataclass
class ParsedQuery:
    raw: str
    text: str
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_len: float = DEFAULT_MIN_LEN  # per clip
    max_len: float = DEFAULT_MAX_LEN
    count: int = DEFAULT_COUNT  # target number of clips; with a budget and no explicit count, derived from it
    mood: str | None = None
    budget: tuple[float, float] | None = None  # total runtime of all clips together, (min, max) seconds
    count_explicit: bool = False  # the query or caller named a count (otherwise budget/mode may change it)
    exhaustive: bool = False  # "every moment ...": all matches above a threshold instead of a fixed count
    opener: str | None = None  # "open with an establishing shot" -> "establishing shot"
    closer: str | None = None  # "end with the fireworks" -> "fireworks"

    def to_dict(self) -> dict:
        return asdict(self)


def _seconds(value: str, unit: str) -> float:
    return float(value) * (60.0 if unit.startswith("min") else 1.0)


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", text) is not None


def _strip(text: str, phrase: str) -> str:
    return re.sub(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", " ", text)


def _cut(text: str, a: int, b: int) -> str:
    return text[:a] + " " + text[b:]


def _around(v: float) -> tuple[float, float]:
    return round(v * (1 - BUDGET_TOL), 2), round(v * (1 + BUDGET_TOL), 2)


def _set_mood(p: ParsedQuery, mood: str | None) -> None:
    if mood in MOODS:
        p.mood = mood
        p.weights.update(MOODS[mood])


def _normalise(q: str) -> str:
    q = re.sub(r"[‒–—−]", "-", q.lower())  # en/em dashes, minus
    q = re.sub(r"\b(\d+):([0-5]\d)\b(?:\s*(?:min|mins|minutes?|minuten)\b)?",
               lambda m: f"{int(m.group(1)) * 60 + int(m.group(2))} s", q)  # 1:30 -> 90 s
    q = re.sub(r"\b(?:a|an|one|eine|einer)\s+(?=(?:min|minute|minuten)\b)", "1 ", q)
    words = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    return re.sub(rf"\b({words})\b(?=[\s-]+(?:{UNIT}|{WORDS}{{0,3}}?{CLIP_NOUNS}\b))",
                  lambda m: str(NUMBER_WORDS[m.group(1)]), q)


def _durations(rest: str, q: str) -> tuple[str, dict, dict]:
    """Find every duration, decide whether it is the total runtime or a per-clip length, and remove it."""
    budget: dict = {}
    clip: dict = {}
    while (m := re.search(VALUE, rest)):
        a = _seconds(m.group("a"), m.group("ua") or m.group("u"))
        b = _seconds(m.group("b"), m.group("u")) if m.group("b") else None
        pre, post = rest[:m.start()], rest[m.end():]
        mod = re.search(r"(?<![\w-])" + MODIFIERS, pre)
        kind = "total" if mod and mod.group("total") else (next(k for k in ("le", "ge", "about") if mod.group(k))
                                                            if mod else None)
        start = mod.start() if mod else m.start()
        pre_bare = pre[:start]
        end = m.end()
        each = re.match(r"\s*(?:each|apiece|per (?:clip|shot|scene)|jeweils|pro (?:clip|szene))\b", post)
        total_after = re.match(r"\s*(?:in total|total|overall|altogether|insgesamt|gesamt)\b", post)
        noun_after = re.match(rf"[\s-]*(?:long\s+)?(?P<w>{WORDS}{{0,3}}?)(?P<n>{BUDGET_NOUNS}|{CLIP_NOUNS})\b", post)
        if noun_after and re.search(CLIP_NOUNS + r"\b", noun_after.group("w")):
            noun_after = None  # "10 second clips for a recap": the nearer noun decides
        noun_before = re.search(rf"(?P<n>{BUDGET_NOUNS}|{CLIP_NOUNS})\s+(?:(?:of|von|mit|lasting|dauer)\s+)?$",
                                pre_bare)
        if each:
            is_budget, end = False, m.end() + each.end()
        elif kind == "total" or total_after:
            is_budget, end = True, m.end() + (total_after.end() if total_after else 0)
        elif noun_before:
            is_budget = re.fullmatch(BUDGET_NOUNS, noun_before.group("n")) is not None
        elif noun_after:
            is_budget = re.fullmatch(BUDGET_NOUNS, noun_after.group("n")) is not None
            if is_budget:  # the format noun goes too; words in between ("energetic") stay for the mood
                n0 = m.end() + noun_after.start("n")
                rest = rest[:n0] + " " + rest[n0 + len(noun_after.group("n")):]
        elif re.search(rf"(?<![\w-]){BUDGET_NOUNS}(?![\w-])", q):
            is_budget = True
        elif re.search(rf"(?<![\w-]){CLIP_NOUNS}(?![\w-])", q):
            is_budget = False
        else:
            is_budget = (b or a) >= BARE_BUDGET_S
        target = budget if is_budget else clip
        if b is not None:
            target["lo"], target["hi"] = a, b
        elif kind == "le":
            target["hi"] = a
        elif kind == "ge":
            target["lo"] = a
        elif is_budget:
            target["lo"], target["hi"] = _around(a)
        else:  # "10 second clips": about ten seconds
            target["lo"], target["hi"] = max(1.0, 0.7 * a), 1.3 * a
        rest = _cut(rest, start, end)
    return rest, budget, clip


def _clean_text(rest: str) -> str:
    words = re.sub(r"[,;:.!?\"()]", " ", rest).replace(" - ", " ").split()
    while words and words[0] in EDGE_STOP:
        words.pop(0)
    while words and words[-1] in EDGE_STOP:
        words.pop()
    return " ".join(w for w in words if w != "-")


def parse(query: str, count: int | None = None, min_len: float | None = None, max_len: float | None = None,
          budget_s: float | tuple | None = None) -> ParsedQuery:
    raw = query.strip()
    q = _normalise(raw)
    p = ParsedQuery(raw=raw, text="")
    rest = q

    for attr, pattern in (("opener", OPENER), ("closer", CLOSER)):
        m = re.search(pattern, rest)
        if m:
            setattr(p, attr, m.group("x").strip())
            rest = _cut(rest, m.start(), m.end())

    rest, budget, clip = _durations(rest, q)
    if budget:
        lo = budget.get("lo", budget.get("hi", 0) * (1 - BUDGET_TOL))
        hi = budget.get("hi", lo * (1 + BUDGET_TOL))
        p.budget = (round(min(lo, hi), 2), round(max(lo, hi), 2))
    explicit_len = bool(clip)
    p.min_len = clip.get("lo", p.min_len)
    p.max_len = clip.get("hi", p.max_len)

    m = re.search(EXHAUSTIVE, rest)
    if m:
        p.exhaustive = True
        rest = _cut(rest, m.start(), m.end())

    # "3 clips", "3 short high-energy scenes", "3 ruhige Szenen": up to three words between number and noun.
    m = re.search(rf"\b(\d+)\s+{WORDS}{{0,3}}?{CLIP_NOUNS}\b", rest)
    if m:
        p.count, p.count_explicit = max(1, min(MAX_COUNT, int(m.group(1)))), True
        rest = _cut(rest, m.start(1), m.end(1))

    m = re.search(r"\b(?:without|no|ohne|kein|keine)\s+(?:[^\W\d][\w-]*\s+){0,2}?"
                  r"(?:talking|speaking|speech|voices?|dialogue|narration|sprache|reden|gespräch|stimmen?)\b", rest)
    if m:
        p.weights["speech"] = -1.0
        rest = _cut(rest, m.start(), m.end())
    if p.weights["speech"] == 0.0 and any(_contains(rest, w) for w in SPEECH_ON):
        p.weights["speech"] = 1.0
    for w in SPEECH_VERBS:
        rest = _strip(rest, w)

    calm = [w for w in CALM if _contains(rest, w)]
    energy = [w for w in ENERGY if _contains(rest, w)]
    _set_mood(p, "energetic" if energy and len(energy) >= len(calm) else "calm" if calm else None)
    for w in energy + calm:
        rest = _strip(rest, w)

    hint = "short" if any(_contains(rest, w) for w in SHORT) else "long" if any(_contains(rest, w) for w in LONG) \
        else None
    if not explicit_len and not p.budget:  # explicit seconds always win over "short"/"long"
        if hint == "short":
            p.max_len = min(p.max_len, 10.0)
            p.min_len = min(p.min_len, 2.0)
        elif hint == "long":
            p.min_len = max(p.min_len, 20.0)
            p.max_len = max(p.max_len, 60.0)
    for w in SHORT + LONG + FILLER + FORMAT:
        rest = _strip(rest, w)

    # "scenic" is both a mood and content: keep it in the text so "scenic view" still matches landscapes.
    keep = [w for w in ("scenic",) if _contains(q, w)]
    p.text = " ".join([*keep, _clean_text(rest)]).strip()
    return _finalise(p, explicit_len, hint, count, min_len, max_len, budget_s)


def _finalise(p: ParsedQuery, explicit_len: bool, hint: str | None, count: int | None, min_len: float | None,
              max_len: float | None, budget_s: float | tuple | None) -> ParsedQuery:
    """Caller overrides, then everything derived from the budget, then clamps. Shared by rules and LLM."""
    if budget_s is not None:
        p.budget = tuple(float(x) for x in budget_s) if isinstance(budget_s, (tuple, list)) else _around(budget_s)
    if count is not None:
        p.count, p.count_explicit = int(count), True
    explicit_len = explicit_len or min_len is not None or max_len is not None
    mid = sum(p.budget) / 2 if p.budget else 0.0
    if p.budget and not explicit_len:
        if p.count_explicit:  # "5 clips, 60 s total": each about 12 s
            per = mid / max(1, p.count)
            p.min_len, p.max_len = 0.75 * per, 1.25 * per
        else:
            p.min_len, p.max_len = TYPICAL_LEN[hint or p.mood]
    if min_len is not None:
        p.min_len = min_len
    if max_len is not None:
        p.max_len = max_len
    p.min_len = round(max(1.0, p.min_len), 2)
    p.max_len = round(max(p.min_len + 1.0, p.max_len), 2)
    if p.count_explicit:
        p.count = max(1, min(MAX_COUNT, p.count))
    elif p.exhaustive:
        p.count = EXHAUSTIVE_CAP
    elif p.budget:
        p.count = max(1, min(MAX_CLIPS, round(mid / ((p.min_len + p.max_len) / 2))))
    if not p.text:
        p.weights["relevance"] = 0.0
    return p


LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "what should be visible, without mood, length or format words"},
        "mood": {"type": "string", "enum": ["calm", "energetic", "neutral"]},
        "speech": {"type": "string", "enum": ["wanted", "unwanted", "any"]},
        "min_seconds": {"type": "number", "description": "per clip, 0 if not given"},
        "max_seconds": {"type": "number", "description": "per clip, 0 if not given"},
        "count": {"type": "integer", "description": "number of clips, 0 if not given"},
        "budget_min_seconds": {"type": "number", "description": "total runtime of all clips, 0 if not given"},
        "budget_max_seconds": {"type": "number", "description": "total runtime of all clips, 0 if not given"},
        "exhaustive": {"type": "boolean", "description": "true for 'every/all moments ...'"},
        "opener": {"type": "string", "description": "what the edit should open with, '' if not given"},
        "closer": {"type": "string", "description": "what the edit should end with, '' if not given"},
    },
    "required": ["content", "mood", "speech", "min_seconds", "max_seconds", "count", "budget_min_seconds",
                 "budget_max_seconds", "exhaustive", "opener", "closer"],
}
LLM_PROMPT = ("Extract video edit parameters from an editor's brief as JSON. Distinguish the total runtime of the "
              "edit (budget) from the length of each clip. Use 0 or '' for anything not stated.")


def parse_with_llm(query: str, model: str, host: str = "http://127.0.0.1:11434", **overrides) -> ParsedQuery:
    """Ask a local Ollama model for the same fields; fall back to the rules on any failure."""
    base = parse(query, **overrides)
    body = {"model": model, "stream": False, "format": LLM_SCHEMA, "think": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": query}]}
    try:
        req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(json.loads(resp.read())["message"]["content"])
        if not isinstance(data, dict):
            return base
        p = ParsedQuery(raw=query, text=str(data.get("content") or "").strip())
        lo, hi = float(data.get("min_seconds") or 0), float(data.get("max_seconds") or 0)
        p.min_len, p.max_len = lo or DEFAULT_MIN_LEN, hi or DEFAULT_MAX_LEN
        if int(data.get("count") or 0) > 0:
            p.count, p.count_explicit = int(data["count"]), True
        blo, bhi = float(data.get("budget_min_seconds") or 0), float(data.get("budget_max_seconds") or 0)
        if blo or bhi:
            p.budget = (blo or round(bhi * (1 - BUDGET_TOL), 2), bhi or round(blo * (1 + BUDGET_TOL), 2))
        p.exhaustive = bool(data.get("exhaustive"))
        p.opener = str(data.get("opener") or "").strip() or None
        p.closer = str(data.get("closer") or "").strip() or None
    except Exception:
        return base
    _set_mood(p, data.get("mood"))
    p.weights["speech"] = {"wanted": 1.0, "unwanted": -1.0}.get(data.get("speech"), 0.0)
    return _finalise(p, bool(lo or hi), None, overrides.get("count"), overrides.get("min_len"),
                     overrides.get("max_len"), overrides.get("budget_s"))
