"""Turn a query into signal weights, length limits and the content text used for relevance.

"chill scenic view" and "short high-energy scenes" differ mostly in motion, camera steadiness and loudness,
which image-text similarity barely measures. So mood words become weights on those signals instead of being
matched as text. English and German. An optional local LLM (Ollama, JSON schema enforced) can replace the
rules; the rules are always the fallback.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field

DEFAULT_MIN_LEN = 3.0
DEFAULT_MAX_LEN = 30.0
DEFAULT_COUNT = 5

DEFAULT_WEIGHTS = {"relevance": 1.0, "quality": 0.4, "energy": 0.0, "steady": 0.2, "speech": 0.0}

CALM = ["chill", "calm", "relaxed", "relaxing", "peaceful", "quiet", "slow", "serene", "tranquil", "cozy", "scenic",
        "ruhig", "ruhige", "entspannt", "entspannte", "gemütlich", "friedlich", "langsam", "chillig"]
ENERGY = ["high-energy", "high energy", "energetic", "action", "action-packed", "exciting", "intense", "fast",
          "hype", "wild", "crazy", "epic", "energiegeladen", "energiegeladene", "actiongeladen", "spannend",
          "spannende", "schnell", "schnelle", "aufregend", "wilde"]
SPEECH_ON = ["talking", "speaking", "speech", "interview", "conversation", "says", "spricht", "sprechen", "rede",
             "gespräch", "redet"]
SPEECH_OFF = ["no talking", "without talking", "no speech", "without speech", "ohne sprache", "ohne reden",
              "ohne gespräch"]
SHORT = ["short", "quick", "kurz", "kurze", "kurzen"]
LONG = ["long", "longer", "lang", "lange", "langen"]
FILLER = ["show me", "find me", "find", "give me", "i want", "i need", "zeig mir", "finde", "ich will", "ich brauche",
          "scenes", "scene", "clips", "clip", "moments", "moment", "parts", "szenen", "szene", "momente", "stellen",
          "with", "mit", "and", "und", "the", "die", "der", "das", "some", "ein paar", "einige"]
UNIT = r"(?:s|sec|secs|second|seconds|sek|sekunde|sekunden)\b"


@dataclass
class ParsedQuery:
    raw: str
    text: str
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_len: float = DEFAULT_MIN_LEN
    max_len: float = DEFAULT_MAX_LEN
    count: int = DEFAULT_COUNT
    mood: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", text) is not None


def _strip(text: str, phrase: str) -> str:
    return re.sub(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", " ", text)


def parse(query: str, count: int | None = None, min_len: float | None = None,
          max_len: float | None = None) -> ParsedQuery:
    raw = query.strip()
    q = raw.lower()
    p = ParsedQuery(raw=raw, text="")
    rest = q

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|bis)\s*(\d+(?:\.\d+)?)\s*" + UNIT, rest)
    if m:
        p.min_len, p.max_len = float(m.group(1)), float(m.group(2))
        rest = rest.replace(m.group(0), " ")
    m = re.search(r"(?:under|less than|max(?:imum)?|at most|unter|höchstens|maximal|bis zu)\s*(\d+(?:\.\d+)?)\s*"
                  + UNIT, rest)
    if m:
        p.max_len = float(m.group(1))
        rest = rest.replace(m.group(0), " ")
    m = re.search(r"(?:over|more than|at least|min(?:imum)?|über|mindestens)\s*(\d+(?:\.\d+)?)\s*" + UNIT, rest)
    if m:
        p.min_len = float(m.group(1))
        rest = rest.replace(m.group(0), " ")
    # "3 clips", "3 calm scenes", "3 ruhige Szenen": up to two words between the number and the noun.
    m = re.search(r"\b(\d+)\s+(?:[^\W\d]+\s+){0,2}?(?:clips|scenes|moments|highlights|szenen|momente|stellen)\b",
                  rest)
    if m:
        p.count = max(1, min(50, int(m.group(1))))
        rest = rest[:m.start(1)] + " " + rest[m.end(1):]

    for phrase in SPEECH_OFF:
        if _contains(rest, phrase):
            p.weights["speech"] = -1.0
            rest = _strip(rest, phrase)
    if p.weights["speech"] == 0.0 and any(_contains(rest, w) for w in SPEECH_ON):
        p.weights["speech"] = 1.0

    calm = [w for w in CALM if _contains(rest, w)]
    energy = [w for w in ENERGY if _contains(rest, w)]
    if energy and len(energy) >= len(calm):
        p.mood = "energetic"
        p.weights.update(energy=1.2, steady=0.0, quality=0.3)
    elif calm:
        p.mood = "calm"
        p.weights.update(energy=-0.9, steady=0.8, quality=0.6)
    for w in energy + calm:
        rest = _strip(rest, w)

    if any(_contains(rest, w) for w in SHORT):
        p.max_len = min(p.max_len, 10.0)
        p.min_len = min(p.min_len, 2.0)
    elif any(_contains(rest, w) for w in LONG):
        p.min_len = max(p.min_len, 20.0)
        p.max_len = max(p.max_len, 60.0)
    for w in SHORT + LONG + FILLER:
        rest = _strip(rest, w)

    # "scenic" is both a mood and content: keep it in the text so "scenic view" still matches landscapes.
    keep = [w for w in ("scenic",) if _contains(q, w)]
    p.text = " ".join([*keep, *rest.replace(",", " ").split()]).strip()
    if not p.text:
        p.weights["relevance"] = 0.0

    if count is not None:
        p.count = count
    if min_len is not None:
        p.min_len = min_len
    if max_len is not None:
        p.max_len = max_len
    p.min_len = max(1.0, p.min_len)
    p.max_len = max(p.min_len + 1.0, p.max_len)
    return p


LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "what should be visible, without mood or length words"},
        "mood": {"type": "string", "enum": ["calm", "energetic", "neutral"]},
        "speech": {"type": "string", "enum": ["wanted", "unwanted", "any"]},
        "min_seconds": {"type": "number"}, "max_seconds": {"type": "number"},
        "count": {"type": "integer"},
    },
    "required": ["content", "mood", "speech", "min_seconds", "max_seconds", "count"],
}


def parse_with_llm(query: str, model: str, host: str = "http://127.0.0.1:11434", **overrides) -> ParsedQuery:
    """Ask a local Ollama model for the same fields; fall back to the rules on any failure."""
    base = parse(query, **overrides)
    body = {"model": model, "stream": False, "format": LLM_SCHEMA, "think": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": "Extract video clip search parameters as JSON."},
                         {"role": "user", "content": query}]}
    try:
        req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(json.loads(resp.read())["message"]["content"])
    except Exception:
        return base
    p = ParsedQuery(raw=query, text=data.get("content", "").strip())
    if data.get("mood") == "energetic":
        p.mood = "energetic"
        p.weights.update(energy=1.2, steady=0.0, quality=0.3)
    elif data.get("mood") == "calm":
        p.mood = "calm"
        p.weights.update(energy=-0.9, steady=0.8, quality=0.6)
    p.weights["speech"] = {"wanted": 1.0, "unwanted": -1.0}.get(data.get("speech"), 0.0)
    p.min_len = float(overrides.get("min_len") or data.get("min_seconds") or DEFAULT_MIN_LEN)
    p.max_len = float(overrides.get("max_len") or data.get("max_seconds") or DEFAULT_MAX_LEN)
    p.count = int(overrides.get("count") or data.get("count") or DEFAULT_COUNT)
    p.min_len = max(1.0, p.min_len)
    p.max_len = max(p.min_len + 1.0, p.max_len)
    if not p.text:
        p.weights["relevance"] = 0.0
    return p
