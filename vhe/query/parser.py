"""Turn an editor's brief into signal weights, length limits, a runtime budget and the content text used for
relevance.

"chill scenic view" and "short high-energy scenes" differ mostly in motion, camera steadiness and loudness,
which image-text similarity barely measures. So mood words become weights on those signals instead of being
matched as text. Durations are either a total runtime ("60-90 s energetic recap", "a 1 minute recap") or a
per-clip length ("3-6 s each", "clips of 2-4 seconds"); every parsed phrase is removed from the content text so
only visual words reach the image-text model. English and German.

A real brief is several sentences: a list of wanted shots, things to leave out, spoken topics, a sound requirement.
These become a plan on the ParsedQuery (shots, avoid, topics, need_live_audio, soundbite length). A local LLM
(Ollama, JSON schema enforced, cached on disk per model and brief) is the default parser when it is reachable; the
rules are the fallback and also ground the model: numbers the brief does not contain are replaced by the rules'
values, and avoid/topic items the rules find are added. Both paths end in the same _finalise.

Visual cues the index v3 signals can answer become fields too: framing (close / wide), want_faces, hook, legible_text
and centered, read on the brief without its opener/closer descriptions and avoid clauses (an LLM value counts only
where that text has a matching stem).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
CLIP_NOUNS = (r"(?:clips?|scenes?|shots?|moments?|highlights|parts|stretch(?:es)?|cutaways?|selects?|holds?|cuts|"
              r"hooks?|first-shots?|"
              r"szenen?|momente|stellen|aufnahmen)")
# A platform named next to a duration ("30 sek für insta") makes it the runtime of the whole post.
BUDGET_NOUNS = (r"(?:recaps?|edit|video|reels?|montage|highlights?|cut|film|trailer|teaser|sizzle|summary|supercut|"
                r"compilation|rückblick|zusammenfassung|zusammenschnitt|schnitt|insta|instagram|tiktok|youtube|piece|"
                r"package|pkg|spot|segment)")
FORMAT = ["recap", "recaps", "edit", "reel", "montage", "highlight", "highlights", "supercut", "compilation",
          "summary", "trailer", "teaser", "sizzle", "video", "b-roll", "broll", "rückblick", "zusammenfassung",
          "zusammenschnitt"]
FILLER = ["show me", "find me", "find", "give me", "i want", "i need", "make", "create", "zeig mir", "finde",
          "ich will", "ich brauche", "scenes", "scene", "clips", "clip", "shots", "shot", "moments", "moment",
          "parts", "szenen", "szene", "momente", "stellen", "aufnahmen",
          "with", "mit", "and", "und", "the", "die", "der", "das", "some", "ein paar", "einige", "absolutely",
          "definitely", "bitte", "please"]
EDGE_STOP = {"a", "an", "the", "of", "with", "for", "from", "in", "on", "at", "and", "or", "to", "some", "where",
             "when", "ein", "eine", "einer", "einem", "der", "die", "das", "den", "dem", "des", "von", "vom", "im",
             "am", "mit", "und"}
UNIT = r"(?:s|sec|secs|second|seconds|sek|sekunde|sekunden|min|mins|minute|minutes|minuten)\b"
NUM = r"\d+(?:\.\d+)?"
VALUE = rf"(?P<a>{NUM})\s*(?P<ua>{UNIT})?(?:\s*(?:-|to|bis)\s*(?P<b>{NUM}))?[\s-]*(?P<u>{UNIT})"
WORDS = r"(?:[^\W\d][\w-]*\s+)"  # one word and its trailing space; never crosses digits or punctuation
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
                "fifty": 50, "sixty": 60, "ninety": 90, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6,
                "sieben": 7, "acht": 8, "zehn": 10, "zwölf": 12, "fünfzehn": 15, "zwanzig": 20, "dreißig": 30,
                "vierzig": 40, "fünfzig": 50, "sechzig": 60, "neunzig": 90}
MODIFIERS = (r"(?:(?P<total>in total|total|overall|altogether|insgesamt|gesamt)|"
             r"(?P<le>under|less than|below|at most|max(?:imum)?|no (?:longer|more) than|up to|within|unter|"
             r"höchstens|maximal|bis zu)|"
             r"(?P<ge>over|more than|at least|min(?:imum)?|über|mindestens)|"
             r"(?P<about>about|around|roughly|approx(?:imately)?|circa|ca\.?|ungefähr|etwa|rund))\s*$")
# The description runs to the next punctuation, "then", or the other edge's verb phrase ("... and end with"), so
# "end on the giant flag and the chanting crowd at the end of the route" keeps its "and ..." and its "end of".
EDGE_VERBS = r"(?:end|ends|close|closes|finish|finishes|start|starts|open|opens|begin|begins|ende|starte|beginne)"
EDGE_OF = (r"\s+(?:(?:it|the\s+(?:edit|video|recap|reel|film|montage))\s+)?(?:with|on|mit|auf)\s+"
           r"(?:(?:a|an|the|some|ein|eine|einem|einer|dem|der)\s+)?(?P<x>[^,;.]+?)\s*"
           rf"(?=[,;.]|\b(?:then|dann)\b|\b(?:(?:and|und)\s+)?{EDGE_VERBS}\s+(?:it\s+|the\s+\w+\s+)?"
           r"(?:with|on|mit|auf)\b|$)")
OPENER = (r"\b(?:open|opens|opening|start|starts|starting|begin|begins|beginning|kick off|lead|beginne|beginnen|"
          r"starte|starten|fange an)" + EDGE_OF)
CLOSER = (r"\b(?:end|ends|ending|close|closes|closing|finish|finishes|finishing|wrap up|wrap it up|ende|enden|"
          r"schließe|schließen|höre auf)" + EDGE_OF)
# "every toast" is exhaustive on its own; "all"/"each" only with a generic noun ("all smiles" is not).
GENERIC = r"(?:moments?|times?|clips?|shots?|scenes?|instances?|parts?|bits?|occurrences?|momente|szenen|stellen|male?)"
EXHAUSTIVE = (rf"\bevery(?:\s+single)?(?:\s+{GENERIC})?\b|\b(?:all(?: the)?|any|each|jede[nrs]?|alle)\s+{GENERIC}\b|"
              r"\bwhenever\b|\bimmer wenn\b")
# "every shot where ...", "every usable shot of ...": all distinct matches; "every float: A, B, C": one per item.
ALL_MATCHES = (rf"\b(?:every|all(?: the)?|any|each|jede[nrs]?|alle)(?:\s+single)?\s+(?:[^\W\d][\w-]*\s+){{0,2}}?"
               rf"{GENERIC}\b|\bwhenever\b|\bimmer wenn\b")
# "the best moments", "die besten Momente", "highlights" (with no shot list): a generic highlight prior; also "which
# ones should I keep" / "welche ... behalten" (the keepers of a pile of clips).
HIGHLIGHT = (r"\b(?:best|top|greatest|coolest|nicest|beste[nmrs]?|schönste[nmrs]?|coolste[nmrs]?|tollste[nmrs]?)\s+"
             r"(?:[^\W\d][\w-]*\s+)?(?:moments?|bits|parts|highlights|scenes|clips|shots|momente|szenen|stellen)\b|"
             r"\bhighlights?\b|\bkeepers\b|\bworth\s+keeping\b|\b(?:which|welche)\b[^.?!]{0,60}?\b(?:keep|behalten)\b")
# A count with no clip noun, before a relative clause: "die 5 die ich behalten soll", "the 3 that I should keep".
COUNT_RELATIVE = (r"\b(?:the|die|der|das|den|top)\s+(\d+)\s+(?=(?:die|that|which|i|ich|you|du|we|wir|worth|to\s+keep)"
                  r"\b)")


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
    # The brief as a plan. shots: wanted visual items, short image-text phrases (relevance is the best item, and
    # selection covers distinct items first). avoid: visual things to leave out. topics: spoken topics, matched
    # against the transcript. need_live_audio: drop clips whose sound is muted. soundbite_len: a spoken clip asked
    # for next to the visual ones ("the best 5-8 s soundbite where the host sums up").
    shots: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    need_live_audio: bool = False
    soundbite_len: tuple[float, float] | None = None
    # exclude: the avoid items the brief forbids outright ("no X", "never X", "absolutely no X", "leave out X"): a hard
    # gate on top of avoid's soft penalty (every exclude item is also in avoid). no_speech: "no talking / no host
    # voice over it": speech-heavy clips are dropped. per_item: an exhaustive shot list asks for the best clip per
    # named item ("every float: A, B, C"); otherwise ("every shot where ...") for every distinct match. highlight:
    # "the best moments" with no shot list: a generic highlight prior (energy, people and faces, variety).
    exclude: list[str] = field(default_factory=list)
    no_speech: bool = False
    per_item: bool = True
    highlight: bool = False
    # Visual cues answered by the index v3 signals (shot scale, faces, approach, OCR), brief-wide: framing "close"
    # (close-ups, faces, filling the frame, right at the camera, centred, vertical/9:16, hook, POV) or "wide"
    # (establishing, wide views, the place as the main character); want_faces; hook: the first clip must grab in its
    # first second (the opener slot becomes a hook pool); legible_text: names/signs/text must be readable; centered:
    # the subject in the middle (9:16 or square crop).
    framing: str = "none"
    want_faces: bool = False
    hook: bool = False
    legible_text: bool = False
    centered: bool = False
    parser: str = "rules"  # "rules" or "llm:<model>"
    parser_note: str | None = None  # "cached", or why the LLM was not used

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


DASHES = re.compile(r"[‒–—−]")  # en/em dashes and minus sign -> '-' (also used by the CLI's --budget)


def _normalise(q: str) -> str:
    q = DASHES.sub("-", q.lower())  # en/em dashes, minus
    q = re.sub(r"(?<![\d:])9\s*:\s*16(?![\d:])", "vertical", q)  # an aspect ratio, not 9 min 16 s
    # "not too long" / "nich zu lang" asks for short clips; it must not read as "long" (or as something to avoid).
    q = re.sub(r"\b(?:not|nicht|nich|net)\s+(?:too|so|zu|allzu)\s+(?:long|lang|lange)\b", "short", q)
    q = re.sub(r"\b(?:not|nicht|nich|net)\s+(?:too|so|zu|allzu)\s+(?:short|kurz|kurze)\b", " ", q)
    q = re.sub(r"\b(\d+):([0-5]\d)\b(?:\s*(?:min|mins|minutes?|minuten)\b)?",
               lambda m: f"{int(m.group(1)) * 60 + int(m.group(2))} s", q)  # 1:30 -> 90 s
    q = re.sub(r"\b(?:a|an|one|eine|einer)\s+(?=(?:min|minute|minuten)\b)", "1 ", q)
    words = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    def number(m: re.Match) -> str:  # "for each one give the clip": a pronoun, not a count
        pronoun = m.group(1) == "one" and re.search(r"\b(?:each|every|any|no|which|this|that|the)\s+$", q[:m.start()])
        return m.group(0) if pronoun else str(NUMBER_WORDS[m.group(1)])

    q = re.sub(rf"\b({words})\b(?=[\s-]+(?:{UNIT}|{WORDS}{{0,3}}?{CLIP_NOUNS}\b))", number, q)
    # "six to 12 seconds" (the second number word was converted above): the first one of the range too
    return re.sub(rf"\b({words})\b(?=\s*(?:-|to|bis|or|oder)\s*\d)", lambda m: str(NUMBER_WORDS[m.group(1)]), q)


# A duration that describes what to leave out or delete, not a length: "2 minutes of nothing", "2 minuten nix".
NOTHING_AFTER = (r"\s*(?:(?:of|lang|long)\s+)?(?:nix|nichts|nothing|gar\s+nichts|garnichts|where\s+nothing|"
                 r"wo\s+(?:nix|nichts)|in\s+denen\s+nichts|of\s+(?:dead\s+air|empty))\b")


def _durations(rest: str, q: str) -> tuple[str, dict, dict]:
    """Find every duration, decide whether it is the total runtime or a per-clip length, and remove it."""
    budget: dict = {}
    clip: dict = {}
    while (m := re.search(VALUE, rest)):
        if re.match(NOTHING_AFTER, rest[m.end():]):  # "2 minutes of nothing", "einfach 2 minuten nix": content
            rest = _cut(rest, m.start(), m.end())
            continue
        a = _seconds(m.group("a"), m.group("ua") or m.group("u"))
        b = _seconds(m.group("b"), m.group("u")) if m.group("b") else None
        pre, post = rest[:m.start()], rest[m.end():]
        mod = re.search(r"(?<![\w-])" + MODIFIERS, pre)
        kind = "total" if mod and mod.group("total") else (next(k for k in ("le", "ge", "about") if mod.group(k))
                                                            if mod else None)
        start = mod.start() if mod else m.start()
        pre_bare = pre[:start]
        end = m.end()
        # "each", "per clip", "per select", "per cutaway", "pro Szene": a per-clip length
        each = re.match(r"\s*(?:each|apiece|per\s+[^\W\d][\w-]*|jeweils|pro\s+[^\W\d][\w-]*)\b", post)
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
        else:  # a clip noun elsewhere ("long holds, ... about two minutes") does not own a duration it is not next to
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


# ---- the plan: wanted shots, things to avoid, spoken topics, live sound ---------------------------------------------
MAX_ITEMS = 12  # per list
MAX_ITEM_WORDS = 16
SOUNDBITE_LEN = (4.0, 12.0)  # a soundbite asked for without a length
# A brief that lists the shots it wants is about content first: relevance outweighs mood. With the energetic
# weight 1.2 a swinging phone (high motion) beat the actual police band (relevance +3 vs +1.5) on real footage.
SHOT_LIST_RELEVANCE = 2.0
HIGHLIGHT_ENERGY = 0.6  # "the best moments" without a shot list or mood: energy weight (energetic mood is 1.2)
HIGHLIGHT_MAX_LEN = 10.0  # ... and clips at most this long when the brief gives no length or budget
LIVE_AUDIO = (r"(?:\bwith\s+)?\b(?:live|natural|ambient|original|real|nat)[\s-]+(?:natural\s+)?(?:sound|audio)\b"
              r"(?:\s*\(\s*not\s+muted\s*\))?|\(\s*not\s+muted\s*\)|\bnot\s+muted\b|\bunmuted\b|\bsound\s+on\b|"
              r"\bnat[\s-]?sound\b|\bmit\s+originalton\b")
SOUNDBITE = rf"(?:{VALUE})?[\s-]*(?:long\s+)?\b(?:sound[\s-]?bites?|soundbites?)\b"
# "the strongest 20-30 s stretch", "the best shot": one clip
SINGLE = (r"\b(?:the|one|a)\s+(?:single\s+)?(?:best|strongest|top|key|most\s+[\w-]+)\s+(?:[^\W\d][\w-]*\s+){0,2}?"
          r"(?:stretch|clip|shot|moment|sequence|scene|take)\b")
AVOID_STRONG = (r"\b(?:leave out|leaving out|avoid(?:ing)?|exclude|excluding|skip(?:ping)?|cut out|"
                r"(?:don't|do not|never) (?:use|include|show))\b")
AVOID_WEAK = (r"\b(?:no|never|without|nothing|ohne|kein|keine|keinen|nichts|niemals|"
              r"nicht(?!\s+(?:zu|so|mehr|länger|kürzer)\b))\b")
# Hardness of an avoid clause: "no / never / absolutely no / leave out / exclude X" forbids X (a hard gate); "avoid",
# "preferably no", "try not to", "if possible" only prefer to (the soft penalty). AVOID_* above decide the clause's
# extent, these its hardness.
SOFT_AVOID = (r"\b(?:avoid(?:ing)?|preferably|ideally|if possible|where possible|try (?:to|not)|rather not|less|fewer|"
              r"too many|too much|möglichst|lieber|eher|wenn möglich|weniger)\b")
HARD_CUE = (r"\b(?:no|not|never|none|nothing|without|exclude|excluding|leave out|leaving out|cut out|skip|don't|"
            r"do not|kein|keine|keinen|nicht|nichts|ohne|niemals|nie|weglassen|weg lassen)\b")
# An avoid item about speech ("the host's talking", "no voice-over") is not a picture: it drops speech-heavy clips.
SPEECH_ITEM = (r"\b(?:talk|talks|talking|speak|speaks|speaking|speech|voice|voices|voice-?over|narration|commentary|"
               r"chatter|dialogue|sprache|reden|redet|stimme|stimmen|gerede|kommentar)\b")
# A montage whose sound is laid on later: "the host's talking (it goes under music)", "music bed to be added
# later", "to sit under a voice-over", "I will lay music over ...". Clips with speech in them would fight that
# track, so they are gated like "no host talking over it" (ParsedQuery.no_speech), unless the brief wants speech.
SPEECH_FREE = (r"\b(?:goes|go|going|sits?|sitting|runs?|plays?|lay|laid|put|placed)\s+under\s+(?:the\s+|a\s+|some\s+)?"
               r"(?:music|song|track|voice-?over|vo|narration)\b|\bmusic\s+bed\b|"
               r"\b(?:lay|laying|put|putting|add|adding)\s+(?:the\s+|some\s+|my\s+)?music\s+(?:over|on|under)\b|"
               r"\b(?:music|voice-?over|vo)\s+(?:will\s+be|is\s+going\s+to\s+be|to\s+be|gets|goes)\s+"
               r"(?:added|laid|recorded|put|on)\b|"
               r"\bunder\s+(?:a\s+|the\s+)?(?:voice-?over|vo|music)\b|"
               r"\bno\s+(?:host\s+|one\s+)?(?:talking|speaking|speech|voices?)\s+over\b|"
               r"\bmusik\s+(?:drunter|darunter|drüber|darüber)\b|\bunter\s+(?:die\s+|der\s+)?musik\b")
# Spoken topics: "talks (directly to his viewers) about ...", "explains ...", "sums up ...", "says that ...".
TOPIC = (r"\b(?:(?:talks?|talking|speaks?|speaking|chats?|chatting)\s+(?:directly\s+|briefly\s+)?"
         r"(?:(?:to|with)\s+(?:the\s+|his\s+|her\s+|their\s+)?[\w-]+\s+)?(?:about|on the subject of)|"
         r"(?:says?|saying|tells?|telling)\s+(?:[\w-]+\s+)?(?:that|about|why|how|what|where)|"
         r"explains?|explaining|mentions?|mentioning|describes?|describing|discuss(?:es)?|discussing|"
         r"sums?\s+up|summing\s+up|summari[sz](?:es|ing|e)|introduces?|introducing|announces?|announcing|"
         r"spricht\s+über|redet\s+über|erzählt|erklärt)\b")
# Instructions about the output, not content: a sentence that starts like this is dropped whole; a clause like
# "for a news package" or "so they can be used as ..." is dropped on its own.
META_SENTENCE = (r"^(?:also\s+)?(?:give|return|provide|include|output|mark|export|note|add)\b.*"
                 r"\b(?:clips?|times?|time\s*codes?|timecodes?|timestamps?|ranges?)\b")
META_CLAUSE = (r"^(?:so\s+(?:that\s+)?(?:they|it|we|i|you)\s+can\b|"
               r"to\s+(?:sit|go|play|run|lay)\s+(?:under|over|with)\b|for\s+(?:a|an|the|our|my)\s+[\w\s-]*?\b(?:package|voice-?over|segment|story|piece|post|deck|"
               r"channel|show|report)\b|for\s+example\b|e\.?g\.?$)")
LEAD = (r"^(?:(?:also|please|then|plus|and|or)\s+)*(?:(?:cut|make|create|build(?:\s+up)?(?:\s+through)?|show|include|"
        r"feature|cover|pick|find|get|give(?:\s+me)?|list|use|add|want|need|i\s+want|i\s+need|we\s+need)\b\s*)?")
SHOT_LEAD = (r"^(?:(?:shots?|clips?|scenes?|footage|frames?|moments?|anything|parts?)\s+"
             r"(?:where|when|with|of|that|in which|showing)?\s*)")
SUPERLATIVE = ["the most", "most", "best", "strongest", "nicest", "greatest", "recommended", "single"]
# An item made only of these words is camera style or a generic clip noun ("steady or locked-off framing", "clips").
STYLE = {"steady", "locked-off", "locked", "off", "static", "tripod", "handheld", "framing", "stable", "smooth", "or",
         "and", "camera", "work", "shots", "shot", "clips", "clip", "scenes", "scene", "moments", "moment",
         "stretch", "stretches", "parts", "highlights"}
GENERIC_HEAD = {"stretch", "example", "examples", "thing", "things", "item", "items", "b-roll", "broll", "clip", "shot",
                "scene", "moment", "part", "bit"}
ITEM_EDGE = EDGE_STOP | {"where", "when", "that", "which", "who", "also", "then", "each", "every", "all", "one", "per"}
FRAMING_HEADS = {"close-up", "close-ups", "closeup", "closeups", "wide", "wides", "establishing", "wide-angle"}
FRAMING_WORDS = FRAMING_HEADS | {"close", "ups", "tight", "view", "views", "angle", "lots", "of", "more"}

# ---- visual cues: framing, faces, hook, readable text, centring -------------------------------------------------
# Read on the brief minus its opener/closer descriptions and avoid clauses: "no bored faces" asks for no faces, and
# "open on an establishing wide" describes the opener (its relevance picks the wide), not the whole edit. Framing is
# the majority of close vs wide cues ("wide view ... rather than close-ups": wide; a tie: none).
HOOK_CUE = (r"\bhooks?\b|\bhooky\b|\bscroll[\s-]?stopp\w*|\bthumb[\s-]?stopp\w*|\bfirst-(?:shots?|frames?)\b|"
            r"\b(?:in|within)\s+(?:the\s+)?(?:very\s+)?first\s+(?:second|1\s*s\b|(?:two|2|few)\s+seconds)|"
            r"\bgrabs?\s+(?:the\s+)?(?:attention|viewers?|people)\b|\battention[\s-]grabb\w*")
CENTRED_CUE = (r"\bcent(?:er|r)ed\b|\bcent(?:er|re)\s+of\s+(?:the\s+)?frame\b|\bin\s+the\s+(?:middle|cent(?:er|re))\b|"
               r"\bvertical\b|\bcroppable\b|\bcrop(?:s|ped|ping)?\s+(?:it\s+|them\s+)?(?:to|for)\s+(?:a\s+)?"
               r"(?:square|vertical|portrait)\b|\bmittig\b|\bhochkant\b")
FACES_CUE = r"\bfaces?\b|\bgesicht(?:er)?\b"
CLOSE_CUE = (r"\bclose[\s-]?ups?\b|\bcloseups?\b|\bextreme\s+close\b|\btight(?:ly)?\s+(?:shots?|framed|framing)\b|"
             r"\bfill(?:s|ed|ing)?\s+(?:up\s+)?(?:the\s+)?(?:whole\s+|entire\s+|full\s+)?(?:frame|screen)\b|"
             r"\b(?:frame|screen)[\s-]filling\b|"
             r"\b(?:right|straight|directly)\s+(?:at|into|toward|towards|in\s+front\s+of)\s+"
             r"(?:the\s+)?(?:camera|lens)\b|"
             r"\b(?:com|runn|walk|charg|mov|head|march|ris|fly)\w*\s+(?:right\s+|straight\s+)?(?:at|toward|towards|"
             r"into)\s+(?:the\s+)?(?:camera|lens)\b|\bpunchy\b|\bpov\b|\bnahaufnahmen?\b")
WIDE_CUE = (r"\bestablishing\b|\bwides?\b|\blong\s+(?:views?|shots?)\b|\bsense\s+of\s+place\b|\bas\s+a\s+place\b|"
            r"\b(?:street|place|city|avenue|town|location|setting|landscape|venue)\b(?:\s+[\w'-]+){0,3}?\s+(?:is|as)\s+"
            r"(?:the\s+)?main\s+character\b|\btotale\b|\bweitwinkel\w*")
# "rather than close-ups", "not too wide": the cue word names what the brief does not want.
CUE_NEGATION = (r"\b(?:rather\s+than|instead\s+of|not|fewer|less|no\s+more)\s+(?:(?:any|the|a|too|so|many)\s+)?"
                r"(?:[\w-]+\s+){0,2}?(?:close[\s-]?ups?|closeups?|wides?|faces?|tight\s+shots?)\b|"
                # "one establishing wide that shows the avenue": a single shot of the list, not the edit's framing
                r"\b(?:one|a|an|single)\s+(?:[\w-]+\s+){0,2}?(?:establishing(?:\s+(?:wide|shot))?|wides?|"
                r"close[\s-]?ups?)\b")
LEGIBLE_CUE = (r"\blegib\w*|\breadab\w*|\bread(?:s|ing)?\s+(?:the|its|their|clearly)\b|\bon-screen\s+text\b|"
               r"\bexact\s+(?:text|wording)\b|\bspellings?\b|\bfact-check\w*|\blesbar\w*|\bschrift\w*")
# Words a clause can hold besides its cues and still be only an instruction ("keep everything centered cause im
# cropping to vertical", "first clip has to be the hook"): such a clause gives no shot item.
CUE_FILLER = {"keep", "keeping", "make", "sure", "everything", "anything", "cause", "because", "since", "im", "i'm",
              "it", "its", "them", "they", "has", "have", "had", "be", "is", "are", "must", "should", "need", "needs",
              "got", "gotta", "first", "clip", "clips", "shot", "shots", "frame", "frames", "second", "seconds",
              "work", "works", "slow", "build", "edit", "video", "always", "really", "very", "just", "so", "up",
              "subject", "subjects", "square", "crop", "cropping", "cropped"}
# What an LLM answer may add on top of the rules, per field: its value counts only when the (scoped) brief contains
# one of these stems.
CUE_GROUND = {"close": r"close|fill|face|camera|lens|tight|cent(?:er|re)|hook|vertical|pov|punch|nahauf|gesicht",
              "wide": r"wide|establish|place|long view|landscape|panoram|scenery|totale|weitwinkel",
              "want_faces": r"\bfaces?\b|gesicht", "hook": r"hook|first|scroll|grab|attention|thumb",
              "legible_text": r"legib|readab|\bread|text|spell|fact|lesbar|schrift", "centered":
              r"cent(?:er|re)|middle|vertical|crop|square|mittig|hochkant"}


# A list of hooks as the deliverable ("rank me 3 hooks", "the 3 best possible first-shots"): every clip is ranked like
# the hook slot (vhe.rank.scorer.HOOK_WEIGHTS, steadiness kept at the floor every brief has): motion and the close /
# approach bonus over the brief's content words, which in such a brief name no shots.
HOOK_LIST_WEIGHTS = {"relevance": 0.5, "energy": 0.8, "quality": 0.3}


def framing_of(text: str | None) -> str:
    """'close', 'wide' or 'none' for a phrase (a brief's scoped text, or an opener/closer description): the majority
    of its close and wide cues, negated ones ("rather than close-ups") left out."""
    if not text:
        return "none"
    t = re.sub(CUE_NEGATION, " ", _normalise(text))
    close = sum(len(re.findall(pat, t)) for pat in (CLOSE_CUE, CENTRED_CUE, HOOK_CUE, FACES_CUE))
    wide = len(re.findall(WIDE_CUE, t))
    return "close" if close > wide else "wide" if wide > close else "none"


def _set_cues(p: ParsedQuery, text: str) -> None:
    """Brief-level cue fields from the scoped brief text (opener/closer and avoid clauses already removed)."""
    t = re.sub(CUE_NEGATION, " ", text)
    p.framing = framing_of(text)
    p.want_faces = re.search(FACES_CUE, t) is not None
    p.hook = p.hook or re.search(HOOK_CUE, t) is not None
    p.legible_text = re.search(LEGIBLE_CUE, t) is not None
    p.centered = re.search(CENTRED_CUE, t) is not None


def _cue_only(clause: str) -> bool:
    """A clause that is a framing/hook/centring instruction and names nothing to show (faces are content, kept)."""
    t = clause.lower()
    hit = False
    for pat in (HOOK_CUE, CENTRED_CUE, CLOSE_CUE, LEGIBLE_CUE):
        t, n = re.subn(pat, " ", t)
        hit = hit or n > 0
    if not hit:
        return False
    words = [w for w in re.findall(r"[^\W\d_][\w'-]*", t) if len(w) > 2]
    return all(w in CUE_FILLER or w in STYLE or w in ITEM_EDGE for w in words)


def _depth_map(text: str) -> list[bool]:
    """True for characters outside quotes and parentheses (where punctuation separates clauses)."""
    out, depth, quote = [], 0, False
    for ch in text:
        if ch == "\"":
            quote = not quote
        elif ch in "“„”":
            quote = ch != "”"
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        out.append(depth == 0 and not quote and ch not in "\"“”„()")
    return out


def _top_colon(text: str) -> int | None:
    return next((i for i, (ch, top) in enumerate(zip(text, _depth_map(text))) if ch == ":" and top), None)


def _split_top(text: str, seps: str) -> list[str]:
    top, parts, a = _depth_map(text), [], 0
    for i, ch in enumerate(text):
        if ch in seps and top[i] and not (ch == "." and i + 1 < len(text) and not text[i + 1].isspace()):
            parts.append(text[a:i])
            a = i + 1
    return [x for x in (*parts, text[a:]) if x.strip()]


def _words(text: str) -> str:
    """Quote and parenthesis characters and loose punctuation removed, the words kept."""
    text = re.sub(r"[\"“”„()\[\]:;!?]", " ", text)
    text = re.sub(r"(?<![\w])\.+|\.+(?![\w])|,", " ", text)
    return " ".join(text.split())


def _trim(words: list[str], stop: set) -> list[str]:
    while words and words[0] in stop:
        words.pop(0)
    while words and words[-1] in stop:
        words.pop()
    return words


def _clean_item(text: str) -> str:
    """A shot/avoid phrase: lead verbs ("build through", "show"), mood, format and superlative words removed."""
    s = _words(text)
    for _ in range(3):
        s = re.sub(LEAD, "", s).strip()
        s = re.sub(SHOT_LEAD, "", s).strip()
        s = " ".join(_trim(s.split(), ITEM_EDGE))
    for w in CALM + ENERGY + FORMAT + SUPERLATIVE:
        s = _strip(s, w)
    words = _trim(s.split(), ITEM_EDGE)
    if all(w in STYLE or w in ITEM_EDGE for w in words):  # "steady or locked-off framing" is style, not a shot
        return ""
    if any(w in FRAMING_HEADS for w in words) and all(w in STYLE | FRAMING_WORDS | ITEM_EDGE for w in words):
        return ""  # "close-ups", "wide establishing views": a framing cue (ParsedQuery.framing), not a shot
    return " ".join(words[:MAX_ITEM_WORDS])


def _list_items(text: str) -> list[str]:
    """'a, b and c' -> [a, b, c]; 'and' splits only the last element of a comma list ('bride and groom' stays)."""
    parts = _split_top(text, ",")
    if len(parts) > 1:
        last = re.split(r"\s+(?:and|or|und|oder)\s+", parts[-1].strip())
        parts = parts[:-1] + last
    return [x.strip() for x in parts if x.strip()]


def _avoid_items(clause: str) -> list[str]:
    """'shots where the camera points at the ground or sky, or where something blocks the lens' ->
    ['camera points at the ground', 'camera points at the sky', 'something blocks the lens']."""
    out: list[list[str]] = []
    # Commas split before _words (which drops them); a parenthesis explains, it names nothing to leave out.
    frags = [f for part in _split_top(re.sub(r"\([^)]*\)", " ", clause), ",")
             for f in re.split(r"\s+(?:or|oder|and|und)\s+", _words(part))]
    for frag in frags:
        frag = re.sub(r"^(?:or|and|oder|und)\s+", "", frag.strip())
        frag = re.sub(SHOT_LEAD, "", frag)
        frag = re.sub(r"^(?:where|when|that|which|if)\s+", "", frag)
        frag = re.sub(r"^(?:any|every|all)\s+(?=\w)", "", frag)
        frag = re.sub(SHOT_LEAD, "", frag)
        frag = re.sub(r"^(?:nothing|anything)\b", "something", frag)
        words = _trim(frag.split(), ITEM_EDGE - {"something"})
        if not words:
            continue
        if out and len(words) <= 2 and len(out[-1]) >= 3:  # "at the ground or sky": "sky" takes "ground"'s place
            words = out[-1][:-len(words)] + words
        out.append(words)
    return [" ".join(w[:MAX_ITEM_WORDS]) for w in out]


def _topic_items(verb: str, obj: str) -> list[str]:
    """Spoken topics after the verb: a list ('the stream: where he is (...), why he mutes (...), and what ...') gives
    one topic per element; a single object keeps the verb ('sums up the parade "short parade ..."')."""
    colon = _top_colon(obj)
    tail = obj[colon + 1:] if colon is not None else ""
    parts = _list_items(tail) if tail.strip() else _list_items(obj if colon is None else obj[:colon])
    items = [" ".join(re.sub(r"^(?:and|or)\s+", "", _words(x)).split()) for x in parts]
    items = [x for x in items if x]
    if len(items) == 1:
        items = [" ".join(_words(f"{verb} {obj}").split())]
    return items[:MAX_ITEMS]


def _plan(rest: str, q: str, p: ParsedQuery) -> str:
    """Fill p.shots/avoid/topics from the remaining brief (punctuation intact); return the rest without the avoid,
    topic and output-instruction parts. Sentences with a topic give no shots (their subject is the speaker)."""
    m = re.search(rf"(?<![\w-]){BUDGET_NOUNS}\s+(?:of|from|about|for)\s+(?P<x>[^.,;:!?]+)", q)
    subject = _clean_item(m.group("x")) if m else ""
    shots: list[str] = []
    kept: list[str] = []
    cue_parts: list[str] = []
    p._cue_text = ""  # the brief without opener/closer and avoid clauses: what the visual cues are read on
    for sentence in _split_top(re.sub(r"\.{2,}|…", " ", rest), ".!?;"):
        s = sentence.strip()
        if re.search(META_SENTENCE, s):
            continue
        soft = re.search(SOFT_AVOID, s) is not None
        m = re.search(AVOID_STRONG, s)
        if m:
            items = _avoid_items(s[m.end():])
            p.avoid += items
            p.exclude += [] if soft else items
            s = s[:m.start()]
        while (m := re.search(AVOID_WEAK, s)):
            # the clause runs to the next comma or the next "(and) (absolutely) no ..."
            end = next((m.end() + k for k, ch in enumerate(s[m.end():]) if ch == ","), len(s))
            nxt = re.search(r"\s+(?:(?:and|or|und|oder)\s+)?(?:(?:absolutely|definitely|certainly|really|garantiert|"
                            r"absolut|bitte)\s+)?" + AVOID_WEAK, s[m.end():end])
            end = m.end() + nxt.start() if nxt else end
            lead = "something " if m.group(0) == "nothing" else ""
            items = _avoid_items(lead + s[m.end():end])
            p.avoid += items
            p.exclude += [] if soft else items
            s = s[:m.start()] + s[end:]
        cue_parts.append(s)
        m = re.search(TOPIC, s)
        if m:
            p.topics += _topic_items(m.group(0), s[m.end():])
            kept.append(re.sub(r"\b(?:where|when|while|in which)\s*$", "", s[:m.start()].strip()))
            continue
        clauses = []
        for c in _split_top(s, ","):
            colon = _top_colon(c)
            if re.search(META_CLAUSE, c[:colon].strip() if colon is not None else c.strip()):
                c = c[colon + 1:] if colon is not None else ""  # "for example: a, b" keeps the list
            clauses.append("" if _cue_only(c) else c)  # "keep everything centered": a cue, not a shot
        s = ",".join(c for c in clauses if c.strip())
        colon = _top_colon(s)
        body = s if colon is None or not s[colon + 1:].strip() else s[colon + 1:]
        shots += [x for x in map(_clean_item, _list_items(body)) if x and x != subject]
        kept.append(s)
    seen = set()
    p.shots = [x for x in shots if not (x in seen or seen.add(x))] if len(set(shots)) >= 2 else []
    if p.shots:
        p.text = subject
    p._cue_text = ". ".join(cue_parts)
    return ". ".join(kept)


def _categorise(items: list[str], raw: str) -> list[str]:
    """Named items get the category the other items share: if two or more items end in 'float(s)', a bare proper
    name from the brief ('Pasargad', 'Gate of the Nations') becomes 'Pasargad float' for the image-text model."""
    heads = [re.sub(r"s$", "", x.split()[-1]) for x in items if len(x.split()) >= 2]
    common = [h for h in set(heads) if heads.count(h) >= 2 and h not in GENERIC_HEAD and len(h) > 2]
    if len(common) != 1:
        return items
    head = common[0]
    out = []
    for x in items:
        m = re.search(r"(?<![\w-])" + r"\W+".join(map(re.escape, x.split())) + r"(?![\w-])", raw, re.IGNORECASE)
        proper = m is not None and all(w[0].isupper() for w in re.findall(r"[^\W\d_][\w'-]*", m.group(0))
                                       if w.lower() not in EDGE_STOP)
        out.append(f"{x} {head}" if proper and not re.search(rf"\b{re.escape(head)}", x, re.IGNORECASE) else x)
    return out


# Nothing to see: "has to work with no context", "no slow build" (a pacing rule, not a picture).
ABSTRACT = {"something", "anything", "any", "context", "build", "buildup", "build-up", "intro", "setup", "explanation"}


def _nameable(item: str) -> bool:
    """An avoid item names something visible: a content word besides clip nouns, camera style and connectives ("no
    clip under 3 s" leaves just "clip")."""
    return any(len(w) > 2 and w not in STYLE | GENERIC_HEAD | ITEM_EDGE | ABSTRACT
               for w in re.findall(r"[^\W\d_][\w'-]*", item.lower()))


def _dedupe(items: list[str], also: tuple = ()) -> list[str]:
    """Drop empties, repeats and items that restate one of `also` (opener/closer) or an earlier item."""
    def toks(s):
        return {w[:5] for w in re.findall(r"\w+", s.lower()) if w not in EDGE_STOP}

    out, refs = [], [toks(a) for a in also if a]
    for x in (" ".join(str(i).split()) for i in items):
        t = toks(x)
        if not t or any(len(t & r) / len(t | r) >= 0.6 for r in refs):
            continue
        out.append(x)
        refs.append(t)
    return out[:MAX_ITEMS]


def parse(query: str, count: int | None = None, min_len: float | None = None, max_len: float | None = None,
          budget_s: float | tuple | None = None) -> ParsedQuery:
    """Rules only (deterministic, no network)."""
    p, explicit_len, hint = _parse_rules(query)
    return _finalise(p, explicit_len, hint, count, min_len, max_len, budget_s)


def _parse_rules(query: str) -> tuple[ParsedQuery, bool, str | None]:
    raw = query.strip()
    q = _normalise(raw)
    p = ParsedQuery(raw=raw, text="")
    rest = q

    for attr, pattern in (("opener", OPENER), ("closer", CLOSER)):
        m = re.search(pattern, rest)
        if m:
            setattr(p, attr, m.group("x").strip())
            rest = _cut(rest, m.start(), m.end())

    m = re.search(LIVE_AUDIO, rest)
    while m:
        p.need_live_audio = True
        rest = _cut(rest, m.start(), m.end())
        m = re.search(LIVE_AUDIO, rest)
    m = re.search(SOUNDBITE, rest)
    if m:  # "the best 5-8 s soundbite": its own length, not the visual clips'
        if m.group("a"):
            a = _seconds(m.group("a"), m.group("ua") or m.group("u"))
            b = _seconds(m.group("b"), m.group("u")) if m.group("b") else None
            p.soundbite_len = (a, b) if b is not None else (max(1.0, 0.7 * a), 1.3 * a)
        else:
            p.soundbite_len = SOUNDBITE_LEN
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
        p.per_item = re.search(ALL_MATCHES, q) is None
        rest = _cut(rest, m.start(), m.end())
    p.highlight = re.search(HIGHLIGHT, q) is not None

    # "3 clips", "3 short high-energy scenes", "3 ruhige Szenen": up to three words between number and noun.
    # "no two clips of the same subject back to back" is a rule, not a count.
    m = next((x for x in re.finditer(rf"\b(\d+)\s+{WORDS}{{0,3}}?{CLIP_NOUNS}\b", rest)
              if not re.search(r"\b(?:no|not|never|kein|keine|nicht|at least|mindestens)\s+$", rest[:x.start()])),
             None)
    if m:
        p.count, p.count_explicit = max(1, min(MAX_COUNT, int(m.group(1)))), True
        p._hook_list = re.search(r"(?:hooks?|first-shots?)$", m.group(0)) is not None  # "the 3 best first-shots"
        rest = _cut(rest, m.start(1), m.end(1))
    elif not p.exhaustive and (m := re.search(SINGLE, rest)):
        p.count, p.count_explicit = 1, True
    elif not p.exhaustive and (m := re.search(COUNT_RELATIVE, rest)):  # "die 5 die ich behalten soll"
        p.count, p.count_explicit = max(1, min(MAX_COUNT, int(m.group(1)))), True
        rest = _cut(rest, m.start(1), m.end(1))

    m = re.search(r"\b(?:without|no|ohne|kein|keine)\s+(?:[^\W\d][\w-]*\s+){0,2}?"
                  r"(?:talking|speaking|speech|voices?|dialogue|narration|sprache|reden|gespräch|stimmen?)\b", rest)
    if m:
        p.weights["speech"] = -1.0
        p.no_speech = True
        rest = _cut(rest, m.start(), m.end())
    if p.weights["speech"] == 0.0 and any(_contains(rest, w) for w in SPEECH_ON):
        p.weights["speech"] = 1.0
    rest = _plan(rest, q, p)
    if re.search(SPEECH_FREE, q) and not p.topics and p.weights["speech"] <= 0:
        p.no_speech, p.weights["speech"] = True, -1.0
    _set_cues(p, p._cue_text)
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
    if not p.shots:  # with a shot list the text is the subject ("recap of the parade"), set by _plan
        p.text = " ".join([*keep, _clean_text(rest)]).strip()
    return p, explicit_len, hint




def _finalise(p: ParsedQuery, explicit_len: bool, hint: str | None, count: int | None, min_len: float | None,
              max_len: float | None, budget_s: float | tuple | None) -> ParsedQuery:
    """Caller overrides, then everything derived from the budget, then clamps, then the plan lists. Shared by rules
    and LLM."""
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

    if getattr(p, "_hook_list", False):  # "rank me 3 hooks": every clip is a hook candidate, not just the first
        p.hook = True
        p.framing = "close" if p.framing == "none" else p.framing
        p.weights.update(HOOK_LIST_WEIGHTS)
    if p.opener and re.search(HOOK_CUE, p.opener.lower()):  # "open with a hook": the hook slot, with what it shows
        p.hook = True
        if _cue_only(p.opener):
            p.opener = None
    if p.framing not in ("none", "close", "wide"):
        p.framing = "none"
    p.shots = _categorise(_dedupe(p.shots, (p.opener, p.closer)), p.raw)
    # Speech is not a picture: "no host talking" drops speech-heavy clips instead of matching an image to it.
    spoken = [x for x in p.avoid + p.exclude if re.search(SPEECH_ITEM, x.lower())]
    if spoken:
        p.weights["speech"] = min(p.weights["speech"], -1.0)
        p.no_speech = p.no_speech or any(x in p.exclude for x in spoken)
    if p.no_speech:  # gated (engine) and preferred
        p.weights["speech"] = min(p.weights["speech"], -1.0)
    # Forbidden items first, so a soft restatement of one ("camera pointing at the ground" next to "camera points at
    # the ground") cannot make it soft; every exclude item is also an avoid item (the soft penalty applies too).
    p.exclude = _dedupe([x for x in p.exclude if x not in spoken and _nameable(x)])
    p.avoid = _dedupe(p.exclude + [x for x in p.avoid if x not in spoken and _nameable(x)])
    p.exclude = [x for x in p.avoid if x in set(p.exclude)]
    p.topics = _dedupe(p.topics)
    if p.highlight and not p.shots and not p.topics:
        if p.mood is None:  # "the best moments": lively over static, unless the brief sets a mood
            p.weights["energy"] = max(p.weights["energy"], HIGHLIGHT_ENERGY)
        if not explicit_len and not p.budget:  # many short moments, never one long static take
            p.min_len, p.max_len = min(p.min_len, TYPICAL_LEN[None][0]), min(p.max_len, HIGHLIGHT_MAX_LEN)
    else:
        p.highlight = False
    if p.shots and not p.text:
        p.text = ", ".join(p.shots)
    if len(p.shots) >= 2 and not getattr(p, "_hook_list", False):
        p.weights["relevance"] = max(p.weights["relevance"], SHOT_LIST_RELEVANCE)
    if p.topics and not p.shots:  # "every moment the host talks about ...": speech clips matched to the topics
        p.weights["speech"] = max(p.weights["speech"], 1.0)
    if p.topics and p.shots and p.soundbite_len is None:
        p.soundbite_len = SOUNDBITE_LEN
    if p.soundbite_len is not None:
        lo, hi = sorted(float(x) for x in p.soundbite_len)
        p.soundbite_len = (round(max(1.0, lo), 2), round(max(lo + 1.0, hi), 2))
    if not p.text and not p.shots and not p.highlight:  # with highlight, relevance is the engine's highlight prior
        p.weights["relevance"] = 0.0
    return p


# ---- LLM parsing (local Ollama) -------------------------------------------------------------------------------------
DEFAULT_LLM = "qwen3:4b"
OLLAMA_HOST = "http://127.0.0.1:11434"
LLM_TIMEOUT = 90.0  # a cold load of a 4B model takes seconds; a hung server must not hang the query
# Unload the model as soon as the answer is back: the query then loads SigLIP and CG-DETR on the same (6 GB) GPU, and
# a resident qwen3:4b (~2.8 GB) made 3 of 17 dev briefs fail with CUDA OOM. Measured 2026-10-18 (16 cores, GTX 1660
# Super, two dev briefs, cold): GPU 12-22 s per parse, CPU-only (num_gpu 0) 44-57 s. So the parse runs on the GPU and
# unloads at once (keep_alive 0: Ollama's /api/ps is empty when the response returns); the engine's CPU fallback
# covers any other GPU tenant.
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
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
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
    nums = {float(x) for x in re.findall(NUM, _normalise(raw))}
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
    words = re.findall(r"[^\W\d_][\w'-]*", item.lower())
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
    low = _normalise(base.raw)

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
    _set_mood(p, base.mood or data.get("mood"))
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
    scope = getattr(base, "_cue_text", "") or ""
    p._cue_text, p._hook_list = scope, getattr(base, "_hook_list", False)
    fr = data.get("framing")
    p.framing = base.framing if base.framing != "none" else \
        fr if fr in ("close", "wide") and re.search(CUE_GROUND[fr], scope) else "none"
    for key in ("want_faces", "hook", "legible_text", "centered"):
        ground = re.search(CUE_GROUND[key], low if key == "hook" else scope) is not None
        setattr(p, key, getattr(base, key) or (data.get(key) is True and ground))
    return p, explicit_len


def parse_with_llm(query: str, model: str, host: str = OLLAMA_HOST, **overrides) -> ParsedQuery:
    """Ask a local Ollama model for the plan, grounded by the rules; the rules alone on any failure."""
    base, explicit_len, hint = _parse_rules(query)
    ov = (overrides.get("count"), overrides.get("min_len"), overrides.get("max_len"), overrides.get("budget_s"))
    try:
        data, cached = _ask_llm(base.raw, model, host)
        p, explicit_len = _from_llm(data, base, explicit_len)
    except Exception as exc:  # unreachable server, unknown model, timeout, malformed answer
        base.parser_note = f"{model} unavailable ({type(exc).__name__}): rules used"
        return _finalise(base, explicit_len, hint, *ov)
    p.parser, p.parser_note = f"llm:{model}", "cached" if cached else None
    return _finalise(p, explicit_len, hint, *ov)


def parse_brief(query: str, llm: str | None = "auto", **overrides) -> ParsedQuery:
    """The parser entry point. llm: an Ollama model, 'auto' (default_llm(), the default) or None (rules only)."""
    model = default_llm() if llm == "auto" else llm
    return parse_with_llm(query, model, **overrides) if model else parse(query, **overrides)
