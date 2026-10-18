"""The brief rules (deterministic, no network): an editor's brief to signal weights, length limits, a runtime budget,
the content text used for relevance, and a plan.

"chill scenic view" and "short high-energy scenes" differ mostly in motion, camera steadiness and loudness,
which image-text similarity barely measures. So mood words become weights on those signals instead of being
matched as text. Durations are either a total runtime ("60-90 s energetic recap", "a 1 minute recap") or a
per-clip length ("3-6 s each", "clips of 2-4 seconds"); every parsed phrase is removed from the content text so
only visual words reach the image-text model. English and German; the word lists are in vhe.query.lexicon.

A real brief is several sentences: a list of wanted shots, things to leave out, spoken topics, a sound requirement.
These become the plan fields of the ParsedQuery (shots, avoid, exclude, topics, need_live_audio, soundbite_len).
Visual cues the index v3 signals can answer become fields too: framing (close / wide), want_faces, hook, legible_text
and centered, read on the brief without its opener/closer descriptions and avoid clauses (ParsedQuery.cue_text).
finalise is shared with the LLM path (vhe.query.llm): caller overrides, budget-derived lengths and counts, clamps and
the plan lists."""
from __future__ import annotations

import re

from vhe.query.lexicon import (
    ABSTRACT,
    ALL_MATCHES,
    AVOID_STRONG,
    AVOID_WEAK,
    BUDGET_NOUNS,
    CALM,
    CENTRED_CUE,
    CLIP_NOUNS,
    CLOSE_CUE,
    CLOSER,
    COUNT_RELATIVE,
    CUE_FILLER,
    CUE_NEGATION,
    DASHES,
    EDGE_STOP,
    ENERGY,
    EXHAUSTIVE,
    FACES_CUE,
    FILLER,
    FORMAT,
    FRAMING_HEADS,
    FRAMING_WORDS,
    GENERIC_HEAD,
    HIGHLIGHT,
    HOOK_CUE,
    ITEM_EDGE,
    LEAD,
    LEGIBLE_CUE,
    LIVE_AUDIO,
    LONG,
    META_CLAUSE,
    META_SENTENCE,
    MODIFIERS,
    NOTHING_AFTER,
    NUMBER_WORDS,
    OPENER,
    SHORT,
    SHOT_LEAD,
    SINGLE,
    SOFT_AVOID,
    SOUNDBITE,
    SPEECH_FREE,
    SPEECH_ITEM,
    SPEECH_ON,
    SPEECH_VERBS,
    STYLE,
    SUPERLATIVE,
    TOPIC,
    UNIT,
    VALUE,
    WIDE_CUE,
    WORDS,
)
from vhe.query.model import (
    BUDGET_TOL,
    EXHAUSTIVE_CAP,
    HIGHLIGHT_ENERGY,
    HIGHLIGHT_MAX_LEN,
    HOOK_LIST_WEIGHTS,
    MAX_CLIPS,
    MAX_COUNT,
    MAX_ITEM_WORDS,
    MAX_ITEMS,
    MOODS,
    SHOT_LIST_RELEVANCE,
    SOUNDBITE_LEN,
    TYPICAL_LEN,
    ParsedQuery,
)
from vhe.query.text_match import WORD, word_tokens

BARE_BUDGET_S = 30.0  # a duration with no clip/recap noun anywhere: at least this long reads as total runtime


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


def set_mood(p: ParsedQuery, mood: str | None) -> None:
    if mood in MOODS:
        p.mood = mood
        p.weights.update(MOODS[mood])


def normalise(q: str) -> str:
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


# ---- visual cues ----------------------------------------------------------------------------------------------------
def framing_of(text: str | None) -> str:
    """'close', 'wide' or 'none' for a phrase (a brief's scoped text, or an opener/closer description): the majority
    of its close and wide cues, negated ones ("rather than close-ups") left out."""
    if not text:
        return "none"
    t = re.sub(CUE_NEGATION, " ", normalise(text))
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
    words = [w for w in re.findall(WORD, t) if len(w) > 2]
    return all(w in CUE_FILLER or w in STYLE or w in ITEM_EDGE for w in words)


# ---- the plan: wanted shots, things to avoid, spoken topics -----------------------------------------------------
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
    return next((i for i, (ch, top) in enumerate(zip(text, _depth_map(text), strict=True)) if ch == ":" and top), None)


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
    p.cue_text = ""  # the brief without opener/closer and avoid clauses: what the visual cues are read on
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
    p.cue_text = ". ".join(cue_parts)
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
        proper = m is not None and all(w[0].isupper() for w in re.findall(WORD, m.group(0))
                                       if w.lower() not in EDGE_STOP)
        out.append(f"{x} {head}" if proper and not re.search(rf"\b{re.escape(head)}", x, re.IGNORECASE) else x)
    return out


def _nameable(item: str) -> bool:
    """An avoid item names something visible: a content word besides clip nouns, camera style and connectives ("no
    clip under 3 s" leaves just "clip")."""
    return any(len(w) > 2 and w not in STYLE | GENERIC_HEAD | ITEM_EDGE | ABSTRACT
               for w in word_tokens(item))


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


# ---- parsing -------------------------------------------------------------------------------------------------------
def parse(query: str, count: int | None = None, min_len: float | None = None, max_len: float | None = None,
          budget_s: float | tuple | None = None) -> ParsedQuery:
    """Rules only (deterministic, no network)."""
    p, explicit_len, hint = parse_rules(query)
    return finalise(p, explicit_len, hint, count, min_len, max_len, budget_s)


def parse_rules(query: str) -> tuple[ParsedQuery, bool, str | None]:
    raw = query.strip()
    q = normalise(raw)
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
        p.hook_list = re.search(r"(?:hooks?|first-shots?)$", m.group(0)) is not None  # "the 3 best first-shots"
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
    _set_cues(p, p.cue_text)
    for w in SPEECH_VERBS:
        rest = _strip(rest, w)

    calm = [w for w in CALM if _contains(rest, w)]
    energy = [w for w in ENERGY if _contains(rest, w)]
    set_mood(p, "energetic" if energy and len(energy) >= len(calm) else "calm" if calm else None)
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


def finalise(p: ParsedQuery, explicit_len: bool, hint: str | None, count: int | None, min_len: float | None,
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

    if p.hook_list:  # "rank me 3 hooks": every clip is a hook candidate, not just the first
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
    if len(p.shots) >= 2 and not p.hook_list:
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
