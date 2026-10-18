"""What a parsed brief is (ParsedQuery), and the numbers that shape every brief: clip count and length limits,
the signal-weight presets (default, moods, shot lists, highlights) and the weights the opener/closer and hook slots
are scored with. The rules (vhe.query.rules) and the LLM path (vhe.query.llm) fill a ParsedQuery; everything
downstream reads it."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# ---- limits ---------------------------------------------------------------------------------------------------
DEFAULT_MIN_LEN = 3.0
DEFAULT_MAX_LEN = 30.0
DEFAULT_COUNT = 5
MAX_COUNT = 50       # explicit counts
MAX_CLIPS = 150      # counts derived from a runtime budget
EXHAUSTIVE_CAP = 50  # "every moment ...": all matches, up to this many
BUDGET_TOL = 0.15    # a single budget value means +-15 %


MAX_ITEMS = 12  # per list
MAX_ITEM_WORDS = 16
SOUNDBITE_LEN = (4.0, 12.0)  # a soundbite asked for without a length


# ---- signal weights ---------------------------------------------------------------------------------------------
DEFAULT_WEIGHTS = {"relevance": 1.0, "quality": 0.4, "energy": 0.0, "steady": 0.3, "speech": 0.0}
# Steadiness stays >= 0.3 for every mood: otherwise a pocket shot (swinging camera, loud muffled audio) wins
# energetic queries.
MOODS = {"energetic": {"energy": 1.2, "steady": 0.3, "quality": 0.3},
         "calm": {"energy": -0.9, "steady": 0.8, "quality": 0.6}}
# Typical clip length (s) when a runtime budget is given but no per-clip length.
TYPICAL_LEN = {"energetic": (2.0, 4.0), "calm": (4.0, 8.0), None: (3.0, 6.0), "short": (2.0, 4.0),
               "long": (8.0, 15.0), "moment": (8.0, 25.0)}
# "moment": a reaction brief without a per-clip length, and any highlight brief whose runtime budget is at least
# HIGHLIGHT_LONG_BUDGET_S: a moment of a person reacting or talking (the setup, the reaction, the line after it)
# takes several sentences; a 10-minute highlights video is a few dozen such moments, not a hundred 4 s cuts.
HIGHLIGHT_LONG_BUDGET_S = 180.0


# A brief that lists the shots it wants is about content first: relevance outweighs mood. With the energetic
# weight 1.2 a swinging phone (high motion) beat the actual police band (relevance +3 vs +1.5) on real footage.
SHOT_LIST_RELEVANCE = 2.0
HIGHLIGHT_ENERGY = 0.6  # "the best moments" without a shot list or mood: energy weight (energetic mood is 1.2)
HIGHLIGHT_MAX_LEN = 10.0  # ... and clips at most this long when the brief gives no length or budget


# ---- slot weights: the opener/closer and hook pools, and a list of hooks ------------------------------------------
# Opener/closer: the shot the description asks for, held steady.
EDGE_WEIGHTS = {"relevance": 1.0, "steady": 0.8, "quality": 0.6}
# The hook slot (vhe.rank.pool.edge_clip): the opener is drawn from a pool scored with these weights and framing
# "close". No steadiness weight: a hook moves (the flag swept at the lens reads steady -3 from the motion it causes);
# the usable gate still drops covered, blurred and very shaky spans, and unclean edges cost the scorer's EDGE_WEIGHT.
HOOK_WEIGHTS = {"relevance": 0.5, "energy": 0.8, "steady": 0.0, "quality": 0.3}
# A list of hooks as the deliverable ("rank me 3 hooks", "the 3 best possible first-shots"): every clip is ranked like
# the hook slot (HOOK_WEIGHTS, steadiness kept at the floor every brief has): motion and the close / approach bonus
# over the brief's content words, which in such a brief name no shots.
HOOK_LIST_WEIGHTS = {"relevance": 0.5, "energy": 0.8, "quality": 0.3}


# ---- the parsed brief ---------------------------------------------------------------------------------------------
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
    # reaction: a highlight brief about a person's reactions ("the funniest moments", "every jump scare, death and
    # clutch play", "a cold open with the biggest reaction"): its prior leans on the audio reaction and excited speech
    # (vhe.signals.reaction) and any shot list it names is one content text (events, not pictures). lines: spoken
    # lines matched against the transcript: quoted lines ("I made it into Harvard"; quotes in the opener/closer
    # description serve that slot instead) and a reaction brief's named moments ("the bear trap").
    exclude: list[str] = field(default_factory=list)
    no_speech: bool = False
    per_item: bool = True
    highlight: bool = False
    reaction: bool = False
    lines: list[str] = field(default_factory=list)
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
    # Parser state, not part of the brief's dict: hook_list, the brief asks for a ranked list of hooks ("the 3 best
    # first-shots", HOOK_LIST_WEIGHTS); cue_text, the brief without its opener/closer and avoid clauses (what the
    # visual cues are read on).
    hook_list: bool = field(default=False, repr=False, compare=False)
    cue_text: str = field(default="", repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in INTERNAL_FIELDS:
            del d[key]
        return d


INTERNAL_FIELDS = ("hook_list", "cue_text")


def item_name(p: ParsedQuery, item) -> str | None:
    """The name of the item a clip serves: an index into p.shots (-1: none), or already a name (a spoken topic)."""
    if isinstance(item, (int, np.integer)):
        return p.shots[int(item)] if 0 <= int(item) < len(p.shots) else None
    return item
