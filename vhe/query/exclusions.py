"""A brief's hard exclusions (ParsedQuery.exclude) as per-step masks: where an excluded item shows (SigLIP relevance
against a per-item calibrated threshold, exclusion_steps), where an excluded camera defect shows (the index's defect
signals, defect_steps), and where on-screen text holds a forbidden word (exclusion_text, ocr_term_steps). An exclude
item that names only what a wanted shot asks for stays a soft penalty (conflicts). vhe.rank.gate applies the masks."""
from __future__ import annotations

import math
import re

import numpy as np

from vhe.query.refine import EDGE_MAX_SHAKE
from vhe.query.text_match import STOPWORDS, WORD, squash, word_tokens
from vhe.signals.quality import reason_mask

# "no protest signs", "absolutely no sanitation trucks": a clip showing an excluded item is dropped, not penalised.
# SigLIP relevance is not comparable across phrases (on the irl-event footage "empty street" has a corpus median of
# +0.75, "portraits of victims" never exceeds -0.7), so an item counts as present at a step when its relevance is in
# the item's own top decile over the corpus and above an absolute floor (log-odds about -7.4). Calibrated 2026-10-18
# on clips the judge failed for showing an excluded thing: protest signs +1.25..+2.17 (gate 0.5), sanitation trucks
# +2.27 (0.68), floats +2.34 (0.84), flag +0.65 (0.55); judged-fine clips of the same brief stayed below their gates
# except one flag-waving float at 0.59 for "protest signs". Abstract phrases ("portraits of victims") are not
# detected at all at this scale: the soft avoid penalty still applies to them.
EXCLUDE_QUANTILE = 0.9
EXCLUDE_MIN_REL = 0.25
EXCLUDE_CLEAR_REL = 1.5  # a clear match (log-odds about -4.3) always counts, also for an item common in the footage
EXCLUDE_MIN_RUN = 2  # grid steps (0.67 s at 3 fps)


WHIP_SPEED = 0.15  # cam_speed of a whip-pan (follow pans are 0.02-0.06; vhe.query.refine EDGE_MAX_SPEED)
# Exclude items about the camera, not the content: the index's own defect signals answer them, not SigLIP.
DEFECTS = {
    "whip": r"\b(?:whip\w*|swish\w*|fast pans?|reißschwenks?)\b",
    "shaky": r"\b(?:shak\w*|wobbl\w*|jitter\w*|unsteady|verwackel\w*|wackler)\b",
    "blocked": r"\b(?:block\w*|cover\w*|obstruct\w*|lens|linse|finger\w*|hands?|in front of)\b",
    "framing": (r"\b(?:ground|floor|pavement|asphalt|boden|straight up)\b|"
                r"\b(?:camera|pointing|points|pointed)\b.*\bsky\b"),
}
GENERIC_WORDS = {"shot", "shots", "clip", "clips", "frame", "frames", "anything", "something", "any", "the", "with",
                 "filling", "fills", "fill", "across", "close", "closeup", "close-up", "close-ups", "view", "scene"}


def defect_kind(item: str) -> str | None:
    """'whip', 'shaky', 'blocked' or 'framing' for an exclude item about a camera defect, else None."""
    low = item.lower()
    return next((k for k, pat in DEFECTS.items() if re.search(pat, low)), None)


def defect_steps(grid: dict, kind: str) -> np.ndarray:
    """bool per grid step where the camera defect `kind` shows (from the usable-mask reasons and motion)."""
    def above(key: str, limit: float) -> np.ndarray:
        return np.nan_to_num(np.asarray(grid[key], dtype=float), nan=0.0) > limit

    if kind == "whip":
        return above("cam_speed", WHIP_SPEED)
    if kind == "shaky":
        return reason_mask(grid["unusable"], "shaky") | above("shake", EDGE_MAX_SHAKE)
    if kind == "blocked":
        return reason_mask(grid["unusable"], "covered", "uniform", "obstructed")
    return reason_mask(grid["unusable"], "bad_framing")


def _stems(text: str) -> set[str]:
    return {w[:5] for w in word_tokens(text) if len(w) > 2 and w not in STOPWORDS
            and w not in GENERIC_WORDS}


def conflicts(item: str, wanted: list[str]) -> bool:
    """The exclude item names only things a wanted shot asks for ('frame with a barricade' against 'establishing wide
    with the barricades'): it then stays a soft penalty, never a gate that would remove the wanted shot."""
    stems = _stems(item)
    return bool(stems) and any(stems <= _stems(w) for w in wanted if w)


def exclusion_steps(rel_excl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rel_excl: (items, steps) SigLIP relevance of each exclude item. (bool per step where any item shows, the
    per-item thresholds): an item shows where its relevance is in its corpus top decile and above EXCLUDE_MIN_REL,
    or above EXCLUDE_CLEAR_REL in any case, for at least EXCLUDE_MIN_RUN steps in a row (a one-frame blip is not a
    sighting)."""
    rel_excl = np.atleast_2d(np.asarray(rel_excl, dtype=float))
    if not len(rel_excl) or not rel_excl.shape[1]:
        return np.zeros(rel_excl.shape[1], dtype=bool), np.zeros(len(rel_excl))
    thr = np.clip(np.quantile(rel_excl, EXCLUDE_QUANTILE, axis=1), EXCLUDE_MIN_REL, EXCLUDE_CLEAR_REL)
    hit = (rel_excl >= thr[:, None]).any(0)
    edges = np.diff(np.r_[0, hit.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if b - a < EXCLUDE_MIN_RUN:
            hit[a:b] = False
    return hit, thr


# ---- exclusions read on screen --------------------------------------------------------------------------------------
# "absolutely no protest signs, no portraits of victims, nothing with 'executed' or 'killed' on it", "nothing
# political": SigLIP alone missed what the r3 judge failed (an "Executed by IRGC" poster behind the giant flag scored
# -0.4..0.0 for "protest signs", far under its gate), but the index read the text (ocr.json). So an exclude item about
# text or politics also excludes every span whose on-screen text holds one of its terms:
#   - words quoted in the brief inside an exclude item ('executed', "killed") are literal terms;
#   - an item about politics, protest or victims ("political", "protest signs", "portraits of victims") adds
#     POLITICAL_TERMS, the words protest and memorial signs are made of, and its SigLIP prompts (POLITICAL_PROMPTS,
#     PORTRAIT_PROMPTS) join the SigLIP exclusion (exclusion_steps, its calibrated thresholds). Measured on the
#     irl-event footage: "a banner with a printed photo of a face" is in its top decile on 70-100 % of the "killed",
#     "executed" and "WOMAN LIFE FREEDOM" sign spans and on none of the floats, band or dancers; "protest sign" also
#     put the Gate of the Nations float (flags) over its gate, so it is not added (an item that says "protest signs"
#     still is one). The victim portraits on the SHIRAZ float are under every prompt's floor and carry no text: missed.
# Terms match the OCR text with spaces and punctuation removed, case-insensitive, and as a fragment of at least
# OCR_TERM_FRAGMENT of the term (from 6 letters on): RapidOCR loses letters at a sign's edge ("tecutedby",
# "r50.000killed"). Any OCR sighting within OCR_HOLD_S of a span's steps drops the span (text is read once a second).
# A long-text rule ("signs with paragraphs") was measured and left out: on the irl-event footage multi-line blocks of
# text are as common on the floats' labels, shop signs and the police band banner as on protest signs.
POLITICAL_ITEM = (r"\b(?:politic\w*|protest\w*|propaganda|activis\w*|slogans?|demonstrat\w*|victims?|martyrs?|"
                  r"memorials?)\b")
PORTRAIT_ITEM = r"\b(?:portraits?|victims?|martyrs?|memorials?|photos? of (?:the )?(?:dead|victims?))\b"
POLITICAL_TERMS = ("killed", "killing", "executed", "execution", "murder", "murdered", "martyr", "regime", "dictator",
                   "freedom", "revolution", "protest", "genocide", "massacre", "prisoner", "political", "terrorist",
                   "terrorism", "torture", "slavery", "humanrights", "oppression")
POLITICAL_PROMPTS = ("a banner with a printed photo of a face",)
PORTRAIT_PROMPTS = ("a poster with a portrait photo of a person", "a banner with a printed photo of a face")
OCR_TERM_FRAGMENT = 0.75
OCR_HOLD_S = 0.5
QUOTED = r"(?<![\w])['‘\"“„]([^'‘’\"“”„]{2,40}?)['’\"”](?![\w])"
TERM_FILLER = {"something", "anything", "nothing", "with", "on", "it", "them", "that", "says", "saying", "written",
               "the", "a", "an", "or", "and", "word", "words", "text", "content", "any", "in", "of", "at", "to", "for",
               "from", "all", "every", "each", "no", "not", "never"}


def quoted_terms(text: str) -> list[str]:
    """Phrases in quotes ('executed', "killed"); apostrophes inside words ("host's") are not quotes."""
    return [q.strip() for q in re.findall(QUOTED, text) if q.strip()]


def exclusion_text(exclude: list[str], raw: str = "", wanted: list[str] = ()) -> tuple[list[str], list[str], list[str]]:
    """(OCR terms, extra SigLIP prompts, items left to SigLIP) for the brief's exclude items. OCR terms: the brief's
    quoted words that an exclude item contains, plus POLITICAL_TERMS for an item about politics, protest or victims.
    Prompts: POLITICAL_PROMPTS / PORTRAIT_PROMPTS for such items, unless a wanted shot asks for the same thing. An
    item that is only a quoted term ("'killed' on it") is matched on screen only, not by SigLIP."""
    quotes = quoted_terms(raw)
    terms, prompts, visual = [], [], []
    for x in exclude:
        low = x.lower()
        mine = [q for q in quotes + quoted_terms(x) if q.lower() in low]
        terms += mine
        political = re.search(POLITICAL_ITEM, low) is not None
        if political:
            terms += POLITICAL_TERMS
            prompts += POLITICAL_PROMPTS
        if re.search(PORTRAIT_ITEM, low):
            prompts += PORTRAIT_PROMPTS
        rest = low
        for q in mine:
            rest = rest.replace(q.lower(), " ")
        words = [w for w in re.findall(WORD, rest) if w not in TERM_FILLER and w not in STOPWORDS]
        if words and not (political and all(re.fullmatch(POLITICAL_ITEM, w) or w in GENERIC_WORDS for w in words)):
            visual.append(x)  # "nothing political in any of them" names nothing to look for besides its prompts
    prompts = [x for x in dict.fromkeys(prompts) if not conflicts(x, list(wanted))]
    return list(dict.fromkeys(t for t in terms if len(squash(t)) >= 3)), prompts, visual


def _term_hit(term: str, text: str) -> bool:
    """A squashed term in squashed OCR text, whole or (from 6 letters on) as a fragment of OCR_TERM_FRAGMENT of it."""
    if term in text:
        return True
    if len(term) < 6:
        return False
    k = math.ceil(OCR_TERM_FRAGMENT * len(term))
    return any(term[a:a + k] in text for a in range(len(term) - k + 1))


def ocr_term_steps(corpus, terms: list[str]) -> np.ndarray | None:
    """bool per corpus grid step: on-screen text holding one of `terms` read within OCR_HOLD_S (never across a file
    boundary); None without OCR text or terms, or when no term is read anywhere."""
    texts = getattr(corpus, "text_items", None)
    sq = [t for t in (squash(x) for x in terms) if t]
    if not texts or not sq:
        return None
    mask = np.zeros(corpus.n, dtype=bool)
    hold = int(round(OCR_HOLD_S * corpus.fps))
    for it in texts:
        text = squash(str(it.get("text", "")))
        if text and any(_term_hit(t, text) for t in sq):
            k, step = int(it.get("file", 0)), int(it["step"])
            lo, hi = int(corpus.bounds[k]), int(corpus.bounds[k + 1])
            mask[max(lo, step - hold):min(hi, step + hold + 1)] = True
    return mask if mask.any() else None
