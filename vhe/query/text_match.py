"""Brief words against text: the one stopword list and the squashing every on-screen-text match uses, the shot items'
words read by OCR inside a span (ocr_item_scores), and the legibility prior for briefs that ask for readable text."""
from __future__ import annotations

import math
import re

import numpy as np

from vhe.query.lexicon import LEGIBLE_CUE, QUOTE_MIN_WORDS, QUOTED

WORD = r"[^\W\d_][\w'-]*"  # a word: a letter, then letters, digits, apostrophes or hyphens


def word_tokens(text: str) -> list[str]:
    """The words of `text`, lower-cased."""
    return re.findall(WORD, text.lower())


# Function words no match counts: topic and exclusion words, shot-item words read on screen, and `vhe search`'s
# spoken-match terms (one list since v3: search used to skip only articles, prepositions and German articles, so
# "where is he going with his music" now matches a sentence on "music" alone instead of also on "going").
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "these", "those", "his", "her", "their", "our", "you",
             "your", "are", "was", "were", "is", "be", "been", "being", "have", "has", "had", "him", "she", "they",
             "them", "what", "where", "when", "why", "how", "who", "which", "about", "into", "onto", "out", "some",
             "any", "all", "just", "going", "gonna", "here", "there", "not", "but", "can", "will", "would", "one",
             "its", "it's", "i'm", "we're", "you're", "der", "die", "das", "und", "mit", "ein", "eine", "von", "auf",
             "für", "den", "dem", "des"}


def squash(text: str) -> str:
    return re.sub(r"[^0-9a-zäöüß]", "", text.lower())


# ---- shot items read on screen ---------------------------------------------------------------------------------------
# Words of an item that name what carries the text, or ask for it to be readable, not the text itself ("the PASARGAD
# float with its name legible" -> "pasargad"): never counted wherever an item is matched against OCR. The one ignore
# rule since v3 (item matching, the ocr_item_match feature and the heuristic's on-screen-name bonuses all read the
# same matrix, vhe.rank.features.ocr_matrix).
OCR_GENERIC = {"name", "names", "named", "text", "texts", "readable", "legible", "legibly", "clearly", "clear",
               "visible", "sign", "signs", "banner", "banners", "float", "floats", "label", "labels", "lettering",
               "written", "words", "wording", "colorful", "colourful", "most", "with"}


def ocr_item_scores(corpus, items: list[str], S: np.ndarray, E: np.ndarray, K: np.ndarray,
                    ignore: set[str] | frozenset = frozenset()) -> np.ndarray | None:
    """Hook for on-screen text: (spans, items) the share of each item's words found in the text read inside
    each span (0..1, matched without spaces), or None when the corpus exposes no text (OCR did not
    run): callers then change nothing.

    Reads `corpus.text_items` when it exists: [{"t": virtual seconds, "text", "conf", "file", ...}] (vhe.corpus, from
    each index's ocr.json). Item matching counts it as evidence of which item a span shows (vhe.rank.items.evidence), so
    a float whose name is legible serves its item ("PASARGAD float"); features carry it raw (ocr_item_match) and only
    HeuristicScorer turns it into score. S, E: virtual seconds; K: file index per span. ignore: words never counted
    (an item left with none scores 0); every caller passes OCR_GENERIC."""
    texts = getattr(corpus, "text_items", None)
    if not texts or not items or not len(S):
        return None
    S, E, K = np.asarray(S, dtype=float), np.asarray(E, dtype=float), np.asarray(K, dtype=int)
    t = np.array([float(x["t"]) for x in texts])
    read = [squash(str(x.get("text", ""))) for x in texts]
    files = np.array([int(x.get("file", 0)) for x in texts])
    # OCR often drops spaces ("TOMBof CYRUS"): words are matched as substrings of the text read inside the span with
    # spaces removed. A word is in the span's text iff it is in one of its reads (squashed text holds no separator).
    terms = [[squash(w) for w in word_tokens(x)
              if len(w) > 2 and w not in STOPWORDS and w not in ignore] for x in items]
    vocab = sorted({w for ws in terms for w in ws})
    col = {w: j for j, w in enumerate(vocab)}
    hit = np.array([[w in r for w in vocab] for r in read], dtype=np.int64).reshape(len(read), len(vocab))
    nonempty = np.array([bool(r) for r in read], dtype=np.int64)
    out = np.zeros((len(S), len(items)))
    for k in np.unique(K):
        mine = np.flatnonzero(files == k)
        mine = mine[np.argsort(t[mine], kind="stable")]
        spans = np.flatnonzero(K == k)
        a = np.searchsorted(t[mine], S[spans], "left")
        b = np.searchsorted(t[mine], E[spans], "right")
        found = np.concatenate([np.zeros((1, len(vocab)), np.int64), np.cumsum(hit[mine], axis=0)])
        found = (found[b] - found[a]) > 0
        full = np.concatenate([[0], np.cumsum(nonempty[mine])])
        has_text = (b - a >= 2) | (full[b] - full[a] > 0)  # the reads, joined, are not an empty string
        for i, ws in enumerate(terms):
            if ws:
                out[spans, i] = np.where(has_text, found[:, [col[w] for w in ws]].sum(1) / len(ws), 0.0)
    return out


# ---- readable text ---------------------------------------------------------------------------------------------------
def legibility_steps(grid: dict, wanted: bool | str) -> np.ndarray | None:
    """Per step 0..1: confident legible text in frame (grid text_conf where text_area > 0, index v3), for a brief that
    asks for readable text (ParsedQuery.legible_text, or a brief string matched with the parser's LEGIBLE_CUE; True:
    always, the raw legibility_mean feature); None otherwise or without OCR in the index. The bonus for it lives in
    vhe.rank.scorer (LEGIBLE_BONUS). Not scaled by text size: weighting small text down cost the fact-check
    brief (p3-newsedit-1) its street signs, 84 % -> 72 % of picks with text read inside."""
    if isinstance(wanted, str):
        wanted = re.search(LEGIBLE_CUE, wanted.lower()) is not None
    if not wanted or "text_conf" not in grid or "text_area" not in grid:
        return None
    conf = np.nan_to_num(np.asarray(grid["text_conf"], dtype=float), nan=0.0)
    area = np.nan_to_num(np.asarray(grid["text_area"], dtype=float), nan=0.0)
    return np.clip(np.where(area > 0, conf, 0.0), 0.0, 1.0)


def legible_items(items: list[str]) -> np.ndarray:
    """bool per shot item that itself asks for readable text ("float with its name legible", "the band with the
    banner readable"); all True when none does (the whole brief asks: "every shot where text is legible")."""
    ask = np.array([re.search(LEGIBLE_CUE, x.lower()) is not None for x in items], dtype=bool)
    return ask if ask.any() else np.ones(len(items), dtype=bool)


# ---- quoted lines against the transcript -------------------------------------------------------------------------
# A brief that quotes what someone says ("the 'I made it into Harvard' speech", "she thought I was a watermelon")
# wants the moment the line is said: its words are looked up in the transcript (ParsedQuery.lines; the heuristic's
# QUOTE_BONUS, the quote_match feature). Whisper rarely has the exact wording ("I finally made the Harvard"), so a
# line is matched on its content words (5-letter stems) inside a window of its length plus LINE_SLACK transcript
# words, each word weighted by how rare it is in the transcript (a match on "watermelon" counts more than one on
# "thought"). The step value ramps from 0 at LINE_FLOOR of the line's weight to 1 at all of it; matches below
# LINE_MIN_MATCH are dropped.
LINE_MIN_MATCH = 0.5
LINE_MIN_ALL = 0.75  # a line of function words and a name ("we are going to Granny"): most of it must be said
LINE_FLOOR = 0.35
LINE_SLACK = 4


def quoted_lines(text: str) -> list[str]:
    """Quoted passages of at least QUOTE_MIN_WORDS words ('the "I made it into Harvard" speech' -> "I made it into
    Harvard"); a quote with a slash is a label ("scare/death"), not a line."""
    return [" ".join(m.group(1).split()) for m in re.finditer(QUOTED, text)
            if "/" not in m.group(1) and len(word_tokens(m.group(1))) >= QUOTE_MIN_WORDS]


def _stem(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())[:5]


def line_steps(words: list[dict], lines: list[str], n: int, fps: float) -> np.ndarray:
    """(len(lines), n) per quoted line, 0..1 per grid step (module notes above): the best rarity-weighted share of the
    line's content words (STOPWORDS left out unless that leaves fewer than two; then a match needs one of its content
    words too and LINE_MIN_ALL of it: "we are going to Granny" is not any "where are you, Granny") said within a
    window of the line's length plus LINE_SLACK transcript words, ramped from LINE_FLOOR to 1, on the steps from the
    first to the last matched word. Words of one or two letters are no content words.
    words: transcript words with virtual-second "start" / "end"."""
    out = np.zeros((len(lines), n))
    stems = [_stem(str(w.get("word", ""))) for w in words]
    counts: dict[str, int] = {}
    for s in stems:
        counts[s] = counts.get(s, 0) + 1
    idf = lambda s: 1.0 + math.log((len(stems) + 1) / (counts.get(s, 0) + 1))  # noqa: E731
    for j, line in enumerate(lines):
        toks = word_tokens(line)
        content = {_stem(t) for t in toks if t not in STOPWORDS and len(t) > 2}
        want = [_stem(t) for t in toks if _stem(t) in content] if len(content) >= 2 else [_stem(t) for t in toks]
        if not want:
            continue
        total = sum(idf(s) for s in want)
        width = len(toks) + LINE_SLACK
        need = LINE_MIN_MATCH if len(content) >= 2 else LINE_MIN_ALL
        for i in range(len(words)):
            left = list(want)
            got, first, last = 0.0, None, None
            key = not content or len(content) >= 2  # the all-words fallback needs a content word
            for k in range(i, min(len(words), i + width)):
                if stems[k] and stems[k] in left:
                    left.remove(stems[k])
                    got += idf(stems[k])
                    first = k if first is None else first
                    last = k
                    key = key or stems[k] in content
            share = got / total
            if first is None or share < need or not key:
                continue
            a = max(0, int(words[first]["start"] * fps))
            b = min(n, int(np.ceil(words[last]["end"] * fps)))
            out[j, a:b] = np.maximum(out[j, a:b], (share - LINE_FLOOR) / (1.0 - LINE_FLOOR))
    return out
