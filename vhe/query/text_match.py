"""Brief words against text: the one stopword list and the squashing every on-screen-text match uses, the shot items'
words read by OCR inside a span (ocr_item_scores), and the legibility prior for briefs that ask for readable text."""
from __future__ import annotations

import re

import numpy as np

from vhe.query.lexicon import LEGIBLE_CUE

WORD = r"[^\W\d_][\w'-]*"  # a word: a letter, then letters, digits, apostrophes or hyphens


def word_tokens(text: str) -> list[str]:
    """The words of `text`, lower-cased."""
    return re.findall(WORD, text.lower())


# Function words no match counts: topic and exclusion words, shot-item words read on screen.
STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "these", "those", "his", "her", "their", "our", "you",
             "your", "are", "was", "were", "is", "be", "been", "being", "have", "has", "had", "him", "she", "they",
             "them", "what", "where", "when", "why", "how", "who", "which", "about", "into", "onto", "out", "some",
             "any", "all", "just", "going", "gonna", "here", "there", "not", "but", "can", "will", "would", "one",
             "its", "it's", "i'm", "we're", "you're", "der", "die", "das", "und", "mit", "ein", "eine", "von", "auf"}
# vhe search's spoken-match terms skip only these (words of 3+ letters). Kept apart because STOPWORDS there changes
# search results ("where is he going with his music" would match on "music" alone): unify in a semantic pass.
SEARCH_STOPWORDS = {"the", "and", "with", "for", "from", "that", "this", "der", "die", "das", "und", "mit", "ein",
                    "eine", "von", "auf", "für", "den", "dem", "des"}


def squash(text: str) -> str:
    return re.sub(r"[^0-9a-zäöüß]", "", text.lower())


# ---- shot items read on screen ---------------------------------------------------------------------------------------
OCR_BONUS = 1.0  # relevance units for a span whose legible text contains all of an item's words


# Words of an item that name what carries the text, or ask for it to be readable, not the text itself ("the PASARGAD
# float with its name legible" -> "pasargad"): left out of the ocr_item_match feature (vhe.rank.features).
OCR_GENERIC = {"name", "names", "named", "text", "texts", "readable", "legible", "legibly", "clearly", "clear",
               "visible", "sign", "signs", "banner", "banners", "float", "floats", "label", "labels", "lettering",
               "written", "words", "wording", "colorful", "colourful", "most", "with"}


def ocr_item_scores(corpus, items: list[str], S: np.ndarray, E: np.ndarray, K: np.ndarray,
                    ignore: set[str] | frozenset = frozenset()) -> np.ndarray | None:
    """Hook for on-screen text: (spans, items) the share of each item's words found in the text read inside
    each span (0..1, matched without spaces), or None when the corpus exposes no text (OCR did not
    run): callers then change nothing.

    Reads `corpus.text_items` when it exists: [{"t": virtual seconds, "text", "conf", "file", ...}] (vhe.corpus, from
    each index's ocr.json). vhe.rank.pool adds OCR_BONUS times this to each span's item relevance, so a float whose name
    is legible serves its item ("PASARGAD float"). A legibility prior for "every shot where text is readable" briefs
    can come from the grid's text_area / text_conf the same way. S, E: virtual seconds; K: file index per span.
    ignore: words never counted (an item left with none scores 0)."""
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
LEGIBLE_BONUS = 1.0  # relevance units at full OCR confidence, for briefs that ask for readable text


def legibility_steps(grid: dict, wanted: bool | str) -> np.ndarray | None:
    """Per step 0..1: confident legible text in frame (grid text_conf where text_area > 0, index v3), for a brief that
    asks for readable text (ParsedQuery.legible_text, or a brief string matched with the parser's LEGIBLE_CUE); None
    otherwise or without OCR in the index. Not scaled by text size: weighting small text down cost the fact-check
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
