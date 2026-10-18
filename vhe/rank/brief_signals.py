"""A parsed brief -> the per-step signals every later stage reads (candidate peaks, gates, features, scorer, clip
reports): the corpus signals (vhe.rank.signals.timestep_signals) plus the brief's relevance, its visual-cue bonus and
its hard-exclusion masks.

relevance is the per-step relevance candidate peaks and the reported clip relevance use, with the heuristic's
constants (vhe.rank.scorer): per shot item (or the content text) SigLIP averaged with the highlight saliency where a
file has it, plus the legibility bonus of a readable-text brief, the best item per step, minus the bounded avoid
penalty; with "the best moments" and no shot list, blended with a highlight prior from the zero-shot tags, faces and
approach and the reaction prior (audio reaction and excited speech, vhe.signals.reaction; a reaction brief leans on
it); plus QUOTE_BONUS where a quoted line of the brief is said. No scorer reads it: features get the parts apart
(features v3/v4) - siglip, avoid_pen, nc_sal (the slot an owned highlight head will fill), h_rel (blended relevance
minus avoid, no legibility, no prior), legibility, highlight, reaction and excitement (raw, whatever the brief asks),
quote (the brief's quoted lines), and per shot item items (SigLIP) and items_avoid (its avoid penalty).

The models stay with the caller (vhe.engine): relevances(texts) -> (texts, steps) SigLIP relevance, and
nc_scores(text) -> (raw highlight saliency or None, proposals per file)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from vhe.query.exclusions import (
    conflicts,
    defect_kind,
    defect_steps,
    exclusion_steps,
    exclusion_text,
    ocr_term_steps,
)
from vhe.query.model import ParsedQuery
from vhe.query.refine import clean_steps
from vhe.query.text_match import legibility_steps, legible_items, line_steps
from vhe.rank.features import grid_steps
from vhe.rank.items import combine, item_centre
from vhe.rank.scorer import HIGHLIGHT_BLEND, LEGIBLE_BONUS, QUOTE_BONUS, REACTION_BLEND, compose_prior, cue_bonus
from vhe.rank.signals import blend_nc, nc_signal, timestep_signals
from vhe.signals import robust_z, smooth
from vhe.signals.reaction import excitement_steps

# "the best moments" with no shot list: relevance is a highlight prior from the zero-shot tags (people, faces,
# dancing, a crowd, a stage), z-scored over the corpus and averaged with the content text's relevance when there is one.
HIGHLIGHT_TAGS = ("crowd", "group of people", "close-up of a face", "dancing", "concert or stage", "party", "sports",
                  "running")
HIGHLIGHT_GRID = ("face_area_max", "approach")  # per-step grid keys (index v3) that join the prior

Relevances = Callable[[list[str]], np.ndarray]


def brief_signals(corpus, p: ParsedQuery, relevances: Relevances, nc_scores: Callable, muted: np.ndarray
                  ) -> tuple[dict, np.ndarray | None, list[list]]:
    """(per-step signals, raw highlight-model saliency or None, its proposals per file) for a parsed brief.

    items: SigLIP relevance per shot item (which item a clip serves is decided on SigLIP alone: the query-wide
    saliency says nothing about a single item and would shift the absolute ITEM_MIN_REL scale)."""
    c = corpus
    nc_raw, proposals = nc_scores(nc_query(p))
    nc = nc_signal(nc_raw)
    sig = timestep_signals(c.grid, c.fps, None, c.audio_steps, c.file_id)
    sig.update(siglip=None, avoid_pen=None, nc_sal=nc, items=None, items_avoid=None, item_centre=None,
               muted=muted, h_rel=np.zeros(c.n), legibility=legibility_steps(c.grid, True),
               highlight=highlight_prior(c), clean=clean_steps(c.grid),
               excitement=excitement_steps(c.words, c.n, c.fps, c.file_id), quote=quote_steps(c, p.lines))
    items = p.shots or ([p.text] if p.text else [])
    # "every shot where the text is legible", "the float with its name legible": for candidate peaks, confident OCR
    # text counts as relevant to the items that ask for it (all of them when none does)
    legible = legibility_steps(c.grid, p.legible_text)
    if items:
        rel_items = relevances(items)
        rel_avoid = relevances(p.avoid) if p.avoid else None
        want = combine(rel_items)
        blended = blend_nc(rel_items, nc)
        peaks = blended
        if legible is not None:
            peaks = blended + LEGIBLE_BONUS * legible_items(items)[:, None] * legible[None, :]
        sig.update(relevance=combine(peaks, rel_avoid), h_rel=combine(blended, rel_avoid), siglip=want,
                   avoid_pen=want - combine(rel_items, rel_avoid))
        if p.shots:
            sig.update(items=rel_items, item_centre=item_centre(rel_items),
                       items_avoid=np.stack([rel_items[j] - combine(rel_items[j:j + 1], rel_avoid)
                                             for j in range(len(rel_items))]))
    if p.highlight and not p.shots:  # "the best moments": people, faces, dancing, reactions ... count as relevant
        visual = sig["highlight"] if sig["highlight"] is not None else np.zeros(c.n)
        prior = compose_prior(visual, sig["reaction"], sig["excitement"], p.reaction)
        rel, a = sig["relevance"], REACTION_BLEND if p.reaction else HIGHLIGHT_BLEND
        pen = a * sig["avoid_pen"] if sig["avoid_pen"] is not None else 0.0  # the avoid penalty in full
        sig["relevance"] = (1 - a) * rel + a * prior - pen if items else prior
        if sig["siglip"] is None:  # the rel_* features stay SigLIP-only (no text: zero, as before)
            sig["siglip"] = np.zeros(c.n)
    if sig["quote"] is not None:  # "the 'I made it into Harvard' speech": where the line is said
        sig["relevance"] = sig["relevance"] + QUOTE_BONUS * sig["quote"]
    if legible is not None and not items:
        sig["relevance"] = sig["relevance"] + LEGIBLE_BONUS * legible
    sig["cue"] = cue_steps(c, p)
    sig.update(exclusions(c, p, relevances))
    return sig, nc_raw, proposals


def quote_steps(corpus, lines: list[str]) -> np.ndarray | None:
    """Per step: the best share of any quoted line said there (vhe.query.text_match.line_steps), 0..1; None without
    lines."""
    if not lines:
        return None
    return line_steps(corpus.words, lines, corpus.n, corpus.fps).max(0)


def nc_query(p: ParsedQuery) -> str:
    """The highlight model was trained on short English sentences: the raw query when it is one, the shot list
    when there is one (an LLM's one-line summary such as "news package" names no visible content), else the
    content text. highlight_nc fits whatever it gets into CLIP's 77 tokens."""
    if p.shots:
        return ", ".join(p.shots)
    simple = not (p.avoid or p.topics) and len(p.raw.split()) <= 25
    return p.raw if simple and p.text else p.text


def cue_steps(corpus, p: ParsedQuery) -> np.ndarray | None:
    """Per step: the scorer's visual-cue bonus (vhe.rank.scorer.cue_bonus on the step's own shot scale, face and
    approach) for a brief with framing / faces / centring cues, so candidate peaks sit on close-ups or wides too;
    None without cues. Readable text is in relevance already."""
    if p.framing == "none" and not (p.want_faces or p.centered):
        return None
    c = corpus
    cols = {"scale_close_mean": grid_steps(c, {}, "scale_close"),
            "scale_wide_mean": grid_steps(c, {}, "scale_wide"),
            "face_area_max_mean": grid_steps(c, {}, "face_area_max"), "approach_max": grid_steps(c, {}, "approach"),
            "face_center_mean": grid_steps(c, {}, "face_center"), "ocr_item_match": np.zeros(c.n)}
    return cue_bonus(cols, p.framing, p.want_faces, p.centered, False)


def highlight_prior(corpus) -> np.ndarray | None:
    """Per step: how much a generic highlight shows people, faces and action (HIGHLIGHT_TAGS, HIGHLIGHT_GRID),
    z-scored and clipped to +-3 like relevance; None without zero-shot tags."""
    c = corpus
    labels = list(c.meta.get("labels") or [])
    cols = [labels.index(x) for x in HIGHLIGHT_TAGS if x in labels]
    tags = c.grid.get("tags")
    parts = []
    if tags is not None and cols:
        parts.append(np.asarray(tags[:, cols], dtype=np.float32).max(1))
    # the largest face (absent when the face detector did not run) and something approaching the camera
    parts += [np.nan_to_num(np.asarray(c.grid[k], dtype=np.float32)) for k in HIGHLIGHT_GRID if k in c.grid]
    if not parts:
        return None
    z = [robust_z(smooth(x, c.fps)) for x in parts]
    return np.clip(np.mean(z, axis=0), -3, 3)


def exclusions(corpus, p: ParsedQuery, relevances: Relevances) -> dict:
    """Per-step masks for the hard gates (vhe.rank.gate): 'excluded' where an exclude item shows (SigLIP, per-item
    calibrated threshold; items about politics or victims add their prompts), 'excluded_text' where on-screen text
    holds one of the exclude items' terms (quoted words, political terms; any sighting drops a clip), 'defect' where an
    excluded camera defect shows (whip-pan, shake, blocked lens, ground/sky). An exclude item that names only what a
    wanted shot asks for stays a soft penalty (vhe.query.exclusions.conflicts)."""
    c = corpus
    wanted = [*p.shots, p.text, p.opener or "", p.closer or ""]
    content, defect = [], np.zeros(c.n, dtype=bool)
    for x in p.exclude:
        kind = defect_kind(x)
        if kind is not None:
            defect |= defect_steps(c.grid, kind)
        elif not conflicts(x, wanted):
            content.append(x)
    terms, prompts, visual = exclusion_text(content, p.raw, wanted)
    siglip = list(dict.fromkeys(visual + prompts))
    excluded = exclusion_steps(relevances(siglip))[0] if siglip else None
    return {"excluded": excluded, "excluded_text": ocr_term_steps(c, terms),
            "defect": defect if defect.any() else None, "exclude_items": siglip, "exclude_terms": terms}
