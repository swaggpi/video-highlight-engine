"""A parsed brief -> the per-step signals every later stage reads (candidate peaks, gates, features, scorer, clip
reports): the corpus signals (vhe.rank.signals.timestep_signals) plus the brief's relevance, its visual-cue bonus and
its hard-exclusion masks.

relevance is the hand-weighted relevance the candidates, the heuristic scorer and the reported scores use: per shot
item (or the content text) SigLIP averaged with the highlight saliency where a file has it, the best item per step,
minus the bounded avoid penalty; with "the best moments" and no shot list, a highlight prior from the zero-shot tags,
faces and approach. siglip / avoid_pen / nc_sal keep the same parts apart for the features: nc_sal is the slot an
owned highlight head will fill.

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
from vhe.query.text_match import LEGIBLE_BONUS, legibility_steps, legible_items
from vhe.rank.features import grid_steps
from vhe.rank.items import combine, item_centre
from vhe.rank.scorer import cue_bonus
from vhe.rank.signals import blend_nc, nc_signal, timestep_signals
from vhe.signals import robust_z, smooth

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
    sig.update(siglip=None, avoid_pen=None, nc_sal=nc, items=None, item_centre=None, muted=muted)
    items = p.shots or ([p.text] if p.text else [])
    # "every shot where the text is legible", "the float with its name legible": confident OCR text counts as
    # relevant, to the items that ask for it (all of them when none does)
    legible = legibility_steps(c.grid, p.legible_text)
    if items:
        rel_items = relevances(items)
        rel_avoid = relevances(p.avoid) if p.avoid else None
        want = combine(rel_items)
        blended = blend_nc(rel_items, nc)
        if legible is not None:
            blended = blended + LEGIBLE_BONUS * legible_items(items)[:, None] * legible[None, :]
        sig.update(relevance=combine(blended, rel_avoid), siglip=want,
                   avoid_pen=want - combine(rel_items, rel_avoid))
        if p.shots:
            sig.update(items=rel_items, item_centre=item_centre(rel_items))
    prior = highlight_prior(c) if p.highlight and not p.shots else None
    if prior is not None:  # "the best moments": people, faces, dancing ... count as relevant
        rel = sig["relevance"]
        sig["relevance"] = 0.5 * rel + 0.5 * prior if items else prior
        if sig["siglip"] is None:  # the rel_* features stay SigLIP-only (no text: zero, as before)
            sig["siglip"] = np.zeros(c.n)
    if legible is not None and not items:
        sig["relevance"] = sig["relevance"] + LEGIBLE_BONUS * legible
    sig["cue"] = cue_steps(c, p)
    sig.update(exclusions(c, p, relevances))
    return sig, nc_raw, proposals


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
