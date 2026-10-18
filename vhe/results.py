"""How a picked span is reported: the clip dict of a result (file and file-local times, the scorer's total, the
per-step signal summary, labels, the reason it was picked, its transcript) and the warnings for what an edit could
not deliver."""
from __future__ import annotations

import numpy as np

from vhe.query.model import ParsedQuery, item_name
from vhe.rank.features import tag_name
from vhe.rank.signals import segment_scores

TAG_MIN_PROB = 0.3  # a zero-shot tag joins a clip's labels at this mean probability
REASON_TERMS = 2  # the reason names the clip's largest positive score contributions, at most this many
REASON_MIN_SHARE = 0.05  # ... each at least this share of the largest contribution's size

# What a score contribution says, by the scorer's term or feature name (HeuristicScorer terms, the feature names a
# learned model's contributions carry, vhe.rank.features.FEATURES). Prefixes match feature families; tag_* name the
# zero-shot tag itself. A name not listed says nothing.
TERM_ASPECT = {
    "relevance": "relevance", "item_rel": "relevance", "avoid_pen": "relevance", "nc_present": "relevance",
    "highlight_prior_mean": "relevance", "legibility_mean": "text", "ocr_item_match": "text", "ocr": "text",
    "quality": "quality", "blown_frac": "quality", "unusable_frac": "quality", "detail_z_mean": "quality",
    "framing_frac": "quality", "energy": "energy", "steady": "steady", "steady_mean": "steady", "speech": "speech",
    "speech_frac": "speech", "has_audio": "sound", "muted_frac": "sound", "length": "length", "length_s": "length",
    "length_rel": "length", "cuts": "cuts", "n_cuts_inside": "cuts", "sentences": "sentences",
    "src_speech": "sentences",
    "close": "close", "approach": "close", "approach_max": "close", "face": "faces", "wide": "wide",
    "scale_wide_mean": "wide", "center": "centred", "face_center_mean": "centred", "edges": "edges",
    "edges_clean": "edges",
}
PREFIX_ASPECT = (("rel_", "relevance"), ("h_rel_", "relevance"), ("nc_sal_", "relevance"),
                 ("quality_", "quality"), ("energy_", "energy"), ("motion_", "energy"), ("cam_speed_", "energy"),
                 ("loud_", "energy"), ("onset_", "energy"), ("scale_close_", "close"), ("face_", "faces"),
                 ("text_", "text"))


def _aspect(name: str) -> str | None:
    if name in TERM_ASPECT:
        return TERM_ASPECT[name]
    if name.startswith("tag_"):
        return name
    return next((a for pre, a in PREFIX_ASPECT if name.startswith(pre)), None)


def clip(corpus, o: dict, sig: dict, p: ParsedQuery, role: str | None, scorer=None) -> dict:
    """The result dict of span o ({"file", "start", "end", "total", "source"}, optional "item", "text", and its
    features "x" and query vector "q"). score (and rank_score, the same number) is the scorer's total; relevance,
    quality and energy summarise the span's per-step signals; reason names the largest contributions to the total
    (scorer.explain, when the scorer has it)."""
    c = corpus
    k, s, e = o["file"], round(float(o["start"]), 3), round(float(o["end"]), 3)
    j0, j1 = (int(x) for x in c.steps(s, e, k))
    parts = segment_scores(sig, j0, j1, p.weights)
    names = labels(c, j0, j1, parts)
    item = item_name(p, o.get("item"))
    contrib = contributions(scorer, o)
    return c.locate(k, s, e) | {
        "duration": round(e - s, 3), "rank_score": round(o["total"], 3), "score": round(o["total"], 3),
        "relevance": round(parts["relevance"], 3), "quality": round(parts["quality"], 3),
        "energy": round(parts["energy"], 3), "labels": names, "source": o["source"], "role": role,
        "item": item, "reason": reason(contrib, p, names, role, o.get("text"), item, corpus.meta.get("labels")),
        "transcript": c.words_between(s, e)}


def contributions(scorer, o: dict) -> dict[str, float]:
    """{aspect: summed contribution to the clip's total}: scorer.explain over the clip's features, grouped by what
    each term or feature says (TERM_ASPECT); empty without features or an explain()."""
    explain = getattr(scorer, "explain", None)
    if explain is None or o.get("x") is None or o.get("q") is None:
        return {}
    out: dict[str, float] = {}
    for name, v in explain(np.asarray(o["x"], dtype=float)[None], np.asarray(o["q"], dtype=float)).items():
        aspect = _aspect(name)
        if aspect is not None:
            out[aspect] = out.get(aspect, 0.0) + float(np.asarray(v)[0])
    return out


def reason(contrib: dict[str, float], p: ParsedQuery, labels: list[str], role: str | None, text: str | None,
           item: str | None = None, tag_labels: list[str] | None = None) -> str:
    """"opener: <text>; <largest contributions to the total, with their size in score units>; <labels>"."""
    what = f"'{item}'" if item else f"'{text}'" if role and text else "the shot list" if p.shots else \
        f"'{p.text}'" if p.text else "the brief"
    phrase = {"relevance": f"matches {what}", "quality": "clean footage", "steady": "steady camera",
              "energy": "high energy" if p.weights.get("energy", 0.0) >= 0 else "calm", "speech": "speech",
              "sound": "live sound", "length": "fits the clip length", "cuts": "no cut inside",
              "sentences": "whole sentences", "close": "close-up", "faces": "faces", "wide": "wide view",
              "centred": "subject centred", "text": "readable text", "edges": "clean in and out points"}
    for label in tag_labels or []:
        phrase.setdefault(tag_name(label), label)
    top = sorted(((v, a) for a, v in contrib.items() if v > 0 and a in phrase), reverse=True)
    bits = [f"{role}: {text}"] if role else []
    if top:
        bits += [f"{phrase[a]} ({v:+.1f})" for v, a in top[:REASON_TERMS]
                 if v >= REASON_MIN_SHARE * top[0][0] and round(v, 1) > 0]
    return "; ".join(bits + ([", ".join(labels)] if labels else [])) or "best available"


def labels(corpus, i0: int, i1: int, parts: dict) -> list[str]:
    """The span's top zero-shot tags, plus "high energy" / "calm" and "speech" from its segment scores."""
    probs = corpus.grid["tags"][i0:max(i1, i0 + 1)].astype(np.float32).mean(0)
    out = [corpus.meta["labels"][j] for j in np.argsort(-probs)[:3] if probs[j] >= TAG_MIN_PROB]
    if parts["energy"] > 0.5:
        out.append("high energy")
    elif parts["energy"] < -0.5:
        out.append("calm")
    if parts["speech"] > 0.0:
        out.append("speech")
    return out


def shortfall(clips: list[dict], p: ParsedQuery) -> list[str]:
    """What the edit could not deliver: a runtime below the budget minimum, fewer clips than asked for."""
    out = []
    runtime = sum(o["duration"] for o in clips)
    if p.budget is not None and runtime < p.budget[0] - 0.5:
        out.append(f"runtime {runtime:.1f} s is below the requested {p.budget[0]:.0f}-{p.budget[1]:.0f} s: the "
                   f"footage has no more distinct usable moments for this brief (after duplicates and exclusions)")
    if p.count_explicit and not p.exhaustive and p.budget is None and len(clips) < p.count:
        out.append(f"{len(clips)} of the {p.count} requested clips: no more distinct usable moments match")
    return out
