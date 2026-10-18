"""How a picked span is reported: the clip dict of a result (file and file-local times, scores, labels, the reason it
was picked, its transcript) and the warnings for what an edit could not deliver."""
from __future__ import annotations

import numpy as np

from vhe.query.model import ParsedQuery, item_name
from vhe.rank.signals import segment_scores

TAG_MIN_PROB = 0.3  # a zero-shot tag joins a clip's labels at this mean probability


def clip(corpus, o: dict, sig: dict, p: ParsedQuery, role: str | None) -> dict:
    """The result dict of span o ({"file", "start", "end", "total", "source"}, optional "item", "text")."""
    c = corpus
    k, s, e = o["file"], round(float(o["start"]), 3), round(float(o["end"]), 3)
    j0, j1 = (int(x) for x in c.steps(s, e, k))
    parts = segment_scores(sig, j0, j1, p.weights)
    names = labels(c, j0, j1, parts)
    item = item_name(p, o.get("item"))
    return c.locate(k, s, e) | {
        "duration": round(e - s, 3), "rank_score": round(o["total"], 3), "score": round(parts["total"], 3),
        "relevance": round(parts["relevance"], 3), "quality": round(parts["quality"], 3),
        "energy": round(parts["energy"], 3), "labels": names, "source": o["source"], "role": role,
        "item": item, "reason": reason(parts, p, names, role, o.get("text"), item),
        "transcript": c.words_between(s, e)}


def reason(parts: dict, p: ParsedQuery, labels: list[str], role: str | None, text: str | None,
           item: str | None = None) -> str:
    what = f"'{item}'" if item else "the shot list" if p.shots else f"'{p.text}'"
    names = {"relevance": f"matches {what}", "quality": "clean footage", "steady": "steady camera",
             "energy": "high energy" if p.weights.get("energy", 0.0) >= 0 else "calm", "speech": "speech"}
    contrib = sorted(((p.weights.get(k, 0.0) * parts[k], k) for k in names), reverse=True)
    bits = [f"{role}: {text}"] if role else []
    bits += [f"{names[k]} ({parts[k]:+.1f})" for v, k in contrib[:2] if v > 0.05]
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
