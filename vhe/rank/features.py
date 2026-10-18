"""Per-candidate feature vectors and the query vector: the inputs of every scorer (vhe.rank.scorer), and what a run
dumps for training (vhe.rank.dump).

A candidate is a span of one file (virtual seconds, file index, generator source). Its features are statistics of the
per-step signals (vhe.rank.signals.timestep_signals plus the query's relevance curves) over the span's grid steps,
computed vectorised for every candidate at once. FEATURE_VERSION names the columns, their order and their meaning: a
learned model records the version it was trained on and is refused on another, so changing a definition, adding or
reordering a column means a new version. Features are float64 here (the hand-weighted scorer is an exact function
of them); dumps store them at float32 precision."""
from __future__ import annotations

import math
import re

import numpy as np

from vhe.query.candidates import cuts_inside
from vhe.query.plan import OCR_GENERIC, assign, item_scores, muted_steps, ocr_item_scores
from vhe.rank.signals import span_max, span_mean, span_min, span_top_mean
from vhe.signals.tags import LABELS

# v2 (2026-10-18): the index v3 framing, face, approach and on-screen text columns appended (FEATURES_V2) and the
# brief's visual cues appended to the query's fixed block (QUERY_V2); v1 is the prefix of both.
FEATURE_VERSION = "v2"
QUERY_VERSION = "v2"
SOURCES = ("shot", "window", "peak", "model", "speech")  # vhe.query.candidates.generate
PERSONAS = ("p1-videographer", "p2-tiktok", "p3-newsedit", "p4-vlogger", "p5-marketing", "p6-casual")
SIGNALS = ("relevance", "quality", "energy", "steady", "speech")  # the weighted signals of ParsedQuery.weights


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


FEATURES = (
    # Relevance to the brief, SigLIP only: the best shot item per step (or the content text), no avoid penalty.
    "rel_mean", "rel_top60", "rel_max",
    # The brief plan: centred relevance of the shot item the span serves (else of its best item; 0 without a shot
    # list), and the mean avoid penalty (how far an avoid item beats the wanted relevance).
    "item_rel", "avoid_pen",
    # Non-commercial highlight saliency (CG-DETR), z-scored over the corpus; 0 where a file has none. The slot an
    # owned highlight head will fill.
    "nc_sal_mean", "nc_sal_top60", "nc_present",
    # The hand-weighted relevance (SigLIP and nc saliency averaged per step, minus the avoid penalty): kept so
    # HeuristicScorer is an exact function of X. A learned model should not use these two.
    "h_rel_mean", "h_rel_top60",
    # Technical quality and defects.
    "quality_mean", "quality_min", "blown_frac", "unusable_frac",
    # Energy (the weighted composite and its pooled z-scored components) and camera steadiness.
    "energy_mean", "motion_z", "cam_speed_z", "loud_z", "onset_z", "steady_mean",
    # Sound.
    "speech_frac", "has_audio", "muted_frac",
    # Shape: length, signed deviation from the brief's target length (/ max(target, 1)), cuts inside, position.
    "length_s", "length_rel", "n_cuts_inside", "pos_in_file",
    # Candidate generator.
    *(f"src_{s}" for s in SOURCES),
    # Zero-shot tag probabilities (vhe.signals.tags.LABELS), span means.
    *(f"tag_{_slug(label)}" for label in LABELS),
)
# v2, appended. Index v3 grid keys over the span's steps, raw units: shot-scale margins in logits (scale_close >= 1
# close-up / fills the frame, scale_wide >= 2.5 wide), the largest face as a fraction of the frame, how centred it
# is (0..1, 0 without a face), the face count, the looming rate (> 0.3 approaching), confident OCR text area and
# confidence; ocr_item_match: the best share of a shot item's words read by OCR inside the span (0..1). A grid key an
# index lacks reads as an ordinary medium shot with no face or text (GRID_NEUTRAL).
FEATURES_V2 = (
    "scale_close_mean", "scale_close_max", "scale_wide_mean", "face_area_max_mean", "face_area_max_max",
    "face_center_mean", "face_n_mean", "approach_max", "text_area_mean", "text_conf_mean", "ocr_item_match",
)
FEATURES = FEATURES + FEATURES_V2
INDEX = {name: i for i, name in enumerate(FEATURES)}
GRID_NEUTRAL = {"scale_close": -6.0, "scale_wide": 0.0, "face_area_max": 0.0, "face_center": 0.0, "face_n": 0.0,
                "approach": 0.0, "text_area": 0.0, "text_conf": 0.0}

QUERY_BASE = (
    "w_relevance", "w_quality", "w_energy", "w_steady", "w_speech",  # signal weights (mood, speech intent, plan)
    "mood_energetic", "mood_calm", "exhaustive", "has_text", "has_budget", "target_len", "log_target_len",
    "has_shots", "n_shots", "has_avoid", "has_topics", "need_live_audio",
)
# v2: the brief's visual cues (ParsedQuery.framing one-hot, want_faces, hook, legible_text, centered), appended to the
# fixed block; the persona one-hot stays last (its length depends on the scorer's persona list).
FRAMINGS = ("none", "close", "wide")
QUERY_V2 = (*(f"framing_{x}" for x in FRAMINGS), "want_faces", "hook", "legible_text", "centered")


def query_names(personas: tuple[str, ...] | list[str] = PERSONAS) -> tuple[str, ...]:
    return QUERY_BASE + QUERY_V2 + tuple(f"persona_{x}" for x in personas)


def target_len(p) -> float:
    """The clip length the brief aims at: the middle of its allowed range."""
    return (p.min_len + p.max_len) / 2


def query_vector(p, persona: str | None = None, personas: tuple[str, ...] | list[str] = PERSONAS,
                 weights: dict | None = None) -> np.ndarray:
    """(len(query_names(personas)),) float64 for a parsed brief (vhe.query.parser.ParsedQuery). weights: the signal
    weights to score with when they are not the brief's (the opener/closer pools). An unknown persona is all zeros."""
    w = p.weights if weights is None else weights
    target = target_len(p)
    v = [*(float(w.get(k, 0.0)) for k in SIGNALS),
         float(p.mood == "energetic"), float(p.mood == "calm"), float(p.exhaustive), float(bool(p.text)),
         float(p.budget is not None), target, math.log(max(target, 1e-3)),
         float(bool(p.shots)), float(len(p.shots)), float(bool(p.avoid)), float(bool(p.topics)),
         float(p.need_live_audio)]
    framing = getattr(p, "framing", "none")
    v += [float(framing == x) for x in FRAMINGS]
    v += [float(bool(getattr(p, k, False))) for k in ("want_faces", "hook", "legible_text", "centered")]
    v += [float(persona == x) for x in personas]
    return np.array(v, dtype=np.float64)


def featurize(corpus, sig: dict, S: np.ndarray, E: np.ndarray, K: np.ndarray, src: list[str], p,
              item_rel: np.ndarray | None = None) -> np.ndarray:
    """(n, len(FEATURES)) float64 for spans S..E (virtual seconds) of files K from generator `src`.

    sig: per-step signals. Required: relevance (the hand-weighted relevance), quality, energy, steady, speech. Used
    when present: siglip (else relevance), avoid_pen, nc_sal (NaN where absent), items / item_centre (shot list),
    motion_z / cam_speed_z / loud_z / onset_z, audio, unusable, blown, muted (else from the corpus grid).
    item_rel: (n, items) relevance of each span to each shot item when the caller already has it."""
    c = corpus
    S, E, K = np.asarray(S, dtype=float), np.asarray(E, dtype=float), np.asarray(K, dtype=int)
    X = np.zeros((len(S), len(FEATURES)))
    if not len(S):
        return X
    i0, i1 = c.steps(S, E, K)

    def put(name: str, values) -> None:
        X[:, INDEX[name]] = values

    h = sig["relevance"]
    rel = sig.get("siglip")
    rel = h if rel is None else rel
    put("rel_mean", span_mean(rel, i0, i1))
    put("rel_top60", span_top_mean(rel, i0, i1))
    put("rel_max", span_max(rel, i0, i1))
    if sig.get("items") is not None and len(sig["items"]):
        scores = item_scores(sig["items"], i0, i1) if item_rel is None else np.asarray(item_rel)
        centre = sig.get("item_centre")
        centre = np.zeros(scores.shape[1]) if centre is None else np.asarray(centre)
        served = assign(scores, centre)
        lift = scores - centre[None, :]
        put("item_rel", np.where(served >= 0, lift[np.arange(len(S)), np.maximum(served, 0)], lift.max(1)))
    if sig.get("avoid_pen") is not None:
        put("avoid_pen", span_mean(sig["avoid_pen"], i0, i1))
    nc = sig.get("nc_sal")
    if nc is not None:
        present = np.isfinite(nc)
        z = np.where(present, nc, 0.0)
        put("nc_sal_mean", span_mean(z, i0, i1))
        put("nc_sal_top60", span_top_mean(z, i0, i1))
        put("nc_present", span_mean(present.astype(float), i0, i1))
    put("h_rel_mean", span_mean(h, i0, i1))
    put("h_rel_top60", span_top_mean(h, i0, i1))

    put("quality_mean", span_mean(sig["quality"], i0, i1))
    put("quality_min", span_min(sig["quality"], i0, i1))
    blown = sig.get("blown")
    blown = np.nan_to_num(np.asarray(c.grid.get("blown", np.zeros(c.n)), dtype=float)) if blown is None else blown
    put("blown_frac", span_mean(blown, i0, i1))
    unusable = sig.get("unusable")
    put("unusable_frac", span_mean((~c.usable).astype(float) if unusable is None else unusable, i0, i1))
    put("energy_mean", span_mean(sig["energy"], i0, i1))
    for key in ("motion_z", "cam_speed_z", "loud_z", "onset_z"):
        if sig.get(key) is not None:
            put(key, span_mean(np.nan_to_num(sig[key]), i0, i1))
    put("steady_mean", span_mean(sig["steady"], i0, i1))

    put("speech_frac", span_mean(sig["speech"], i0, i1))
    audio = sig.get("audio")
    put("has_audio", span_mean(c.audio_steps.astype(float) if audio is None else audio, i0, i1))
    muted = sig.get("muted")
    muted = muted_steps(c.grid["loudness"], c.audio_steps) if muted is None else muted
    put("muted_frac", span_mean(np.asarray(muted, dtype=float), i0, i1))

    length = E - S
    target = target_len(p)
    put("length_s", length)
    put("length_rel", (length - target) / max(target, 1.0))
    put("n_cuts_inside", cuts_inside(S, E, np.asarray(c.cuts)))
    put("pos_in_file", ((S + E) / 2 - c.offsets[K]) / np.maximum(c.durations[K], 1e-9))
    for s in SOURCES:
        put(f"src_{s}", [x == s for x in src])

    labels = list(c.meta.get("labels") or [])
    tags = c.grid.get("tags")
    if tags is not None and labels:
        cum = np.concatenate([np.zeros((1, tags.shape[1])), np.cumsum(tags, axis=0, dtype=np.float64)])
        means = (cum[i1] - cum[i0]) / (i1 - i0)[:, None]
        for j, label in enumerate(labels):
            name = f"tag_{_slug(label)}"
            if name in INDEX and j < means.shape[1]:
                put(name, means[:, j])

    # v2: shot scale, faces, approach, on-screen text (index v3)
    g = {key: grid_steps(c, sig, key) for key in GRID_NEUTRAL}
    put("scale_close_mean", span_mean(g["scale_close"], i0, i1))
    put("scale_close_max", span_max(g["scale_close"], i0, i1))
    put("scale_wide_mean", span_mean(g["scale_wide"], i0, i1))
    put("face_area_max_mean", span_mean(g["face_area_max"], i0, i1))
    put("face_area_max_max", span_max(g["face_area_max"], i0, i1))
    put("face_center_mean", span_mean(g["face_center"], i0, i1))
    put("face_n_mean", span_mean(g["face_n"], i0, i1))
    put("approach_max", span_max(g["approach"], i0, i1))
    put("text_area_mean", span_mean(g["text_area"], i0, i1))
    put("text_conf_mean", span_mean(g["text_conf"], i0, i1))
    items = list(p.shots) or ([p.text] if p.text else [])
    ocr = ocr_item_scores(c, items, S, E, K, OCR_GENERIC) if items else None
    if ocr is not None and ocr.size:
        put("ocr_item_match", ocr.max(1))
    return X


def grid_steps(corpus, sig: dict, key: str) -> np.ndarray:
    """Per-step float64 values of an index v3 grid key (from sig when it has them, else the corpus grid); missing
    steps and indexes without the key read as GRID_NEUTRAL[key]."""
    x = sig.get(key)
    x = corpus.grid.get(key) if x is None else x
    if x is None:
        return np.full(corpus.n, GRID_NEUTRAL[key])
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=GRID_NEUTRAL[key])
