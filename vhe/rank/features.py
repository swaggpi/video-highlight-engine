"""Per-candidate feature vectors and the query vector: the inputs of every scorer (vhe.rank.scorer), and what a run
dumps for training (vhe.rank.dump).

A candidate is a span of one file (virtual seconds, file index, generator source). Its features are statistics of the
per-step signals (vhe.rank.signals.timestep_signals plus the query's relevance curves) over the span's grid steps,
computed vectorised for every candidate at once. FEATURE_VERSION names the columns, their order and their meaning: a
learned model records the version it was trained on and is refused on another, so changing a definition, adding or
reordering a column means a new version. Features are float64 here (the hand-weighted scorer is an exact function
of each row); dumps store them at float32 precision.

Features carry raw signals; every hand-set weight, bonus and penalty lives in a scorer (HeuristicScorer). The only
pool statistics are the length-bucket columns (bucket_*): per CANDIDATE_LENGTHS bucket, the size and the medians of a
few signal means over the brief's stage-1 candidates (pool_stats), attached to any span by its own length."""
from __future__ import annotations

import math
import re
from typing import Iterator

import numpy as np

from vhe.config import CANDIDATE_LENGTHS
from vhe.query.candidates import cuts_inside
from vhe.query.refine import spans_edges_clean
from vhe.query.text_match import OCR_GENERIC, ocr_item_scores
from vhe.rank.items import assign, item_scores
from vhe.rank.signals import muted_steps, span_max, span_mean, span_min, span_top_mean
from vhe.signals.tags import LABELS

# v2 (2026-10-18): the index v3 framing, face, approach and on-screen text columns appended (FEATURES_V2) and the
# brief's visual cues appended to the query's fixed block (QUERY_V2); v1 is the prefix of both.
# v3 (2026-10-18): raw signals only. h_rel_* no longer hold the legibility bonus or the highlight-prior blend (both
# are raw columns now, the heuristic composes them); FEATURES_V3 appended (clean edges, legibility, highlight prior,
# quality components, length-bucket pool statistics); ocr_item_match has the one ignore rule (OCR_GENERIC); the
# query vector gains the highlight flag (QUERY_V3). The first 82 names are v2's, in v2's order.
# v4 (2026-10-18): reaction signals appended (FEATURES_V4: the audio reaction curve, the transcript's excited speech,
# quoted-line matches, their length-bucket medians) and the reaction flag to the query (QUERY_V4). The first 95 names
# are v3's, in v3's order.
FEATURE_VERSION = "v4"
QUERY_VERSION = "v4"
SOURCES = ("shot", "window", "peak", "model", "speech")  # vhe.query.candidates.generate
PERSONAS = ("p1-videographer", "p2-tiktok", "p3-newsedit", "p4-vlogger", "p5-marketing", "p6-casual")
SIGNALS = ("relevance", "quality", "energy", "steady", "speech")  # the weighted signals of ParsedQuery.weights


def tag_name(label: str) -> str:
    """The feature name of a zero-shot tag label ("close-up of a face" -> "tag_close_up_of_a_face")."""
    return "tag_" + _slug(label)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


FEATURES = (
    # Relevance to the brief, SigLIP only: the best shot item per step (or the content text), no avoid penalty.
    "rel_mean", "rel_top60", "rel_max",
    # The brief plan: centred relevance of the shot item the span serves (else of its best item; 0 without a shot
    # list), SigLIP only, and the mean avoid penalty (how far an avoid item beats the wanted relevance).
    "item_rel", "avoid_pen",
    # Non-commercial highlight saliency (CG-DETR), z-scored over the corpus; 0 where a file has none. The slot an
    # owned highlight head will fill.
    "nc_sal_mean", "nc_sal_top60", "nc_present",
    # The blended relevance (SigLIP and nc saliency averaged per step where a file has saliency, the best item, minus
    # the avoid penalty): per-step blending is not a function of the span statistics above, so it is its own column;
    # HeuristicScorer reads it. A learned model should not use these two (masked in training).
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
    *(tag_name(label) for label in LABELS),
)
# v2, appended. Index v3 grid keys over the span's steps, raw units: shot-scale margins in logits (scale_close >= 1
# close-up / fills the frame, scale_wide >= 2.5 wide), the largest face as a fraction of the frame, how centred it
# is (0..1, 0 without a face), the face count, the looming rate (> 0.3 approaching), confident OCR text area and
# confidence; ocr_item_match: the best share of a shot item's words read by OCR inside the span (0..1, OCR_GENERIC
# words never counted). A grid key an index lacks reads as an ordinary medium shot with no face or text (GRID_NEUTRAL).
FEATURES_V2 = (
    "scale_close_mean", "scale_close_max", "scale_wide_mean", "face_area_max_mean", "face_area_max_max",
    "face_center_mean", "face_n_mean", "approach_max", "text_area_mean", "text_conf_mean", "ocr_item_match",
)
# v3, appended. edges_clean: 1 when the span starts and ends on clean frames (vhe.query.refine.edges_clean; speech
# clips always 1: they end in word gaps). legibility_mean: confident legible text per step (text_conf where text was
# found, vhe.query.text_match.legibility_steps), whether or not the brief asks for it. highlight_prior_mean: the
# generic highlight prior (people, faces, dancing, approach; vhe.rank.brief_signals.highlight_prior), z units,
# whether or not the brief asks for the best moments. detail_z_mean: pooled robust z of the log re-blur ratio (0
# where unmeasured); framing_frac: steps with the soft bad-framing reason (camera at the ground or sky). The quality
# composite's other parts are blown_frac and steady_mean. bucket_*: the stage-1 pool's statistics of the span's
# length bucket (CANDIDATE_LENGTHS): its candidate count, and the medians of h_rel_mean, quality_mean, energy_mean,
# steady_mean, speech_frac, legibility_mean and highlight_prior_mean over it (how a typical candidate of this
# length scores: short windows on peaks have higher means than long ones).
BUCKET_COLS = {"bucket_h_rel": "h_rel_mean", "bucket_quality": "quality_mean", "bucket_energy": "energy_mean",
               "bucket_steady": "steady_mean", "bucket_speech": "speech_frac", "bucket_legibility": "legibility_mean",
               "bucket_highlight": "highlight_prior_mean"}
FEATURES_V3 = ("edges_clean", "legibility_mean", "highlight_prior_mean", "detail_z_mean", "framing_frac",
               "bucket_size", *BUCKET_COLS)
# v4, appended (vhe.signals.reaction; whatever the brief asks). reaction_mean / reaction_max: the audio reaction curve
# (loudness and onset bursts over the file's running baseline, robust z); excitement_mean: the transcript's excited
# speech (exclamations, laughter tokens, strong words, shouted words, repeats, speech-rate bursts; robust z, 0
# without a transcript); quote_match: the best share of a quoted line of the brief (ParsedQuery.lines) said inside
# the span (0..1, vhe.query.text_match.line_steps; 0 without lines); bucket_reaction / bucket_excitement: their
# length-bucket medians over the stage-1 pool, like the v3 bucket_* columns.
BUCKET_COLS_V4 = {"bucket_reaction": "reaction_mean", "bucket_excitement": "excitement_mean"}
FEATURES_V4 = ("reaction_mean", "reaction_max", "excitement_mean", "quote_match", *BUCKET_COLS_V4)
BUCKETS = {**BUCKET_COLS, **BUCKET_COLS_V4}  # every bucket_* column and its source
FEATURES = FEATURES + FEATURES_V2 + FEATURES_V3 + FEATURES_V4
INDEX = {name: i for i, name in enumerate(FEATURES)}
# The columns that describe a span's relevance to the brief: per shot item, item_features gives each item's own.
ITEM_COLS = ("rel_mean", "rel_top60", "rel_max", "item_rel", "avoid_pen", "h_rel_mean", "h_rel_top60",
             "ocr_item_match")
GRID_NEUTRAL = {"scale_close": -6.0, "scale_wide": 0.0, "face_area_max": 0.0, "face_center": 0.0, "face_n": 0.0,
                "approach": 0.0, "text_area": 0.0, "text_conf": 0.0}

QUERY_BASE = (
    "w_relevance", "w_quality", "w_energy", "w_steady", "w_speech",  # signal weights (mood, speech intent, plan)
    "mood_energetic", "mood_calm", "exhaustive", "has_text", "has_budget", "target_len", "log_target_len",
    "has_shots", "n_shots", "has_avoid", "has_topics", "need_live_audio",
)
# v2: the brief's visual cues (ParsedQuery.framing one-hot, want_faces, hook, legible_text, centered), appended to the
# fixed block; v3: the highlight flag ("the best moments", ParsedQuery.highlight). The persona one-hot stays last (its
# length depends on the scorer's persona list).
FRAMINGS = ("none", "close", "wide")
QUERY_V2 = (*(f"framing_{x}" for x in FRAMINGS), "want_faces", "hook", "legible_text", "centered")
QUERY_V3 = ("highlight",)
QUERY_V4 = ("reaction",)  # a reaction brief (ParsedQuery.reaction: "the funniest moments", "every jump scare")


def query_names(personas: tuple[str, ...] | list[str] = PERSONAS) -> tuple[str, ...]:
    return QUERY_BASE + QUERY_V2 + QUERY_V3 + QUERY_V4 + tuple(f"persona_{x}" for x in personas)


def target_len(p) -> float:
    """The clip length the brief aims at: the middle of its allowed range."""
    return (p.min_len + p.max_len) / 2


def query_vector(p, persona: str | None = None, personas: tuple[str, ...] | list[str] = PERSONAS,
                 weights: dict | None = None) -> np.ndarray:
    """(len(query_names(personas)),) float64 for a parsed brief (vhe.query.model.ParsedQuery). weights: the signal
    weights to score with when they are not the brief's (the opener/closer pools). An unknown persona is all zeros."""
    w = p.weights if weights is None else weights
    target = target_len(p)
    v = [*(float(w.get(k, 0.0)) for k in SIGNALS),
         float(p.mood == "energetic"), float(p.mood == "calm"), float(p.exhaustive), float(bool(p.text)),
         float(p.budget is not None), target, math.log(max(target, 1e-3)),
         float(bool(p.shots)), float(len(p.shots)), float(bool(p.avoid)), float(bool(p.topics)),
         float(p.need_live_audio)]
    v += [float(p.framing == x) for x in FRAMINGS]
    v += [float(bool(x)) for x in (p.want_faces, p.hook, p.legible_text, p.centered)]
    v += [float(bool(getattr(p, "highlight", False)))]
    v += [float(bool(getattr(p, "reaction", False)))]
    v += [float(persona == x) for x in personas]
    return np.array(v, dtype=np.float64)


def length_bucket(length) -> np.ndarray:
    """The CANDIDATE_LENGTHS bucket of each span length (HeuristicScorer's length-bucket rule)."""
    return np.searchsorted(np.asarray(CANDIDATE_LENGTHS, dtype=float), np.asarray(length, dtype=float) - 1e-6)


def pool_stats(X: np.ndarray) -> dict[str, np.ndarray]:
    """Per length bucket of a candidate pool's features X: "bucket_size" (count) and, for each BUCKET_COLS entry, the
    median of its source column (0 in an empty bucket)."""
    b = length_bucket(X[:, INDEX["length_s"]]) if len(X) else np.zeros(0, int)
    nb = len(CANDIDATE_LENGTHS) + 1
    out = {"bucket_size": np.bincount(b, minlength=nb).astype(float)}
    for name, src in BUCKETS.items():
        med = np.zeros(nb)
        for k in np.unique(b):
            med[k] = np.median(X[b == k, INDEX[src]])
        out[name] = med
    return out


def apply_pool(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Fill X's bucket_* columns from pool statistics (pool_stats), by each row's own length bucket; in place."""
    if len(X):
        b = length_bucket(X[:, INDEX["length_s"]])
        for name in ("bucket_size", *BUCKETS):
            X[:, INDEX[name]] = np.asarray(stats[name])[b]
    return X


def ocr_matrix(corpus, p, S: np.ndarray, E: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """(spans, items) share of each item's words read on screen inside each span (items: the shot list, else the
    content text; OCR_GENERIC never counted), or None without OCR or items."""
    items = list(p.shots) or ([p.text] if p.text else [])
    return ocr_item_scores(corpus, items, S, E, K, OCR_GENERIC) if items else None


def span_edges_clean(corpus, clean: np.ndarray | None, S: np.ndarray, E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """bool per span: vhe.query.refine.edges_clean on each file's clean-step mask (clean: per corpus step, e.g.
    vhe.query.refine.clean_steps(corpus.grid); None: everything clean), vectorised over spans."""
    if clean is None or not len(S):
        return np.ones(len(S), dtype=bool)
    c = corpus
    lo = c.bounds[K]
    return spans_edges_clean(clean, lo, c.bounds[K + 1] - lo, S - c.offsets[K], E - c.offsets[K], c.fps)


def featurize(corpus, sig: dict, S: np.ndarray, E: np.ndarray, K: np.ndarray, src: list[str], p,
              item_rel: np.ndarray | None = None, ocr: np.ndarray | None = None,
              stats: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """(n, len(FEATURES)) float64 for spans S..E (virtual seconds) of files K from generator `src`.

    sig: per-step signals. Required: relevance, quality, energy, steady, speech. Used when present: h_rel (the blended
    relevance, else relevance), siglip (else relevance), avoid_pen, nc_sal (NaN where absent), items / item_centre
    (shot list), motion_z / cam_speed_z / loud_z / onset_z, detail_z, framing, legibility, highlight, clean, reaction,
    excitement, quote, audio, unusable, blown, muted (else from the corpus grid). item_rel: (n, items) SigLIP
    relevance of each span to each shot item and ocr: (n, items) ocr_matrix, when the caller already has them. stats:
    the stage-1 pool's statistics (pool_stats); None: these spans are the pool."""
    c = corpus
    S, E, K = np.asarray(S, dtype=float), np.asarray(E, dtype=float), np.asarray(K, dtype=int)
    X = np.zeros((len(S), len(FEATURES)))
    if not len(S):
        return X
    i0, i1 = c.steps(S, E, K)

    def put(name: str, values) -> None:
        X[:, INDEX[name]] = values

    h = sig.get("h_rel")
    h = sig["relevance"] if h is None else h
    rel = sig.get("siglip")
    rel = sig["relevance"] if rel is None else rel
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
            name = tag_name(label)
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
    ocr = ocr_matrix(c, p, S, E, K) if ocr is None else ocr
    if ocr is not None and np.size(ocr):
        put("ocr_item_match", np.asarray(ocr).max(1))

    # v3: clean edges, legibility, highlight prior, quality components, pool statistics
    clean = span_edges_clean(c, sig.get("clean"), S, E, K)
    put("edges_clean", clean | np.array([x == "speech" for x in src], dtype=bool))
    for name, key in (("legibility_mean", "legibility"), ("highlight_prior_mean", "highlight"),
                      ("detail_z_mean", "detail_z"), ("framing_frac", "framing")):
        if sig.get(key) is not None:
            put(name, span_mean(np.asarray(sig[key], dtype=float), i0, i1))

    # v4: reaction, excited speech, quoted lines
    if sig.get("reaction") is not None:
        r = np.asarray(sig["reaction"], dtype=float)
        put("reaction_mean", span_mean(r, i0, i1))
        put("reaction_max", span_max(r, i0, i1))
    if sig.get("excitement") is not None:
        put("excitement_mean", span_mean(np.asarray(sig["excitement"], dtype=float), i0, i1))
    if sig.get("quote") is not None:
        put("quote_match", span_max(np.asarray(sig["quote"], dtype=float), i0, i1))
    return apply_pool(X, pool_stats(X) if stats is None else stats)


def item_features(corpus, sig: dict, S: np.ndarray, E: np.ndarray, K: np.ndarray, X: np.ndarray,
                  ocr: np.ndarray | None = None) -> Iterator[np.ndarray]:
    """Per shot item i (sig["items"]), the features X of spans S..E as if the brief named item i alone: ITEM_COLS
    from item i's SigLIP relevance (sig["items"]; h_rel_* too: the query-wide highlight saliency says nothing about
    one item, as in item assignment), its avoid penalty (sig["items_avoid"]) and ocr[:, i]; every other column (the
    pool statistics and nc saliency included) as in X. Yields one (n, F) array per item (Scorer.score_items reads them
    one at a time)."""
    items = sig.get("items")
    if items is None or not len(items) or not len(S):
        return
    c = corpus
    i0, i1 = c.steps(np.asarray(S, dtype=float), np.asarray(E, dtype=float), np.asarray(K, dtype=int))
    centre = sig.get("item_centre")
    centre = np.zeros(len(items)) if centre is None else np.asarray(centre)
    avoid = sig.get("items_avoid")
    for j, row in enumerate(items):
        Xi = X.copy()
        mean, top = span_mean(row, i0, i1), span_top_mean(row, i0, i1)
        Xi[:, INDEX["rel_mean"]] = Xi[:, INDEX["h_rel_mean"]] = mean
        Xi[:, INDEX["rel_top60"]] = Xi[:, INDEX["h_rel_top60"]] = top
        Xi[:, INDEX["rel_max"]] = span_max(row, i0, i1)
        Xi[:, INDEX["item_rel"]] = top - centre[j]
        Xi[:, INDEX["avoid_pen"]] = span_mean(avoid[j], i0, i1) if avoid is not None else 0.0
        Xi[:, INDEX["ocr_item_match"]] = ocr[:, j] if ocr is not None and np.size(ocr) else 0.0
        yield Xi


def grid_steps(corpus, sig: dict, key: str) -> np.ndarray:
    """Per-step float64 values of an index v3 grid key (from sig when it has them, else the corpus grid); missing
    steps and indexes without the key read as GRID_NEUTRAL[key]."""
    x = sig.get(key)
    x = corpus.grid.get(key) if x is None else x
    if x is None:
        return np.full(corpus.n, GRID_NEUTRAL[key])
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=GRID_NEUTRAL[key])
