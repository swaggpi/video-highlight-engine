"""The ranking seam: a Scorer maps candidate features X (vhe.rank.features.featurize) and the query vector q
(query_vector) to one score per candidate. The engine ranks every gated candidate with score(), refines the edges
of the best, featurises the refined spans again and asks rescore() for the totals selection runs on.

HeuristicScorer is the hand-weighted total the engine has always used, plus the visual-cue bonus (features v2).
LinearRankScorer is a learned model in a JSON file (format "vhe-linear-rank/1", trained outside the package):
standardised features plus their products with a few query entries, one weight each; a model of another feature or
query version (v1) is refused. load_scorer picks one from a CLI/API argument."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from vhe.query.candidates import LENGTHS
from vhe.rank.features import (
    FEATURE_VERSION,
    FEATURES,
    FEATURES_V2,
    INDEX,
    PERSONAS,
    QUERY_VERSION,
    SIGNALS,
    query_names,
)

CUT_PENALTY = 0.15
LENGTH_PRIOR = 0.15
SPEECH_BONUS = 0.3  # utterance-aligned candidates beat windows that start mid-sentence, scaled by speech weight
BUCKET_MIN = 5  # length buckets smaller than this are not median-centred (a singleton would become exactly 0)
LINEAR_FORMAT = "vhe-linear-rank/1"
Q_SMALL = tuple(f"w_{k}" for k in SIGNALS)  # query entries a linear model crosses with every feature by default

# ---- visual-cue bonuses (features v2; ParsedQuery framing / want_faces / centered / legible_text) ------------------
# Score units: about one robust z of energy or one relevance unit. Each term is a 0..1 ramp of a span feature times
# its bonus, so a brief without the cue (or an index without the signal: features.GRID_NEUTRAL) gets exactly 0.
# Ramps calibrated 2026-10-18 on the irl-event dev footage (6039 steps; festival-commons held out):
#   scale_close (logits): median step -6.3, 75th pct -4.0, 95th -0.3; close-ups / a flag filling the frame >= 1.0.
#   scale_wide: median 1.9 (mostly street wides), establishing views 2.5..6.2.
#   face_area_max: faces are small in event footage (95th pct 0.36 % of the frame, 0.13 % of steps above 3 %).
#   approach: 99th pct 0.69 per s; the giant flag coming at the lens 0.3..1.2.
CLOSE_RAMP = (-2.0, 2.0)  # scale_close_mean: 0 below the top ~10 % of steps (scaffolding at the lens: -1.3), 1 at
#                            a clear close-up (the giant flag at the lens 2.3..5.3)
WIDE_RAMP = (1.0, 5.0)  # scale_wide_mean: 0 below an ordinary street view, 1 at an establishing wide
FACE_RAMP = (0.002, 0.04)  # face_area_max_mean (log ramp): a face in the crowd -> a face filling a good part of it
APPROACH_FULL = 0.6  # approach_max at which the looming term is full
CLOSE_BONUS = 0.8  # framing close: shot scale
FACE_BONUS = 0.5  # framing close or want_faces: the largest face
APPROACH_BONUS = 0.5  # framing close: something growing toward the lens
CENTER_BONUS = 0.3  # centered: the largest face near the frame centre (subjects without a face are not measured)
WIDE_BONUS = 0.8  # framing wide: shot scale
WIDE_CLOSE_PENALTY = 0.6  # framing wide: a close-up costs this at full close score
OCR_MATCH_BONUS = 0.6  # legible_text: an item's words read by OCR in the span (the legibility prior is in relevance)
# The hook slot (vhe.engine): the opener is drawn from a pool scored with these weights and framing "close". No
# steadiness weight: a hook moves (the flag swept at the lens reads steady -3 from the motion it causes); the
# usable gate still drops covered, blurred and very shaky spans, and EDGE_PENALTY unclean edges.
HOOK_WEIGHTS = {"relevance": 0.5, "energy": 0.8, "steady": 0.0, "quality": 0.3}


def _ramp(x, lo: float, hi: float):
    return np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def cue_terms(cols: dict) -> dict:
    """0..1 ramps of the v2 span features (or of per-step values under the same names)."""
    face = np.maximum(np.asarray(cols["face_area_max_mean"], dtype=float), 1e-9)
    return {"close": _ramp(cols["scale_close_mean"], *CLOSE_RAMP), "wide": _ramp(cols["scale_wide_mean"], *WIDE_RAMP),
            "face": _ramp(np.log(face), *np.log(FACE_RAMP)),
            "approach": _ramp(cols["approach_max"], 0.0, APPROACH_FULL),
            "center": np.clip(np.asarray(cols["face_center_mean"], dtype=float), 0.0, 1.0),
            "ocr": np.clip(np.asarray(cols["ocr_item_match"], dtype=float), 0.0, 1.0)}


def cue_bonus(cols: dict, framing: str, want_faces: bool, centered: bool, legible_text: bool):
    """The heuristic's visual-cue bonus for spans (cols: v2 feature columns by name) or steps (the same names over
    per-step values); 0 for a brief without cues."""
    t = cue_terms(cols)
    out = np.zeros(np.shape(t["close"]))
    if framing == "close":
        out += CLOSE_BONUS * t["close"] + APPROACH_BONUS * t["approach"]
    if framing == "close" or want_faces:
        out += FACE_BONUS * t["face"]
    if framing == "wide":
        out += WIDE_BONUS * t["wide"] - WIDE_CLOSE_PENALTY * t["close"]
    if centered:
        out += CENTER_BONUS * t["center"]
    if legible_text:
        out += OCR_MATCH_BONUS * t["ocr"]
    return out


@runtime_checkable
class Scorer(Protocol):
    name: str
    feature_version: str
    personas: tuple[str, ...]  # the persona one-hot the query vector is built with

    def score(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        """(n,) scores for candidates X (n, F) under query q; higher is better. Used to pick the shortlist."""
        ...

    def rescore(self, s: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray, q: np.ndarray) -> np.ndarray:
        """(m,) final scores of m shortlisted candidates after edge refinement: s their score() values, X_pre their
        features as generated, X_post the features of the refined spans."""
        ...


def _col(X: np.ndarray, name: str) -> np.ndarray:
    return X[:, INDEX[name]]


class HeuristicScorer:
    """The hand-weighted total, exactly as the engine computed it before the seam: the weighted means of relevance
    (SigLIP and nc saliency averaged per step, minus the avoid penalty), quality, energy, steadiness and speech,
    centred on the median of each candidate length bucket (short windows on a peak have higher means than long
    ones), a mild prior towards the target length, a penalty per cut inside and a bonus for utterance-aligned spans,
    plus (features v2) the visual-cue bonus of a brief with framing / faces / centring / readable-text cues
    (cue_bonus; exactly 0 without cues). The final total swaps the relevance mean for the mean of the best 60 % of
    steps, on the span as generated."""

    name = "heuristic"
    feature_version = FEATURE_VERSION
    personas = PERSONAS

    def __init__(self) -> None:
        self._q = {n: i for i, n in enumerate(query_names(self.personas))}

    def _w(self, q: np.ndarray, key: str) -> float:
        return float(q[self._q[f"w_{key}"]])

    def score(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        means = {"relevance": _col(X, "h_rel_mean"), "quality": _col(X, "quality_mean"),
                 "energy": _col(X, "energy_mean"), "steady": _col(X, "steady_mean"),
                 "speech": _col(X, "speech_frac") * 2 - 1}
        total = sum(self._w(q, k) * means[k] for k in SIGNALS)
        length = _col(X, "length_s")
        bucket = np.searchsorted(np.asarray(LENGTHS, dtype=float), length - 1e-6)
        for b in np.unique(bucket):
            if np.count_nonzero(bucket == b) >= BUCKET_MIN:
                total[bucket == b] -= np.median(total[bucket == b])
        target = float(q[self._q["target_len"]])
        total += -LENGTH_PRIOR * np.abs(length - target) / max(target, 1.0) - CUT_PENALTY * _col(X, "n_cuts_inside")
        total += SPEECH_BONUS * max(0.0, self._w(q, "speech")) * _col(X, "src_speech")
        return total + self.cue_bonus(X, q)

    def cue_bonus(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        """cue_bonus from the v2 feature columns and the query's cue entries (0 for a brief without cues)."""
        flag = {k: float(q[self._q[k]]) > 0.5 for k in ("framing_close", "framing_wide", "want_faces", "centered",
                                                          "legible_text")}
        if not any(flag.values()):
            return np.zeros(len(X))
        framing = "close" if flag["framing_close"] else "wide" if flag["framing_wide"] else "none"
        return cue_bonus({n: _col(X, n) for n in FEATURES_V2}, framing, flag["want_faces"], flag["centered"],
                         flag["legible_text"])

    def rescore(self, s: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray, q: np.ndarray) -> np.ndarray:
        """The span as generated, with exact relevance; the refined edges do not change the total."""
        return s + self._w(q, "relevance") * (_col(X_pre, "h_rel_top60") - _col(X_pre, "h_rel_mean"))


class LinearRankScorer:
    """score = w . phi(x, q), phi = [z, outer(z, q_small)], z = (x - mu) / sd. A model JSON:
    {"format": "vhe-linear-rank/1", "feature_version", "query_version", "features": [names, optional check],
     "personas": [...], "q_small": [query names, default the signal weights], "mu": [F], "sd": [F],
     "w": [F] or [F * (1 + len(q_small))], "mask": [F] optional 0/1, "trained_on", "created"}.
    mask: the features the model uses; a masked-out column is ignored (its z is 0), whatever its value."""

    def __init__(self, mu, sd, w, personas=PERSONAS, q_small=Q_SMALL, name: str = "linear",
                 feature_version: str = FEATURE_VERSION, query_version: str = QUERY_VERSION,
                 meta: dict | None = None, mask=None) -> None:
        if feature_version != FEATURE_VERSION:
            raise ValueError(f"model features {feature_version!r}, this engine computes {FEATURE_VERSION!r}")
        if query_version != QUERY_VERSION:
            raise ValueError(f"model query vector {query_version!r}, this engine builds {QUERY_VERSION!r}")
        F = len(FEATURES)
        self.mu, self.sd = np.asarray(mu, dtype=float), np.asarray(sd, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.personas = tuple(personas)
        self.q_small = tuple(q_small)
        names = query_names(self.personas)
        missing = [n for n in self.q_small if n not in names]
        if missing:
            raise ValueError(f"q_small names not in the query vector: {missing}")
        self._qi = [names.index(n) for n in self.q_small]
        if self.mu.shape != (F,) or self.sd.shape != (F,):
            raise ValueError(f"mu and sd need {F} entries (features {FEATURE_VERSION})")
        self.mask = None if mask is None else np.asarray(mask, dtype=float)
        if self.mask is not None and (self.mask.shape != (F,) or not np.isin(self.mask, (0.0, 1.0)).all()):
            raise ValueError(f"mask needs {F} entries of 0 or 1")
        if len(self.w) not in (F, F * (1 + len(self.q_small))):
            raise ValueError(f"w needs {F} or {F * (1 + len(self.q_small))} entries, got {len(self.w)}")
        self.name, self.feature_version, self.query_version = name, feature_version, query_version
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path | str) -> "LinearRankScorer":
        path = Path(path)
        m = json.loads(path.read_text())
        if m.get("format") != LINEAR_FORMAT:
            raise ValueError(f"{path}: format {m.get('format')!r}, expected {LINEAR_FORMAT!r}")
        fv, qv = m.get("feature_version"), m.get("query_version")
        if fv != FEATURE_VERSION or qv != QUERY_VERSION:
            raise ValueError(f"{path}: model trained on features {fv!r} / query vector {qv!r}; this engine computes "
                             f"features {FEATURE_VERSION!r} / query {QUERY_VERSION!r}. Retrain it on "
                             f"{FEATURE_VERSION} candidate dumps (tools/train) or use --scorer heuristic")
        if "features" in m and list(m["features"]) != list(FEATURES):
            raise ValueError(f"{path}: feature names differ from features {FEATURE_VERSION}")
        return cls(m["mu"], m["sd"], m["w"], m.get("personas", PERSONAS), m.get("q_small", Q_SMALL),
                   name=f"linear:{path.name}", feature_version=m.get("feature_version"),
                   query_version=m.get("query_version"),
                   meta={k: m.get(k) for k in ("trained_on", "created")}, mask=m.get("mask"))

    def to_json(self) -> dict:
        return {"format": LINEAR_FORMAT, "feature_version": self.feature_version, "query_version": self.query_version,
                "features": list(FEATURES), "personas": list(self.personas), "q_small": list(self.q_small),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(),
                **({} if self.mask is None else {"mask": self.mask.astype(int).tolist()}), **self.meta}

    def phi(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        z = np.nan_to_num((np.asarray(X, dtype=float) - self.mu) / np.where(self.sd > 0, self.sd, 1.0))
        if self.mask is not None:
            z = z * self.mask
        if len(self.w) == z.shape[1]:
            return z
        qs = np.asarray(q, dtype=float)[self._qi]
        return np.concatenate([z, (z[:, :, None] * qs[None, None, :]).reshape(len(z), -1)], axis=1)

    def score(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        return self.phi(X, q) @ self.w

    def rescore(self, s: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray, q: np.ndarray) -> np.ndarray:
        return self.score(X_post, q)


def load_scorer(spec: "Scorer | str | Path | None" = None) -> Scorer:
    """None or "heuristic": the hand-weighted scorer; a path: a learned model file (format by its "format" key)."""
    if spec is None or spec == "heuristic":
        return HeuristicScorer()
    if isinstance(spec, Scorer):
        return spec
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"scorer {spec!r}: not 'heuristic' and no such model file")
    fmt = json.loads(path.read_text()).get("format")
    if fmt == LINEAR_FORMAT:
        return LinearRankScorer.load(path)
    raise ValueError(f"{path}: unknown scorer format {fmt!r}")
