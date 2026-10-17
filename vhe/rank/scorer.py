"""The ranking seam: a Scorer maps candidate features X (vhe.rank.features.featurize) and the query vector q
(query_vector) to one score per candidate. The engine ranks every gated candidate with score(), refines the edges
of the best, featurises the refined spans again and asks rescore() for the totals selection runs on.

HeuristicScorer is the hand-weighted total the engine has always used. LinearRankScorer is a learned model in a JSON
file (format "vhe-linear-rank/1", trained outside the package): standardised features plus their products with a
few query entries, one weight each. load_scorer picks one from a CLI/API argument."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from vhe.query.candidates import LENGTHS
from vhe.rank.features import FEATURE_VERSION, FEATURES, INDEX, PERSONAS, QUERY_VERSION, SIGNALS, query_names

CUT_PENALTY = 0.15
LENGTH_PRIOR = 0.15
SPEECH_BONUS = 0.3  # utterance-aligned candidates beat windows that start mid-sentence, scaled by speech weight
BUCKET_MIN = 5  # length buckets smaller than this are not median-centred (a singleton would become exactly 0)
LINEAR_FORMAT = "vhe-linear-rank/1"
Q_SMALL = tuple(f"w_{k}" for k in SIGNALS)  # query entries a linear model crosses with every feature by default


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
    ones), a mild prior towards the target length, a penalty per cut inside and a bonus for utterance-aligned spans.
    The final total swaps the relevance mean for the mean of the best 60 % of steps, on the span as generated."""

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
        return total

    def rescore(self, s: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray, q: np.ndarray) -> np.ndarray:
        """The span as generated, with exact relevance; the refined edges do not change the total."""
        return s + self._w(q, "relevance") * (_col(X_pre, "h_rel_top60") - _col(X_pre, "h_rel_mean"))


class LinearRankScorer:
    """score = w . phi(x, q), phi = [z, outer(z, q_small)], z = (x - mu) / sd. A model JSON:
    {"format": "vhe-linear-rank/1", "feature_version", "query_version", "features": [names, optional check],
     "personas": [...], "q_small": [query names, default the signal weights], "mu": [F], "sd": [F],
     "w": [F] or [F * (1 + len(q_small))], "trained_on", "created"}."""

    def __init__(self, mu, sd, w, personas=PERSONAS, q_small=Q_SMALL, name: str = "linear",
                 feature_version: str = FEATURE_VERSION, query_version: str = QUERY_VERSION,
                 meta: dict | None = None) -> None:
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
        if "features" in m and list(m["features"]) != list(FEATURES):
            raise ValueError(f"{path}: feature names differ from features {FEATURE_VERSION}")
        return cls(m["mu"], m["sd"], m["w"], m.get("personas", PERSONAS), m.get("q_small", Q_SMALL),
                   name=f"linear:{path.name}", feature_version=m.get("feature_version"),
                   query_version=m.get("query_version"),
                   meta={k: m.get(k) for k in ("trained_on", "created")})

    def to_json(self) -> dict:
        return {"format": LINEAR_FORMAT, "feature_version": self.feature_version, "query_version": self.query_version,
                "features": list(FEATURES), "personas": list(self.personas), "q_small": list(self.q_small),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(), **self.meta}

    def phi(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        z = np.nan_to_num((np.asarray(X, dtype=float) - self.mu) / np.where(self.sd > 0, self.sd, 1.0))
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
