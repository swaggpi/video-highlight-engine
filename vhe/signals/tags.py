"""Zero-shot tags and framing defects from the cached frame embeddings (SigLIP sigmoid probabilities / logits).
No extra model."""
from __future__ import annotations

import numpy as np

LABELS = [
    "beach", "ocean", "lake or river", "mountains", "forest", "field or meadow", "desert", "snow", "sky and clouds",
    "sunset", "night", "city street", "building interior", "home interior", "office", "restaurant or bar",
    "road and cars", "boat", "airplane view", "drone shot from above", "crowd", "group of people", "one person talking",
    "close-up of a face", "child", "dog", "cat", "bird", "animal", "food", "sports", "running", "swimming", "dancing",
    "concert or stage", "party", "screen or computer", "text or sign", "cartoon or animation",
]
PROMPT = "a photo of {}"

# Framing defects of raw handheld footage. Each score is a margin in logits: logsumexp over the defect prompts minus
# logsumexp over ordinary scene prompts, so a frame only counts as a defect when it looks more like the defect than
# like any normal shot (SigLIP's bias cancels; the scale stays). Calibrated with OBSTRUCTED_MARGIN and
# BAD_FRAMING_MARGIN in config.py.
OBSTRUCTED_PROMPTS = [
    "a blurry close-up of fabric covering the camera lens", "a hand covering the camera lens",
    "a close-up of the back of a person's shirt blocking the view", "a blurry close-up of clothing",
    "the inside of a pocket",
]
BAD_FRAMING_PROMPTS = [  # pocket shots at the ground, or the camera pointed straight up
    "a photo of the pavement at my feet", "a photo of asphalt road surface",
    "a blurry tilted photo of the street ground", "a photo of the ground taken by accident",
    "a photo looking straight up at a tall building", "a photo of the sky with nothing else",
    "a photo looking straight up at the sky",
]
SCENE_PROMPTS = [
    "a photo of a crowd of people", "a photo of a city street", "a photo of people at an event", "a photo of a person",
    "a photo of a landscape", "a photo of a room",
]


def label_texts() -> list[str]:
    return [PROMPT.format(label) for label in LABELS]


def tag_probs(emb: np.ndarray, label_emb: np.ndarray, scale: float, bias: float) -> np.ndarray:
    logits = emb.astype(np.float32) @ label_emb.T * scale + bias
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float16)


def defect_texts() -> list[str]:
    """Text prompts for defect_scores, in the order it expects their embeddings."""
    return OBSTRUCTED_PROMPTS + BAD_FRAMING_PROMPTS + SCENE_PROMPTS


def _logsumexp(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]


def defect_scores(emb: np.ndarray, text_emb: np.ndarray, scale: float, bias: float) -> dict[str, np.ndarray]:
    """{'obstruction', 'framing'} margins per frame (float32). text_emb: embeddings of defect_texts()."""
    logits = emb.astype(np.float32) @ np.asarray(text_emb, dtype=np.float32).T * scale + bias
    no, nf = len(OBSTRUCTED_PROMPTS), len(BAD_FRAMING_PROMPTS)
    scene = _logsumexp(logits[:, no + nf:])
    return {"obstruction": (_logsumexp(logits[:, :no]) - scene).astype(np.float32),
            "framing": (_logsumexp(logits[:, no:no + nf]) - scene).astype(np.float32)}
