"""Zero-shot tags from the cached frame embeddings (SigLIP sigmoid probabilities). No extra model."""
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


def label_texts() -> list[str]:
    return [PROMPT.format(label) for label in LABELS]


def tag_probs(emb: np.ndarray, label_emb: np.ndarray, scale: float, bias: float) -> np.ndarray:
    logits = emb.astype(np.float32) @ label_emb.T * scale + bias
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float16)
