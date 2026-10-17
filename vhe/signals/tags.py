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


# Shot scale, the same margin construction against ordinary shots (calibrated with SCALE_CLOSE_MARGIN and
# SCALE_WIDE_MARGIN in config.py). The close set needs the flag/banner prompt: SigLIP does not call a giant flag
# filling the frame a close-up; banners seen at mid distance are in its reference set so they do not count.
CLOSE_PROMPTS = [
    "an extreme close-up photo", "a close-up photo of a person's face",
    "a close-up of a large object filling the entire frame",
    "a huge flag or banner right in front of the camera filling the frame",
    "a close-up shot of a person's head and shoulders",
]
CLOSE_REF_PROMPTS = [
    "a medium shot of people", "a medium shot of a person", "a photo of people at an event",
    "a photo of a group of people", "a photo of a city street", "a photo looking up at buildings and trees",
    "a photo of a street parade", "a photo", "a photo of a crowd holding signs and banners",
    "a medium shot of people holding a banner",
]
WIDE_PROMPTS = [
    "a wide establishing shot of a street", "a long shot of a crowd seen from far away",
    "a wide angle view of a whole city street", "a wide shot showing the whole scene from a distance",
    "a view down a long avenue with buildings on both sides",
]
WIDE_REF_PROMPTS = [
    "a medium shot of people", "a medium shot of a person", "a photo of people at an event",
    "a photo of a group of people", "a photo of a person walking", "a photo",
]
_SCALE_SETS = (CLOSE_PROMPTS, CLOSE_REF_PROMPTS, WIDE_PROMPTS, WIDE_REF_PROMPTS)


def scale_texts() -> list[str]:
    """Text prompts for scale_scores, in the order it expects their embeddings."""
    return [p for group in _SCALE_SETS for p in group]


def scale_scores(emb: np.ndarray, text_emb: np.ndarray, scale: float, bias: float) -> dict[str, np.ndarray]:
    """{'scale_close', 'scale_wide'} margins per frame (float32, logits). text_emb: embeddings of scale_texts()."""
    logits = emb.astype(np.float32) @ np.asarray(text_emb, dtype=np.float32).T * scale + bias
    edges = np.cumsum([0] + [len(g) for g in _SCALE_SETS])
    close, close_ref, wide, wide_ref = (_logsumexp(logits[:, a:b]) for a, b in zip(edges[:-1], edges[1:]))
    return {"scale_close": (close - close_ref).astype(np.float32), "scale_wide": (wide - wide_ref).astype(np.float32)}
