"""SigLIP2 image and text embeddings (Apache-2.0). One encoder family for the whole index; the cached
vectors are only valid for the encoder that produced them, so its name is stored in the index."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Encoder:
    def __init__(self, name: str, device: str) -> None:
        from transformers import AutoModel, AutoProcessor

        self.name = name
        self.device = device
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(name)
        self.dtype = dtype

    @torch.no_grad()
    def images(self, paths: list[Path], batch_size: int = 32, progress=None) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            feats = self.model.get_image_features(**inputs)
            out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
            if progress:
                progress(min(i + batch_size, len(paths)), len(paths))
        return np.concatenate(out).astype(np.float16) if out else np.zeros((0, 1), np.float16)

    @torch.no_grad()
    def texts(self, texts: list[str]) -> np.ndarray:
        # SigLIP2 was trained on lower-cased text padded to 64 tokens.
        inputs = self.processor(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()

    def logit_params(self) -> tuple[float, float]:
        scale = float(self.model.logit_scale.exp().item())
        bias = float(self.model.logit_bias.item()) if hasattr(self.model, "logit_bias") else 0.0
        return scale, bias
