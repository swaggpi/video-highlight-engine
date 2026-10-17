"""On-disk index: meta.json (scalars, cuts, change points, VAD, timings, licence flags), grid.npz (every
per-timestep array on the 3 fps grid) and speech.json (transcript). Vectors are tied to the encoder name
stored in meta; a different encoder means re-indexing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INDEX_VERSION = 1


def default_index_dir(video: Path) -> Path:
    return video.parent / (video.name + ".vhe")


def save_index(root: Path, meta: dict, grid: dict[str, np.ndarray], transcript: dict | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "grid.npz", **grid)
    if transcript is not None:
        (root / "speech.json").write_text(json.dumps(transcript, ensure_ascii=False))
    (root / "meta.json").write_text(json.dumps({"version": INDEX_VERSION, **meta}, indent=1, ensure_ascii=False))


class Index:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        if self.meta.get("version") != INDEX_VERSION:
            raise ValueError(f"index version {self.meta.get('version')} != {INDEX_VERSION}; re-index the video")
        with np.load(self.root / "grid.npz") as g:
            self.grid = {k: g[k] for k in g.files}
        speech = self.root / "speech.json"
        self.transcript = json.loads(speech.read_text()) if speech.exists() else None

    @property
    def video(self) -> Path:
        return Path(self.meta["video"])

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    @property
    def duration(self) -> float:
        return float(self.meta["duration"])

    @property
    def n(self) -> int:
        return len(self.grid["t"])

    @property
    def cuts(self) -> list[float]:
        return self.meta["cuts"]

    @property
    def change_points(self) -> list[float]:
        return self.meta.get("change_points", [])

    @property
    def vad(self) -> list[tuple[float, float]]:
        return [tuple(x) for x in self.meta.get("vad", [])]

    @property
    def words(self) -> list[dict]:
        if not self.transcript:
            return []
        return [w for seg in self.transcript["segments"] for w in seg["words"]]
