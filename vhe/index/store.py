"""On-disk index: meta.json (scalars, cuts, change points, VAD, timings, licence flags), grid.npz (every
per-timestep array on the 3 fps grid) and speech.json (transcript). Vectors are tied to the encoder name
stored in meta; a different encoder means re-indexing."""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

INDEX_VERSION = 1


def usable_mask(grid: dict, n: int) -> np.ndarray:
    """bool per grid step; all True when an older index has no 'usable' array. Shared by Index and Corpus."""
    return grid["usable"].astype(bool) if "usable" in grid else np.ones(n, dtype=bool)


def default_index_dir(video: Path) -> Path:
    return video.parent / (video.name + ".vhe")


def save_index(root: Path, meta: dict, grid: dict[str, np.ndarray], transcript: dict | None) -> None:
    """Also records the source's size and mtime_ns, so callers can tell whether the index is current."""
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "grid.npz", **grid)
    speech = root / "speech.json"
    if transcript is not None:
        speech.write_text(json.dumps(transcript, ensure_ascii=False))
    else:
        speech.unlink(missing_ok=True)  # a transcript from an earlier run would no longer match
    src = Path(meta["video"]) if meta.get("video") else None
    if src and src.exists():
        st = src.stat()
        meta = {**meta, "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns}
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

    @functools.cached_property
    def words(self) -> list[dict]:
        if not self.transcript:
            return []
        return sorted((w for seg in self.transcript["segments"] for w in seg["words"]), key=lambda w: w["start"])

    @property
    def has_audio(self) -> bool:
        # Indexes before the project format also kept a copy in grid.npz.
        return bool(self.meta.get("has_audio", self.grid.get("has_audio", True)))

    @property
    def usable(self) -> np.ndarray:
        """bool per grid step: no absolute defect (signals/quality.usable_mask). All True for older indexes."""
        return usable_mask(self.grid, self.n)

    @property
    def unusable_reasons(self) -> np.ndarray:
        """uint8 bits per grid step; bit i = signals.quality.REASONS[i] (decode with quality.reason_names)."""
        return self.grid["unusable"] if "unusable" in self.grid else np.zeros(self.n, dtype=np.uint8)

    @property
    def recorded_at(self) -> str | None:
        return self.meta.get("recorded_at")
