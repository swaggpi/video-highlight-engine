"""A corpus: per-file indexes concatenated on one virtual timeline, so the query side has a single code path for one
video and for a folder of raw footage (a single video is a corpus of one).

File k occupies grid steps [bounds[k], bounds[k+1]) and virtual seconds [offsets[k], offsets[k] + durations[k]].
Offsets are whole grid steps (n_k / fps), so virtual time t is grid step t * fps everywhere. Every file boundary is
a hard cut; candidates and refinement run per file, so no clip ever spans two files."""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from vhe.index.store import Index, default_index_dir, usable_mask
from vhe.project import PROJECT_DIR, load_project

# Per-step arrays an older index may lack: fill values that mean "no defect" / "unknown" (NaN z-scores to 0).
_FILL = {"usable": True, "unusable": 0}


class Corpus:
    def __init__(self, indexes: list[Index], files: list[str] | None = None, sources: list[Path] | None = None,
                 root: Path | None = None) -> None:
        if not indexes:
            raise ValueError("no indexed videos")
        if len({ix.fps for ix in indexes}) > 1 or len({ix.meta.get("encoder") for ix in indexes}) > 1:
            raise ValueError("files were indexed with different grid rates or encoders; re-index them")
        self.members = indexes
        self.files = files or [ix.video.name for ix in indexes]  # labels: path relative to the project root
        self.sources = sources or [ix.video for ix in indexes]  # absolute source paths, for export and playback
        self.root = root
        self.fps = indexes[0].fps
        ns = np.array([ix.n for ix in indexes])
        self.bounds = np.concatenate([[0], np.cumsum(ns)])
        self.offsets = self.bounds[:-1] / self.fps
        self.durations = np.array([min(ix.duration, ix.n / self.fps) for ix in indexes])
        self.file_id = np.repeat(np.arange(len(indexes)), ns)
        self.meta = indexes[0].meta  # encoder, labels, logit scale/bias: shared (same encoder)
        self.grid = self._concat()

    def _concat(self) -> dict[str, np.ndarray]:
        keys = {k for ix in self.members for k, v in ix.grid.items() if np.ndim(v) >= 1 and len(v) == ix.n} - {"t"}
        grid = {"t": np.arange(self.n) / self.fps}
        for key in sorted(keys):
            ref = next(ix.grid[key] for ix in self.members if key in ix.grid)
            fill = _FILL.get(key, np.nan if np.issubdtype(ref.dtype, np.floating) else 0)
            grid[key] = np.concatenate([ix.grid[key] if key in ix.grid else
                                        np.full((ix.n, *ref.shape[1:]), fill, dtype=ref.dtype) for ix in self.members])
        return grid

    # ---- the Index interface the engine uses ----------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.bounds[-1])

    @property
    def duration(self) -> float:
        return self.n / self.fps

    @property
    def has_audio(self) -> bool:
        return any(ix.has_audio for ix in self.members)

    @functools.cached_property
    def audio_steps(self) -> np.ndarray:
        return np.repeat([ix.has_audio for ix in self.members], np.diff(self.bounds))

    @property
    def usable(self) -> np.ndarray:
        return usable_mask(self.grid, self.n)

    @functools.cached_property
    def cuts(self) -> list[float]:
        """Virtual: every member's cuts plus each file boundary."""
        out = [float(o) for o in self.offsets[1:]]
        for ix, off in zip(self.members, self.offsets):
            out += [off + c for c in ix.cuts]
        return sorted(out)

    @functools.cached_property
    def change_points(self) -> list[float]:
        return sorted(off + c for ix, off in zip(self.members, self.offsets) for c in ix.change_points)

    @functools.cached_property
    def vad(self) -> list[tuple[float, float]]:
        return [(off + s, off + e) for ix, off in zip(self.members, self.offsets) for s, e in ix.vad]

    @functools.cached_property
    def words(self) -> list[dict]:
        return [dict(w, start=off + w["start"], end=off + w["end"])
                for ix, off in zip(self.members, self.offsets) for w in ix.words]

    @functools.cached_property
    def text_items(self) -> list[dict]:
        """On-screen text (Index.text_items) on the virtual timeline: step and t shifted, plus the file index."""
        return [dict(it, step=it["step"] + int(b), t=off + it["t"], file=k)
                for k, (ix, b, off) in enumerate(zip(self.members, self.bounds, self.offsets)) for it in ix.text_items]

    @functools.cached_property
    def segments(self) -> list[dict]:
        """Transcript segments on the virtual timeline, each with its file index."""
        return [dict(s, start=off + s["start"], end=off + s["end"], file=k,
                     words=[dict(w, start=off + w["start"], end=off + w["end"]) for w in s.get("words") or []])
                for k, (ix, off) in enumerate(zip(self.members, self.offsets))
                for s in (ix.transcript or {}).get("segments", [])]

    @property
    def source_map(self) -> dict[str, Path]:
        """file label -> source path (what export needs)."""
        return dict(zip(self.files, self.sources))

    # ---- virtual <-> file-local --------------------------------------------------------------------------------
    def file_of(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.offsets, t + 1e-9, side="right") - 1, 0, len(self.members) - 1))

    def local(self, k: int, t: float) -> float:
        return round(float(t - self.offsets[k]), 3)

    def steps(self, s: np.ndarray, e: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid step range [i0, i1) of virtual spans in file k, never outside that file."""
        lo, hi = self.bounds[k], self.bounds[np.asarray(k) + 1]
        i0 = np.clip(np.floor(np.asarray(s) * self.fps + 1e-6).astype(int), lo, hi - 1)
        i1 = np.clip(np.ceil(np.asarray(e) * self.fps - 1e-6).astype(int), i0 + 1, hi)
        return i0, i1


def open_corpus(path: Path) -> Corpus:
    """A project folder (<DIR>/.vhe/project.json), a per-video index dir, or a video with its index next to it."""
    path = Path(path)
    if (path / PROJECT_DIR / "project.json").exists():
        manifest, indexes = load_project(path)
        ok = [e["rel_path"] for e in manifest["files"] if e["status"] == "ok"]
        if not ok:
            raise FileNotFoundError(f"no successfully indexed videos in {path}; see `vhe index {path}`")
        return Corpus(indexes, ok, [path.resolve() / r for r in ok], root=path)
    if (path / "meta.json").exists():
        return Corpus([Index(path)])
    if path.is_file() and (default_index_dir(path) / "meta.json").exists():
        return Corpus([Index(default_index_dir(path))], sources=[path.resolve()])
    raise FileNotFoundError(f"{path} is not indexed; run `vhe index {path}` first")
