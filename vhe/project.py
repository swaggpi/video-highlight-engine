"""Projects: a folder of raw footage (10-40 camera and phone files) indexed file by file with `vhe index DIR`.

<DIR>/.vhe/project.json lists every video found (recursively) with its status; each indexed file gets the usual
per-file index in <DIR>/.vhe/files/<slug>/. Re-running only indexes new or changed files. Speech is transcribed
in one project language: --lang, else the majority of confident per-file detections.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from vhe.config import MIN_DURATION_S, VIDEO_EXTENSIONS, Config, resolve_device
from vhe.index.store import Index
from vhe.timing import hms

PROJECT_VERSION = 1
PROJECT_DIR = ".vhe"
# Runtime choices that do not change the index contents.
_UNHASHED = {"device", "hwaccel", "batch_size"}


def natural_key(s: str) -> list:
    """C2 before C10, GX010012 before GX020012."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def discover_videos(root: Path) -> list[Path]:
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        rel = p.relative_to(root).parts
        if any(part == PROJECT_DIR or part.endswith(".vhe") for part in rel[:-1]):
            continue
        # "._x.mp4": macOS metadata files on exFAT cards, not videos.
        if p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith("._") and p.is_file():
            out.append(p)
    return sorted(out, key=lambda p: natural_key(p.relative_to(root).as_posix()))


def slug(rel_path: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", rel_path).strip("._")[:80]
    return f"{safe}-{hashlib.sha1(rel_path.encode()).hexdigest()[:8]}"


def config_hash(cfg: Config) -> str:
    d = {k: v for k, v in cfg.to_dict().items() if k not in _UNHASHED}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def order_key(entry: dict) -> tuple:
    """Recording time first (undated files last), then natural name order."""
    return entry.get("recorded_at") is None, entry.get("recorded_at") or "", natural_key(entry["rel_path"])


def is_current(entry: dict | None, size: int, mtime_ns: int, chash: str, root: Path) -> bool:
    return bool(entry and entry.get("status") == "ok" and entry.get("size") == size
                and entry.get("mtime_ns") == mtime_ns and entry.get("config_hash") == chash
                and (root / entry["index_dir"] / "meta.json").exists())


def read_manifest(root: Path) -> dict | None:
    path = Path(root) / PROJECT_DIR / "project.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_manifest(root: Path, manifest: dict) -> None:
    path = Path(root) / PROJECT_DIR / "project.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written manifest


def detect_file_language(video: Path, cfg: Config, device: str) -> tuple[str | None, float]:
    """Language of a file's speech (VAD regions only), from a separate audio-only decode."""
    from vhe.signals.audio import load_audio, voice_activity
    from vhe.signals.speech import detect_language

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-map", "0:a:0", "-ac", "1",
                        "-ar", "16000", str(wav)], check=True)
        y = load_audio(wav)
    return detect_language(y, voice_activity(y), cfg.asr_model, device)


def project_language(cfg: Config, todo: list[tuple[Path, dict]], kept: list[dict], say,
                     detector: Callable) -> str | None:
    """--lang if given; else the majority of confident detections over the files to index (files kept from an
    earlier run vote with the language they were transcribed in). None = each file detects its own."""
    from vhe.signals.speech import majority_language

    if cfg.asr_language or cfg.asr_model == "none":
        return cfg.asr_language
    speech_todo = [(p, e) for p, e in todo if e["has_audio"]]
    votes = [(e["language"], 1.0) for e in kept if e.get("language")]
    device = resolve_device(cfg.device)
    for k, (path, entry) in enumerate(speech_todo, 1):
        try:
            lang, prob = detector(path, cfg, device)
        except Exception as exc:  # the file's own indexing will report real problems
            lang, prob = None, 0.0
            say(f"language detection failed for {entry['rel_path']}: {exc}")
        say(f"language [{k}/{len(speech_todo)}] {entry['rel_path']}: {lang or '-'} ({prob:.2f})")
        votes.append((lang, prob))
    lang = majority_language(votes)
    if speech_todo:
        say(f"project language: {lang or 'none confident, detecting per file'}")
    return lang


def index_project(root: Path, cfg: Config | None = None, progress: Callable[[str], None] | None = None,
                  keep_cache: bool = False, indexer: Callable | None = None,
                  detector: Callable | None = None) -> dict:
    """Index every video under `root` that is new or changed; returns the manifest written to .vhe/project.json.
    A file that fails is recorded as failed and the batch continues."""
    from vhe.ingest.decode import probe

    if indexer is None:
        from vhe.pipeline import build_index as indexer
    cfg = cfg or Config()
    root = Path(root).resolve()
    say = progress or (lambda msg: None)
    chash = config_hash(cfg)
    old = {e["rel_path"]: e for e in (read_manifest(root) or {}).get("files", [])}
    entries: dict[str, dict] = {}
    todo: list[tuple[Path, dict]] = []
    kept: list[dict] = []
    for path in discover_videos(root):
        rel = path.relative_to(root).as_posix()
        st = path.stat()
        if is_current(old.get(rel), st.st_size, st.st_mtime_ns, chash, root):
            entries[rel] = old[rel]
            kept.append(old[rel])
            continue
        entry = {"rel_path": rel, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "config_hash": chash,
                 "duration": None, "recorded_at": None, "has_audio": None, "language": None, "status": "failed",
                 "usable_fraction": None, "index_dir": f"{PROJECT_DIR}/files/{slug(rel)}"}
        entries[rel] = entry
        try:
            p = probe(path)
        except Exception as exc:
            entry["error"] = f"probe failed: {exc}"
            continue
        entry.update(duration=round(p.duration, 3), recorded_at=p.recorded_at, has_audio=p.has_audio)
        if p.duration < MIN_DURATION_S:
            entry["status"] = "too_short"
            continue
        entry["error"] = "not indexed yet"  # stays failed if the run is interrupted
        todo.append((path, entry))
    say(f"{len(entries)} videos: {len(kept)} up to date, {len(todo)} to index, "
        f"{sum(e['status'] == 'too_short' for e in entries.values())} too short, "
        f"{len(entries) - len(kept) - len(todo) - sum(e['status'] == 'too_short' for e in entries.values())} "
        f"unreadable")

    manifest = {"version": PROJECT_VERSION, "config_hash": chash, "language": None, "files": []}

    def save() -> None:
        manifest["files"] = sorted(entries.values(), key=order_key)
        write_manifest(root, manifest)

    todo.sort(key=lambda t: order_key(t[1]))
    manifest["language"] = project_language(cfg, todo, kept, say, detector or detect_file_language)
    file_cfg = dataclasses.replace(cfg, asr_language=manifest["language"])
    save()
    total = sum(e["duration"] for _, e in todo)
    done_s, t0 = 0.0, time.perf_counter()
    for k, (path, entry) in enumerate(todo, 1):
        elapsed = time.perf_counter() - t0
        eta = f", eta {hms(elapsed / done_s * (total - done_s))}" if done_s > 0 else ""
        say(f"[{k}/{len(todo)}] {entry['rel_path']} ({hms(entry['duration'])} of video{eta})")
        out = root / entry["index_dir"]
        try:
            indexer(path, out, file_cfg, progress=lambda m: say(f"  {m}"), keep_cache=keep_cache)
            meta = json.loads((out / "meta.json").read_text())
            entry.update(status="ok", language=meta.get("language"), usable_fraction=meta.get("usable_fraction"),
                         has_audio=meta.get("has_audio", entry["has_audio"]))
            entry.pop("error", None)
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
            say(f"  FAILED: {entry['error']}")
        done_s += entry["duration"]
        save()
    return manifest


def load_project(root: Path) -> tuple[dict, list[Index]]:
    """(manifest, per-file Index for every status-ok file, in manifest order)."""
    root = Path(root)
    manifest = read_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"no project index in {root}; run `vhe index {root}` first")
    return manifest, [Index(root / e["index_dir"]) for e in manifest["files"] if e["status"] == "ok"]
