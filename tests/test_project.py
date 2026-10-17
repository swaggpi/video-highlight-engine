import json
import os
from pathlib import Path

import pytest

from vhe import project
from vhe.config import Config
from vhe.ingest.decode import Probe
from vhe.project import discover_videos, index_project, load_project, natural_key, order_key


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fake_probe(durations: dict, fail: set = frozenset(), dates: dict | None = None):
    def probe(path: Path) -> Probe:
        if path.name in fail:
            raise RuntimeError("moov atom not found")
        return Probe(duration=durations.get(path.name, 10.0), fps=25.0, width=64, height=36, color_transfer=None,
                     has_audio=path.name != "silent.mp4", recorded_at=(dates or {}).get(path.name))
    return probe


class FakeIndexer:
    def __init__(self, fail: set = frozenset()):
        self.calls, self.fail = [], fail

    def __call__(self, video: Path, out: Path, cfg: Config, progress=None, keep_cache=False):
        self.calls.append((video.name, cfg.asr_language))
        if video.name in self.fail:
            raise RuntimeError("decoder exploded")
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps({"language": cfg.asr_language, "usable_fraction": 0.9,
                                                   "has_audio": video.name != "silent.mp4"}))
        return out


def test_discover_recursive_skips_index_dirs(tmp_path):
    for rel in ["C0002.MP4", "C0010.mov", "sub/GX010012.MP4", "notes.txt", ".vhe/files/x/clip.mp4",
                "old.mp4.vhe/frames.mp4", "._C0002.MP4", "sub/clip.LRV"]:
        _touch(tmp_path / rel)
    found = [p.relative_to(tmp_path).as_posix() for p in discover_videos(tmp_path)]
    assert found == ["C0002.MP4", "C0010.mov", "sub/GX010012.MP4"]


def test_natural_and_recorded_order():
    assert sorted(["C10.mp4", "C2.mp4", "c1.mp4"], key=natural_key) == ["c1.mp4", "C2.mp4", "C10.mp4"]
    entries = [{"rel_path": "C2.mp4", "recorded_at": None}, {"rel_path": "C10.mp4", "recorded_at": None},
               {"rel_path": "Z.mp4", "recorded_at": "2024-06-01T10:00:00Z"},
               {"rel_path": "A.mp4", "recorded_at": "2024-06-01T11:00:00Z"}]
    assert [e["rel_path"] for e in sorted(entries, key=order_key)] == ["Z.mp4", "A.mp4", "C2.mp4", "C10.mp4"]


def test_index_project_statuses_and_incremental(tmp_path, monkeypatch):
    for name in ["C0001.mp4", "C0002.mp4", "tap.mp4", "broken.mp4", "crash.mp4", "silent.mp4"]:
        _touch(tmp_path / name)
    monkeypatch.setattr("vhe.ingest.decode.probe", _fake_probe(
        {"tap.mp4": 1.5}, fail={"broken.mp4"}, dates={"C0002.mp4": "2024-06-01T10:00:00Z"}))
    cfg = Config(asr_language="de", use_nc_highlight=False)
    idx = FakeIndexer(fail={"crash.mp4"})
    m = index_project(tmp_path, cfg, indexer=idx)

    status = {e["rel_path"]: e["status"] for e in m["files"]}
    assert status == {"C0001.mp4": "ok", "C0002.mp4": "ok", "tap.mp4": "too_short", "broken.mp4": "failed",
                      "crash.mp4": "failed", "silent.mp4": "ok"}
    assert m["files"][0]["rel_path"] == "C0002.mp4"  # dated files first, then natural name order
    assert sorted(c[0] for c in idx.calls) == ["C0001.mp4", "C0002.mp4", "crash.mp4", "silent.mp4"]
    assert all(lang == "de" for _, lang in idx.calls) and m["language"] == "de"
    errors = {e["rel_path"]: e.get("error", "") for e in m["files"]}
    assert "decoder exploded" in errors["crash.mp4"] and "probe failed" in errors["broken.mp4"]
    ok = next(e for e in m["files"] if e["rel_path"] == "C0001.mp4")
    assert ok["index_dir"].startswith(".vhe/files/") and ok["usable_fraction"] == 0.9 and ok["duration"] == 10.0
    assert json.loads((tmp_path / ".vhe" / "project.json").read_text()) == m

    # Second run: only the failed file is retried.
    idx2 = FakeIndexer()
    m2 = index_project(tmp_path, cfg, indexer=idx2)
    assert [c[0] for c in idx2.calls] == ["crash.mp4"]
    assert {e["rel_path"]: e["status"] for e in m2["files"]}["crash.mp4"] == "ok"

    # A changed file (size/mtime) and a new file are indexed; a removed file leaves the manifest.
    _touch(tmp_path / "C0001.mp4", b"longer")
    _touch(tmp_path / "C0003.mp4")
    (tmp_path / "silent.mp4").unlink()
    idx3 = FakeIndexer()
    m3 = index_project(tmp_path, cfg, indexer=idx3)
    assert sorted(c[0] for c in idx3.calls) == ["C0001.mp4", "C0003.mp4"]
    assert "silent.mp4" not in {e["rel_path"] for e in m3["files"]}

    # A different config re-indexes everything that is indexable.
    idx4 = FakeIndexer()
    index_project(tmp_path, Config(asr_language="en", use_nc_highlight=False), indexer=idx4)
    assert sorted(c[0] for c in idx4.calls) == ["C0001.mp4", "C0002.mp4", "C0003.mp4", "crash.mp4"]


def test_mtime_alone_triggers_reindex(tmp_path, monkeypatch):
    f = _touch(tmp_path / "a.mp4")
    monkeypatch.setattr("vhe.ingest.decode.probe", _fake_probe({}))
    cfg = Config(asr_language="en")
    index_project(tmp_path, cfg, indexer=FakeIndexer())
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    idx = FakeIndexer()
    index_project(tmp_path, cfg, indexer=idx)
    assert [c[0] for c in idx.calls] == ["a.mp4"]


def test_project_language_majority_and_rerun_votes(tmp_path, monkeypatch):
    for name in ["a.mp4", "b.mp4", "c.mp4", "silent.mp4"]:
        _touch(tmp_path / name)
    monkeypatch.setattr("vhe.ingest.decode.probe", _fake_probe({}))
    detections = {"a.mp4": ("de", 0.95), "b.mp4": ("de", 0.8), "c.mp4": ("en", 0.99)}
    asked = []

    def detector(path, cfg, device):
        asked.append(path.name)
        return detections[path.name]

    idx = FakeIndexer()
    m = index_project(tmp_path, Config(device="cpu"), indexer=idx, detector=detector)
    assert sorted(asked) == ["a.mp4", "b.mp4", "c.mp4"]  # files without audio are not asked
    assert m["language"] == "de" and {lang for _, lang in idx.calls} == {"de"}

    # A new English file does not flip the project: earlier files vote with their language.
    _touch(tmp_path / "d.mp4")
    detections["d.mp4"] = ("en", 0.99)
    idx2 = FakeIndexer()
    m2 = index_project(tmp_path, Config(device="cpu"), indexer=idx2, detector=detector)
    assert idx2.calls == [("d.mp4", "de")] and m2["language"] == "de"


def test_load_project_returns_ok_indexes(tmp_path, monkeypatch):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "tap.mp4")
    monkeypatch.setattr("vhe.ingest.decode.probe", _fake_probe({"tap.mp4": 1.0}))
    index_project(tmp_path, Config(asr_language="en"), indexer=FakeIndexer())
    loaded = []
    monkeypatch.setattr(project, "Index", lambda root: loaded.append(Path(root).name) or root)
    manifest, indexes = load_project(tmp_path)
    assert len(manifest["files"]) == 2 and len(indexes) == 1 and loaded[0].startswith("a.mp4-")
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "nothing")
