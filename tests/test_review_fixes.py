"""Regression tests for review findings: EDL source frame rates, index-version freshness, --budget dashes,
language-detection decode limit."""
from test_project import FakeIndexer, _fake_probe, _touch

from vhe import project
from vhe.config import Config
from vhe.project import index_project


def _edl_events(tmp_path, monkeypatch, fps_by_file, clips, tcs=None):
    from vhe.export import clips as ex
    from vhe.ingest.decode import Probe

    monkeypatch.setattr(ex, "probe", lambda p: Probe(duration=7200.0, fps=fps_by_file[p.name], width=64, height=36,
                                                     color_transfer=None, has_audio=True, recorded_at=None))
    monkeypatch.setattr(ex, "_source_tc", lambda p, base: (tcs or {}).get(p.name, 0))
    srcs = {n: tmp_path / n for n in fps_by_file}
    edl = ex.write_edl(clips, srcs, tmp_path / "e.edl").read_text().splitlines()
    return [line.split() for line in edl if line[:3].isdigit()]


def test_edl_source_times_use_true_fps_29_97(tmp_path, monkeypatch):
    ev = _edl_events(tmp_path, monkeypatch, {"a.mp4": 30000 / 1001},
                     [{"file": "a.mp4", "file_start": 3600.0, "file_end": 3601.0}])
    # round(3600 * 29.97) = 107892 frames = 00:59:56:12 on the 30-frame base (not 01:00:00:00).
    assert ev[0][4:] == ["00:59:56:12", "00:59:57:12", "01:00:00:00", "01:00:01:00"]


def test_edl_mixed_25p_and_30p_use_own_rate(tmp_path, monkeypatch):
    clips = [{"file": "a.mp4", "file_start": 10.0, "file_end": 12.0},
             {"file": "b.mp4", "file_start": 10.0, "file_end": 12.0}]
    ev = _edl_events(tmp_path, monkeypatch, {"a.mp4": 25.0, "b.mp4": 30.0}, clips, tcs={"b.mp4": 30 * 3600})
    assert ev[0][4:6] == ["00:00:10:00", "00:00:12:00"]  # 25p: 25-frame seconds
    assert ev[1][4:6] == ["01:00:10:00", "01:00:12:00"]  # 30p with 01:00:00:00 start timecode
    assert ev[0][6:] == ["01:00:00:00", "01:00:02:00"] and ev[1][6] == "01:00:02:00"  # one 25 fps record side


def test_index_version_bump_triggers_reindex(tmp_path, monkeypatch):
    _touch(tmp_path / "a.mp4")
    monkeypatch.setattr("vhe.ingest.decode.probe", _fake_probe({}))
    cfg = Config(asr_language="en")
    index_project(tmp_path, cfg, indexer=FakeIndexer())
    same = FakeIndexer()
    index_project(tmp_path, cfg, indexer=same)
    assert same.calls == []
    monkeypatch.setattr(project, "INDEX_VERSION", project.INDEX_VERSION + 1)
    bumped = FakeIndexer()
    index_project(tmp_path, cfg, indexer=bumped)
    assert [c[0] for c in bumped.calls] == ["a.mp4"]


def test_cli_budget_accepts_unicode_dashes():
    from vhe.cli import _budget

    assert _budget("60—90") == _budget("60–90") == _budget("60-90") == (60.0, 90.0) and _budget("75") == 75.0


def test_language_detection_decodes_only_the_start(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(project.subprocess, "run", lambda cmd, **kw: cmds.append(cmd))
    monkeypatch.setattr("vhe.signals.audio.load_audio", lambda p: None)
    monkeypatch.setattr("vhe.signals.audio.voice_activity", lambda y: [])
    monkeypatch.setattr("vhe.signals.speech.detect_language", lambda *a: ("de", 0.9))
    assert project.detect_file_language(tmp_path / "a.mp4", Config(), "cpu") == ("de", 0.9)
    assert cmds[0][cmds[0].index("-t") + 1] == str(project.LANG_DETECT_DECODE_S)
