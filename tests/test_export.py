"""EDL / CSV export (vhe.export.clips): source frame rates per file."""


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
