"""The UI server's clip previews (/preview): span validation, cache key, the same allow-list as /media, and a real
VP9/Opus transcode of a short generated clip (skipped when ffmpeg lacks the encoders)."""
import http.client
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
from http.server import ThreadingHTTPServer

import pytest

from vhe.export import preview
from vhe.ui import server


@pytest.fixture
def ui(tmp_path):
    """A running UI server whose only allowed source is tmp/src/clip.mp4 (a stand-in file; the transcode test makes
    a real one): (port, state, source)."""
    src = tmp_path / "src" / "clip.mp4"
    src.parent.mkdir()
    src.write_bytes(b"\0" * 1024)
    state = server.State("cpu")
    state.media.add(src.resolve())
    state.previews[src.resolve()] = tmp_path / "proj" / ".vhe" / "previews"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = server.make_handler(state, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield port, state, src.resolve()
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(port: int, path: str, host: str | None = None, headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("GET", path, headers={"Host": host or f"127.0.0.1:{port}", **(headers or {})})
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), body


def _q(**kw) -> str:
    return "/preview?" + urllib.parse.urlencode(kw)


def test_span_validation():
    assert preview.span("1.25", "3") == (1.25, 3.0)
    assert preview.span(0, 0.0004 + 1) == (0.0, 1.0)  # milliseconds
    for start, end in [("abc", "2"), ("1", ""), ("nan", "2"), ("1", "inf"), ("-1", "2"), ("3", "3"), ("4", "2"),
                       ("0", str(preview.MAX_PREVIEW_S + 1))]:
        with pytest.raises(ValueError):
            preview.span(start, end)


def test_preview_key_tracks_source_span_and_settings(tmp_path, monkeypatch):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    key = preview.preview_key(src, 1.0, 2.5)
    assert len(key) == 32 and preview.preview_key(src, 1.0, 2.5) == key
    assert preview.preview_key(src, 1.0001, 2.5) == key  # the span counts in milliseconds
    assert preview.preview_key(src, 1.0, 2.6) != key and preview.preview_key(src, 1.1, 2.5) != key
    assert preview.preview_path(tmp_path / "c", src, 1.0, 2.5) == tmp_path / "c" / f"{key}.webm"
    monkeypatch.setattr(preview, "PREVIEW_WIDTH", 320)
    assert preview.preview_key(src, 1.0, 2.5) != key
    monkeypatch.undo()
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # the source changed: a new preview
    assert preview.preview_key(src, 1.0, 2.5) != key
    other = tmp_path / "b.mp4"
    other.write_bytes(b"x")
    assert preview.preview_key(other, 1.0, 2.5) != preview.preview_key(src, 1.0, 2.5)


def test_preview_serves_only_indexed_sources_and_valid_spans(ui, tmp_path, monkeypatch):
    port, state, src = ui
    calls = []
    monkeypatch.setattr(server, "make_preview", lambda *a, **k: calls.append(a))  # never reached below
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(b"x")
    for path in ["/etc/passwd", str(secret), str(src.parent / ".." / "secret.mp4"), str(tmp_path), "",
                 str(src) + "/../../secret.mp4"]:
        status, _, body = _get(port, _q(path=path, start=1, end=2))
        assert status == 403, (path, body)
    for start, end in [("x", "2"), ("1", ""), ("nan", "2"), ("-1", "1"), ("2", "1"), ("0", "1e9")]:
        status, _, body = _get(port, _q(path=str(src), start=start, end=end))
        assert status == 400 and b"bad span" in body, (start, end)
    status, _, _ = _get(port, "/preview?path=" + urllib.parse.quote(str(src)))  # no span at all
    assert status == 400
    status, _, _ = _get(port, _q(path=str(src), start=1, end=2), host="evil.example:80")  # DNS rebinding
    assert status == 403
    assert not calls and not (tmp_path / "proj").exists()


def test_preview_reports_a_failed_transcode(ui, monkeypatch):
    port, _, src = ui

    def fail(*a, **k):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="Invalid data found when processing input\n")

    monkeypatch.setattr(server, "make_preview", fail)
    status, _, body = _get(port, _q(path=str(src), start=0, end=1))
    assert status == 500 and "Invalid data" in json.loads(body)["error"]


def _encoders() -> bool:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe") or not preview.encoders_available():
        return False
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return " libx264 " in listed and " aac " in listed


@pytest.mark.skipif(not _encoders(), reason="ffmpeg without libvpx-vp9 / libopus / libx264")
def test_preview_transcodes_the_span_once_to_vp9_opus(ui, monkeypatch):
    port, state, src = ui
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=5", "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=48000:duration=5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    made = []
    real = server.make_preview
    monkeypatch.setattr(server, "make_preview", lambda *a, **k: made.append(a) or real(*a, **k))
    t0 = time.perf_counter()
    status, headers, body = _get(port, _q(path=str(src), start=1.0, end=3.5))
    took = time.perf_counter() - t0
    assert status == 200 and headers["Content-Type"] == "video/webm", body[:200]
    out = next(state.previews[src].glob("*.webm"))
    assert out.read_bytes() == body and not list(state.previews[src].glob("*.part"))
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                                      "-show_streams", str(out)], capture_output=True, text=True, check=True).stdout)
    codecs = {s["codec_type"]: s for s in info["streams"]}
    assert codecs["video"]["codec_name"] == "vp9" and codecs["audio"]["codec_name"] == "opus"
    assert int(codecs["video"]["width"]) == preview.PREVIEW_WIDTH and int(codecs["video"]["height"]) == 360
    assert abs(float(info["format"]["duration"]) - 2.5) < 0.1
    assert took < 30
    # cached: a second request and a ranged one (as a <video> element sends) do not transcode again
    status, _, again = _get(port, _q(path=str(src), start=1.0, end=3.5))
    status2, headers2, part = _get(port, _q(path=str(src), start="1.0000", end="3.5"), headers={"Range": "bytes=0-99"})
    assert status == 200 and again == body and len(made) == 1
    assert status2 == 206 and part == body[:100] and headers2["Content-Range"] == f"bytes 0-99/{len(body)}"
    # /media still serves the source itself
    status, headers, raw = _get(port, "/media?" + urllib.parse.urlencode({"path": str(src)}))
    assert status == 200 and raw == src.read_bytes() and headers["Content-Type"] == "video/mp4"
