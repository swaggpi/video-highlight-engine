"""Minimal local UI: stdlib HTTP server on 127.0.0.1, one HTML page, JSON endpoints, ranged video serving."""
from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vhe.config import Config
from vhe.index.store import default_index_dir

PAGE = Path(__file__).with_name("index.html")


class State:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self.job = {"state": "idle", "stage": "", "error": None, "video": None}
        self.engines: dict[str, object] = {}

    def engine(self, video: str):
        from vhe.engine import Engine

        key = str(Path(video).resolve())
        with self.lock:
            if key not in self.engines:
                self.engines[key] = Engine(default_index_dir(Path(key)), self.device)
            return self.engines[key]


def _index_job(state: State, video: str) -> None:
    from vhe.engine import MODEL_LOCK
    from vhe.pipeline import build_index

    def progress(msg: str) -> None:
        state.job["stage"] = msg

    try:
        with MODEL_LOCK:  # queries wait while indexing holds the accelerator
            build_index(Path(video), cfg=Config(device=state.device), progress=progress)
        state.engines.pop(str(Path(video).resolve()), None)
        state.job.update(state="done", stage="done")
    except Exception as exc:  # surfaced in the UI
        traceback.print_exc()
        state.job.update(state="error", error=str(exc))


def make_handler(state: State, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _trusted(self, post: bool) -> bool:
            # Blocks DNS rebinding (foreign Host header) and simple cross-site form posts (non-JSON body).
            if self.headers.get("Host") not in allowed_hosts:
                self._json({"error": "forbidden host"}, 403)
                return False
            if post and not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._json({"error": "JSON required"}, 415)
                return False
            return True

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:
            if not self._trusted(post=False):
                return
            url = urllib.parse.urlparse(self.path)
            if url.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/status":
                self._json(state.job)
            elif url.path == "/media":
                self._media(urllib.parse.parse_qs(url.query).get("path", [""])[0])
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if not self._trusted(post=True):
                return
            path = urllib.parse.urlparse(self.path).path
            try:
                data = self._body()
                video = data.get("video", "")
                if path == "/api/index":
                    if not Path(video).is_file():
                        return self._json({"error": f"file not found: {video}"}, 400)
                    with state.lock:
                        if state.job["state"] == "running":
                            return self._json({"error": "an index job is already running"}, 409)
                        state.job = {"state": "running", "stage": "starting", "error": None, "video": video}
                    threading.Thread(target=_index_job, args=(state, video), daemon=True).start()
                    return self._json({"ok": True})
                if not default_index_dir(Path(video)).joinpath("meta.json").exists():
                    return self._json({"error": "video is not indexed yet"}, 400)
                eng = state.engine(video)
                if path == "/api/clips":
                    return self._json(eng.clips(data.get("query", ""), count=data.get("count") or None))
                if path == "/api/search":
                    return self._json({"results": eng.search(data.get("query", ""))})
                if path == "/api/export":
                    from vhe.export.clips import export_clips

                    out = Path(video).parent / (Path(video).stem + "_clips")
                    paths = export_clips(Path(video), data.get("clips", []), out)
                    return self._json({"folder": str(out), "files": [p.name for p in paths]})
                self._json({"error": "not found"}, 404)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc)}, 500)

        def _media(self, raw: str) -> None:
            path = Path(raw).resolve()
            # Only serve videos that have an index next to them; this is not a general file server.
            if not path.is_file() or not default_index_dir(path).joinpath("meta.json").exists():
                return self._json({"error": "not an indexed video"}, 403)
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            if rng:
                try:
                    a, _, b = rng.removeprefix("bytes=").split(",")[0].partition("-")
                    start = int(a) if a else max(0, size - int(b))
                    end = min(int(b), size - 1) if a and b else size - 1
                    if start > end or start >= size:
                        raise ValueError
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                try:
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


def serve(port: int = 8765, device: str = "auto") -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(State(device), port))
    print(f"vhe UI on http://127.0.0.1:{port}")
    server.serve_forever()
