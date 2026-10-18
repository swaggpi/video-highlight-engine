"""Minimal local UI: stdlib HTTP server on 127.0.0.1, one HTML page, JSON endpoints, ranged video serving, and
per-clip WebM previews (vhe.export.preview) that play in browsers without an H.264 decoder."""
from __future__ import annotations

import json
import mimetypes
import subprocess
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vhe.config import Config
from vhe.export.preview import make_preview, preview_path, span
from vhe.index.store import default_index_dir
from vhe.project import PROJECT_DIR

PAGE = Path(__file__).with_name("index.html")
PREVIEW_CONCURRENCY = 2  # transcodes at once (each uses several cores); the page also asks for previews lazily


class State:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self.job = {"state": "idle", "stage": "", "file": "", "error": None, "video": None}
        self.engines: dict[str, object] = {}
        self.media: set[Path] = set()  # source files of opened engines: the only files /media serves
        self.previews: dict[Path, Path] = {}  # source file -> its project's preview cache folder
        self.preview_slots = threading.BoundedSemaphore(PREVIEW_CONCURRENCY)
        self.preview_locks: dict[str, threading.Lock] = {}  # one transcode per preview, however many requests ask

    def engine(self, path: str):
        """Engine for a project folder or a video file; models come from the process-wide cache (vhe.models)."""
        from vhe.engine import Engine

        key = str(Path(path).resolve())
        with self.lock:
            if key not in self.engines:
                self.engines[key] = Engine.open(Path(key), self.device)
                c = self.engines[key].corpus
                for k, s in enumerate(c.sources):
                    s = Path(s).resolve()
                    self.media.add(s)
                    # <DIR>/.vhe/previews for a project, the video's own index folder otherwise
                    self.previews[s] = (Path(c.root) / PROJECT_DIR if c.root else c.members[k].root) / "previews"
            return self.engines[key]

    def allowed(self, path: Path) -> bool:
        """Only source videos of an opened index or project are served; this is not a general file server."""
        return path.is_file() and (path in self.media or default_index_dir(path).joinpath("meta.json").exists())

    def preview_dir(self, path: Path) -> Path:
        return self.previews.get(path) or default_index_dir(path) / "previews"

    def preview_lock(self, key: str) -> threading.Lock:
        with self.lock:
            return self.preview_locks.setdefault(key, threading.Lock())


def _index_job(state: State, path: str) -> None:
    from vhe.models import MODEL_LOCK

    def progress(msg: str) -> None:
        if msg.startswith("["):  # "[k/N] file (...)" from the project indexer
            state.job["file"] = msg
        state.job["stage"] = msg.strip()

    try:
        with MODEL_LOCK:  # queries wait while indexing holds the accelerator
            if Path(path).is_dir():
                from vhe.project import index_project

                index_project(Path(path), Config(device=state.device), progress=progress)
            else:
                from vhe.pipeline import build_index

                build_index(Path(path), cfg=Config(device=state.device), progress=progress)
        with state.lock:
            state.engines.pop(str(Path(path).resolve()), None)
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
            elif url.path == "/preview":
                q = urllib.parse.parse_qs(url.query)
                self._preview(*(q.get(k, [""])[0] for k in ("path", "start", "end")))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if not self._trusted(post=True):
                return
            path = urllib.parse.urlparse(self.path).path
            try:
                data = self._body()
                video = data.get("video", "")  # a video file or a project folder
                if path == "/api/index":
                    if not Path(video).exists():
                        return self._json({"error": f"not found: {video}"}, 400)
                    with state.lock:
                        if state.job["state"] == "running":
                            return self._json({"error": "an index job is already running"}, 409)
                        state.job = {"state": "running", "stage": "starting", "file": "", "error": None,
                                     "video": video}
                    threading.Thread(target=_index_job, args=(state, video), daemon=True).start()
                    return self._json({"ok": True})
                try:
                    eng = state.engine(video)
                except FileNotFoundError as exc:
                    return self._json({"error": str(exc)}, 400)
                if path == "/api/clips":
                    # The count box is empty by default: then the brief's own count/budget decides.
                    return self._json(eng.clips(data.get("query", ""), count=data.get("count") or None))
                if path == "/api/search":
                    return self._json({"results": eng.search(data.get("query", ""))})
                if path == "/api/export":
                    from vhe.export.clips import export_clips

                    src = Path(video).resolve()
                    # Next to the footage folder, not inside it: re-indexing would take the clips for new takes.
                    out = src.parent / (src.name + "_vhe_export" if src.is_dir() else src.stem + "_clips")
                    res = export_clips(data.get("clips", []), out, eng.corpus.source_map, with_rough_cut=True)
                    return self._json({"folder": str(out), "files": [p.name for p in res["clips"]],
                                       "edl": res["edl"].name, "rough_cut": res["rough_cut"] and res["rough_cut"].name})
                self._json({"error": "not found"}, 404)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc)}, 500)

        def _media(self, raw: str) -> None:
            path = Path(raw).resolve()
            if not state.allowed(path):
                return self._json({"error": "not an indexed video"}, 403)
            self._send_file(path, mimetypes.guess_type(path.name)[0] or "video/mp4")

        def _preview(self, raw: str, start: str, end: str) -> None:
            """The clip source[start:end] (file-local seconds) as WebM, transcoded once and cached on disk."""
            path = Path(raw).resolve() if raw else None
            if path is None or not state.allowed(path):
                return self._json({"error": "not an indexed video"}, 403)
            try:
                s, e = span(start, end)
            except ValueError as exc:
                return self._json({"error": f"bad span: {exc}"}, 400)
            out = preview_path(state.preview_dir(path), path, s, e)
            with state.preview_lock(out.stem):
                if not out.exists():
                    try:
                        with state.preview_slots:
                            make_preview(path, s, e, out)
                    except (OSError, subprocess.SubprocessError) as exc:
                        err = getattr(exc, "stderr", None)
                        detail = (err if isinstance(err, str) and err.strip() else str(exc)).strip()
                        return self._json({"error": f"preview failed: {detail[-500:]}"}, 500)
            self._send_file(out, "video/webm")

        def _send_file(self, path: Path, content_type: str) -> None:
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
            self.send_header("Content-Type", content_type)
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
