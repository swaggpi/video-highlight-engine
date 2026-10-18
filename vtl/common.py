"""Small shared helpers: JSON/JSONL files, time formatting/parsing, hashing, robust JSON extraction."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mts"}


def read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(p)


def read_jsonl(p: Path) -> list[dict]:
    if not Path(p).exists():
        return []
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def write_jsonl(p: Path, rows) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(p)


def sha(*parts, n: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\0")
    return h.hexdigest()[:n]


def hms(t: float, decimals: int = 0) -> str:
    """3539.4 -> '0:58:59' (decimals=1 -> '0:58:59.4')."""
    scale = 10 ** decimals
    units = int(round(max(0.0, float(t)) * scale))
    secs, frac = divmod(units, scale)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" + (f".{frac:0{decimals}d}" if decimals else "")


def parse_time(v) -> float | None:
    """'1:02:03.5', '62:03', '3723.5' or a number -> seconds; None if unparseable."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().split(":")
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return None
    if not 1 <= len(nums) <= 3:
        return None
    t = 0.0
    for x in nums:
        t = t * 60 + x
    return t


def extract_json(text: str):
    """First JSON object in an LLM reply: a ```json fence, else the first decodable {...}."""
    text = (text or "").strip()
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        try:
            v = json.loads(m.group(1))
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                v = dec.raw_decode(text[i:])[0]
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                pass
    raise ValueError("no JSON object in reply")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]


def video_files(path: Path) -> tuple[Path, list[Path]]:
    """(root holding .vtl/, videos). A file indexes into its parent folder."""
    path = Path(path).resolve()
    if path.is_file():
        return path.parent, [path]
    vids = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXT and ".vtl" not in p.parts
                  and not any(part.startswith(".") for part in p.relative_to(path).parts))
    return path, vids
