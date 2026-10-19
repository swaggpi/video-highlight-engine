"""On-screen game text: RapidOCR (via vhe.signals.ocr) on the game-area crop at 1 fps, folded into text cards.

A card is one piece of text from when it appears to when it is gone ("DAY5 | (THE LASTDAY)" at 59:16 for 3 s).
OCR of the same text flickers between frames ("BIRDSEED", "BIRDSE", "BIRD SEED"), so a line continues a card while a
fuzzy-similar line was read within GAP_S; lines that start together form one card. Stream alerts that overlap the
game area (donations, subs, cheers) and "user: message" chat lines are not game text and are dropped, and so is
text read only once (scrolling-chat fragments). Cards whose text keeps coming back are marked
"chrome" (item prompts, the pause menu) and are not shown to the timeline LLM."""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from pathlib import Path

VERSION = "ocr-1"         # the OCR reads (ocr.jsonl)
CARDS_VERSION = "cards-2"  # the cards made from them (cards.jsonl); rebuilt from ocr.jsonl when this changes
MIN_CONF = 0.8       # RapidOCR recognition confidence for a line to count
MIN_CHARS = 3        # ... and at least this many letters/digits
GAP_S = 2.5          # a line continues a card if the same text was read at most this long ago
SAME_TEXT = 0.6      # SequenceMatcher ratio of normalised texts at or above this is the same text
CARD_LINES = 4       # a card shows at most this many lines ...
CARD_CHARS = 100     # ... and this many characters (menus and full-screen chat are walls of text)
CHROME_N = 10        # a card text that appears more often than this in one file is interface chrome
ALERT = re.compile(r"donat|super ?chat|subscri|cheer|\bbits?\b|follow|gift|raid|@\w", re.I)
CHAT = re.compile(r"^\s*[\w.\-]{2,30}\s?[:\uff1a]")  # "user: message" lines (chat shown in the game area)
MIN_READS = 2        # a text read only once (1 s) is mostly a fragment of scrolling chat or motion blur


def norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def read(paths: list[Path], times: list[float], crop: list[int] | None, log=print) -> list[dict]:
    """[{t, lines: [[text, conf], ...]}] for the frames `paths` (at `times`), cropped to [x, y, w, h]."""
    import cv2

    from vhe.signals.ocr import confident, load_engine

    engine, rows, t0, shown = None, [], time.time(), time.time()
    for k, (p, t) in enumerate(zip(paths, times, strict=True)):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if crop:
            x, y, w, h = crop
            img = img[y:y + h, x:x + w]
        if engine is None:
            engine = load_engine(img.shape[0])
        dets = confident(engine(img), MIN_CONF, MIN_CHARS)
        rows.append({"t": round(t, 3), "lines": [[text, round(conf, 3)] for _, text, conf in dets]})
        if time.time() - shown > 60:
            shown = time.time()
            log(f"  ocr {k + 1}/{len(paths)} ({(k + 1) / (time.time() - t0):.1f}/s)")
    return rows


def cards(rows: list[dict], gap: float = GAP_S) -> list[dict]:
    """Text cards [{t, end, text, conf}] from OCR rows, in time order."""
    tracks: list[dict] = []
    open_: list[dict] = []
    for r in sorted(rows, key=lambda r: r["t"]):
        t = r["t"]
        open_ = [k for k in open_ if t - k["end"] <= gap]
        for text, conf in r["lines"]:
            key = norm(text)
            if conf < MIN_CONF or len(key) < MIN_CHARS or ALERT.search(text) or CHAT.search(text):
                continue
            match = max(open_, key=lambda k: SequenceMatcher(None, key, k["key"]).ratio(), default=None)
            if match is not None and SequenceMatcher(None, key, match["key"]).ratio() >= SAME_TEXT:
                match["end"], match["n"] = t, match["n"] + 1
                match["seen"][text] = match["seen"].get(text, 0) + 1
                match["conf"] = max(match["conf"], conf)
            else:
                k = {"t": t, "end": t, "key": key, "seen": {text: 1}, "conf": conf, "n": 1}
                tracks.append(k)
                open_.append(k)
    out: list[dict] = []
    for k in (k for k in tracks if k["n"] >= MIN_READS):  # lines that appear in the same read form one card
        text = max(k["seen"].items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        if out and out[-1]["t"] == k["t"]:
            c = out[-1]
            c["lines"].append(text)
            c["end"], c["conf"] = max(c["end"], k["end"]), max(c["conf"], k["conf"])
        else:
            out.append({"t": k["t"], "end": k["end"], "lines": [text], "conf": k["conf"]})
    res = [{"t": c["t"], "end": c["end"], "text": card_text(c["lines"]), "conf": round(c["conf"], 2)} for c in out]
    seen: dict[str, int] = {}
    for c in res:
        seen[norm(c["text"])] = seen.get(norm(c["text"]), 0) + 1
    for c in res:  # the same text coming back over and over is interface chrome ("(Space) to drop"), not news
        c["chrome"] = seen[norm(c["text"])] > CHROME_N
    return res


def card_text(lines: list[str]) -> str:
    text = " | ".join(lines[:CARD_LINES]) + (" | ..." if len(lines) > CARD_LINES else "")
    return text if len(text) <= CARD_CHARS else text[:CARD_CHARS - 3] + "..."
