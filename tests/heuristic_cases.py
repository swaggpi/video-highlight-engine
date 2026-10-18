"""Synthetic engine cases for the ranking-seam regression test (test_rank.py).

tests/data/heuristic_clips.json holds what Engine.clips returns for every case; HeuristicScorer must reproduce it
exactly. First written at 138dcd6 (before the ranking seam, vhe/rank); regenerated on purpose after the r2 field-test
fixes changed the selection itself (near-duplicates are never picked, visual edges move onto clean frames, speech
clips are whole sentences, "leave out X" is a hard gate, a short budget is reported), and again for features v2 (the
parsed query of every case gained the five visual-cue keys at their defaults, every clip and every other key stayed
identical; the cue_* cases were appended), and for the r3 speech padding (only the speech-index cases topics,
speaks_every and soundbite moved: speech-clip edges sit SPEECH_LEAD before the first and SPEECH_TAIL after the last
word, in the pause; every other case stayed identical), and for features v3 (the scorer seam made a drop-in: the
final total scores the refined span itself, length-bucket centring by per-signal medians, the clean-edge term,
legibility / highlight / OCR bonuses composed by the heuristic, the merged stopword list, pool-relative selection
units; dogs_every appended): any other change to it is a regression. The cases cover the paths the hand-weighted
total runs through: length buckets, cut and length priors, the speech bonus, the usable and live-sound gates,
opener/closer pools, shot lists (per-item pool extras, round robin, exhaustive per-item), a visual "every X" list,
avoid items, spoken topics, the non-commercial saliency blend (mocked: NaN on one file), and the visual cues (close /
wide framing, faces, centring, the hook slot, readable text) on index v3 signals. The module-layout pass (vhe/query,
vhe/rank split into one module per job) kept it byte-identical."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from test_corpus import DIM, FPS, make_index, make_project
from vhe.engine import Engine
from vhe.index.store import save_index
from vhe.query import topics

TIMING = ("seconds", "load_seconds", "timings")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _text_vec(t: str) -> np.ndarray:
    """A fixed random unit vector per text (texts the case does not name)."""
    return _unit(np.random.default_rng(zlib.crc32(t.encode())).normal(size=DIM))


def _engine(root: Path, vec: dict | None = None, nc=None) -> Engine:
    """Engine with SigLIP text vectors from `vec` (else a fixed random vector per text) and, with `nc`, a mocked
    highlight model: nc(engine) -> (saliency per corpus step, NaN where a file has none; proposals per file)."""

    class Eng(Engine):
        def text_vectors(self, texts):
            return np.stack([(vec or {}).get(t, _text_vec(t)) for t in texts]).astype(np.float32)

        def _nc_scores(self, text):
            if nc is None or not text:
                return None, [[] for _ in self.corpus.members]
            return nc(self)

    return Eng.open(root, device="cpu", use_nc=False)


def _emb(rng, n_steps, dirs, spans=()):
    """Unit embeddings orthogonal to every direction, with cosine 0.2 to `d` over each (a, b, d) in spans."""
    e = rng.normal(size=(n_steps, DIM))
    for v in dirs:
        e -= np.outer(e @ v, v)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    for a, b, d in spans:
        i, j = int(a * FPS), int(b * FPS)
        e[i:j] = 0.2 * d + np.sqrt(1 - 0.04) * e[i:j]
    return e


def _dirs(n, seed=7):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(DIM, n)))
    return q.T


def _speech_index(root: Path, dur: float = 200.0) -> Path:
    lines = [(10.0, "Hello everyone and welcome back to the stream today."), (30.0, "I'm on Madison Avenue right now."),
             (80.0, "The weather is nice today, really sunny."), (120.0, "Look at all these people over here."),
             (150.0, "We mute the music because of copyright."), (175.0, "That is it for today, see you soon.")]
    segs = []
    for s, text in lines:
        words = text.split()
        segs.append({"start": s, "end": s + 0.5 * len(words), "text": text,
                     "words": [{"start": s + 0.5 * k, "end": s + 0.5 * k + 0.4, "word": w}
                               for k, w in enumerate(words)]})
    n = int(dur * FPS)
    speech = np.zeros(n)
    for sg in segs:
        speech[int(sg["start"] * FPS):int(sg["end"] * FPS)] = 1.0
    ix = make_index(root, dur, seed=5, speech=speech)
    meta = dict(ix.meta, vad=[[s["start"], s["end"]] for s in segs])
    meta.pop("version", None)
    save_index(root, meta, ix.grid, {"language": "en", "segments": segs})
    return root


def _nc_mock(seed: int, absent: int | None = None):
    def nc(eng):
        c = eng.corpus
        sal = np.random.default_rng(seed).normal(size=c.n).cumsum() * 0.1
        props = []
        for k in range(len(c.members)):
            if k == absent:
                sal[c.bounds[k]:c.bounds[k + 1]] = np.nan
                props.append([])
            else:
                d = float(c.durations[k])
                props.append([(0.2 * d, 0.2 * d + 4.0, 0.9), (0.6 * d, 0.6 * d + 7.0, 0.7)])
        return sal, props
    return nc


CUE_SPANS = {  # file -> kind -> [(start, end) seconds] where the cue project's v3 signals show it
    0: {"close": [(10.0, 14.0)], "wide": [(30.0, 40.0)]},
    1: {"text": [(20.0, 26.0)], "close": [(44.0, 48.0)]},
}
CUE_BRIEFS = {
    "cue_none": "6 clips of the parade",
    "cue_close": "6 clips of the parade, close-ups and faces, keep everything centered",
    "cue_wide": "6 clips of the parade, wide establishing views, the street as the main character",
    "cue_hook": "a 20 s edit of the parade, 2-3 s per clip. the first clip has to be the hook",
    "cue_legible": "4 clips. Show the Pasargad float with its name readable, and the crowd.",
}


def _cue_project(root: Path) -> Path:
    """Two 60 s files whose index v3 signals (shot scale, faces, approach, OCR text) show CUE_SPANS; everything
    else is an ordinary medium shot. The embeddings carry one direction everywhere (the parade)."""
    (parade,) = _dirs(1, seed=9)
    specs = []
    for k in (0, 1):
        n = int(60 * FPS)
        g = {"scale_close": np.full(n, -6.0, np.float32), "scale_wide": np.full(n, 1.5, np.float32),
             "face_n": np.zeros(n, np.int16), "face_area_max": np.zeros(n, np.float32),
             "face_center": np.zeros(n, np.float32), "approach": np.zeros(n, np.float32),
             "text_area": np.zeros(n, np.float32), "text_conf": np.zeros(n, np.float32)}
        for a, b in CUE_SPANS[k].get("close", []):
            i, j = int(a * FPS), int(b * FPS)
            g["scale_close"][i:j], g["scale_wide"][i:j] = 3.0, -3.0
            g["face_n"][i:j], g["face_area_max"][i:j], g["face_center"][i:j] = 1, 0.05, 0.9
            g["approach"][i:i + 6] = 0.8
        for a, b in CUE_SPANS[k].get("wide", []):
            i, j = int(a * FPS), int(b * FPS)
            g["scale_wide"][i:j], g["scale_close"][i:j] = 5.0, -8.0
        for a, b in CUE_SPANS[k].get("text", []):
            i, j = int(a * FPS), int(b * FPS)
            g["text_area"][i:j], g["text_conf"][i:j] = 0.03, 0.95
        g["emb"] = _emb(np.random.default_rng(20 + k), n, (parade,), [(0, 60, parade)])
        specs.append((f"take{k}.mp4", 60.0, g))
    make_project(root, specs)
    for d in sorted((root / ".vhe" / "files").iterdir()):  # the text store of take1 (OCR read every 3rd step)
        if d.name.startswith("take1"):
            a, b = CUE_SPANS[1]["text"][0]
            items = [[s, "PASARGAD", 0.95, [0.3, 0.4, 0.7, 0.5]] for s in range(int(a * FPS), int(b * FPS), 3)]
            (d / "ocr.json").write_text(json.dumps({"version": 1, "engine": "fake", "every": 3, "min_conf": 0.7,
                                                    "items": items}))
    return root


def cue_engine(root: Path) -> Engine:
    """The cue project's engine: every text the rules parse from CUE_BRIEFS (and the hook/opener slots) matches the
    parade direction, so the cue signals decide between otherwise equal footage."""
    from vhe.query.rules import parse

    (parade,) = _dirs(1, seed=9)
    texts = {"parade"}
    for q in CUE_BRIEFS.values():
        p = parse(q)
        texts |= {*p.shots, p.text, p.opener or "", p.closer or ""}
    return _engine(_cue_project(root), {t: parade for t in texts if t})


def cases(tmp: Path):
    """Yield (name, engine, query, clips kwargs)."""
    root = make_project(tmp / "recap", [("a.mp4", 61.0, {}), ("b/c.mp4", 45.0, {"has_audio": False}),
                                        ("d.mp4", 50.0, {})])
    eng = _engine(root)
    yield "recap", eng, "60-90 s energetic recap", {}
    yield "calm_budget", eng, "5 calm B-roll shots 3-6 s each", {"budget_s": (20.0, 30.0)}

    rng = np.random.default_rng(2)
    (dogs,) = _dirs(1, seed=2)
    on = _emb(rng, 120, (dogs,), [(0, 40, dogs)])
    root = make_project(tmp / "dogs", [("noise.mp4", 40.0, {"emb": _emb(rng, 120, (dogs,))}),
                                       ("dogs.mp4", 40.0, {"emb": on})])
    yield "dogs", _engine(root, {"dogs": dogs}), "3 clips of dogs", {}
    yield "dogs_nc", _engine(root, {"dogs": dogs}, _nc_mock(11, absent=0)), "3 clips of dogs", {}
    # a visual "every X" list (strong_matches: the score margin and floor, in the pool's spreads; added for v3)
    yield "dogs_every", _engine(root, {"dogs": dogs}), "every shot of dogs", {}

    n = int(90 * FPS)
    loud, usable, motion = np.full(n, 0.1), np.ones(n, bool), np.full(n, 0.1)
    loud[90:150], usable[90:150], motion[90:150] = 5.0, False, 5.0
    root = make_project(tmp / "gate", [("a.mp4", 90.0, {"loudness": loud, "motion": motion, "usable": usable})])
    yield "gate", _engine(root), "5 energetic clips", {}

    root = make_project(tmp / "silent", [("loud.mp4", 120.0, {}), ("silent.mp4", 120.0, {"has_audio": False})])
    yield "audioless", _engine(root), "8 energetic clips", {}

    wide, crowd = _dirs(2, seed=4)
    specs = [(f"take{k}.mp4", 60.0, {"emb": _emb(np.random.default_rng(k), 180, (wide, crowd),
                                                 [(10, 20, {2: wide, 0: crowd}[k])] if k in (0, 2) else [])})
             for k in range(4)]
    root = make_project(tmp / "edges", specs)
    yield "edges", _engine(root, {"wide shot": wide, "the crowd": crowd, "crowd": crowd}), \
        "45-60 s recap, open with a wide shot, end with the crowd", {}

    dogs, cats, horses, birds, wide = _dirs(5)
    dirs = (dogs, cats, horses, birds, wide)
    rng = np.random.default_rng(0)
    specs = [("a.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(5, 50, dogs)])}),
             ("b.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(20, 30, cats), (40, 50, birds)])}),
             ("c.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(5, 25, dogs), (30, 40, birds), (45, 55, wide)])}),
             ("d.mp4", 60.0, {"emb": _emb(rng, 180, dirs, [(0, 12, horses), (30, 58, cats)])})]
    root = make_project(tmp / "shots", specs)
    vec = {"dogs": dogs, "cats": cats, "horses": horses, "birds": birds, "wide shot": wide}
    eng = _engine(root, vec)
    yield "shots", eng, "3 clips. Show the dogs, the cats and the horses.", {}
    yield "shots_every", eng, "Every animal: the dogs, the cats and the horses.", {}
    yield "shots_avoid", eng, "4 clips. Open with a wide shot. Show the dogs, the cats and the horses. " \
                              "Leave out shots of the birds.", {}
    yield "shots_budget", eng, "A 30-40 s calm piece. Show the dogs, the cats and the horses. No birds. " \
                               "End with the cats.", {}
    eng = _engine(root, vec, _nc_mock(3, absent=2))
    yield "shots_avoid_nc", eng, "4 clips. Open with a wide shot. Show the dogs, the cats and the horses. " \
                                 "Leave out shots of the birds.", {}
    yield "shots_every_nc", eng, "Every animal: the dogs, the cats and the horses.", {}

    (dogs,) = _dirs(1, seed=1)
    rng = np.random.default_rng(1)
    loud = rng.random(180)
    loud[:90] = -120.0
    root = make_project(tmp / "live", [("a.mp4", 60.0, {"emb": _emb(rng, 180, (dogs,), [(0, 60, dogs)]),
                                                        "loudness": loud})])
    eng = _engine(root, {"dogs": dogs})
    yield "live", eng, "4 clips of dogs with natural sound", {}
    yield "not_live", eng, "8 clips of dogs", {}

    root = _speech_index(tmp / "talk.mp4.vhe")
    eng = _engine(root)
    yield "topics", eng, "Find every moment where the host talks about where he is (Madison Avenue), and why he " \
                         "mutes the music (copyright).", {}
    yield "talks", eng, "5 clips where someone talks to camera", {}
    yield "speaks_every", eng, "every moment someone speaks", {}
    yield "soundbite", eng, "3 clips of the crowd. Also give the best 5-8 s soundbite where the host sums up " \
                            "the stream (\"that is it for today\").", {}

    # Visual cues (features v2, index v3 signals): added with the cue bonuses, after the cue-free cases above.
    eng = cue_engine(tmp / "cues")
    for name, q in CUE_BRIEFS.items():
        yield name, eng, q, {}


def strip(res: dict, tmp: Path) -> dict:
    """Without timings, clip paths relative to the temp dir."""
    out = {k: v for k, v in res.items() if k not in TIMING}
    out["clips"] = [dict(c, path=Path(c["path"]).relative_to(tmp).as_posix()) for c in res["clips"]]
    return out


def run_all(tmp: Path, scorer=None, **clips_kw) -> dict:
    """{case name: Engine.clips result without timings}; rules parser, keyword topic matching (no servers). scorer:
    rank every case with this Scorer instead of the engine's default (the heuristic)."""
    embed = topics.embed
    topics.embed = lambda texts, **kw: None
    out = {}
    try:
        for name, eng, query, kw in cases(tmp):
            if scorer is not None:
                eng.scorer = scorer
            out[name] = strip(eng.clips(query, llm=None, **kw, **clips_kw), tmp)
        return out
    finally:
        topics.embed = embed
