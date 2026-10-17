"""Synthetic engine cases for the ranking-seam regression test (test_rank.py).

tests/data/heuristic_clips.json holds what Engine.clips returned for every case before the ranking seam
(vhe/rank) existed, generated at commit 138dcd6; HeuristicScorer must reproduce it exactly. The cases cover the
paths the hand-weighted total runs through: length buckets, cut and length priors, the speech bonus, the usable
and live-sound gates, opener/closer pools, shot lists (per-item pool extras, round robin, exhaustive per-item),
avoid items, spoken topics, and the non-commercial saliency blend (mocked: NaN on one file)."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from test_corpus import DIM, FPS, make_index, make_project
from vhe.engine import Engine
from vhe.index.store import save_index
from vhe.query import plan

TIMING = ("seconds", "load_seconds")


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


def strip(res: dict, tmp: Path) -> dict:
    """Without timings, clip paths relative to the temp dir."""
    out = {k: v for k, v in res.items() if k not in TIMING}
    out["clips"] = [dict(c, path=Path(c["path"]).relative_to(tmp).as_posix()) for c in res["clips"]]
    return out


def run_all(tmp: Path, **clips_kw) -> dict:
    """{case name: Engine.clips result without timings}; rules parser, keyword topic matching (no servers)."""
    embed = plan.embed
    plan.embed = lambda texts, **kw: None
    try:
        return {name: strip(eng.clips(query, llm=None, **kw, **clips_kw), tmp) for name, eng, query, kw in cases(tmp)}
    finally:
        plan.embed = embed
