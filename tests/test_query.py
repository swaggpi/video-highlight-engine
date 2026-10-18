import itertools
import json
import urllib.request

import numpy as np
import pytest

from vhe.query.candidates import cuts_inside, generate, peak_times, speech_spans
from vhe.query.refine import SPEECH_TAIL, Context, refine
from vhe.query.rules import parse
from vhe.rank.select import (
    EXHAUSTIVE_MIN_TOTAL,
    EXHAUSTIVE_SCORE_MARGIN,
    select,
    select_for,
)
from vhe.rank.signals import segment_scores
from vhe.signals import robust_z


def test_parse_energy_and_short():
    p = parse("short high-energy scenes")
    assert p.mood == "energetic"
    assert p.max_len == 10.0
    assert p.weights["energy"] > 0
    assert p.text == ""
    assert p.weights["relevance"] == 0.0


def test_parse_calm_keeps_content():
    p = parse("chill scenic view of the mountains")
    assert p.mood == "calm"
    assert p.weights["energy"] < 0 and p.weights["steady"] > 0
    assert "mountains" in p.text and "scenic" in p.text and "chill" not in p.text


def test_parse_german_and_limits():
    p = parse("3 ruhige Szenen am Strand unter 12 Sekunden ohne Sprache")
    assert p.count == 3
    assert p.max_len == 12.0
    assert p.mood == "calm"
    assert p.weights["speech"] == -1.0
    assert "strand" in p.text


def test_parse_hyphenated_count():
    p = parse("3 short high-energy scenes")
    assert p.count == 3 and p.text == "" and p.max_len == 10.0


def test_parse_length_not_count():
    p = parse("10 second clips of the beach")
    assert p.count == 5 and 7.0 <= p.min_len and p.max_len <= 13.0
    assert "second" not in p.text and "beach" in p.text
    p = parse("2 minute clips")
    assert p.count == 5 and p.max_len > 100


def test_parse_negated_speech_and_explicit_length_wins():
    assert parse("forest without people talking").weights["speech"] == -1.0
    p = parse("kurze Szenen 5 bis 8 Sekunden")
    assert (p.min_len, p.max_len) == (5.0, 8.0)
    assert parse("dogs", count=500).count == 50


def test_parse_budget():
    # Was `.budget_s == 60.0`: the budget is now a (min, max) range; a single value means +-15 %.
    assert parse("highlights total 60 seconds").budget == (51.0, 69.0)


def test_parse_range():
    p = parse("dog playing 5-15s")
    assert (p.min_len, p.max_len) == (5.0, 15.0)
    assert p.text == "dog playing"


def test_robust_z_constant_is_zero():
    assert np.all(robust_z(np.ones(10)) == 0)


def test_segment_scores_total_uses_weights():
    sig = {k: np.zeros(30) for k in ("relevance", "quality", "energy", "steady", "speech")}
    sig["energy"][10:20] = 2.0
    s = segment_scores(sig, 10, 20, {"energy": 1.0})
    assert s["energy"] == 2.0 and s["total"] == 2.0


def test_generate_respects_length_limits():
    cands = generate(60.0, cuts=[20.0, 40.0], change_points=[], min_len=4, max_len=10)
    assert cands
    assert all(4 - 1e-6 <= e - s <= 10 + 1e-6 for s, e, _ in cands)
    assert all(0 <= s and e <= 60 for s, e, _ in cands)


def test_cuts_inside():
    assert cuts_inside(0, 10, np.array([5.0, 9.9, 0.1])) == 1


def test_peak_times_min_gap():
    x = np.zeros(100)
    x[[10, 12, 50]] = [5, 4, 3]
    peaks = peak_times(x, fps=3.0, min_gap_s=2.0, pct=0)
    assert 10 / 3 in peaks and 12 / 3 not in peaks and 50 / 3 in peaks


def test_select_avoids_overlap_and_duplicates():
    cands = [{"start": 0, "end": 10, "total": 3.0}, {"start": 2, "end": 12, "total": 2.9},
             {"start": 20, "end": 30, "total": 2.0}, {"start": 40, "end": 50, "total": 1.9}]
    v = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=float)
    picked = select(cands, v, n=2)
    assert [c["start"] for c in picked] == [0, 40]  # 2-12 overlaps, 20-30 looks identical to the first


def test_refine_snaps_to_cut_and_word():
    ctx = Context(duration=60, cuts=[10.0], vad=[(20.0, 26.0)],
                  words=[{"start": 24.5, "end": 25.2, "word": "hello"}])
    s, e = refine(10.4, 25.0, ctx, min_len=3, max_len=30)
    assert abs(s - 10.04) < 1e-6  # just after the cut
    assert abs(e - (25.2 + SPEECH_TAIL)) < 1e-6  # after the word with its tail pad, not mid-word


def test_refine_extension_respects_cuts():
    ctx = Context(duration=60, cuts=[22.5], vad=[], words=[])
    s, e = refine(20.0, 23.0, ctx, min_len=3, max_len=30)
    assert e <= 22.5 and e - s >= 3 - 1e-6  # grows backwards instead of crossing the cut


def test_select_budget():
    cands = [{"start": 0, "end": 10, "total": 3.0}, {"start": 20, "end": 30, "total": 2.0},
             {"start": 40, "end": 45, "total": 1.0}]
    v = np.eye(3)
    picked = select(cands, v, n=3, budget_s=15)
    assert [c["start"] for c in picked] == [0, 40]


def test_refine_enforces_length():
    ctx = Context(duration=60, cuts=[], vad=[], words=[])
    s, e = refine(5.0, 6.0, ctx, min_len=4, max_len=10)
    assert e - s >= 4 - 1e-6
    s, e = refine(5.0, 40.0, ctx, min_len=4, max_len=10)
    assert e - s <= 10 + 1e-6


# ---- editor briefs ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("brief", ["60-90 s energetic recap, open with an establishing shot",
                                   "60–90 s energetic recap, open with an establishing shot",
                                   "60—90 sec energetic recap, open with an establishing shot",
                                   "60 to 90 seconds energetic recap, open with an establishing shot"])
def test_brief_budget_range_not_clip_length(brief):
    p = parse(brief)
    assert p.budget == (60.0, 90.0)
    assert (p.min_len, p.max_len) == (2.0, 4.0)  # energetic: typical 2-4 s per clip
    assert p.count == 25 and not p.count_explicit  # 75 s / 3 s, a target rather than a cap
    assert p.mood == "energetic" and p.opener == "establishing shot"
    assert p.text == "" and p.weights["relevance"] == 0.0  # nothing of the brief leaks into the SigLIP text


@pytest.mark.parametrize("brief,budget", [("1 minute recap of the evening", (51.0, 69.0)),
                                          ("a 1 minute recap", (51.0, 69.0)), ("a one minute recap", (51.0, 69.0)),
                                          ("a recap under 2 minutes", (102.0, 120.0)),
                                          ("under 2 minutes", (102.0, 120.0)), ("about 45 s", (38.25, 51.75)),
                                          ("a 1:30 highlight reel", (76.5, 103.5))])
def test_brief_single_budget(brief, budget):
    p = parse(brief)
    assert p.budget == budget
    assert p.text in ("", "evening")
    assert p.count > 5 and (p.min_len, p.max_len) == (3.0, 6.0)


def test_brief_per_clip_lengths():
    p = parse("5 calm scenic shots, 3–6 s each")
    assert p.budget is None and (p.min_len, p.max_len) == (3.0, 6.0) and p.count == 5 and p.count_explicit
    assert p.mood == "calm" and p.text == "scenic"
    p = parse("clips of 2-4 seconds")
    assert p.budget is None and (p.min_len, p.max_len) == (2.0, 4.0) and p.text == ""
    p = parse("short clips")
    assert p.budget is None and p.max_len == 10.0 and p.text == ""
    p = parse("10 second clips for a recap")  # the nearer noun decides
    assert p.budget is None and 7.0 <= p.min_len and p.max_len <= 13.0


def test_brief_budget_with_count_sets_clip_length():
    p = parse("5 clips, 60 s total")
    assert p.budget == (51.0, 69.0) and p.count == 5 and p.count_explicit
    assert (p.min_len, p.max_len) == (9.0, 15.0)  # 60 s / 5 clips, +-25 %


def test_brief_budget_override_and_explicit_count():
    p = parse("energetic recap", budget_s=60)
    assert p.budget == (51.0, 69.0) and p.count == 20 and not p.count_explicit
    assert parse("energetic recap", budget_s=(50, 70)).budget == (50.0, 70.0)
    assert parse("60-90 s recap", count=10).count == 10


def test_brief_exhaustive_speech():
    p = parse("every moment someone speaks to camera")
    assert p.exhaustive and not p.count_explicit and p.count == 50
    assert p.weights["speech"] == 1.0
    assert p.text == "someone to camera"  # the verb is intent, "to camera" is also visible
    p = parse("all the times people talk to the camera")
    assert p.exhaustive and p.weights["speech"] == 1.0 and "talk" not in p.text
    p = parse("every toast")
    assert p.exhaustive and p.text == "toast" and p.weights["speech"] == 1.0
    assert not parse("everyone dancing").exhaustive and not parse("all smiles").exhaustive


@pytest.mark.parametrize("verb", ["speaks", "speaking", "talks", "talking", "says", "saying", "addresses",
                                  "speech", "announcement"])
def test_brief_speech_verbs(verb):
    p = parse(f"the host {verb} on stage")
    assert p.weights["speech"] == 1.0 and verb not in p.text.split() and "host" in p.text


@pytest.mark.parametrize("phrase", ["to camera", "interview", "toast", "presenter"])
def test_brief_speech_visual_nouns(phrase):
    p = parse(f"bride {phrase}")
    assert p.weights["speech"] == 1.0 and phrase in p.text


def test_brief_opener_closer():
    p = parse("2 minute montage of the dance floor, start with a drone shot and end with the fireworks")
    assert p.opener == "drone shot" and p.closer == "fireworks"
    assert p.budget == (102.0, 138.0) and p.text == "dance floor"
    p = parse("a 1:30 highlight reel, end with the fireworks")
    assert p.opener is None and p.closer == "fireworks" and p.text == ""


def test_energetic_keeps_steadiness():
    assert parse("energetic recap").weights["steady"] >= 0.3
    assert parse("dogs").weights["steady"] >= 0.3


# ---- selection under a budget ----------------------------------------------------------------------------------

class _P:  # stands in for ParsedQuery in select_for
    def __init__(self, count, budget=None, count_explicit=False, exhaustive=False, speech=0.0):
        self.count, self.budget, self.count_explicit, self.exhaustive = count, budget, count_explicit, exhaustive
        self.weights = {"speech": speech}


def _pool(lengths, gap=1.0, totals=None):
    """Distinct moments, 1 s apart on the virtual timeline. Each is in its own file: clips of one file within
    DUP_GAP_S of each other are near-duplicates (select.duplicates), which these budget/count tests are not about."""
    out, t = [], 0.0
    for k, L in enumerate(lengths):
        out.append({"start": t, "end": t + L, "total": float(totals[k] if totals is not None else 10 - k * 0.01),
                    "file": k})
        t += L + gap
    return out


def test_select_fills_budget_past_count():
    picked = select_for(_pool([3.0] * 60), np.eye(60), _P(count=5, budget=(60.0, 90.0)))
    used = sum(c["end"] - c["start"] for c in picked)
    assert 60.0 <= used <= 90.0 and len(picked) >= 20  # the default count no longer caps the recap


def test_select_budget_never_exceeds_max():
    cands = _pool([20.0, 20.0, 20.0, 25.0, 4.0, 4.0])
    picked = select(cands, np.eye(6), n=1, budget_s=(60.0, 70.0), n_max=50)
    used = sum(c["end"] - c["start"] for c in picked)
    assert 60.0 <= used <= 70.0
    assert 25.0 not in [c["end"] - c["start"] for c in picked]  # would overshoot; smaller clips fill instead


def test_select_explicit_count_with_budget_is_a_cap():
    picked = select_for(_pool([3.0] * 60), np.eye(60), _P(count=5, budget=(60.0, 90.0), count_explicit=True))
    assert len(picked) == 5


def test_select_exhaustive_threshold_no_overlap():
    # visual "every X": above the absolute threshold and within EXHAUSTIVE_SCORE_MARGIN of the best (strong matches
    # only since r3), no overlap, no MMR cut
    m = EXHAUSTIVE_SCORE_MARGIN
    cands = _pool([4.0] * 5, totals=[3.0, 2.0, 0.4, 3.0 - m + 0.1, 0.1]) + [{"start": 1.0, "end": 5.0, "total": 2.9}]
    picked = select_for(cands, np.ones((6, 1)), _P(count=50, exhaustive=True))
    assert sorted(c["total"] for c in picked) == [3.0 - m + 0.1, 2.0, 3.0]
    assert all(c["total"] >= EXHAUSTIVE_MIN_TOTAL for c in picked)
    weak = _pool([4.0] * 5, totals=[3.0 + m, 2.0 + m, 0.4, 3.0 - 0.1, 0.1])
    picked = select_for(weak, np.ones((5, 1)), _P(count=50, exhaustive=True))
    assert sorted(c["total"] for c in picked) == [2.0 + m, 3.0 + m]  # 2.9: more than the margin below the best
    cands = _pool([4.0] * 5, totals=[3.0, 2.0, 0.4, 1.0, 0.1]) + [{"start": 1.0, "end": 5.0, "total": 2.9}]
    for c, f in zip(cands, [1.0, 0.2, 1.0, 0.9, 1.0, 0.0], strict=True):
        c["speech_frac"] = f
    picked = select_for(cands, np.ones((6, 1)), _P(count=50, exhaustive=True, speech=1.0))
    assert sorted(c["total"] for c in picked) == [1.0, 3.0]  # speech intent: mostly-speech clips only


def test_budget_holds_after_refine():
    """Refinement changes lengths (snaps to words/cuts, grows to min_len); selecting on refined edges keeps the
    total inside the budget, which selecting first and refining afterwards did not guarantee."""
    words = [{"start": t, "end": t + 0.4, "word": "w"} for t in np.arange(0.2, 200.0, 0.5)]
    ctx = Context(duration=200, cuts=list(np.arange(7.0, 200.0, 7.0)), vad=[], words=words)
    raw = _pool([2.2, 3.7, 1.5, 3.9] * 10, gap=1.3)
    refined = [dict(c, start=s, end=e) for c in raw for s, e in [refine(c["start"], c["end"], ctx, 2.0, 4.0)]]
    assert any(abs((r["end"] - r["start"]) - (c["end"] - c["start"])) > 0.05 for r, c in zip(refined, raw, strict=True))
    picked = select_for(refined, np.eye(len(refined)), _P(count=10, budget=(40.0, 45.0)))
    used = sum(c["end"] - c["start"] for c in picked)
    assert 40.0 <= used <= 45.0
    spans = sorted((c["start"], c["end"]) for c in picked)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))


# ---- speech candidates and word-safe refinement ----------------------------------------------------------------

def _seg(words):
    return {"start": words[0][0], "end": words[-1][1], "text": " ".join(w for *_, w in words),
            "words": [{"start": s, "end": e, "word": w} for s, e, w in words]}


def test_speech_spans_merge_and_split_at_sentences():
    segs = [_seg([(10.0, 10.4, "Hello"), (10.5, 11.0, "everyone.")]),
            _seg([(11.3, 11.8, "Thanks"), (11.9, 12.5, "for"), (12.6, 13.0, "coming!")]),  # 0.3 s pause: merged
            _seg([(20.0, 20.5, "Cheers.")])]  # 7 s later: separate
    spans = speech_spans(segs, vad=[], max_len=30)
    assert spans == [(pytest.approx(9.85), pytest.approx(13.15)), (pytest.approx(19.85), pytest.approx(20.65))]
    spans = speech_spans(segs, vad=[], max_len=2.5)  # too long together: split where the first sentence ends
    assert spans[0] == (pytest.approx(9.85), pytest.approx(11.15))  # padding: half the 0.3 s pause at most
    assert spans[1] == (pytest.approx(11.15), pytest.approx(13.15))


def test_speech_spans_from_vad_without_transcript():
    spans = speech_spans([], vad=[(5.0, 7.0), (7.3, 9.0), (30.0, 50.0)], max_len=10.3)
    assert spans[0] == (pytest.approx(4.85), pytest.approx(9.15))  # merged across the 0.3 s pause
    assert len(spans) == 3 and all(e - s <= 10.3 + 1e-6 for s, e in spans)  # 20 s span split in two
    assert spans[1][0] == pytest.approx(29.85) and spans[2][1] == pytest.approx(50.15)


def test_generate_speech_candidates_start_with_speaker():
    cands = generate(60.0, cuts=[], change_points=[], min_len=3, max_len=30, speech=[(9.85, 13.15), (19.85, 20.65)])
    speech = [(s, e) for s, e, src in cands if src == "speech"]
    assert (9.85, 13.15) in speech
    short = next(c for c in speech if c[0] < 20.0 < c[1])  # 0.8 s "Cheers." grows evenly to min_len
    assert short[1] - short[0] == pytest.approx(3.0) and short[0] < 19.85 and short[1] > 20.65


def test_refine_never_inside_words():
    words = [{"start": t, "end": t + 0.45, "word": "w"} for t in np.arange(0.0, 60.0, 0.5)]
    ctx = Context(duration=60, cuts=[12.2, 30.0], vad=[(0.0, 60.0)], words=words)

    def inside(t):
        return any(w["start"] < t < w["end"] for w in words)
    for s0 in np.arange(0.0, 50.0, 0.37):
        for L in (1.0, 3.3, 7.7, 40.0):
            s, e = refine(float(s0), float(s0 + L), ctx, min_len=3, max_len=8)
            assert not inside(s) and not inside(e), (s0, L, s, e)


def test_refine_max_trim_does_not_end_after_cut():
    ctx = Context(duration=60, cuts=[14.7], vad=[], words=[])
    _, e = refine(5.0, 30.0, ctx, min_len=3, max_len=10)  # trim lands at 14.8, 0.1 s into the next shot
    assert e == pytest.approx(14.7 - 0.04)


def test_refine_ignores_cut_inside_word():
    ctx = Context(duration=60, cuts=[10.3], vad=[], words=[{"start": 10.0, "end": 10.6, "word": "hello"}])
    s, _ = refine(10.5, 20.0, ctx, min_len=3, max_len=30)
    assert s <= 10.0  # starts before the word instead of at the cut in the middle of it


# ---- engine end to end on a synthetic index ---------------------------------------------------------------------

def _synthetic_index(root, transcript=None, vad=()):
    from vhe.index.store import save_index

    fps, dur = 3.0, 240.0
    n = int(dur * fps)
    rng = np.random.default_rng(0)
    grid = {"t": np.arange(n) / fps, "emb": rng.normal(size=(n, 8)).astype(np.float16),
            "blur_ratio": 4 + rng.random(n), "blown": np.zeros(n), "motion": rng.random(n),
            "loudness": rng.random(n), "onset": rng.random(n), "cam_speed": rng.random(n), "shake": rng.random(n),
            "speech": np.zeros(n), "usable": np.ones(n, bool), "unusable": np.zeros(n, np.uint8),
            "tags": np.zeros((n, 2), np.float16)}
    for s, e in vad:
        grid["speech"][int(s * fps):int(e * fps)] = 1.0
    meta = {"video": str(root / "v.mp4"), "fps": fps, "duration": dur, "cuts": [float(c) for c in range(20, 240, 20)],
            "vad": [list(x) for x in vad], "encoder": "none", "labels": ["a", "b"], "has_audio": True,
            "logit_scale": 112.9, "logit_bias": -16.8}
    save_index(root, meta, grid, transcript)
    return root


def test_engine_recap_fills_budget(tmp_path):
    from vhe.engine import Engine

    res = Engine(_synthetic_index(tmp_path / "x.vhe"), device="cpu", use_nc=False).clips("60-90 s energetic recap")
    total = sum(c["end"] - c["start"] for c in res["clips"])
    assert tuple(res["query"]["budget"]) == (60.0, 90.0)
    assert 60.0 <= total <= 90.0 and len(res["clips"]) > 5
    spans = sorted((c["start"], c["end"]) for c in res["clips"])
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))
    json.dumps(res)


def test_engine_every_speech_moment_starts_with_speaker(tmp_path, monkeypatch):
    from vhe.engine import Engine

    monkeypatch.setattr(Engine, "text_vector", lambda self, text: np.ones(8, np.float32) / np.sqrt(8))  # no model
    utter = [(31.0, 34.5), (82.0, 86.0), (150.5, 153.0)]
    segs = [_seg([(s, s + 1.0, "Good"), (s + 1.1, e, "evening.")]) for s, e in utter]
    root = _synthetic_index(tmp_path / "y.vhe", transcript={"language": "en", "segments": segs}, vad=utter)
    res = Engine(root, device="cpu", use_nc=False).clips("every moment someone speaks")
    assert res["query"]["exhaustive"]
    clips = sorted(res["clips"], key=lambda c: c["start"])
    assert len(clips) == 3  # every utterance, nothing else (the footage between is random noise)
    for (s, e), c in zip(utter, clips, strict=True):
        assert s - 0.5 <= c["start"] <= s and e <= c["end"] <= e + 0.5  # whole utterance, first to last word
        assert c["transcript"] == "Good evening."


def test_llm_path_carries_new_fields(monkeypatch):
    import io

    from vhe.query import llm

    answer = {"content": "dance floor", "mood": "energetic", "speech": "any", "min_seconds": 0, "max_seconds": 0,
              "count": 0, "budget_min_seconds": 60, "budget_max_seconds": 90, "exhaustive": False,
              "opener": "drone shot", "closer": ""}
    body = json.dumps({"message": {"content": json.dumps(answer)}}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: io.BytesIO(body))
    # The model's numbers count only when the brief contains them (grounding); the rules find none here.
    p = llm.parse_with_llm("drone opener, then the dance floor: 60 to 90", model="m")
    assert p.budget == (60.0, 90.0) and (p.min_len, p.max_len) == (2.0, 4.0) and p.count == 25
    assert p.opener == "drone shot" and p.closer is None and p.text == "dance floor"
    assert p.weights["steady"] >= 0.3
