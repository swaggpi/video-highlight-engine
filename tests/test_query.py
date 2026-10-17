import numpy as np

from vhe.query.candidates import cuts_inside, generate, peak_times
from vhe.query.parser import parse
from vhe.query.refine import Context, refine
from vhe.query.score import robust_z, segment_scores
from vhe.query.select import overlap_fraction, select


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
    assert parse("highlights total 60 seconds").budget_s == 60.0


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
    assert overlap_fraction((0, 10), (5, 15)) == 0.5


def test_refine_snaps_to_cut_and_word():
    ctx = Context(duration=60, cuts=[10.0], vad=[(20.0, 26.0)],
                  words=[{"start": 24.5, "end": 25.2, "word": "hello"}])
    s, e = refine(10.4, 25.0, ctx, min_len=3, max_len=30)
    assert abs(s - 10.04) < 1e-6  # just after the cut
    assert abs(e - 25.3) < 1e-6  # after the word, not mid-word


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
