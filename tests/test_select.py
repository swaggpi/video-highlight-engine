"""Selection (vhe.rank.select): near-duplicates, item coverage, budgets and exhaustive lists."""
from types import SimpleNamespace

import numpy as np

from vhe.query.rules import parse
from vhe.rank import select as S
from vhe.rank.select import duplicates, select, select_for, strong_matches


# ---- near-duplicates and item coverage -------------------------------------------------------------------------------
def _cand(file, s, e, total, source="window", item=-1):
    return {"file": file, "start": s, "end": e, "total": total, "source": source, "item": item}


def test_same_action_and_lookalikes_are_picked_once():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(8, 16))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[6] = v[5]  # same file, 60 s apart, but the same static view
    cands = [_cand(0, 10, 14, 9.0), _cand(0, 15, 19, 8.9), _cand(0, 18.5, 22.5, 8.8), _cand(0, 6, 9.5, 8.7),  # one pass
             _cand(1, 40, 44, 5.0), _cand(2, 100, 104, 7.0), _cand(2, 160, 164, 6.9), _cand(3, 200, 204, 4.0)]
    picked = select(cands, v, n=5, max_overlap=0.0)
    files = sorted((c["file"], c["start"]) for c in picked)
    assert sum(f == 0 for f, _ in files) == 1  # one clip of the mounted-police-like pass, not four
    assert sum(f == 2 for f, _ in files) == 1  # the static twin is dropped
    assert len(picked) == 4  # nothing else left: stops short instead of padding with duplicates
    assert len(select(cands, v, n=5, max_overlap=0.0, dedupe=False)) == 5  # the old behaviour, for contrast


def test_adjacent_sentences_of_a_talking_track_are_not_duplicates():
    v = np.tile(np.eye(4)[:1], (3, 1))  # same picture throughout
    cands = [_cand(0, 0, 4, 3.0, "speech"), _cand(0, 4.5, 9, 2.9, "speech"), _cand(0, 9.5, 12, 2.8, "speech")]
    assert len(select(cands, v, n=3, max_overlap=0.0, talk_track=True)) == 3
    assert not duplicates(cands[1:], v[1:], cands[0], v[0], talk_track=True).any()
    # in a visual edit the operator talking over the same shot does not make three clips of it
    assert len(select(cands, v, n=3, max_overlap=0.0)) == 1


def test_round_robin_moves_on_instead_of_padding_an_item():
    v = np.eye(6)
    cands = [_cand(0, 10, 14, 9.0, item=0), _cand(0, 15, 19, 8.9, item=0), _cand(0, 18.5, 22.5, 8.8, item=0),
             _cand(1, 30, 34, 5.0, item=1), _cand(2, 40, 44, 4.0, item=1), _cand(3, 50, 54, 1.0, item=-1)]
    picked = select(cands, v, n=4, max_overlap=0.0, items=np.array([c["item"] for c in cands]))
    assert [c["file"] for c in picked].count(0) == 1 and len(picked) == 4  # item 1 twice, then the unassigned clip


def test_a_count_that_cannot_fill_the_budget_does_not_cap_it():
    cands = [_cand(k, 20.0 * k, 20.0 * k + 10, 10 - 0.01 * k) for k in range(30)]  # one 10 s moment per file
    p = SimpleNamespace(count=3, count_explicit=True, budget=(100.0, 140.0), max_len=12.0, exhaustive=False,
                        weights={}, shots=[])
    picked = select_for(cands, np.eye(30), p, groups=np.arange(30))
    assert sum(c["end"] - c["start"] for c in picked) >= 100.0


def test_every_shot_where_returns_every_distinct_match_not_one_per_item():
    assert not parse("I need every shot where text is legible: float names, banners, street signs.").per_item
    assert parse("List every float: the Lotus, the Dragon and the Tower.").per_item
    cands = [_cand(k, 10.0 * k, 10.0 * k + 4, 3.0 - 0.1 * k, item=k % 2) for k in range(8)]
    item_rel = np.array([[2.0, 0.0] if k % 2 == 0 else [0.0, 2.0] for k in range(8)])
    base = dict(count=50, count_explicit=False, budget=None, max_len=6.0, exhaustive=True, weights={"relevance": 1.0},
                shots=["float names", "banners"])
    every = select_for(cands, np.eye(8), SimpleNamespace(**base, per_item=False), item_rel=item_rel)
    assert len(every) == 8  # all distinct matches above the threshold
    per = select_for(cands, np.eye(8), SimpleNamespace(**base, per_item=True), item_rel=item_rel)
    assert len(per) == 2  # one per named item


# ---- exhaustive lists: precision over recall -------------------------------------------------------------------------
def _p(**kw):
    base = dict(count=50, count_explicit=False, budget=None, max_len=8.0, exhaustive=True, per_item=False,
                weights={"relevance": 2.0, "speech": 0.0}, shots=["dancers", "anyone moving"])
    return SimpleNamespace(**{**base, **kw})


def test_every_shot_keeps_strong_matches_of_each_item_only():
    # item 0 (specific): best 5.0; item 1 (generic): best 4.0. Each in its own file, 20 s apart (no duplicates).
    spec = [  # (item, item relevance, total, penalty, unusable)
        (0, 1.2, 5.0, 0.0, 0.0), (0, 1.0, 5.0 - S.EXHAUSTIVE_SCORE_MARGIN + 0.1, 0.0, 0.0),  # kept: within margins
        (0, 1.1, 5.0 - S.EXHAUSTIVE_SCORE_MARGIN - 0.1, 0.0, 0.0),  # padding: too far below the item's best score
        (0, 0.45, 4.9, 0.0, 0.0),  # under the absolute relevance floor
        (1, 2.6, 4.0, 0.0, 0.0), (1, 2.6 - S.EXHAUSTIVE_ITEM_MARGIN - 0.1, 3.9, 0.0, 0.0),  # weakest generic match
        (0, 1.2, 4.8, 1.5, 0.0),  # unclean edges
        (0, 1.2, 4.8, 0.0, 0.2),  # unusable steps inside
        (-1, 0.0, 6.0, 0.0, 0.0),  # serves no item
    ]
    cands, rows = [], []
    for k, (it, rel, tot, pen, unus) in enumerate(spec):
        cands.append({"file": k, "start": 20.0 * k, "end": 20.0 * k + 5, "total": tot, "penalty": pen,
                      "unusable_frac": unus, "item": it, "source": "window"})
        row = np.zeros(2)
        if it >= 0:
            row[it] = rel
        rows.append(row)
    item_rel = np.array(rows)
    keep = strong_matches(cands, np.array([c["item"] for c in cands]), item_rel)
    assert keep == [0, 1, 4]
    picked = select_for(cands, np.eye(len(cands)), _p(), item_rel=item_rel)
    assert sorted(c["file"] for c in picked) == [0, 1, 4]


def test_every_shot_never_returns_back_to_back_takes_of_one_view():
    # same file: 8 s apart and nearly the same view (cosine 0.9, under DUP_SIM) -> once; 8 s apart, another view ->
    # both; 30 s apart, the same nearly-same view -> both
    v = np.zeros((4, 3))
    v[0] = [1, 0, 0]
    v[1] = v[3] = [0.9, np.sqrt(1 - 0.81), 0]
    v[2] = [0, 0, 1]
    cands = [{"file": 0, "start": 0.0, "end": 4.0, "total": 5.0, "item": 0},
             {"file": 0, "start": 12.0, "end": 16.0, "total": 4.9, "item": 0},
             {"file": 0, "start": 24.0, "end": 28.0, "total": 4.8, "item": 0},
             {"file": 0, "start": 58.0, "end": 62.0, "total": 4.7, "item": 0}]
    item_rel = np.ones((4, 2))
    picked = select_for(cands, v, _p(), item_rel=item_rel)
    assert sorted(c["start"] for c in picked) == [0.0, 24.0, 58.0]
    # the ordinary (non-exhaustive) duplicate rule is unchanged: 8 s apart, same view, two picks
    assert not S.duplicates(cands[1:2], v[1:2], cands[0], v[0])[0]
    assert S.duplicates(cands[1:2], v[1:2], cands[0], v[0], back_to_back=(10.0, 0.9))[0]


def test_every_shot_list_is_capped_unless_a_count_is_given():
    cands = [{"file": k, "start": 10.0 * k, "end": 10.0 * k + 4, "total": 5.0, "relevance": 1.0} for k in range(40)]
    one = SimpleNamespace(**{**vars(_p()), "shots": []})
    assert len(select_for(cands, np.eye(40), one)) == S.EXHAUSTIVE_VISUAL_CAP
    counted = SimpleNamespace(**{**vars(one), "count": 30, "count_explicit": True})
    assert len(select_for(cands, np.eye(40), counted)) == 30


def test_every_moment_someone_speaks_keeps_every_sentence():
    # speech exhaustive lists are not cut by the visual precision margins
    cands = [{"file": 0, "start": 10.0 * k, "end": 10.0 * k + 3, "total": 5.0 - k, "source": "speech",
              "speech_frac": 0.9} for k in range(5)]
    p = _p(shots=[], weights={"relevance": 0.0, "speech": 1.0})
    assert len(select_for(cands, np.eye(5), p)) == 5  # all above EXHAUSTIVE_MIN_TOTAL (5, 4, 3, 2, 1 >= 0.5)
