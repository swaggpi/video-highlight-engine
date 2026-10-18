"""Fixes from the r3 blind-judge field test: precise exhaustive lists, whole first/last words in speech clips,
exclusions read on screen (OCR), speech-free montages, and the LLM parse that drops invented shots."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from heuristic_cases import _dirs, _emb, _engine, _speech_index
from test_corpus import FPS, make_project
from vhe.query import parser as P
from vhe.query import plan
from vhe.query.candidates import sentence_spans
from vhe.query.parser import parse
from vhe.query.refine import (SPEECH_LEAD, SPEECH_TAIL, TALK_MAX_SILENCE, WORD_GUARD, Context, refine,
                              refine_speech, word_clear)
from vhe.rank import select as S
from vhe.rank.select import select_for, strong_matches


# ---- 1. exhaustive lists: strong, distinct matches only -----------------------------------------------------------
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


# ---- 2. speech clips hold their first and last word whole ----------------------------------------------------------
def _words(spec):
    return [{"start": s, "end": e, "word": w} for s, e, w in spec]


TALK = _words([(10.0, 10.3, "We're"), (10.35, 10.7, "here"), (10.75, 11.2, "live."),  # 1.6 s pause
               (12.8, 13.1, "Let"), (13.12, 13.3, "me"), (13.35, 13.9, "pack"), (13.95, 14.4, "up."),  # 0.3 s pause
               (14.7, 15.0, "While"), (15.05, 15.6, "walking."), (25.0, 25.5, "Bye.")])


def test_speech_edges_sit_in_the_pause_with_lead_and_tail():
    ctx = Context(60.0, [], [], TALK)
    s, e = refine_speech(12.9, 14.3, ctx, 1.0, 10.0)  # "Let me pack up." (1.6 s pause before, 0.3 s after)
    assert s == pytest.approx(12.8 - SPEECH_LEAD)  # a full lead-in: the pause is long
    assert e == pytest.approx(14.4 + 0.15)  # the tail stops at the pause midpoint, never into "While"
    s, e = refine_speech(10.1, 11.0, ctx, 1.0, 10.0)  # "We're here live." with a 1.6 s pause after it
    assert e == pytest.approx(11.2 + SPEECH_TAIL)
    for t in (s, e):
        assert word_clear(TALK, t)


def test_no_speech_clip_edge_lies_inside_a_word_or_its_guard():
    rng = np.random.default_rng(3)
    t, words = 1.0, []
    for k in range(300):  # pauses from 0.02 s to 2 s
        L = rng.uniform(0.1, 0.6)
        words.append({"start": t, "end": t + L, "word": "so." if k % 9 == 8 else "word"})
        t += L + rng.choice([0.02, 0.08, 0.2, 0.5, 2.0])
    ctx = Context(t + 5, [12.0, 40.0], [], words)
    for s0 in np.arange(0.0, t - 10, 1.37):
        for L in (2.0, 5.5, 12.0):
            s, e = refine_speech(float(s0), float(s0 + L), ctx, 3.0, 10.0)
            assert word_clear(words, s) and word_clear(words, e), (s0, L, s, e)
            s, e = refine(float(s0), float(s0 + L), ctx, 3.0, 10.0)  # visual clips too
            assert word_clear(words, s) and word_clear(words, e), (s0, L, s, e)


def test_word_clear_is_the_guard_check():
    w = _words([(1.0, 1.5, "a"), (1.6, 2.0, "b"), (5.0, 5.5, "c")])
    assert not word_clear(w, 1.2) and not word_clear(w, 1.5 + WORD_GUARD - 0.02)
    assert word_clear(w, 1.55)  # the midpoint of a 0.1 s pause
    assert not word_clear(w, 2.0 + WORD_GUARD - 0.02) and word_clear(w, 2.0 + WORD_GUARD + 0.02)


def test_talking_track_clip_has_no_long_silent_tail():
    ctx = Context(60.0, [], [], TALK)
    s, e = refine_speech(25.0, 25.5, ctx, 4.0, 10.0)  # "Bye." alone, 9 s of silence before it
    assert e - s == pytest.approx(4.0)  # a visual edit pads the speech clip to its minimum length
    s, e = refine_speech(25.0, 25.5, ctx, 4.0, 10.0, max_silence=TALK_MAX_SILENCE)
    assert 25.0 - s <= SPEECH_LEAD + TALK_MAX_SILENCE + 1e-6 and e - 25.5 <= SPEECH_TAIL + TALK_MAX_SILENCE + 1e-6


def test_a_pause_does_not_end_a_sentence_on_a_hanging_word():
    w = _words([(0.0, 0.3, "I'm"), (0.4, 0.8, "going"), (0.9, 1.1, "to"), (2.6, 3.0, "explore"), (3.1, 3.6, "it."),
                (5.0, 5.4, "Then"), (5.5, 5.9, "we"), (7.5, 7.9, "walk.")])
    assert sentence_spans(w) == [(0, 4), (5, 7)]  # 1.5 s after "to" and "we": not the end of a sentence
    w2 = _words([(0.0, 0.3, "Done"), (2.0, 2.4, "of"), (2.5, 2.9, "course.")])
    assert sentence_spans(w2) == [(0, 2)]  # nor before "of"
    w3 = _words([(0.0, 0.3, "Done"), (2.0, 2.4, "next"), (2.5, 2.9, "one.")])
    assert sentence_spans(w3) == [(0, 0), (1, 2)]


def test_engine_talk_track_edges_are_word_clear(tmp_path):
    root = _speech_index(tmp_path / "talk.mp4.vhe")
    eng = _engine(root)
    words = eng.corpus.words
    res = eng.clips("every moment someone speaks", llm=None)
    assert res["clips"]
    assert any(c["source"] == "speech" for c in res["clips"])
    for c in res["clips"]:
        assert word_clear(words, c["start"]) and word_clear(words, c["end"])
        if c["source"] == "speech":  # whole sentences, in the long pauses between them: full lead and tail
            inside = [w for w in words if c["start"] <= w["start"] and w["end"] <= c["end"]]
            assert inside[0]["start"] - c["start"] == pytest.approx(SPEECH_LEAD, abs=1e-3)
            assert c["end"] - inside[-1]["end"] == pytest.approx(SPEECH_TAIL, abs=1e-3)


# ---- 3. exclusions read on screen ------------------------------------------------------------------------------------
P5 = ("We need a 45 s sizzle. Absolutely no protest signs, no portraits of victims, nothing with 'executed' or "
      "'killed' on it, no empty street.")


def test_exclusion_text_terms_prompts_and_visual_items():
    terms, prompts, visual = plan.exclusion_text(
        ["protest signs", "portraits of victims", "something with 'executed' or 'killed'", "'killed' on it",
         "empty street"], P5)
    assert {"executed", "killed"} <= set(terms) and "freedom" in terms  # quoted + political terms
    assert set(plan.PORTRAIT_PROMPTS) <= set(prompts) and set(plan.POLITICAL_PROMPTS) <= set(prompts)
    assert visual == ["protest signs", "portraits of victims", "empty street"]  # quoted-only items: OCR only
    terms, prompts, visual = plan.exclusion_text(["something political in any of them"], "Nothing political.")
    assert "executed" in terms and visual == [] and prompts
    terms, prompts, visual = plan.exclusion_text(["empty street", "bins"], "no empty street, no bins")
    assert terms == [] and prompts == [] and visual == ["empty street", "bins"]  # unchanged for other items
    # a wanted shot that asks for the same thing keeps it (no prompt that would remove it)
    _, prompts, _ = plan.exclusion_text(["portraits of victims"], "",
                                        wanted=["a poster with a portrait photo of a person in the crowd"])
    assert "a poster with a portrait photo of a person" not in prompts
    assert plan.quoted_terms("the host's talking, 'executed' and \"killed\"") == ["executed", "killed"]


def test_ocr_terms_match_squashed_case_insensitive_and_fragments():
    corpus = SimpleNamespace(n=60, fps=3.0, bounds=np.array([0, 30, 60]), text_items=[
        {"step": 6, "t": 2.0, "text": "tecuted byIRGConApr3", "file": 0},  # a fragment of "executed"
        {"step": 15, "t": 5.0, "text": "SKILLS CENTER", "file": 0},
        {"step": 45, "t": 15.0, "text": "Over 50,000 KILLED", "file": 1},
        {"step": 29, "t": 9.67, "text": "GATE OF THE NATIONS", "file": 0}])
    mask = plan.ocr_term_steps(corpus, ["executed", "killed"])
    hold = int(round(plan.OCR_HOLD_S * 3.0))
    assert mask[6 - hold:7 + hold].all() and mask[45 - hold:46 + hold].all()
    assert mask.sum() == 2 * (2 * hold + 1)  # "SKILLS" is not "killed", the gate float names nothing
    assert plan.ocr_term_steps(corpus, ["regime"]) is None
    assert plan.ocr_term_steps(SimpleNamespace(text_items=[]), ["killed"]) is None


def _ocr_project(root):
    (parade,) = _dirs(1, seed=9)
    specs = [(f"take{k}.mp4", 60.0, {"emb": _emb(np.random.default_rng(30 + k), int(60 * FPS), (parade,),
                                                 [(0, 60, parade)])}) for k in (0, 1)]
    make_project(root, specs)
    for d in sorted((root / ".vhe" / "files").iterdir()):
        if d.name.startswith("take1"):
            items = [[s, "Executed by IRGC on Apr 3", 0.95, [0.1, 0.2, 0.4, 0.3]] for s in range(60, 81, 3)]
            (d / "ocr.json").write_text(json.dumps({"version": 1, "engine": "fake", "every": 3, "min_conf": 0.7,
                                                    "items": items}))
    return root, parade


def test_engine_drops_every_clip_that_shows_a_forbidden_word(tmp_path):
    root, parade = _ocr_project(tmp_path / "ocr")
    vec = {"the parade": parade, "parade": parade}
    eng = _engine(root, vec)
    sign = (eng.corpus.offsets[1] + 20.0 - plan.OCR_HOLD_S, eng.corpus.offsets[1] + 26.67 + plan.OCR_HOLD_S)

    def shows(c):
        return c["file"] == "take1.mp4" and c["start"] < sign[1] and c["end"] > sign[0]
    plain = eng.clips("12 clips of the parade, 4-6 s each", llm=None)
    assert any(shows(c) for c in plain["clips"])  # the sign's stretch is good footage otherwise
    res = eng.clips("12 clips of the parade, 4-6 s each. Nothing with 'executed' on it.", llm=None)
    assert res["clips"] and not any(shows(c) for c in res["clips"])


# ---- 4. speech-free montages -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("brief", [
    "A 90 s recap, music bed to be added later.",
    "Exclude anything shaky and the host's talking (it goes under music).",
    "5 calm shots to sit under a voice-over.",
    "Every performer select; I will lay music over the muted ones.",
    "8 cutaways, no host talking over it.",
    "ein 30 s video, musik drunter"])
def test_speech_free_cues(brief):
    p = parse(brief)
    assert p.no_speech and p.weights["speech"] <= -1.0


def test_speech_free_cue_does_not_override_a_talking_brief():
    p = parse("Every moment the host talks about the parade, music bed later.")
    assert not p.no_speech
    assert not parse("Use the host's own voice as the narration bed.").no_speech


def test_speech_free_brief_gates_out_every_clip_with_speech(tmp_path):
    root = _speech_index(tmp_path / "talk.mp4.vhe")
    eng = _engine(root)
    speech = eng.corpus.grid["speech"]
    assert plan.NO_SPEECH_MAX_FRAC <= 0.05
    res = eng.clips("8 clips 3-5 s each, the music bed goes on later", llm=None)
    assert res["clips"]
    for c in res["clips"]:
        i0, i1 = int(c["start"] * FPS), int(np.ceil(c["end"] * FPS))
        assert np.mean(speech[i0:i1] > 0.5) <= plan.NO_SPEECH_MAX_FRAC


# ---- 5. the LLM's invented shots and the rules' instruction clauses ------------------------------------------------
TIKTOK = ("dont cut anything yet. just give me the 3 best possible first-shots for a parade tiktok, ranked. 2-3 sec "
          "each max, has to work in the first second with no context, big motion or a face right there, centered. "
          "tell me the clip and the exact seconds and one line why it hooks. bonus if the sound on it is actually "
          "loud not muted")


def _llm(monkeypatch, answer):
    monkeypatch.setattr(P, "_ask_llm", lambda query, model, host: (answer, True))


def test_invented_llm_shots_fall_back_to_no_shot_list_not_instruction_clauses(monkeypatch):
    assert len(parse(TIKTOK).shots) >= 4  # the rules read the clauses as shots
    _llm(monkeypatch, {"content": "parade tiktok first shots", "shots": ["confetti explosion", "rider flying at us"],
                       "hook": True, "centered": True})
    p = P.parse_with_llm(TIKTOK, "fake")
    assert p.shots == [] and p.text == "parade tiktok first shots"
    assert p.hook and p.framing == "close" and p.centered and p.count == 3  # the brief's cues still stand


def test_rules_shots_that_name_things_survive_the_fallback(monkeypatch):
    brief = "Cut 5 clips. Show the dancers, the marching band and the giant flag."
    _llm(monkeypatch, {"content": "parade", "shots": ["fireworks over the bay"]})  # invented: dropped
    assert P.parse_with_llm(brief, "fake").shots == parse(brief).shots == ["dancers", "marching band", "giant flag"]


def test_instruction_clauses():
    for x in ("dont cut anything yet", "tell me the clip and the exact seconds", "ranked", "max",
              "mein handy ist voll lol", "welche sind die 5 die ich auf jeden fall behalten soll"):
        assert P._instruction(x), x
    for x in ("big motion", "dancers in costume", "the giant flag", "marching band"):
        assert not P._instruction(x), x


def test_keep_or_delete_brief_reads_its_count_and_no_budget():
    q = ("mein handy ist voll lol. welche von den parade clips sind müll (boden, verwackelt, hand vor der linse, oder "
         "einfach 2 minuten nix) die ich direkt löschen kann? und welche sind die 5 die ich auf jeden fall behalten "
         "soll")
    p = parse(q)
    assert p.budget is None and p.count == 5 and p.count_explicit  # "2 minuten nix" is junk, not a runtime
    _llm_answer = {"content": "parade clips", "shots": []}
    p = P._finalise(*P._from_llm(_llm_answer, P._parse_rules(q)[0], False), None, None, None, None, None)
    assert p.shots == [] and p.highlight and p.count == 5
    assert parse("cut 2 minutes of nothing from the 60-90 s recap").budget == (60.0, 90.0)
