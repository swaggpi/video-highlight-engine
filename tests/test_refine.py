"""Clip edges (vhe.query.refine): whole sentences and whole words for speech, clean frames for visual clips."""
import numpy as np
import pytest

from test_corpus import FPS
from vhe.query.candidates import fit_utterances, sentence_spans, speech_spans
from vhe.query.refine import (
    SPEECH_LEAD,
    SPEECH_TAIL,
    TALK_MAX_SILENCE,
    WORD_GUARD,
    Context,
    edges_clean,
    refine,
    refine_speech,
    word_clear,
)


# ---- whole sentences, clean visual edges -----------------------------------------------------------------------------
def _words(spec):
    """[(start, end, word)] -> transcript words."""
    return [{"start": s, "end": e, "word": w} for s, e, w in spec]


SPEECH = _words([(10.0, 10.4, "So"), (10.5, 11.0, "we"), (11.1, 11.6, "are"), (11.7, 12.3, "here."),
                 (12.6, 13.0, "The"), (13.1, 13.6, "music"), (13.7, 14.0, "is"), (14.1, 14.8, "muted"),
                 (14.9, 15.6, "because"), (15.7, 16.0, "of"), (16.1, 17.0, "copyright."),
                 (17.4, 17.8, "Next"), (17.9, 18.2, "we"), (18.3, 18.8, "go."), (25.0, 25.6, "Bye.")])


def test_speech_clip_is_whole_sentences():
    ctx = Context(60.0, [], [], SPEECH)
    s, e = refine_speech(13.05, 15.0, ctx, 2.0, 10.0)  # starts on "music", ends on "because": both mid-sentence
    assert s <= 12.6 and s >= 12.3  # before "The", after "here."
    assert 17.0 <= e <= 17.4  # after "copyright.", before "Next"
    s, e = refine_speech(13.05, 15.0, ctx, 2.0, 3.0)  # the sentence does not fit 3 s: it is still not cut mid-word
    assert not any(w["start"] < s < w["end"] or w["start"] < e < w["end"] for w in SPEECH)
    assert e - s <= 3.0 + 2 * 0.15 + 1e-6  # and the clip stays within the length limit (plus the lead-in/out)


def test_sentences_run_across_asr_segments_and_short_ones_grow_by_whole_neighbours():
    # ASR split the second sentence over two segments: the second segment starts mid-sentence.
    segs = [{"start": 10.0, "end": 14.0, "words": SPEECH[:7]}, {"start": 14.1, "end": 26.0, "words": SPEECH[7:]}]
    spans = speech_spans(segs, vad=[], max_len=30, gap=0.2)
    assert any(s <= 12.6 and e >= 17.0 for s, e in spans)  # "The music ... copyright." is one unit
    assert not any(12.6 < s < 17.0 for s, e in spans)  # nothing starts inside it
    fitted = fit_utterances(spans, lo=5.0, hi=12.0, duration=60.0)
    for s, e in fitted:  # every fitted clip starts and ends between words
        assert not any(w["start"] < s < w["end"] or w["start"] < e < w["end"] for w in SPEECH)
        assert e - s >= 5.0 - 1e-6


def test_visual_clip_does_not_end_inside_a_whip_pan():
    n = int(60 * FPS)
    clean = np.ones(n, dtype=bool)
    clean[int(19.5 * FPS):int(21.5 * FPS)] = False  # whip-pan 19.5-21.5 s
    clean[int(30 * FPS):int(30.7 * FPS)] = False  # black frames at 30 s
    ctx = Context(60.0, [], [], [], None, FPS, clean=clean)
    s, e = refine(15.0, 21.0, ctx, 3.0, 8.0)
    assert e <= 19.5 + 1e-6 and e - s >= 3.0 and edges_clean(s, e, ctx)
    s, e = refine(30.0, 35.0, ctx, 3.0, 8.0)
    assert s >= 30.66 and edges_clean(s, e, ctx)
    assert not edges_clean(15.0, 21.0, ctx)


# ---- first and last words whole, no dead air -------------------------------------------------------------------------
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
