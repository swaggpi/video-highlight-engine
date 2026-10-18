"""Streamer / VOD briefs: stream phrases in the rules parser, the reaction and excited-speech signals
(vhe.signals.reaction), quoted lines against the transcript (vhe.query.text_match.line_steps) and how the heuristic
composes them (features v4). Synthetic data only."""
import numpy as np

from vhe.query.model import EXHAUSTIVE_CAP, TYPICAL_LEN, ParsedQuery
from vhe.query.rules import finalise, parse
from vhe.query.text_match import line_steps, quoted_lines
from vhe.rank.features import FEATURES, INDEX, query_vector
from vhe.rank.scorer import QUOTE_BONUS, HeuristicScorer, compose_prior
from vhe.rank.select import DUP_SIM_WINDOW_S, duplicates
from vhe.signals.reaction import excitement_steps, reaction_steps, word_points

FPS = 3.0

VOD_HIGHLIGHTS = ('Cut a 8-12 min "first time playing Granny" highlights video from this VOD, with the best moments in '
                  'stream order. Open with the face-filter intro and the "we are going to Granny" hype, then the first '
                  'jump scare, the chases and deaths, the bear trap and the Harvard bit, and end on his "W first time '
                  'playing Granny" verdict. Cut all dead air: the launch and options menus, him leaving the desk, the '
                  'pause menu, slow searching and the shout-out outro.')


# ---- parser: stream phrases ------------------------------------------------------------------------------------------
def test_dead_air_clause_is_a_soft_avoid_list_not_shots_and_sets_no_mood():
    p = parse(VOD_HIGHLIGHTS)
    assert p.avoid[:3] == ["launch and options menus", "him leaving the desk", "pause menu"]
    assert "shout-out outro" in p.avoid and p.exclude == []  # preferably: no hard SigLIP gate on "searching"
    assert not any("menu" in x or "desk" in x for x in p.shots + [p.text])
    assert p.mood is None  # "slow searching" is something to cut, not a calm mood
    assert "stream order" not in p.text and "vod" not in p.text  # ordering rule and the source are no content


def test_highlights_video_is_a_reaction_brief_of_moments():
    p = parse(VOD_HIGHLIGHTS)
    assert p.highlight and p.reaction and p.shots == []  # named events become one content text
    assert "bear trap" in p.text and "harvard bit" in p.text
    assert (p.min_len, p.max_len) == TYPICAL_LEN["moment"] and p.budget == (480.0, 720.0)
    assert 20 <= p.count <= 50  # a few dozen moments, not a hundred 4 s cuts
    assert p.weights["energy"] > 0
    # the opener/closer keep their quotes; they are not body lines. The named moments are looked up in the transcript.
    assert '"we are going to granny"' in p.opener and "verdict" in p.closer
    assert p.lines[:2] == ["first jump scare", "chases and deaths"] and any("bear trap" in x for x in p.lines)
    assert not any("going to" in x or "verdict" in x for x in p.lines)


def test_every_scare_death_and_clutch_is_exhaustive_over_reactions():
    p = parse("List every jump scare, every time Granny catches him, every Game Over screen, and every clutch play. "
              "Give a 3-8 s clip for each, for a fast-cut montage. "
              "Mark each clip as \"scare/death\" or \"clutch/win\".")
    assert p.exhaustive and p.reaction and p.highlight and p.shots == []
    assert p.count == EXHAUSTIVE_CAP and (p.min_len, p.max_len) == (3.0, 8.0)
    assert not any("scare/death" in x or "clutch/win" in x for x in p.lines)  # labels, not spoken lines


def test_shorts_count_per_clip_ending_and_output_instructions():
    p = parse("Pick three 30-60 s moments that work as vertical Shorts. Each must have a strong first second: a scream "
              "or a punchline right at the start. Each must make sense without context and end on a payoff. "
              "For each, give the in and out points and a one-line hook/caption.")
    assert p.count == 3 and p.count_explicit and (p.min_len, p.max_len) == (30.0, 60.0)
    assert p.closer is None  # "each ... end on a payoff" is how every clip ends, not the edit's last shot
    assert p.reaction and not p.topics
    assert "points" not in p.text and "caption" not in p.text


def test_cold_open_is_not_a_legible_text_brief_and_has_no_topic():
    p = parse("Find the single best 10-20 s moment to put before the intro as a cold open: the biggest reaction in the "
              "stream, readable without context. Give the exact in and out points and say why it beats the runner-up.")
    assert p.count == 1 and p.reaction and not p.legible_text and p.topics == []


def test_quoted_lines_and_funniest_moments():
    p = parse('Find the 5 funniest moments, 15-60 s each, with in and out points, and rank them. Look for laughs. '
              'Candidates include the "Granny" face filter, the "I made it into Harvard" speech and "she thought I '
              'was a watermelon".')
    quotes = ["I made it into Harvard", "she thought I was a watermelon"]  # "Granny": a name, not a line
    assert p.lines[:2] == quotes and "granny face filter" in p.lines  # then the named moments
    assert p.reaction and p.count == 5 and (p.min_len, p.max_len) == (15.0, 60.0)
    assert "rank them" not in p.text and "out points" not in p.text and "look for" not in p.text
    # the LLM path ends in finalise too: lines and the reaction flag come from the raw brief
    q = finalise(ParsedQuery(raw=p.raw, text="funny moments", shots=["face filter", "harvard speech"]), False, None,
                 None, None, None, None)
    assert q.reaction and q.highlight and q.shots == [] and q.lines == [*quotes, "face filter", "harvard speech"]


def test_camera_briefs_are_not_reaction_briefs():
    for brief in ("Cut a 60-90 s energetic recap of the Persian Day Parade. Open on an establishing wide.",
                  "mein handy ist voll lol. welche von den parade clips sind müll? und welche sind die 5 die ich "
                  "behalten soll",
                  "Pick 5 calm B-roll shots of 3-6 s each, no whip-pans, nothing blocking the lens."):
        p = parse(brief)
        assert not p.reaction and p.lines == [], brief
    assert parse("Cut a 60-90 s energetic recap of the Persian Day Parade.").mood == "energetic"


# ---- signals ---------------------------------------------------------------------------------------------------------
def test_reaction_marks_a_burst_over_the_files_own_baseline():
    rng = np.random.default_rng(0)
    n = int(300 * FPS)
    quiet = -40 + rng.normal(0, 1.0, n)
    loud = -12 + rng.normal(0, 1.0, n)  # a loud file at a constant level: nothing reacts in it
    quiet[int(150 * FPS):int(153 * FPS)] += 20  # a 3 s scream in the quiet file
    onset = np.abs(rng.normal(1, 0.2, 2 * n))
    file_id = np.repeat([0, 1], n)
    r = reaction_steps(np.r_[quiet, loud], onset, np.ones(2 * n, bool), FPS, file_id)
    assert r.shape == (2 * n,) and np.abs(r).max() <= 3
    peak = int(np.argmax(r))
    assert 148 * FPS <= peak <= 155 * FPS
    assert r[peak] >= 1.5 and r[n:].max() < 0.5  # relative to each file's own level, not absolute loudness
    assert not reaction_steps(np.r_[quiet, loud], onset, np.zeros(2 * n, bool), FPS, file_id).any()  # no audio


def test_excited_speech_counts_exclamations_laughs_caps_and_repeats():
    calm = [{"start": t, "end": t + 0.4, "word": w}
            for t, w in zip(np.arange(0, 60, 0.5), ["we", "walk", "on"] * 40, strict=True)]
    hot = [{"start": 30 + 0.3 * k, "end": 30.25 + 0.3 * k, "word": w}
           for k, w in enumerate(["no", "no", "NO!", "haha", "what?!", "run", "run", "RUN!"])]
    assert word_points(hot[:3]) == [0.5, 1.0, 3.0] and word_points([{"word": "hahaha"}]) == [1.5]
    words = sorted(calm + hot, key=lambda w: w["start"])
    x = excitement_steps(words, int(60 * FPS), FPS)
    assert int(np.argmax(x)) in range(int(28 * FPS), int(36 * FPS))
    assert x[int(31 * FPS)] > 1.5 > x[int(10 * FPS)]
    assert not excitement_steps([], 30, FPS).any()


def test_quoted_line_matches_the_paraphrase_where_it_is_said():
    text = ("so we go upstairs okay where is she I finally made the Harvard I want to share this news "
            "with you granny where are you granny this house has many rooms made of wood").split()
    words = [{"start": 0.5 * k, "end": 0.5 * k + 0.4, "word": w} for k, w in enumerate(text)]
    n = int(len(text) * 0.5 * FPS) + 3
    m = line_steps(words, ["I made it into Harvard", "we are going to granny"], n, FPS)
    said = np.flatnonzero(m[0] > 0) / FPS
    k = text.index("made")
    assert m[0].max() == 1.0 and said.min() >= 0.5 * k - 0.4 and said.max() <= 0.5 * (k + 2) + 0.5
    assert not m[1].any()  # stopword-heavy line: "where are you granny" shares only function words and the name
    assert quoted_lines('the "I made it into Harvard" bit, "Granny", and "scare/death"') == ["I made it into Harvard"]


# ---- heuristic composition -------------------------------------------------------------------------------------------
def test_reaction_brief_ranks_reactions_and_quoted_lines_up():
    X = np.zeros((3, len(FEATURES)))
    X[:, INDEX["length_s"]] = 15.0
    X[1, INDEX["reaction_mean"]] = X[1, INDEX["excitement_mean"]] = 2.0  # a scream with excited speech
    X[2, INDEX["quote_match"]] = 1.0  # the quoted line is said
    X[2, INDEX["highlight_prior_mean"]] = 1.0
    h = HeuristicScorer()
    react = finalise(ParsedQuery(raw="the 5 funniest moments", text=""), False, None, None, None, None, None)
    plain = parse("the best moments of the parade")
    assert react.reaction and plain.highlight and not plain.reaction
    s_react, s_plain = h.score(X, query_vector(react)), h.score(X, query_vector(plain))
    assert s_react[1] - s_react[0] > s_plain[1] - s_plain[0] > 0  # the reaction brief leans on reactions
    rel_react = h.relevance(X, query_vector(react), final=False)[0]
    np.testing.assert_allclose(rel_react[2] - compose_prior(1.0, 0.0, 0.0, True), QUOTE_BONUS)
    assert compose_prior(1.0, 0.0, 0.0, False) > compose_prior(1.0, 0.0, 0.0, True)  # visual prior shares less
    # with a content text, the avoid penalty costs in full ("cut the menus"), not blended away with the text
    menus = finalise(ParsedQuery(raw="the funniest moments, cut the menus", text="funny moments", avoid=["menus"]),
                     False, None, None, None, None, None)
    Y = X[:1].copy()
    Y[0, INDEX["avoid_pen"]] = 1.0
    Y[0, INDEX["h_rel_mean"]] = -1.0  # h_rel holds the penalty already
    np.testing.assert_allclose(h.relevance(Y, query_vector(menus), final=False)[0], -1.0)


# ---- selection on one long recording -----------------------------------------------------------------------------
def test_long_recordings_repeat_a_look_without_repeating_a_moment():
    """One 85-min livestream reads cosine >= 0.92 between almost any two spans: the same-file cosine rule holds only
    within DUP_SIM_WINDOW_S of a pick (the gap rule as before)."""
    o = {"file": 0, "start": 0.0, "end": 10.0}
    cands = [{"file": 0, "start": 3.0, "end": 8.0}, {"file": 0, "start": 40.0, "end": 50.0},
             {"file": 0, "start": 10 + DUP_SIM_WINDOW_S + 30, "end": 10 + DUP_SIM_WINDOW_S + 40},
             {"file": 1, "start": 0.0, "end": 10.0}]
    same = np.ones((4, 3)) / np.sqrt(3)  # every span looks the same
    assert duplicates(cands, same, o, same[0]).tolist() == [True, True, False, False]
