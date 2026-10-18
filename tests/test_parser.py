"""The brief parser: rules (vhe.query.rules) and the grounded LLM path (vhe.query.llm), on synthetic briefs."""
import pytest

from test_plan import _answer, _mock_ollama
from vhe.query import llm, rules
from vhe.query.llm import parse_with_llm
from vhe.query.rules import parse


# ---- LLM reliability and grounding -----------------------------------------------------------------------------------
def test_llm_parse_unloads_the_model_immediately(monkeypatch):
    calls = _mock_ollama(monkeypatch, _answer(content="dogs"))
    parse_with_llm("three clips of dogs", model="qwen3:4b")
    assert calls[0]["keep_alive"] == 0  # not resident while SigLIP / CG-DETR load on the same GPU


def test_llm_items_are_translated_and_kept(monkeypatch):
    calls = _mock_ollama(monkeypatch, _answer(content="parade highlights", shots=["horses", "giant flag", "dancers"],
                                              exclude=["empty streets"]))
    p = parse_with_llm(GERMAN, model="qwen3:4b")
    assert p.shots == ["horses", "giant flag", "dancers"]  # English, not grounded word by word in German
    assert "empty streets" in p.exclude and p.budget == (25.5, 34.5)
    assert "English" in calls[0]["messages"][0]["content"]
    english = parse_with_llm("Show the dogs and the cats.", model="other")  # English briefs stay grounded
    assert "horses" not in english.shots


# ---- multilingual and vague briefs -----------------------------------------------------------------------------------
GERMAN = ("hey kannst du mir die besten momente von der parade raussuchen, so die pferde, die riesen fahne und die "
          "tänzerinnen, nich zu lang so 30 sek für insta. keine leeren strassen")


def test_german_rules_read_budget_length_and_negation():
    p = parse(GERMAN)
    assert p.budget == (25.5, 34.5)  # "so 30 sek für insta": the whole post
    assert p.max_len <= 6.0  # "nich zu lang" is short, not "lang"
    assert p.exclude and "leeren strassen" in " ".join(p.exclude)
    assert parse("3 clips ohne autos").exclude == ["autos"]
    assert parse("szenen von 4-6 sekunden, nicht zu lang").max_len == 6.0


# ---- speech-free montages --------------------------------------------------------------------------------------------
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


# ---- an LLM's invented shots, instruction clauses, keep-or-delete briefs ---------------------------------------------
TIKTOK = ("dont cut anything yet. just give me the 3 best possible first-shots for a parade tiktok, ranked. 2-3 sec "
          "each max, has to work in the first second with no context, big motion or a face right there, centered. "
          "tell me the clip and the exact seconds and one line why it hooks. bonus if the sound on it is actually "
          "loud not muted")


def _llm(monkeypatch, answer):
    monkeypatch.setattr(llm, "_ask_llm", lambda query, model, host: (answer, True))


def test_invented_llm_shots_fall_back_to_no_shot_list_not_instruction_clauses(monkeypatch):
    assert len(parse(TIKTOK).shots) >= 4  # the rules read the clauses as shots
    _llm(monkeypatch, {"content": "parade tiktok first shots", "shots": ["confetti explosion", "rider flying at us"],
                       "hook": True, "centered": True})
    p = llm.parse_with_llm(TIKTOK, "fake")
    assert p.shots == [] and p.text == "parade tiktok first shots"
    assert p.hook and p.framing == "close" and p.centered and p.count == 3  # the brief's cues still stand


def test_rules_shots_that_name_things_survive_the_fallback(monkeypatch):
    brief = "Cut 5 clips. Show the dancers, the marching band and the giant flag."
    _llm(monkeypatch, {"content": "parade", "shots": ["fireworks over the bay"]})  # invented: dropped
    assert llm.parse_with_llm(brief, "fake").shots == parse(brief).shots == ["dancers", "marching band", "giant flag"]


def test_instruction_clauses():
    for x in ("dont cut anything yet", "tell me the clip and the exact seconds", "ranked", "max",
              "mein handy ist voll lol", "welche sind die 5 die ich auf jeden fall behalten soll"):
        assert llm._instruction(x), x
    for x in ("big motion", "dancers in costume", "the giant flag", "marching band"):
        assert not llm._instruction(x), x


def test_keep_or_delete_brief_reads_its_count_and_no_budget():
    q = ("mein handy ist voll lol. welche von den parade clips sind müll (boden, verwackelt, hand vor der linse, oder "
         "einfach 2 minuten nix) die ich direkt löschen kann? und welche sind die 5 die ich auf jeden fall behalten "
         "soll")
    p = parse(q)
    assert p.budget is None and p.count == 5 and p.count_explicit  # "2 minuten nix" is junk, not a runtime
    _llm_answer = {"content": "parade clips", "shots": []}
    p = rules.finalise(*llm._from_llm(_llm_answer, rules.parse_rules(q)[0], False), None, None, None, None, None)
    assert p.shots == [] and p.highlight and p.count == 5
    assert parse("cut 2 minutes of nothing from the 60-90 s recap").budget == (60.0, 90.0)


# ---- the CLI's --budget ----------------------------------------------------------------------------------------------
def test_cli_budget_accepts_unicode_dashes():
    from vhe.cli import _budget

    assert _budget("60—90") == _budget("60–90") == _budget("60-90") == (60.0, 90.0) and _budget("75") == 75.0
