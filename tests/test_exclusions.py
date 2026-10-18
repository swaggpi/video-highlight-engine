"""Hard exclusions (vhe.query.exclusions): hard vs soft wording, defects, terms and prompts read on screen."""
from types import SimpleNamespace

import numpy as np

from vhe.query.exclusions import (
    OCR_HOLD_S,
    POLITICAL_PROMPTS,
    PORTRAIT_PROMPTS,
    conflicts,
    defect_kind,
    exclusion_text,
    ocr_term_steps,
    quoted_terms,
)
from vhe.query.rules import parse


def test_exclusion_wording_is_hard_or_soft():
    hard = parse("5 clips of dogs. No birds, never cats and absolutely no horses.")
    assert set(hard.exclude) >= {"birds", "cats", "horses"} and set(hard.exclude) <= set(hard.avoid)
    soft = parse("5 clips of dogs. Avoid birds, preferably no cats.")
    assert soft.avoid and not soft.exclude
    talk = parse("8 cutaways, steady, no host talking over it.")
    assert talk.no_speech and talk.weights["speech"] < 0
    host = parse("A 90 s recap. Exclude anything shaky and the host's talking (it goes under music).")
    assert host.no_speech and not any("talk" in x for x in host.avoid) and "shaky" in " ".join(host.exclude)
    assert defect_kind("whip-pans") == "whip" and defect_kind("something blocking the lens") == "blocked"
    assert defect_kind("camera points at the sky") == "framing" and defect_kind("protest signs") is None
    assert conflicts("frame with a barricade", ["establishing wide with the barricades"])


P5 = ("We need a 45 s sizzle. Absolutely no protest signs, no portraits of victims, nothing with 'executed' or "
      "'killed' on it, no empty street.")


def test_exclusion_text_terms_prompts_and_visual_items():
    terms, prompts, visual = exclusion_text(
        ["protest signs", "portraits of victims", "something with 'executed' or 'killed'", "'killed' on it",
         "empty street"], P5)
    assert {"executed", "killed"} <= set(terms) and "freedom" in terms  # quoted + political terms
    assert set(PORTRAIT_PROMPTS) <= set(prompts) and set(POLITICAL_PROMPTS) <= set(prompts)
    assert visual == ["protest signs", "portraits of victims", "empty street"]  # quoted-only items: OCR only
    terms, prompts, visual = exclusion_text(["something political in any of them"], "Nothing political.")
    assert "executed" in terms and visual == [] and prompts
    terms, prompts, visual = exclusion_text(["empty street", "bins"], "no empty street, no bins")
    assert terms == [] and prompts == [] and visual == ["empty street", "bins"]  # unchanged for other items
    # a wanted shot that asks for the same thing keeps it (no prompt that would remove it)
    _, prompts, _ = exclusion_text(["portraits of victims"], "",
                                        wanted=["a poster with a portrait photo of a person in the crowd"])
    assert "a poster with a portrait photo of a person" not in prompts
    assert quoted_terms("the host's talking, 'executed' and \"killed\"") == ["executed", "killed"]


def test_ocr_terms_match_squashed_case_insensitive_and_fragments():
    corpus = SimpleNamespace(n=60, fps=3.0, bounds=np.array([0, 30, 60]), text_items=[
        {"step": 6, "t": 2.0, "text": "tecuted byIRGConApr3", "file": 0},  # a fragment of "executed"
        {"step": 15, "t": 5.0, "text": "SKILLS CENTER", "file": 0},
        {"step": 45, "t": 15.0, "text": "Over 50,000 KILLED", "file": 1},
        {"step": 29, "t": 9.67, "text": "GATE OF THE NATIONS", "file": 0}])
    mask = ocr_term_steps(corpus, ["executed", "killed"])
    hold = int(round(OCR_HOLD_S * 3.0))
    assert mask[6 - hold:7 + hold].all() and mask[45 - hold:46 + hold].all()
    assert mask.sum() == 2 * (2 * hold + 1)  # "SKILLS" is not "killed", the gate float names nothing
    assert ocr_term_steps(corpus, ["regime"]) is None
    assert ocr_term_steps(SimpleNamespace(text_items=[]), ["killed"]) is None
