"""Brief parsing, the entry point: an editor's brief -> a ParsedQuery (vhe.query.model).

A local LLM (Ollama, vhe.query.llm) is the default parser when it is reachable; the rules (vhe.query.rules, their
word lists in vhe.query.lexicon) are the fallback and also ground the model: numbers the brief does not contain are
replaced by the rules' values, and avoid/topic items the rules find are added. Both paths end in the same
vhe.query.rules.finalise. $VHE_LLM names the default model ("none" for rules only)."""
from __future__ import annotations

from vhe.query.llm import default_llm, parse_with_llm
from vhe.query.model import ParsedQuery
from vhe.query.rules import parse


def parse_brief(query: str, llm: str | None = "auto", **overrides) -> ParsedQuery:
    """llm: an Ollama model, 'auto' (default_llm(): $VHE_LLM, else qwen3:4b) or None (rules only). overrides: count,
    min_len, max_len and budget_s from the caller, which win over the brief."""
    model = default_llm() if llm == "auto" else llm
    return parse_with_llm(query, model, **overrides) if model else parse(query, **overrides)
