import pytest


@pytest.fixture(autouse=True)
def _hermetic_parsing(tmp_path_factory, monkeypatch):
    """Tests never reach a local Ollama server or the user's parse cache: the default parser is the rules, and
    the LLM answer cache lives in a temp dir. Tests of the LLM path mock the HTTP call."""
    from vhe.query import llm

    monkeypatch.setenv("VHE_LLM", "none")
    monkeypatch.setattr(llm, "PARSE_CACHE", tmp_path_factory.mktemp("parse_cache"))


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    """models.fetch never downloads in tests: a model missing from ~/.cache/vhe/models makes its stage skip."""
    monkeypatch.setenv("VHE_NO_DOWNLOAD", "1")
