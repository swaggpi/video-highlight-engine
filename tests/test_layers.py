"""The package layering (docs/ARCHITECTURE.md): imports only point down, and ranking never imports the parser."""
import ast
from pathlib import Path

VHE = Path(__file__).resolve().parents[1] / "vhe"
# Lower number = lower layer. A module may import its own layer and below.
LAYER = {"vhe.config": 0, "vhe.models": 0, "vhe.timing": 0, "vhe.signals": 0, "vhe.ingest": 0, "vhe.index": 0,
         "vhe.project": 1, "vhe.pipeline": 1, "vhe.corpus": 2, "vhe.export": 2, "vhe.query": 3, "vhe.rank": 4,
         "vhe.results": 5, "vhe.engine": 6, "vhe.cli": 7, "vhe.ui": 7}
PARSER = ("vhe.query.parser", "vhe.query.rules", "vhe.query.llm", "vhe.query.lexicon")


def _layer(module: str) -> int | None:
    hits = [k for k in LAYER if module == k or module.startswith(k + ".")]
    return LAYER[max(hits, key=len)] if hits else None


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vhe"):
            yield node.lineno, [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            yield node.lineno, [a.name for a in node.names if a.name.startswith("vhe")]


def test_imports_point_down_and_rank_does_not_parse():
    bad = []
    for path in sorted(VHE.rglob("*.py")):
        module = ".".join(path.relative_to(VHE.parent).with_suffix("").parts).removesuffix(".__init__")
        here = _layer(module)
        for line, targets in _imports(path):
            for target in targets:
                there = _layer(target)
                if here is not None and there is not None and there > here:
                    bad.append(f"{module}:{line} imports {target}")
                if module.startswith("vhe.rank") and target.startswith(PARSER):
                    bad.append(f"{module}:{line} imports the parser ({target})")
    assert not bad, "\n".join(bad)
