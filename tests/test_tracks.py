"""The two tracks are kept apart in code (CLAUDE.md): each imports the substrate, never the other."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def imports(track: str) -> set[str]:
    found = set()
    for path in (ROOT / track).glob("*.py"):
        found |= set(re.findall(r"^\s*(?:from|import)\s+(\w+)", path.read_text(), re.M))
    return found


def test_product_and_research_never_import_each_other():
    assert "research" not in imports("product")
    assert "product" not in imports("research")


def test_the_substrate_imports_neither_track():
    assert not {"product", "research"} & imports("substrate")
