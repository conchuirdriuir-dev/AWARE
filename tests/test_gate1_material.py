"""Gate 1's statements meet the pre-registration (PLAN.md): 20 topics of 10, declarative, naming no negation."""
import json
import re
from pathlib import Path

NEGATION = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without|cannot|opposite|negat\w*)\b|n't", re.I)
INSTRUCTION = {"write", "design", "plan", "find", "make", "create", "explain", "describe", "list", "give", "tell",
               "draw", "build", "put", "set", "work", "study", "prepare", "develop", "outline", "draft"}


def rows():
    path = Path(__file__).resolve().parent.parent / "material" / "negation_statements.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_twenty_topics_of_ten_distinct_statements():
    r = rows()
    assert len(r) == 20 and len({x["id"] for x in r}) == 20
    assert all(len(x["statements"]) == 10 for x in r)
    every = [s for x in r for s in x["statements"]]
    assert len(set(every)) == 200
    assert all(x["by"] == "model" and x["frozen"] for x in r)


def test_no_statement_names_negation_or_begins_with_an_instruction():
    for x in rows():
        for s in x["statements"]:
            assert not NEGATION.search(s), s
            assert s.split()[0].lower() not in INSTRUCTION, s
            assert s.endswith(".") and s[0].isupper(), s
