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


def test_the_prompt_is_fixed_and_names_no_domain():
    from research.gate1_pairs import PROMPT, clean, prompt, review_sample

    assert prompt("The river is high.") == PROMPT + "\n\nThe river is high."
    assert clean("<think>\n\n</think>\n\nThe river is low. ") == "The river is low."
    chosen = review_sample(200)
    assert len(set(chosen)) == 20 and chosen == review_sample(200)      # the same 20 on any rerun


def test_unfrozen_statements_are_refused(tmp_path):
    import pytest
    from research.gate1_pairs import load_frozen

    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"id": "x", "statements": ["A."], "frozen": False}) + "\n")
    with pytest.raises(SystemExit):
        load_frozen(path)


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


def test_the_owners_reading_matches_the_review_sample():
    from research.gate1_pairs import review_sample
    from research.gate1_run import REVIEW, PAIRS, rows as read, subsets

    pairs, review = read(PAIRS), read(REVIEW)
    assert [r["pair"] for r in review] == review_sample(len(pairs))
    assert all(pairs[r["pair"]]["opposite"] == r["opposite"] and r["by"] == "person" for r in review)
    keep = subsets(pairs, review)
    assert (~keep["without wrong"]).sum() == 2 and (~keep["without wrong or borderline"]).sum() == 4
    assert keep["post hoc: negation word"].sum() == 114 and keep["post hoc: no negation word"].sum() == 86


def test_a_form_that_wins_on_shuffled_pairs_is_void():
    from research.gate1_run import void_against

    real = {"global": {"verdict": "additive"}, "per topic": {"verdict": "mirror"}}
    shuffled = {"global": {"verdict": "additive"}, "per topic": {"verdict": "undecided"}}
    assert void_against(real, shuffled) == {"global": "void", "per topic": "mirror"}
