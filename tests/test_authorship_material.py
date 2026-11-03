"""Causal authorship's conversations meet the pre-registration (PLAN.md): 4 kinds of 10, every source in two wordings."""
import json
from pathlib import Path


def rows():
    path = Path(__file__).resolve().parent.parent / "material" / "authorship_conversations.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_four_kinds_of_ten_each_with_its_topic():
    r = rows()
    assert len(r) == 40 and len({x["id"] for x in r}) == 40
    kinds = [x["kind"] for x in r]
    assert all(kinds.count(k) == 10 for k in ("explicit", "frame once", "frame three times", "no frame"))
    assert all(x["by"] == "model" and x["frozen"] for x in r)


def test_every_source_has_two_wordings_and_frames_repeat_only_where_they_should():
    for x in rows():
        assert len(x["goal"]) == 2 and x["goal"][0] != x["goal"][1]
        assert len(x["partner"]) == 2
        copies = 3 if x["kind"] == "frame three times" else 1
        assert all(len(w) == copies for w in x["partner"]), x["id"]
        assert x["partner"][0] != x["partner"][1]


def test_every_inert_note_has_two_alternates_and_none_repeats():
    r = rows()
    notes = [n for x in r for n in (x["inert"], *x["inert_alternates"])]
    assert all(len(x["inert_alternates"]) == 2 for x in r)
    assert len(set(notes)) == 120


def test_no_partner_message_quotes_a_reply_and_only_explicit_ones_instruct():
    for x in rows():
        for w in x["partner"]:
            for m in w:
                low = m.lower()
                assert "you said" not in low and "you wrote" not in low and "your plan" not in low, x["id"]
                assert ("must" in low) == (x["kind"] == "explicit") or x["kind"] == "explicit", x["id"]
