"""Rung 1, freeform chat (chat/SPEC.md): the shadow harness's plumbing, run with canned replies."""
import json

import pytest

from chat import aims, record, review
from chat.run import main


def script(lines):
    it = iter(lines)
    return lambda prompt="": next(it)


def say(session, lines):
    main(["talk", "--session", str(session), "--fake"], read=script(lines), write=lambda *a: None)


def test_a_session_keeps_the_revision_mark_out_of_everything_but_the_key(tmp_path):
    s = tmp_path / "s01"
    say(s, ["help me plan a garden", "ok", "/r actually, plan a vegetable bed only", "/q"])
    ts = record.turns(s)
    assert [t["id"] for t in ts] == ["P1", "M1", "P2", "M2", "P3", "M3"]
    assert ts[4]["text"] == "actually, plan a vegetable bed only"
    log = (s / "log.jsonl").read_text()
    assert "/r" not in log and "revision" not in log
    assert [(m["turn"], m["mark"], m["when"]) for m in record.marks(s)] == [("P3", "revision", "live")]
    calls = [e for e in record.read_jsonl(s / "log.jsonl") if e["kind"] == "call"]
    assert calls[0]["prompt"][0]["role"] == "system" and calls[-1]["prompt"][-1]["content"] == ts[4]["text"]
    with pytest.raises(SystemExit):
        say(s, ["again", "/q"])                                       # one session per directory


def test_the_reviewer_codes_each_turn_checks_quotes_and_never_sees_the_key(tmp_path):
    s = tmp_path / "s02"
    say(s, ["write a poem about rain", "make it shorter", "no, I meant a haiku", "ok", "/r now a limerick", "/q"])
    main(["review", "--session", str(s), "--fake"], write=lambda *a: None)
    log = record.read_jsonl(s / "reviews" / "review" / "log.jsonl")
    codes = {e["turn"]: e for e in log if e["kind"] == "code"}
    assert len(codes) == 10 and all(c["quote_checked"] for c in codes.values())
    assert codes["P3"]["code"]["function"] == "repair" and codes["P4"]["code"]["function"] == "feedback"
    assert codes["M2"]["code"]["addresses"] == "A2" and codes["M2"]["addresses_checked"]
    assert codes["P2"]["measured_against"] == "P1" and codes["M1"]["sim_to_aim"] is not None
    stacks = [e["stack"] for e in log if e["kind"] == "stack"]
    assert stacks[:4] == [["A1"], ["A1", "A2"], ["A1", "A2"], ["A1", "A2"]]   # a repair and feedback leave it
    prompts = " ".join(json.dumps(e["prompt"]) for e in log if e["kind"] == "call")
    assert "/r" not in prompts and "key" not in prompts.lower().replace("keys", "")
    assert json.loads((s / "reviews" / "review" / "vectors.json").read_text()).keys() >= {"P1", "M5"}
    with pytest.raises(SystemExit):
        main(["show", "--session", str(s)], write=lambda *a: None)    # codes only after the person's marks


def test_marks_come_before_codes_and_go_to_the_key(tmp_path):
    s = tmp_path / "s03"
    say(s, ["explain tides", "no, I meant ocean tides on other planets", "/q"])
    out = []
    main(["mark", "--session", str(s)], read=script(["", "", "r", "d"]), write=out.append)
    shown = "\n".join(map(str, out))
    assert "reviewer" not in shown and "A1" not in shown
    got = [(m["turn"], m["mark"]) for m in record.marks(s) if m["kind"] == "mark"]
    assert got == [("P2", "repair"), ("M2", "drift")]
    with pytest.raises(SystemExit):
        main(["mark", "--session", str(s)], read=script([""]), write=lambda *a: None)
    main(["review", "--session", str(s), "--fake"], write=lambda *a: None)
    main(["show", "--session", str(s)], write=out.append)
    assert "reviewer: repair" in out[-1]


def test_the_aim_stack_follows_the_codes_and_keeps_the_persons_words():
    every, stack = [], []
    turn = lambda i, text: {"id": f"P{i}", "text": text}
    stack = aims.apply(stack, every, turn(1, "plan a trip to Spain"), {"function": "task", "relation": "new"})
    stack = aims.apply(stack, every, turn(2, "just the hotels"), {"function": "task", "relation": "narrower"})
    stack = aims.apply(stack, every, turn(3, "cheaper ones"), {"function": "task", "relation": "same"})
    assert stack == ["A1", "A2"] and every[1]["quotes"] == ["just the hotels", "cheaper ones"]
    stack = aims.apply(stack, every, turn(4, "now, a recipe for paella"), {"function": "task", "relation": "new"})
    assert stack == ["A3"]
    stack = aims.apply(stack, every, turn(5, "back to the hotels"), {"function": "task", "relation": "back",
                                                                     "back_to": "A2"})
    assert stack == ["A2"]
    stack = aims.apply(stack, every, turn(6, "forget hotels, find flights"), {"function": "revision"})
    assert stack == ["A4"] and every[3]["quotes"] == ["forget hotels, find flights"]
    assert aims.apply(stack, every, turn(7, "ok"), {"function": "feedback", "relation": "same"}) == ["A4"]


def test_quotes_are_checked_against_the_whole_turn():
    assert review.quote_ok('"No, I meant"', "no, I meant a haiku")
    assert not review.quote_ok("a sonnet", "no, I meant a haiku") and not review.quote_ok("", "x")


def test_an_old_turn_drops_out_of_the_models_view_and_the_record_says_so(tmp_path):
    s = tmp_path / "s04"
    long = " ".join(["word"] * 120)
    main(["talk", "--session", str(s), "--fake", "--window", "1500"],
         read=script([long + " one", long + " two", long + " three", "/q"]), write=lambda *a: None)
    ts = record.turns(s)
    assert ts[1]["shown_from"] == "P1" and ts[-1]["shown_from"] != "P1"
    last = [e for e in record.read_jsonl(s / "log.jsonl") if e["kind"] == "call"][-1]["prompt"]
    assert last[0]["role"] == "system" and last[1]["role"] == "user" and last[-1]["content"].endswith("three")
