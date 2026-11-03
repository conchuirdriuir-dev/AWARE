"""Step 2's second wording: it reads step 2's own replies back from the record."""
from research import hidden_wording as w
from substrate.record import Record


def test_the_replies_come_back_by_conversation_and_condition(tmp_path):
    rec = Record(tmp_path / "r.jsonl")
    rec.append("generated", "model", conversation="picnic", condition="G", reply="cal", calibration=True)
    rec.append("generated", "model", conversation="hall", condition="G", reply="g", cut_off=True)
    rec.append("generated", "model", conversation="hall", condition="V", reply="v", cut_off=True)
    assert w.recorded(rec.path) == {"hall": {"G": "g", "V": "v"}}


def test_the_wording_differs_from_step_2s_and_keeps_its_two_keys():
    from research import hidden as h

    assert w.WORDING[0] != h.QUESTION and set(w.WORDING[1]) == set(h.CHOICES)
    assert "something I was given" in h.question(("own", "steered"), w.WORDING)
