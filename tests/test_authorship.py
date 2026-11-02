"""Causal authorship's reading (PLAN.md), on planted numbers and a tiny random Qwen3: plumbing, never a result."""
import numpy as np
import pytest

from research import authorship as a

CONV = {"id": "x", "kind": "frame three times", "goal": ["Goal one.", "Goal two."],
        "partner": [["P1.", "P2.", "P3."], ["Q1.", "Q2.", "Q3."]], "inert": "Note.", "inert_alternates": ["Alt1.", "Alt2."]}


def contents(turns):
    return [m["content"] for m in a.render(turns)]


def test_turns_run_in_the_order_written_with_the_labels():
    t = a.turns(CONV, 0, "My reply.")
    assert contents(t) == ["Person: Goal one.", "My reply.", "Note: Note.", "Partner: P1.", "Partner: P2.", "Partner: P3.",
                           a.CLOSING]
    assert [m["role"] for m in a.render(t)] == ["user", "assistant", "user", "user", "user", "user", "user"]
    assert contents(a.turns(CONV, 1, "Other."))[0] == "Person: Goal two." and contents(a.turns(CONV, 1, "Other."))[3] == "Partner: Q1."


def test_removal_deletes_or_replaces_a_group_and_keeps_the_label():
    t = a.turns(CONV, 0, "My reply.")
    gone = a.without(t, "partner", None)
    assert "Partner: P1." not in contents(gone) and len(contents(gone)) == 4
    filled = a.without(t, "partner", lambda s: "blank")
    assert contents(filled)[3:6] == ["Partner: blank"] * 3
    assert contents(a.without(t, "partner1", lambda s: "blank"))[3:6] == ["Partner: P1.", "Partner: blank", "Partner: P3."]
    assert contents(a.without(t, "own", lambda s: "blank"))[1] == "blank"


def test_swapping_the_note_touches_nothing_else():
    t = a.turns(CONV, 0, "My reply.")
    got = contents(a.with_note(t, "Alt1."))
    assert got[2] == "Note: Alt1." and got[:2] == contents(t)[:2] and got[3:] == contents(t)[3:]


class CountingTokenizer:
    """One token per character: decode and encode are exact, so the count is easy to check."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_a_filler_has_the_token_count_of_what_it_replaces():
    tok = CountingTokenizer()
    for n in (1, 7, len(a.FILLER), 3 * len(a.FILLER) + 2):
        assert len(tok(a.filler(tok, n))["input_ids"]) == n


def test_kl_is_zero_for_the_same_prediction_and_positive_otherwise():
    p = np.log(np.array([[0.5, 0.5], [0.9, 0.1]]))
    q = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert a.influence(p, p) == 0
    assert a.influence(p, q) == pytest.approx(0.5 * (0.9 * np.log(1.8) + 0.1 * np.log(0.2)))


def planted(n=40, strong=1.0):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        kind = a.KINDS[i % 4]
        partner = strong if kind == "explicit" else 0.5
        top = {"goal": 0.1, "own": 0.05, "partner": partner}
        rows.append({"id": str(i), "kind": kind, "replaced": top, "deleted": top, "second": top,
                     "alt1": float(rng.uniform(0.005, 0.02)), "alt2": float(rng.uniform(0.005, 0.015)),
                     "reread": top})
    return rows


def test_a_sound_instrument_passes_all_six():
    r = planted()
    assert a.floor(r) == pytest.approx(np.percentile([x["alt1"] for x in r], 95))
    checks = a.instrument(r)
    assert all(checks[k]["pass"] for k in a.CHECKS), checks
    assert a.holds(checks)


@pytest.mark.parametrize("damage, check", [
    (lambda x: x.update(reread={**x["replaced"], "partner": x["replaced"]["partner"] * 1.5}), "read twice"),
    (lambda x: x.update(alt2=1.0), "null"),
    (lambda x: x.update(replaced={"goal": 0.1, "own": 0.05, "partner": 0.06}) if x["kind"] == "explicit" else None, "strong"),
    (lambda x: x.update(deleted={"goal": 9.0, "own": 0.05, "partner": 0.5}), "removal"),
    (lambda x: x.update(second={"goal": 9.0, "own": 0.05, "partner": 0.5}), "wording"),
    (lambda x: x.update(replaced={"goal": 0.0, "own": 0.0, "partner": 0.0}) if x["kind"] == "frame three times" else None, "redundancy"),
])
def test_each_check_fails_when_its_own_thing_is_broken(damage, check):
    r = planted()
    for x in r:
        damage(x)
    checks = a.instrument(r)
    assert not checks[check]["pass"], check
    assert not a.holds(checks)


def test_the_key_is_the_top_source_above_the_floor_otherwise_none():
    assert a.key({"goal": 0.1, "own": 0.5, "partner": 0.2}, 0.3) == "own"
    assert a.key({"goal": 0.1, "own": 0.2, "partner": 0.2}, 0.3) == "none"


def test_options_shuffle_by_seed_and_parse_back():
    letters = a.options("abc")
    assert sorted(letters.values()) == sorted(a.OPTIONS) and letters == a.options("abc")
    assert any(a.options(str(i)) != letters for i in range(20))
    assert a.parse("B", letters) == letters["B"] and a.parse(" c. something", letters) == letters["C"]
    assert a.parse("I am not sure", letters) is None
    assert all(a.OPTIONS[s] in a.question(letters) for s in a.OPTIONS)


def test_the_claim_reads_the_interval_against_the_bar():
    n = 40
    assert a.account_verdict(np.array([1] * 12 + [0] * 28))["verdict"] == "holds"
    assert a.account_verdict(np.array([1] * n))["verdict"] == "model reports well"
    assert a.account_verdict(np.array([1] * 31 + [0] * 9))["verdict"] == "undecided"


def test_tail_reads_the_predictions_for_each_reply_token_only():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)).eval()
    context, reply = [1, 2, 3, 4], [5, 6]
    got = a.tail(hf, context, reply)
    with torch.no_grad():
        whole = hf(torch.tensor([context + reply])).logits[0].float().log_softmax(-1).numpy()
    assert got.shape == (2, 64)
    assert np.allclose(got, whole[3:5], atol=1e-4)


def test_a_row_can_be_recorded(tmp_path):
    """The record's own arguments are not available as field names: this crashed a dry run."""
    from substrate.record import Record

    row = planted(4)[0]
    rec = Record(tmp_path / "r.jsonl")
    rec.append("read", "code", conversation=row["id"], conversation_kind=row["kind"],
               **{k: v for k, v in row.items() if k not in ("reply", "own", "id", "kind")})


def test_a_killed_run_resumes_from_its_record(tmp_path):
    from substrate.record import Record

    assert a.finished(tmp_path / "none.jsonl") == {}
    rec = Record(tmp_path / "r.jsonl")
    row = planted(4)[0]
    rec.append("generated", "model", conversation=row["id"], wording=0, own="O", reply="R", own_cut_off=True, reply_cut_off=False)
    rec.append("generated", "model", conversation=row["id"], wording=1, own="O2", reply="R2", own_cut_off=True, reply_cut_off=False)
    rec.append("read", "code", conversation=row["id"], conversation_kind=row["kind"], note_replaced=0.1, copies=[0.2],
               **{k: v for k, v in row.items() if k not in ("id", "kind")})
    rec.append("generated", "model", conversation="half-read", wording=0, own="O", reply="R", own_cut_off=True, reply_cut_off=False)
    got = a.finished(rec.path)
    assert list(got) == [row["id"]]
    assert got[row["id"]]["own"] == "O" and got[row["id"]]["reply"] == "R" and got[row["id"]]["replaced"] == row["replaced"]
