"""Hidden sources' reading (PLAN.md, direction step 2), on planted numbers and a tiny random Qwen3: plumbing, never a
result."""
import json
from pathlib import Path

import numpy as np
import pytest

from research import hidden as h


def test_markers_match_at_a_word_start_in_any_case():
    assert h.carries("Bring SUNSCREEN for all", ["sunscreen"])
    assert h.carries("we will recycle parts", ["recycl"])
    assert h.carries("the startup", ["star"])                   # a word start matches a longer word: why "star" is not a marker
    assert not h.carries("a teapot", ["pot"])
    assert h.carries("a step-free route", ["step-free"])


def test_the_steered_context_adds_the_partner_line_and_nothing_else():
    g, s = h.goal_messages("Plan it."), h.steered_messages("Plan it.", "the lunch")
    assert g == [{"role": "user", "content": "Person: Plan it."}]
    assert s[0]["content"] == "Person: Plan it.\nPartner: In your reply, focus on the lunch."


def test_both_orders_ask_the_same_question():
    a, b = (h.question(o) for o in h.ORDERS)
    assert "A. my own judgement" in a and "B. I was steered" in a
    assert "A. I was steered" in b and "B. my own judgement" in b


def test_the_score_is_the_steered_letter_in_either_order():
    assert h.score(0.2, 0.6, ("own", "steered")) == pytest.approx(0.75)
    assert h.score(0.6, 0.2, ("steered", "own")) == pytest.approx(0.75)


def test_auc_counts_ties_as_half_and_bootstrap_brackets_it():
    assert h.auc([0, 0], [1, 1]) == 1.0
    assert h.auc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(1)
    g, o = rng.normal(0, 1, 40), rng.normal(2, 1, 40)
    low, high = h.bootstrap(g, o, n_boot=500)
    assert low <= h.auc(g, o) <= high


def test_the_verdict_reads_the_interval_against_both_bars():
    assert h.verdict(0.4, 0.74) == "holds"
    assert h.verdict(0.8, 0.95) == "model reports well"
    assert h.verdict(0.6, 0.9) == "undecided"


def cal_row(carry: dict, fluent: dict) -> dict:
    return {"G_logprob": -1.0, "I": {f"{L}/{a}": {"carries": carry.get((L, a), False),
                                                  "logprob": -1.0 if fluent.get((L, a), True) else -9.0}
                                     for L in h.LAYERS for a in h.ALPHAS}}


def test_calibration_picks_most_carrying_fluent_pair_ties_to_smaller_alpha_then_layer():
    rows = [cal_row({(24, 1.0): True, (12, 2.0): True}, {}) for _ in range(8)]
    assert h.choose(rows) == {"layer": 24, "alpha": 1.0, "carrying": 1.0, "fluent": 1.0}
    rows = [cal_row({(12, 4.0): True}, {(12, 4.0): False}) for _ in range(8)]
    assert h.choose(rows) is None                                   # carrying only where it is not fluent
    rows = [cal_row({(18, 0.5): i < 3}, {}) for i in range(8)]
    assert h.choose(rows) is None                                   # 3 of 8 is under half


def planted(n: int = 40, hidden: float = 0.5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r = {"id": str(i)}
        for c, mean, carry in (("G", 0.2, False), ("V", 0.9, True), ("P", hidden, True), ("I", hidden, True)):
            s = float(np.clip(rng.normal(mean, 0.1), 0, 1))
            r[c] = {"score": s, "by_order": [s, s], "greedy": ["own", "own"], "carries": carry, "logprob": -1.0}
        rows.append(r)
    rows[0]["reread"] = [rows[0][c]["score"] for c in h.CONDITIONS]
    return rows


def test_a_sound_construction_passes_and_a_chance_account_holds():
    rows = planted(hidden=0.2)
    checks = h.construction(rows, injected=True)
    assert h.holds(checks) and checks["fluency"]["pass"]
    assert h.compare(rows, "P")["verdict"] == "holds"


def test_an_account_that_sees_through_the_hiding_reports_well():
    assert h.compare(planted(hidden=0.9), "I")["verdict"] == "model reports well"


def test_each_check_fails_when_its_own_thing_is_broken():
    rows = planted()
    rows[0]["reread"][1] += 0.01
    assert not h.construction(rows, True)["read twice"]["pass"]
    rows = planted()
    for r in rows[:10]:
        r["G"]["carries"] = True
    assert not h.construction(rows, True)["carrying"]["pass"]
    rows = planted()
    for r in rows:
        r["V"]["score"] = r["G"]["score"]
    assert not h.construction(rows, True)["answerable"]["pass"]
    rows = planted()
    for r in rows[:10]:
        r["I"]["logprob"] = -9.0
    assert not h.construction(rows, True)["fluency"]["pass"]


def test_only_conversations_matching_their_keys_enter_and_too_few_is_undecided():
    rows = planted()
    for r in rows[:20]:
        r["V"]["carries"] = False
    assert h.compare(rows, "P") == {"n": 20, "verdict": "undecided", "why": "fewer than 25 conversations enter"}


def test_a_killed_run_resumes_from_its_record(tmp_path):
    from substrate.record import Record

    rec = Record(tmp_path / "r.jsonl")
    assert h.finished(rec.path) == (None, False, {})
    rec.append("injection chosen", "code", chosen=None, layers=h.LAYERS, alphas=h.ALPHAS)
    row = planted(2)[0]
    rec.append("read", "code", conversation=row["id"], **{k: v for k, v in row.items() if k != "id"})
    chosen, calibrated, rows = h.finished(rec.path)
    assert chosen is None and calibrated and rows == {row["id"]: row}


def test_the_steering_hook_moves_the_layer_and_comes_away():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)).eval()
    ids, seen = torch.tensor([[1, 2, 3, 4]]), []

    def grab(_m, args, kwargs):      # what layer 1 receives: transformers records hidden_states before a hook's change
        seen.append((args[0] if args else kwargs["hidden_states"]).clone())

    handle = hf.model.layers[1].register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():
        base = hf(ids).logits
        with h.Steer(hf, 0, torch.ones(32), 2.0):
            steered = hf(ids).logits
        after = hf(ids).logits
    handle.remove()
    assert len(seen) == 3
    assert torch.allclose(seen[1], seen[0] + 2.0, atol=1e-5) and torch.allclose(seen[2], seen[0])
    assert not torch.allclose(steered, base) and torch.allclose(after, base)


ROOT = Path(__file__).resolve().parents[1]


def test_the_directions_match_the_frozen_goals_one_to_one():
    dirs = [json.loads(x) for x in (ROOT / h.DIRECTIONS).read_text().splitlines()]
    goals = [json.loads(x)["id"] for x in (ROOT / h.GOALS).read_text().splitlines()]
    assert [d["id"] for d in dirs] == goals
    assert all(d["by"] == "model" and len(d["markers"]) == 3 and d["direction"] for d in dirs)
    cal = [json.loads(x) for x in (ROOT / h.CALIBRATION).read_text().splitlines()]
    assert len(cal) == 8 and not {c["id"] for c in cal} & set(goals)
