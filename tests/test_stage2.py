"""Stage 2 (toyworld/SPEC.md, "Stage 2"): kinds of failure learned from residue."""
import json

import pytest

from toyworld import monitor, residue, world
from toyworld.run import main, replay


@pytest.fixture(scope="module")
def cmap(tmp_path_factory):
    out = tmp_path_factory.mktemp("explore") / "run"
    main(["explore", "--fake", "--run", str(out)])
    return json.loads((out / "control.json").read_text())


ORDER = {"type": "set_order_qty", "item": "oat milk", "qty": 20, "for": "stockout"}
REPORT = {"type": "report_to_head_office", "problem": "reconciliation", "status": "open", "cause": "",
          "for": "reconciliation"}


def under_ordering(n_runs=6, days=5):
    """Runs where the right action for stockout is always too small, and reconciliation is reported once."""
    states = []
    for i in range(n_runs):
        s = world.new_state(f"bare-{i + 1}", "bare", 100 + i, {"view": "compact", "stage": 1})
        for t in range(1, days):
            world.apply_reply(s, t, {"actions": [ORDER] + ([REPORT] if t == 1 else [])})
            world.observe(s, t + 1)
        states.append(s)
    return states


def test_the_right_action_too_small_is_found_and_shuffling_finds_nothing(cmap):
    learned = residue.learn(under_ordering(), cmap)
    movers = [k for k in learned["kinds"] if k["type"] == "mover"]
    assert movers and movers[0]["tier"] == 1 and movers[0]["target"] == "still_present"
    assert "anom:stockout" in movers[0]["atoms"] or "did:set_order_qty" in movers[0]["atoms"]
    assert not any(k["tier"] == 2 for k in learned["kinds"])          # one such action: no class to posit
    assert learned["control"]["tier1_admitted"] == 0 and learned["control"]["tier2_admitted"] == 0


def test_rows_speak_the_records_terms_and_skip_what_the_design_caused(cmap):
    s = world.new_state("bare-1", "bare", 7, {"view": "compact", "stage": 1, "explore_withhold": 1.0})
    for t in range(1, 4):
        world.apply_reply(s, t, {"actions": [REPORT]})                  # proposed, always withheld by the coin
        world.observe(s, t + 1)
    rs = [r for r in residue.rows([s], cmap) if r["type"] == "neglect" and r["anomaly"] == "reconciliation"]
    assert rs and not any(r["target"] for r in rs)                   # the design withheld it: not neglect
    belief = world.new_state("belief-1", "belief", 7, {"view": "compact", "stage": 1})
    for t in range(1, 4):
        world.apply_reply(belief, t, {})
        world.observe(belief, t + 1)
    assert residue.rows([belief], cmap) == []                         # it may not act: no residue
    fs = {f for r in residue.rows(under_ordering(2), cmap) for f in residue.facts(r)}
    assert fs and not any("oat" in f or "milk" in f or any(c.isdigit() for c in f) for f in fs)   # no world, no numbers


def test_a_kind_offer_shows_the_model_its_own_record_once_settled(cmap):
    kinds = residue.learn(under_ordering(), cmap)
    s = world.new_state("monitor-1", "monitor", 5, {"view": "compact", "stage": 2, "control": cmap, "kinds": kinds})
    goal = {"proposition": "the world is such that oat milk does not sell out", "direction": "goal",
            "anomaly": "stockout"}
    world.apply_reply(s, 1, {"register_changes": {"add": [goal]}, "actions": [ORDER]})
    world.observe(s, 2)
    offers = [o for o in monitor.offer(s, monitor.review(s)) if o["check"].startswith("kind")]
    assert offers and offers[0]["question"].endswith("?") and offers[0]["evidence"] and offers[0]["wrong_if"]
    n = offers[0]["change"]["note"]
    assert "unfilled requests" in n and "set_order_qty" in n and "qty 20" in n   # the model's own action, as written
    assert "moved it in exploration" not in n                                    # and no endorsement of it
    assert not any(c.isdigit() for r in residue.rows([s], cmap) for f in residue.facts(r) for c in f)  # kernel: none
    assert "Settled" not in world.proposer_prompt(s, 2)              # nothing reaches the model until settled
    monitor.settle(s, offers[0], accept=True, by="owner")
    world.apply_reply(s, 2, {"actions": [ORDER]})
    world.observe(s, 3)
    later = [o for o in monitor.offer(s, monitor.review(s)) if o["check"].startswith("kind")]
    assert later                                                       # new days: offered again, with them
    monitor.settle(s, later[0], accept=True, by="owner")
    notes = [l for l in world.proposer_prompt(s, 3).splitlines() if "the record shows" in l]
    assert len(notes) == 1 and "day(s) 1, 2" in notes[0]               # only the kind's latest note is shown


def test_kinds_command_and_stage_2_run_replay(tmp_path, cmap):
    ex = tmp_path / "explore"
    main(["explore", "--fake", "--run", str(ex), "--runs", "4"])
    main(["kinds", "--from", str(ex), "--control", str(ex / "control.json"), "--out", str(tmp_path / "k.json")])
    k = json.loads((tmp_path / "k.json").read_text())
    assert {"rows", "kinds", "control", "sources"} <= set(k)
    out = tmp_path / "s2"
    main(["run", "--fake", "--run", str(out), "--stage", "2", "--control", str(ex / "control.json"),
          "--kinds", str(tmp_path / "k.json"), "--runs", "1", "--days", "4"])
    _, _, differ = replay(out / "log.jsonl")
    assert differ == 0
    with pytest.raises(SystemExit):
        main(["run", "--fake", "--run", str(tmp_path / "x"), "--stage", "2", "--control", str(ex / "control.json")])
