"""The habitat and Step -1 (body/habitat.py, body/stepm1.py): the frozen seeds, the engine's labels and k, the rule
change, news, and the scoring of the three bars, on toy material. The planted room's switch runs only where TextWorld
is installed."""
import random

import numpy as np
import pytest

from body import habitat as H
from body import stepm1 as M


def test_pools_are_disjoint_and_as_registered():
    spec = H.worlds()
    sizes = {name: len(H.pool(name, spec)) for name in ("calibration", "step_minus_1", "step0_habitat",
                                                         "behaviour_test", "self_talk")}
    assert sizes == {"calibration": 100, "step_minus_1": 60, "step0_habitat": 400, "behaviour_test": 100,
                     "self_talk": 20}
    seeds = [s for name in sizes for s in H.pool(name, spec)]
    assert len(seeds) == len(set(seeds))
    assert min(seeds) < spec["pools"]["babbling"]["seeds_from"] > max(seeds)


def test_planted_kinds_and_turns():
    kinds = [None if p is None else p["kind"] for p in map(H.planted, H.pool("step_minus_1"))]
    assert kinds.count("new_room") == kinds.count("rule_change") == kinds.count(None) == 20
    turns = [H.planted(s)["turn"] for s in range(2000, 2040)]
    assert all(15 <= t <= 35 for t in turns) and len(set(turns)) > 5
    assert H.planted(1000) is None and H.planted(2000) == H.planted(2000)


def test_params_from_the_seed_alone():
    for s in (1000, 2059, 3399):
        p = H.params(s)
        assert p == H.params(s)
        assert 6 <= p["rooms"] <= 12 and 10 <= p["objects"] <= 20 and 3 <= p["quest_length"] <= 8


def test_clean_removes_the_prompt_and_status_bar():
    raw = "You see a lamp.\n\n>                     -= Lounge =-0/20  \nIt is dark."
    assert H.clean(raw) == "You see a lamp.\n\nIt is dark."


def test_k_from_resolved_stalls():
    T, F = True, False
    eps = [[T, F, F, T, F, T, F, F, F, T], [F, F, F, F, F]]          # the second ends unresolved: not counted
    assert [(s, n, r) for s, n, r in H.stalls(eps[0])] == [(1, 2, True), (4, 1, True), (6, 3, True)]
    assert H.stalls(eps[1]) == [(0, 5, False)]
    assert H.choose_k(eps) == int(np.floor(np.percentile([2, 1, 3], 95))) + 1 == 3
    with pytest.raises(ValueError):
        H.choose_k([[F, F]])


def _turn(action, room="kitchen", new=False, score=0):
    return {"action": action, "room": room, "new_state": new, "score_change": score}


def test_closures_need_k_flat_turns_and_repeats():
    loop = [_turn("go north", new=True)] + [_turn(a) for a in ["look", "look", "take apple", "look", "look"]]
    assert H.closures(loop, 4) == [(1, 5)]
    assert H.closures(loop, 6) == []
    fresh = [_turn("go north", new=True)] + [_turn(f"examine thing {i}") for i in range(5)]
    assert H.closures(fresh, 4) == []                                    # flat, but nothing repeats
    scored = [_turn("look"), _turn("look"), _turn("look", score=1), _turn("look"), _turn("look")]
    assert H.closures(scored, 3) == []


def test_ascents_are_new_lows_of_the_quest_left():
    left = [3, 3, 2, 3, None, 2, 1, 1, 0]                               # 3 at the opening
    turns = [{"quest_left": n} for n in left]
    assert H.ascents(turns, 3) == [2, 6, 8]                             # back to 2 at turn 5 is not a new low
    assert H.ascents(turns, None) == [2, 6, 8]                          # no opening count: the first count is the low
    assert H.quest_left(["go north", "open box", "go south", "insert key into box"], False) == 2
    assert H.quest_left([], True) == 0 and H.quest_left([], False) is None


def test_rule_change_blocks_the_last_world_changing_command():
    r = H.RuleChange(turn=3)
    assert r.after(0, "go north", True) == ""                            # a move never becomes the rule
    assert r.after(1, "open chest", True) == ""
    assert r.after(2, "take apple", False) == ""                         # changed nothing: not a candidate
    assert "cannot open chest" in r.after(3, "look", False) and r.fired_at == 3
    assert r.before(4, "Open chest ") is not None and r.before(4, "open box") is None


def test_rule_change_waits_for_a_candidate():
    r = H.RuleChange(turn=0)
    assert r.after(0, "look", False) == "" and r.after(1, "go east", True) == ""
    assert r.after(2, "eat apple", True) and r.fired_at == 2


def test_news_is_zero_on_an_echo_and_large_on_news():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(5, 2, 8))
    S[2] = S[1]                                                          # the reply echoes this turn's action
    S[4] = S[0] + 50                                                     # the reply is far from everything earlier
    N = M.news(S, ["world", "self", "world", "self", "world"], np.array([1.0, 1.0]))
    assert np.isnan(N[0]).all() and np.isnan(N[1]).all()
    assert np.allclose(N[2], 0) and (N[4] > 1).all()


def _ep(seed, kind, n=40, planted=25, fire=True, variant=None):
    """A toy episode: replies' news below the floor except on new-state turns and, if `fire`, the planted turn."""
    chunks = [{"slot": "world", "turn": -1, "text": "opening"}]
    keys, rows = {}, [np.full(2, np.nan)]
    for t in range(n):
        new = t % 10 == 0
        here = kind is not None and t == planted
        v = 3.0 if (new or (here and fire)) else 0.3
        text = f"new {t}" if new or here else f"r{t % 4}"                # only quiet replies repeat
        chunks += [{"slot": "self", "turn": t, "text": f"a{t}"}, {"slot": "world", "turn": t, "text": text}]
        rows += [np.full(2, np.nan), np.full(2, v)]
        keys[t] = {"turn": t, "new_state": new and not here, "planted_here": here}
        if here and kind == "new_room":
            keys[t]["switched"] = True
        if here and kind == "rule_change":
            keys[t]["blocked"] = "open chest"
    info = {"seed": seed, "planted_kind": kind, "self_talk": variant}
    return (seed, variant), {"info": info, "chunks": chunks, "keys": keys}, np.array(rows)


def _material(fire_rooms=20, st_value=0.2):
    eps, news = {}, {}
    for i, seed in enumerate(range(2000, 2060)):
        kind = H.planted(seed)["kind"] if H.planted(seed) else None
        k, e, n = _ep(seed, kind, fire=not (kind == "new_room" and i >= fire_rooms))
        eps[k], news[k] = e, n
    for seed in range(2100, 2120):
        for variant in ("exact", "restated"):
            k, e, n = _ep(seed, None, n=30, variant=variant)
            n[2::2] = st_value
            eps[k], news[k] = e, n
    for seed in range(1000, 1010):
        k, e, n = _ep(seed, None)
        eps[k], news[k] = e, n
    return eps, news


def test_score_passes_on_a_sense_that_hears_the_world():
    eps, news = _material()
    out = M.score(eps, news, H.worlds())
    assert out["self_talk"]["exact"]["pass"] and out["self_talk"]["restated"]["pass"]
    assert out["fires"]["fired"] == 40 and out["fires"]["pass"]
    assert out["above_ordinary"]["auc"] == 1.0 and out["above_ordinary"]["pass"]
    assert out["floor_check"]["pass"] and out["pass"]
    assert out["k"] == {"k": 10, "episodes": 10}                         # toy stalls of 9 turns between new states


def test_score_fails_when_planted_rooms_are_missed_or_self_talk_fires():
    eps, news = _material(fire_rooms=16)                                 # 16 of 20 rooms fire: under 17 of a kind
    out = M.score(eps, news, H.worlds())
    assert out["fires"]["fired"] == 36 and not out["fires"]["pass"] and not out["pass"]
    eps, news = _material(st_value=1.5)
    out = M.score(eps, news, H.worlds())
    assert not out["self_talk"]["exact"]["pass"] and not out["pass"]


def test_a_plant_that_did_not_happen_counts_as_not_firing():
    eps, news = _material()
    key = (2000, None)
    t = next(t for t, k in eps[key]["keys"].items() if k["planted_here"])
    eps[key]["keys"][t]["switched"] = False
    out = M.score(eps, news, H.worlds())
    assert out["fires"]["fired"] == 39


def test_the_planted_room_switch_keeps_the_world_the_same(tmp_path):
    pytest.importorskip("textworld")
    built = H.build(2000, tmp_path, "new_room")
    for trial in range(3):
        rng = random.Random(trial)
        e = H.Engine(built, admissible=True)
        for _ in range(20):
            cmds = e.state.admissible_commands
            door = [c for c in cmds if built["door"] in c]
            e.step(rng.choice(door) if door and rng.random() < 0.5 else rng.choice(cmds))
        assert e.switch()
        door = {str(f) for f in e.state.facts if f"{built['door']}: d" in str(f)}
        assert any(f.startswith("closed(") for f in door) and not any(f.startswith("locked(") for f in door)


class _FakeAgent:
    """Plays like a poor player: a few fixed commands, cycled. Enough to drive the harness end to end."""
    name, revision = "fake", None
    COMMANDS = ["look", "inventory", "go north", "go south", "go east", "go west", "take all", "open door"]

    def __init__(self):
        self.t = 0

    def fit(self, messages):
        return messages, 0

    def say(self, messages, max_tokens=24):
        self.t += 1
        return self.COMMANDS[self.t % len(self.COMMANDS)]

    def restate(self, text):
        return f"I said: {text}"


def test_play_appends_turns_and_plants(tmp_path):
    pytest.importorskip("textworld")
    from substrate.record import Record, read

    record = Record(tmp_path / "record.jsonl")
    for seed in (2000, 2020):                                            # a planted room, a planted rule change
        H.play(_FakeAgent(), record, seed, tmp_path / "games", max_turns=40)
    H.self_talk(_FakeAgent(), record, 2100, "restated", turns=5)
    eps = M.episodes(read(record.path))
    room, rule, talk = eps[(2000, None)], eps[(2020, None)], eps[(2100, "restated")]
    assert [c["slot"] for c in room["chunks"][:3]] == ["world", "self", "world"]
    assert all(c["by"] == "model" for c in room["chunks"] if c["slot"] == "self")
    planted_room = [k for k in room["keys"].values() if k["planted_here"]]
    assert len(planted_room) == 1 and planted_room[0]["turn"] == H.planted(2000)["turn"]
    assert planted_room[0]["switched"] is True
    rooms = {k["room"] for k in room["keys"].values()}               # the closure label pairs actions with rooms
    assert None not in rooms and len(rooms) >= 2
    assert sum(k["planted_here"] for k in rule["keys"].values()) <= 1
    assert all(c["provenance"] == "self" and c["by"] == "model" for c in talk["chunks"][1:])
    assert {c["slot"] for c in talk["chunks"][1:]} == {"self", "world"}


def test_the_8b_floor_file_covers_reads_layers():
    import json
    import math
    from pathlib import Path

    floor = json.loads((Path(__file__).resolve().parent.parent / "material" / "echo_floor_8b.json").read_text())
    n = 36                                                               # Qwen3-8B's layers
    assert sorted(map(int, floor)) == list(range(math.ceil(2 * n / 3), n))
    assert all(0 < v < 1 for v in floor.values())                        # in units of D, under D


class _FakeCache:
    def __init__(self, n):
        self.n, self.cropped = n, []

    def get_seq_length(self):
        return self.n

    def crop(self, k):
        assert k < 0                                                     # transformers 5 takes a count to remove
        self.n += k
        self.cropped.append(k)


def test_the_agent_keeps_only_the_cache_it_shares_with_the_prompt():
    torch = pytest.importorskip("torch")
    agent = object.__new__(H.Agent)
    agent._kv, agent._kv_ids = _FakeCache(6), torch.tensor([[1, 2, 3, 4, 5, 6]])
    assert agent._reuse(torch.tensor([[1, 2, 3, 9, 9, 9, 9]])) is agent._kv and agent._kv.n == 3
    agent._kv, agent._kv_ids = _FakeCache(3), torch.tensor([[1, 2, 3]])
    assert agent._reuse(torch.tensor([[1, 2, 3]])).get_seq_length() == 2   # one token is always left to read
    agent._kv, agent._kv_ids = _FakeCache(3), torch.tensor([[1, 2, 3]])
    fresh = agent._reuse(torch.tensor([[7, 2, 3, 4]]))                   # nothing shared: a new cache
    assert fresh is not agent._kv and fresh.get_seq_length() == 0
