"""Step -1C (body/stepm1c.py): its pool, each seed's plan, the drawing rule, and the scoring of its two bars, on toy
material. The planted room's description and an episode's end on entry run only where TextWorld and the Qwen3
tokenizer are available."""
import json

import numpy as np
import pytest

from body import habitat as H
from body import stepm1c as C

ROOT = H.ROOT / "material"


def _registered():
    out = set()
    for name in ("habitat_worlds.json", "habitat_worlds_stage2.json"):
        spec = json.loads((ROOT / name).read_text())
        for p in spec["pools"].values():
            for key in ("seeds",):
                if key in p:
                    out |= set(range(p[key][0], p[key][1] + 1))
            if "self_talk" in p:
                out |= set(range(p["self_talk"]["seeds"][0], p["self_talk"]["seeds"][1] + 1))
    return out


def test_pool_is_disjoint_from_every_registered_pool_and_babbling():
    pool = set(C.seeds())
    assert len(pool) == 1000 and not pool & _registered()
    assert max(pool) < H.worlds()["pools"]["babbling"]["seeds_from"]
    mirror = set(C.seeds(rehearsal=True))
    assert not mirror & pool and not mirror & {s + H.REHEARSAL for s in _registered()}


def test_generator_is_stage_1s():
    keep = ("rooms", "objects", "quest_length", "drawn_by")
    assert {k: C.spec()["generator"][k] for k in keep} == {k: H.worlds()["generator"][k] for k in keep}
    assert C.spec()["episode"]["max_turns"] == H.worlds()["episode"]["max_turns"]


def test_plan_kind_by_parity_turn_as_step_minus_1_rehearsal_mirrors():
    p = [C.plan(s) for s in (6000, 6001)]
    assert [x["vault"] for x in p] == ["on", "off"] and all(x["kind"] == "new_room" for x in p)
    assert all(15 <= C.plan(s)["turn"] <= 35 for s in C.seeds()[:200])
    assert C.plan(6001 + H.REHEARSAL)["vault"] == "off"
    assert C.plan(6000)["turn"] == int(np.random.default_rng(6000 + 500000).integers(15, 36))


def test_draw_takes_seeds_in_order_until_each_kind_is_full():
    fate = {s: ("entered" if s % 5 else "unmatched") for s in range(6000, 6100)}
    fate[6002] = "played"
    played = []

    def play_one(seed):
        played.append(seed)
        return fate[seed]

    out = C.draw(list(range(6000, 6100)), play_one, per_kind=3)
    assert out["full"] and out["usable"] == {"on": 3, "off": 3}
    assert played == sorted(played) and played[-1] < 6012
    on = out["counts"]["on"]
    assert on["entered"] == 3 and on["never entered"] == 1 and on["no length match"] == 1     # 6000, 6002, 6010
    # a seed already in the record is counted and not played again
    played.clear()
    again = C.draw(list(range(6000, 6100)), play_one, per_kind=3, done={6000: "unmatched", 6001: "entered"})
    assert 6000 not in played and 6001 not in played and again["counts"] == out["counts"]


def test_draw_reports_a_pool_that_runs_out():
    out = C.draw([6000, 6001, 6002], lambda s: "entered", per_kind=5)
    assert not out["full"] and out["usable"] == {"on": 2, "off": 1}


def test_outcomes_from_the_record():
    entries = [{"kind": "skip", "seed": 1, "reason": "no length match"},
               {"kind": "skip", "seed": 2, "reason": "over 120 s"},
               {"kind": "episode", "seed": 3}, {"kind": "end", "seed": 3, "entered": True},
               {"kind": "episode", "seed": 4}, {"kind": "end", "seed": 4, "entered": False},
               {"kind": "episode", "seed": 5}]                              # cut off: played again
    assert C.outcomes(entries) == {1: "unmatched", 2: "unbuildable", 3: "entered", 4: "played"}


def _toy(n, shape_off, news_off, seed=0, layers=3):
    """n episodes per kind; the scored chunk is the reply of turn 20. shape_off and news_off shift the off kind."""
    rng = np.random.default_rng(seed)
    eps, news_by, shape_by, tokens_by = {}, {}, {}, {}
    for j in range(2 * n):
        kind = "on" if j % 2 == 0 else "off"
        chunks = [{"slot": "world", "turn": -1, "cause": "spontaneous"}]
        for t in range(21):
            chunks += [{"slot": "self", "turn": t}, {"slot": "world", "turn": t, "cause": "elicited"}]
        keys = {t: {"turn": t, **({"entered": True} if t == 20 else {})} for t in range(21)}
        key = (6000 + j, None)
        eps[key] = {"info": {"plan": {"vault": kind}}, "chunks": chunks, "keys": keys}
        S, N = np.full((len(chunks), layers), np.nan), np.full((len(chunks), layers), np.nan)
        up = kind == "off"
        S[-1] = rng.normal(10 + shape_off * up, 1, layers)
        N[-1] = rng.normal(3 + news_off * up, 1, layers)
        shape_by[key], news_by[key] = S, N
        tokens_by[key] = np.full(len(chunks), 30)
    return eps, news_by, shape_by, tokens_by


def test_score_passes_when_shape_parts_and_boundary_does_not():
    out = C.score(*_toy(40, shape_off=3.0, news_off=0.0), draws=300)
    assert out["shape"]["auc"] > 0.95 and out["shape"]["pass"]
    assert abs(out["apart_from_boundary"]["boundary_auc"] - 0.5) < 0.2
    assert out["apart_from_boundary"]["pass"] and out["apart_from_boundary"]["direction"] == "shape above boundary"
    assert out["pass"] and out["episodes"] == {"on": 40, "off": 40}
    assert out["precondition"]["on"] == "40 of 40 above the floor"


def test_score_fails_apart_when_shape_only_tracks_boundary():
    eps, news_by, shape_by, tokens_by = _toy(40, shape_off=3.0, news_off=0.0)
    for k in eps:                                     # boundary carries the same separation as shape
        news_by[k][-1] = shape_by[k][-1] - 7
    out = C.score(eps, news_by, shape_by, tokens_by, draws=300)
    assert out["shape"]["pass"] and not out["apart_from_boundary"]["pass"] and not out["pass"]


def test_score_fails_apart_when_boundary_parts_better_than_shape():
    """Amendment 1: one-sided. A boundary that separates the kinds better than shape no longer passes bar 2."""
    out = C.score(*_toy(40, shape_off=1.2, news_off=6.0), draws=300)
    d = out["apart_from_boundary"]
    assert d["direction"] == "boundary above shape" and d["interval_90"][1] < 0 and not d["pass"]


def test_shape_given_boundary_sees_through_a_ceiling():
    """Both AUCs near 1, so bar 2 cannot pass, yet shape less its line on boundary still parts the kinds."""
    eps, news_by, shape_by, tokens_by = _toy(60, shape_off=0.0, news_off=0.0, seed=3)
    rng = np.random.default_rng(4)
    for k, ep in eps.items():
        up = ep["info"]["plan"]["vault"] == "off"
        nv = rng.normal(5 + 4 * up, 1)                    # boundary parts the kinds almost fully
        news_by[k][-1] = nv
        shape_by[k][-1] = 2 * nv + 3 * up + rng.normal(0, 0.5)  # shape: boundary's part plus its own
    out = C.score(eps, news_by, shape_by, tokens_by, draws=300)
    assert out["apart_from_boundary"]["boundary_auc"] > 0.95 and not out["apart_from_boundary"]["pass"]
    g = out["reported"]["shape given boundary"]
    assert g["shape_auc"] > 0.65                          # conservative: the line on news takes up some of the effect


def test_shape_given_boundary_at_chance_when_shape_is_boundary():
    eps, news_by, shape_by, tokens_by = _toy(60, shape_off=0.0, news_off=4.0, seed=5)
    rng = np.random.default_rng(6)
    for k in eps:
        shape_by[k][-1] = 2 * news_by[k][-1] + 1 + rng.normal(0, 0.3)
    g = C.score(eps, news_by, shape_by, tokens_by, draws=200)["reported"]["shape given boundary"]
    assert abs(g["shape_auc"] - 0.5) < 0.15


def test_score_fails_shape_at_chance():
    out = C.score(*_toy(40, shape_off=0.0, news_off=0.0), draws=300)
    assert not out["shape"]["pass"] and not out["pass"]


def test_scored_is_the_entry_turns_reply_and_none_without_entry():
    eps, *_ = _toy(1, 0, 0)
    ep = eps[(6000, None)]
    assert C.scored(ep) == len(ep["chunks"]) - 1
    ep["keys"][20].pop("entered")
    assert C.scored(ep) is None


# --- TextWorld and the tokenizer ----------------------------------------------------------------------------------

def _needs_world():
    pytest.importorskip("textworld")
    pytest.importorskip("transformers")
    from body import offtheme
    try:
        offtheme.qwen_tokens("x")
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")


def _desc(path, room="r_vault"):
    infos = json.loads(path.with_suffix(".json").read_text())["infos"]
    return next(v["desc"] for k, v in infos if k == room)


def test_vault_description_off_replaces_house_on_keeps_it(tmp_path):
    _needs_world()
    from pathlib import Path
    on = H.build(906004, tmp_path / "on", "new_room", C.spec(), vault="on")
    off = H.build(906004, tmp_path / "off", "new_room", C.spec(), vault="off")
    assert on["match"] == off["match"]                     # the match is drawn alike for both kinds
    house, theme = _desc(Path(on["B"])), _desc(Path(off["B"]))
    assert house != theme and _desc(Path(off["A"])) == theme
    assert "Room" in theme or "Sector" in theme or "Entered" in theme or "listing" in theme.lower()


def test_unmatched_seed_compiles_nothing(tmp_path):
    _needs_world()
    out = H.build(906001, tmp_path, "new_room", C.spec(), vault="off")
    assert out["unmatched"] and out["match"]["desc"] is None and not list(tmp_path.glob("*.z8"))


def test_episode_ends_on_entering_the_planted_room(tmp_path):
    _needs_world()
    from substrate.record import Record, read
    from body.stepm1 import episodes
    rec = Record(tmp_path / "record.jsonl")
    seed = 906004
    assert H.play(H.Script(), rec, seed, tmp_path / "games", worlds_path=C.WORLDS, plan=C.plan(seed)) == "entered"
    ep = episodes(read(rec.path))[(seed, None)]
    last = max(ep["keys"])
    assert ep["keys"][last].get("entered") and not ep["keys"][last].get("planted_here")
    assert [t for t, k in ep["keys"].items() if k.get("planted_here")] == [C.plan(seed)["turn"]]
    i = C.scored(ep)
    assert ep["chunks"][i]["turn"] == last and ep["chunks"][i]["slot"] == "world"
    assert C.outcomes(read(rec.path)) == {seed: "entered"}


def test_the_kind_changes_only_the_rooms_text_not_the_play(tmp_path):
    """The script never reads a description, so an on- and an off-structure world from one seed are played alike:
    who enters the room, and when, cannot depend on the kind."""
    _needs_world()
    from substrate.record import Record, read
    from body.stepm1 import episodes
    seed, acts, scored_text = 906004, {}, {}
    for vault in ("on", "off"):
        rec = Record(tmp_path / vault / "record.jsonl")
        H.play(H.Script(), rec, seed, tmp_path / vault / "games", worlds_path=C.WORLDS,
               plan={**C.plan(seed), "vault": vault})
        ep = episodes(read(rec.path))[(seed, None)]
        acts[vault] = [c["text"] for c in ep["chunks"] if c["slot"] == "self"]
        scored_text[vault] = ep["chunks"][C.scored(ep)]["text"]
    assert acts["on"] == acts["off"] and scored_text["on"] != scored_text["off"]


def test_cut_off_finds_episodes_begun_but_never_ended():
    from body.stepm1c import cut_off
    e = lambda kind, seed, **k: {"kind": kind, "seed": seed, **k}
    entries = [e("episode", 1), e("end", 1), e("episode", 2), e("chunk", 2), e("skip", 3, reason="no length match"),
               e("episode", 4)]
    assert cut_off(entries) == [2, 4]
    assert cut_off(entries + [{"kind": "resume", "seed": None, "cut_off": [2, 4]}]) == []
    assert cut_off(entries + [e("episode", 2), e("end", 2)]) == [4]


def test_play_capped_skips_an_episode_past_its_caps_and_numbers_on(tmp_path):
    from body.stepm1c import outcomes, play_capped
    from substrate.record import Record, read
    record = Record(tmp_path / "record.jsonl")
    record.append("note", "code", text="before")
    assert play_capped(record, 6001, tmp_path / "games", memory=64 * 2**20) == "unplayable"     # too little to start
    assert play_capped(record, 6001, tmp_path / "games", seconds=0.01) == "unplayable"         # too little time
    entries = read(record.path)
    assert [e["seq"] for e in entries] == list(range(len(entries)))
    skips = [e for e in entries if e["kind"] == "skip"]
    assert len(skips) == 2 and all(e["reason"].startswith("unplayable") for e in skips)
    assert outcomes(entries)[6001] == "unplayable"


def test_play_capped_plays_as_play_does(tmp_path):
    pytest.importorskip("textworld")
    from body.habitat import Script, play
    from body.stepm1c import WORLDS, plan, play_capped
    from substrate.record import Record, read
    (tmp_path / "a" / "games").mkdir(parents=True)
    (tmp_path / "b" / "games").mkdir(parents=True)
    ra, rb = Record(tmp_path / "a" / "r.jsonl"), Record(tmp_path / "b" / "r.jsonl")
    how_a = play(Script(), ra, 6001, tmp_path / "a" / "games", worlds_path=WORLDS, plan=plan(6001))
    how_b = play_capped(rb, 6001, tmp_path / "b" / "games")
    strip = lambda p: [{k: v for k, v in e.items() if k not in ("seq", "at")} for e in read(p)]
    assert how_a == how_b and strip(ra.path) == strip(rb.path)


def test_draw_takes_every_seed_when_the_pool_is_drawn_whole():
    out = C.draw([6000, 6001, 6002, 6003], lambda s: "entered" if s != 6002 else "played", per_kind=None)
    assert out["full"] and out["usable"] == {"on": 1, "off": 2}
    assert out["counts"]["on"]["never entered"] == 1
