"""Rate's calibration in Stage 1: the two scripts, the pairing, and the scoring, on toy material; one real pair of
worlds where TextWorld is installed."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from body import rate_calibration as rc
from substrate.record import Record


def engine(feedback: str, walk=("go north", "take key")):
    return SimpleNamespace(state=SimpleNamespace(feedback=feedback, policy_commands=list(walk)))


def test_named_directions_in_order_once_each():
    assert rc.named("There is an exit to the East. You should try going north, or east.") == ["east", "north"]
    assert rc.named("You have to open the passageway first.") == []
    assert rc.named("A northern light") == []                      # whole words only


def test_seeds_are_pairs_outside_every_pool():
    s = rc.seeds()
    assert len(s) == 40 and all(rc.partner(rc.partner(x)) == x for x in s)
    assert {rc.partner(x) for x in s} == set(s)
    assert len(rc.seeds(limit=5)) == 4                              # whole pairs only
    root = Path(__file__).resolve().parent.parent / "material"
    used = set()
    for name in ("habitat_worlds.json", "habitat_worlds_stage2.json", "habitat_worlds_stepm1c.json"):
        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k == "seeds" and isinstance(v, list) and len(v) == 2 and all(isinstance(i, int) for i in v):
                        used.update(range(v[0], v[1] + 1))
                    else:
                        walk(v)
        walk(json.loads((root / name).read_text()))
    assert used and not used & set(s)


def test_contingent_takes_a_named_direction_and_never_the_last_step():
    c = rc.Contingent()
    c.start(7000)
    picks = set()
    for _ in range(50):
        c.first = False
        picks.add(c.act(engine("-= Hall =-\nThere is an exit to the west. You should try going south.")))
    assert picks == {"go west", "go south"}
    c.first = False
    assert c.act(engine("Nothing here.")) == "look"
    c.first = False
    assert c.act(engine("There is an exit to the north.", walk=("go north",))) == "look"


def test_blind_replays_the_partner_whatever_it_is_told():
    b = rc.Blind({7001: ["go east", "go north", "look"]})
    b.start(7000)
    said = ["There is an exit to the west.", "You can't go that way.", "There is an exit to the south.", "x"]
    assert [b.act(engine(t)) for t in said] == ["go east", "go north", "look", "look"]


def test_compare_reads_the_direction():
    up = {w: {"contingent": 2.0 + w * 0.01, "blind": 0.0 + w * 0.01} for w in range(20)}
    r = rc.compare(up, draws=200)
    assert r["AUC"] == 1.0 and r["lower bound above 0.5"] and r["worlds contingent higher"] == 1.0
    same = {w: {"contingent": float(w), "blind": float(w)} for w in range(20)}
    assert not rc.compare(same, draws=200)["lower bound above 0.5"]


def _toy_run(tmp_path: Path, gap: float) -> Path:
    """Two arms on two worlds, with checkpoints as `read` writes them: X higher under contingent by `gap`."""
    run = tmp_path / "run"
    rng = np.random.default_rng(0)
    for arm in rc.ARMS:
        r = Record(run / arm / "record.jsonl")
        for seed in (7000, 7001):
            r.append("episode", "code", seed=seed)
            r.chunk("opening", "code", "external", cause="spontaneous", seed=seed, turn=-1, slot="world")
            for t in range(12):
                r.chunk("go north", "code", "self", seed=seed, turn=t, slot="self")
                r.chunk("a room", "code", "external", cause="elicited", seed=seed, turn=t, slot="world")
                r.append("key", "code", seed=seed, turn=t, new_state=t % 3 == 0, score_change=0, room=f"r{t % 2}")
            r.append("end", "code", seed=seed, turns=12, done=False)
            n = 25
            arrays = {}
            for nm in ("L24", "L25", "text"):
                for f in ("X", "S", "L_now", "X_w5", "X_w20", "C"):
                    v = np.full(n, np.nan)
                    v[21::2] = rng.normal(gap if arm == "contingent" else 0.0, 0.1, len(v[21::2]))
                    arrays[f"{nm}:{f}"] = v
            (run / "checkpoints").mkdir(parents=True, exist_ok=True)
            np.savez(run / "checkpoints" / f"{arm}-{seed}-a1.npz", **arrays)
    return run


def test_score_on_toy_checkpoints(tmp_path):
    out = rc.score(_toy_run(tmp_path, 3.0), draws=100)
    main = out["X, mean over the read layers"]
    assert out["worlds played by both"] == 2 and main["AUC"] == 1.0
    assert set(out["reported"]["X per layer"]) == {"L24", "L25"}
    assert out["reported"]["contingent: turns, share new state, share moved, share look"][0] == 24


def test_a_real_pair_of_worlds(tmp_path):
    pytest.importorskip("textworld")
    from body.habitat import REHEARSAL
    from body.stepm1 import episodes
    from substrate.record import read

    a = SimpleNamespace(run=str(tmp_path), limit=2, rehearsal=True)
    rc.cmd_play(a)
    acts = {arm: {s: [c["text"] for c in ep["chunks"] if c["slot"] == "self"]
                  for (s, _), ep in episodes(read(tmp_path / arm / "record.jsonl")).items()} for arm in rc.ARMS}
    s0, s1 = 7000 + REHEARSAL, 7001 + REHEARSAL
    assert set(acts["contingent"]) == set(acts["blind"]) == {s0, s1}
    for s in (s0, s1):
        mine, theirs = acts["blind"][s], acts["contingent"][rc.partner(s)]
        assert len(mine) == len(theirs) == 100
        # the blind script replays the partner's moves; only the quest's last step is ever swapped for look
        assert sum(m != t for m, t in zip(mine, theirs)) <= sum(m == "look" for m in mine)
    rc.cmd_play(a)                                           # resumes: nothing played twice
    assert len(episodes(read(tmp_path / "blind" / "record.jsonl"))) == 2


def test_read_then_score_on_a_tiny_model(tmp_path, monkeypatch):
    """The plumbing from the arms' records to the score, on a tiny randomly initialised Qwen3 (no download): every
    episode of both arms is read, checkpointed under its arm, and scored; a second read resumes."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nnterp")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    import substrate.states as states
    from substrate.record import read

    words = "go north south east west look a room exit there is an to the you can't that way".split()
    vocab = {"[UNK]": 0, "<bos>": 1, "<pad>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", bos_token="<bos>", pad_token="<pad>")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)).eval()
    real = states.Reader
    monkeypatch.setattr(states, "Reader", lambda *a, **kw: real(model, tokenizer=tok, sink="first"))

    run = tmp_path / "run"
    rng = np.random.default_rng(1)
    for arm in rc.ARMS:
        r = Record(run / arm / "record.jsonl")
        for seed in (7000, 7001):
            r.append("episode", "code", seed=seed)
            r.chunk("there is an exit to the north", "code", "external", cause="spontaneous", seed=seed, turn=-1,
                    slot="world")
            for t in range(16):
                d = str(rng.choice(["north", "south", "east", "west"]))
                r.chunk(f"go {d}", "code", "self", seed=seed, turn=t, slot="self")
                r.chunk(f"a room there is an exit to the {d}", "code", "external", cause="elicited", seed=seed,
                        turn=t, slot="world")
                r.append("key", "code", seed=seed, turn=t, new_state=False, score_change=0, room="r")
            r.append("end", "code", seed=seed, turns=16, done=False)
    sets = tmp_path / "sets.jsonl"
    sets.write_text("\n".join(json.dumps({"id": f"s{i}", "frozen": True, "texts": [f"go {w} a room" for w in words[1:6]]})
                              for i in range(3)))
    a = SimpleNamespace(run=str(run), model="tiny", device="cpu", dtype="float32", gpu_memory=None, sets=str(sets),
                        draft=False, no_variants=True)
    rc.cmd_read(a)
    names = sorted(p.name for p in (run / "checkpoints").glob("*.npz"))
    assert names == [f"{arm}-{s}-a1.npz" for arm in ("blind", "contingent") for s in (7000, 7001)]
    rc.cmd_read(a)                                          # resumes from the checkpoints
    entries = read(run / "record.jsonl")
    assert sum(e["kind"] == "resumed" and e.get("step", "").startswith(("contingent", "blind")) for e in entries) == 4
    assert [e["kind"] for e in entries].count("read_end") == 2
    out = rc.score(run, draws=50)
    assert out["worlds played by both"] == 2
    assert np.isfinite(out["X, mean over the read layers"]["AUC"])
    assert set(out["reported"]["X per layer"]) == {"L4", "L5"}


def _gate_run(tmp_path: Path, worlds: int, artifact: bool) -> Path:
    """Toy arms over `worlds` worlds of 40 turns. The blind arm hears can't-go on turns 12, 16, 20 and 24, so its
    clean scored turns are 10-12 and 35-39. X is 2 under contingent on every turn and 0 under blind; with `artifact`,
    blind's X is 2 on its clean turns too, so the whole gap lives where blind heard can't-go."""
    run = tmp_path / "run"
    rng = np.random.default_rng(0)
    for arm in rc.ARMS:
        r = Record(run / arm / "record.jsonl")
        for seed in range(7000, 7000 + worlds):
            r.append("episode", "code", seed=seed)
            r.chunk("opening", "code", "external", cause="spontaneous", seed=seed, turn=-1, slot="world")
            for t in range(40):
                r.chunk("go north", "code", "self", seed=seed, turn=t, slot="self")
                cant = arm == "blind" and t in (12, 16, 20, 24)
                r.chunk("You can't go that way." if cant else "a room", "code", "external", cause="elicited",
                        seed=seed, turn=t, slot="world")
                r.append("key", "code", seed=seed, turn=t, new_state=False, score_change=0, room="r")
            r.append("end", "code", seed=seed, turns=40, done=False)
            n = 81
            arrays = {}
            for nm in ("L24", "L25", "text"):
                for f in ("X", "S", "L_now", "X_w5", "X_w20", "C"):
                    v = np.full(n, np.nan)
                    for row in range(21, n, 2):              # the actions of turns 10-39
                        t = (row - 1) // 2
                        high = arm == "contingent" or (artifact and (t <= 12 or t >= 35))
                        v[row] = rng.normal(2.0 if high else 0.0, 0.1)
                    arrays[f"{nm}:{f}"] = v
            (run / "checkpoints").mkdir(parents=True, exist_ok=True)
            np.savez(run / "checkpoints" / f"{arm}-{seed}-a1.npz", **arrays)
    return run


def test_clean_rows_leave_out_any_window_with_a_cant_go():
    chunks = [{"slot": "world", "turn": -1, "text": "o"}]
    for t in range(15):
        chunks += [{"slot": "self", "turn": t, "text": "go north"},
                   {"slot": "world", "turn": t, "text": "You can't go that way." if t == 2 else "a room"}]
    turns = [chunks[i]["turn"] for i in rc.clean_rows({"chunks": chunks})]
    assert turns == [0, 1, 2, 13, 14]                         # turns 3-12 have turn 2's reply in their window


def test_the_gate_passes_only_when_both_hold(tmp_path):
    out = rc.score(_gate_run(tmp_path / "a", 12, artifact=False), draws=200)
    assert out["the can't-go check"]["worlds"] == 12 and out["gate"]["pass"]
    out = rc.score(_gate_run(tmp_path / "b", 12, artifact=True), draws=200)   # the gap lives only where can't-go is
    assert out["gate"]["main, lower bound above 0.5"] and not out["gate"]["pass"]
    assert "stays out" in out["gate"]["meaning"]


def test_too_few_clean_worlds_keep_rate_out(tmp_path):
    out = rc.score(_gate_run(tmp_path, 8, artifact=False), draws=200)
    assert out["the can't-go check"]["too few to decide"] and not out["gate"]["pass"]
    assert out["gate"]["can't-go check, lower bound above 0.5"] is None
