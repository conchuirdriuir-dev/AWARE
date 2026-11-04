"""The habitat (PLAN.md, "Body readout", "The habitat"): TextWorld games rebuilt from their seed, the harness's planted
events, an episode played by the model and appended to the record as it happens, and the engine's labels.

Who said what: the agent's actions are the model's; the world's replies, its true state and score are the code's (the
engine); planted events and labels are the code's (the harness). The engine's state, score, the new-state flag and the
planted turn are keys: written to the record as `key` entries, read only by scoring.

TextWorld is imported only where a game is built or played, so the rest (seeds, labels, k) runs and tests without it.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORLDS = Path(__file__).resolve().parent.parent / "material" / "habitat_worlds.json"
DIRS = {"north_of": "south_of", "south_of": "north_of", "east_of": "west_of", "west_of": "east_of"}
SYSTEM = ("You are playing a text adventure game. Read what the game says, then reply with one command and nothing "
          "else, such as: go north, take lamp, open door, examine table, inventory.")
CONTEXT_TOKENS, MAX_ACTION_TOKENS, TEMPERATURE, TOP_P = 16384, 24, 0.7, 0.95
RESTATE = "Say the following again in other words, as one short line and nothing else:\n\n{text}"


# --- seeds -------------------------------------------------------------------------------------------------------

def worlds(path: Path = WORLDS) -> dict:
    return json.loads(Path(path).read_text())


def _span(pair) -> list[int]:
    return list(range(pair[0], pair[1] + 1))


def pool(name: str, spec: dict | None = None) -> list[int]:
    spec = spec or worlds()
    p = spec["pools"]
    if name == "self_talk":
        return _span(p["step_minus_1"]["self_talk"]["seeds"])
    return _span(p[name]["seeds"])


def params(seed: int, spec: dict | None = None) -> dict:
    """The world's size, drawn from its seed alone, in the frozen order."""
    g = (spec or worlds())["generator"]
    rng = np.random.default_rng(seed)
    return {k: int(rng.integers(g[k][0], g[k][1] + 1)) for k in ("rooms", "objects", "quest_length")}


def planted(seed: int, spec: dict | None = None) -> dict | None:
    """The planted event of a Step -1 world: its kind and turn, or None. A key: read by the harness and scoring only."""
    p = (spec or worlds())["pools"]["step_minus_1"]["planted"]
    for kind in ("new_room", "rule_change"):
        if p[kind][0] <= seed <= p[kind][1]:
            return {"kind": kind, "turn": int(np.random.default_rng(seed + 500000).integers(15, 36))}
    return None


# --- labels and k (keys; scoring only) ---------------------------------------------------------------------------

def stalls(new: list[bool]) -> list[tuple[int, int, bool]]:
    """Runs of turns with no new state: (start, length, resolved), where resolved means a new state followed in the
    same episode."""
    out, start = [], None
    for t, n in enumerate(list(new) + [None]):
        if n is False and start is None:
            start = t
        elif n is not False and start is not None:
            out.append((start, t - start, n is True))
            start = None
    return out


def choose_k(episodes: list[list[bool]]) -> int:
    """The smallest whole number above the 95th percentile of the lengths of stalls that resolved (PLAN.md)."""
    lengths = [n for new in episodes for _, n, resolved in stalls(new) if resolved]
    if not lengths:
        raise ValueError("no resolved stall: k cannot be set")
    return int(np.floor(np.percentile(lengths, 95))) + 1


def closures(turns: list[dict], k: int) -> list[tuple[int, int]]:
    """(onset, length) of each closure: at least k turns with no new state, no score change, and at least half the
    actions repeating an earlier (action, room) pair of the episode."""
    seen, repeat = set(), []
    for t in turns:
        pair = (t["action"].strip().lower(), t["room"])
        repeat.append(pair in seen)
        seen.add(pair)
    out, start = [], None
    flat = [not t["new_state"] and t["score_change"] == 0 for t in turns]
    for i, f in enumerate(flat + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            n = i - start
            if n >= k and sum(repeat[start:i]) * 2 >= n:
                out.append((start, n))
            start = None
    return out


def ascents(turns: list[dict]) -> list[int]:
    return [i for i, t in enumerate(turns) if t["score_change"] > 0]


# --- worlds (TextWorld) -------------------------------------------------------------------------------------------

def build(seed: int, out: Path, kind: str | None = None) -> dict:
    """Compile the world for `seed` into `out`. A new-room world is compiled twice: A, with one extra room behind a
    locked door whose key is not in the world, and B, the same with the door closed but unlocked. Returns the paths
    and the planted room's names as the compiled game holds them."""
    import textworld
    from textworld.generator import Game, World, compile_game, make_game
    from textworld.logic import Proposition, Variable

    p = params(seed)
    opts = textworld.GameOptions()
    opts.seeds, opts.nb_rooms, opts.nb_objects, opts.quest_length = seed, p["rooms"], p["objects"], p["quest_length"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def compile_(game, name):
        o = textworld.GameOptions()
        o.path, o.force_recompile = str(out / f"w{seed}{name}.z8"), True
        return compile_game(game, o)

    base = make_game(opts)
    if kind != "new_room":
        return {"A": compile_(base, "")}
    facts = list(base.world.facts)
    used = {(f.arguments[1].name, f.name) for f in facts if f.name in DIRS}
    room, rel = next((r, d) for r in sorted(x.name for x in base.world.rooms) for d in DIRS if (r, d) not in used)
    R, V, D, O = Variable(room, "r"), Variable("r_vault", "r"), Variable("d_vault", "d"), Variable("o_vault", "o")
    plant = [Proposition(rel, (V, R)), Proposition(DIRS[rel], (R, V)), Proposition("link", (R, D, V)),
             Proposition("link", (V, D, R)), Proposition("at", (O, V))]
    paths, first = {}, None
    for name, state in (("A", "locked"), ("B", "closed")):
        game = Game(World.from_facts(facts + plant + [Proposition(state, (D,))]), base.grammar, base.quests)
        game.metadata = dict(base.metadata)
        if first is not None:                       # B is A with the door unlocked: the same names and text
            game._infos = copy.deepcopy(first.infos)
        paths[name] = compile_(game, name)
        first = first or game
    names = {k: first.infos[k].name for k in ("r_vault", "d_vault", "o_vault")}
    return {**paths, "door": names["d_vault"], "room": names["r_vault"], "object": names["o_vault"], "next_to": room}


STATUS = re.compile(r"-=\s.*?=-\s*\d+/\d+")


def clean(feedback: str) -> str:
    """The engine's text as a player reads it: the prompt line and the status bar (room and move count, which changes
    every turn) removed, so a reply's novelty is its content."""
    lines = [STATUS.sub("", l).rstrip() for l in feedback.splitlines() if not l.lstrip().startswith(">")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def facts_key(facts, drop: str | None = None) -> frozenset:
    return frozenset(str(f) for f in facts if drop is None or drop not in str(f))


# --- the harness --------------------------------------------------------------------------------------------------

@dataclass
class RuleChange:
    """From the planted turn on, a command that has already changed the world (not a move) stops working: the turn
    announces it, and every later use gets a fixed reply and never reaches the engine."""
    turn: int
    blocked: str | None = None
    fired_at: int | None = None
    changing: list[str] = field(default_factory=list)

    def before(self, t: int, command: str) -> str | None:
        if self.blocked is not None and command.strip().lower() == self.blocked:
            return f"You cannot {self.blocked} any more: the rules here have changed."
        return None

    def after(self, t: int, command: str, changed: bool) -> str:
        cmd = command.strip().lower()
        if changed and not cmd.startswith("go "):
            self.changing.append(cmd)
        if self.blocked is None and t >= self.turn and self.changing:
            self.blocked, self.fired_at = self.changing[-1], t
            return f"\n\nSomething in this place shifts. From now on, you cannot {self.blocked}."
        return ""


class Engine:
    """One episode's world. For a new-room world, `switch` moves to B by replaying, silently, the commands that changed
    A's state, then checks that the two states agree apart from the door; on a mismatch it stays in A (counted)."""

    INFOS = dict(facts=True, inventory=False, location=True, score=True, admissible_commands=False)

    def __init__(self, built: dict, admissible: bool = False):
        import textworld

        self._tw, self.built = textworld, built
        self.infos = textworld.EnvInfos(**{**self.INFOS, "admissible_commands": admissible})
        self.env = textworld.start(built["A"], request_infos=self.infos)
        self.state = self.env.reset()
        self.changing: list[str] = []

    def opening(self) -> str:
        return clean(self.state.feedback)

    def step(self, command: str) -> tuple[str, bool]:
        before = facts_key(self.state.facts)
        self.state, _, _ = self.env.step(command)
        changed = facts_key(self.state.facts) != before
        if changed:
            self.changing.append(command)
        return clean(self.state.feedback), changed

    def switch(self) -> bool:
        door = self.built["door"]
        env = self._tw.start(self.built["B"], request_infos=self.infos)
        state = env.reset()
        for c in self.changing:
            state, _, _ = env.step(c)
        same = facts_key(state.facts, door) == facts_key(self.state.facts, door) and state.score == self.state.score
        if same:
            self.env, self.state = env, state
        return same

    def keys(self) -> dict:
        s = self.state
        # The player's room from its at(P, room) fact: TextWorld 1.7's `location` info comes back None on these games.
        room = next((f.arguments[1].name for f in s.facts if f.name == "at" and f.arguments[0].type == "P"), None)
        return {"facts": sorted(facts_key(s.facts)), "score": s.score, "room": room,
                "done": bool(s.done), "won": bool(getattr(s, "won", False))}


# --- the agent ----------------------------------------------------------------------------------------------------

class Agent:
    """Qwen3 with thinking off, sampling one command a turn. Plays only; states are read later from the record."""

    def __init__(self, model: str, device: str = "auto", dtype: str = "bfloat16", seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.name = torch, model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        torch.manual_seed(seed)

    def _ids(self, messages):
        # return_dict: transformers 5 returns a BatchEncoding here even without it, so ask for one and take the ids.
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True)["input_ids"]

    def fit(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Drop the oldest turns (after the system prompt and the opening) until the context fits; returns the
        messages and how many turns were dropped."""
        dropped = 0
        while self._ids(messages).shape[1] > CONTEXT_TOKENS - MAX_ACTION_TOKENS and len(messages) > 4:
            messages = messages[:2] + messages[4:]
            dropped += 1
        return messages, dropped

    def say(self, messages: list[dict], max_tokens: int = MAX_ACTION_TOKENS) -> str:
        ids = self._ids(messages).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=True, temperature=TEMPERATURE,
                                      top_p=TOP_P, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return (text.splitlines() or [""])[0].strip()

    def restate(self, text: str) -> str:
        return self.say([{"role": "user", "content": RESTATE.format(text=text)}], max_tokens=64)


# --- episodes -----------------------------------------------------------------------------------------------------

def play(agent, record, seed: int, games: Path, max_turns: int | None = None, kind: str | None = "auto") -> None:
    """One episode on world `seed`, appended to `record` turn by turn. Each turn is two chunks, the action (the
    model's, self) and the reply (the code's, external, elicited), then one key entry the readout never reads."""
    spec = worlds()
    max_turns = max_turns or spec["episode"]["max_turns"]
    plant = planted(seed, spec) if kind == "auto" else None
    built = build(seed, games, plant["kind"] if plant else None)
    engine = Engine(built)
    rule = RuleChange(plant["turn"]) if plant and plant["kind"] == "rule_change" else None
    record.append("episode", "code", seed=seed, pool_params=params(seed, spec), model=agent.name,
                  revision=agent.revision, planted_kind=plant["kind"] if plant else None)
    opening = engine.opening()
    record.chunk(opening, "code", "external", cause="spontaneous", seed=seed, turn=-1, slot="world")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": opening}]
    seen = {facts_key(engine.state.facts)}
    score = engine.state.score
    for t in range(max_turns):
        messages, dropped = agent.fit(messages)
        action = agent.say(messages)
        record.chunk(action, "model", "self", seed=seed, turn=t, slot="self", dropped=dropped)
        blocked = rule.before(t, action) if rule else None
        if blocked is not None:
            reply, changed = blocked, False
        else:
            reply, changed = engine.step(action)
        note = {}
        if plant and plant["kind"] == "new_room" and t == plant["turn"]:
            note["switched"] = engine.switch()
            if note["switched"]:
                reply += f"\n\nYou hear a click from the {built['door']}."
        if rule:
            reply += rule.after(t, action, changed)
            if rule.fired_at == t:
                note["blocked"] = rule.blocked
        record.chunk(reply, "code", "external", cause="elicited", seed=seed, turn=t, slot="world")
        keys = engine.keys()
        state = frozenset(keys["facts"])
        new = state not in seen
        seen.add(state)
        record.append("key", "code", seed=seed, turn=t, new_state=new, score_change=keys["score"] - score,
                      room=keys["room"], planted_here=bool(note), **note)
        score = keys["score"]
        messages += [{"role": "assistant", "content": action}, {"role": "user", "content": reply}]
        if keys["done"]:
            break


def self_talk(agent, record, seed: int, variant: str, turns: int = 30) -> None:
    """No engine: the reply slot carries the agent's own previous action, verbatim or restated by the same model.
    Recorded honestly: the reply's provenance is self and it is the model's; only its slot says world."""
    record.append("episode", "code", seed=seed, self_talk=variant, model=agent.name, revision=agent.revision)
    opening = "You are alone in a quiet room. Nothing here answers you but your own words."
    record.chunk(opening, "code", "external", cause="spontaneous", seed=seed, turn=-1, slot="world", variant=variant)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": opening}]
    for t in range(turns):
        messages, dropped = agent.fit(messages)
        action = agent.say(messages)
        record.chunk(action, "model", "self", seed=seed, turn=t, slot="self", variant=variant, dropped=dropped)
        reply = action if variant == "exact" else agent.restate(action)
        record.chunk(reply, "model", "self", seed=seed, turn=t, slot="world", variant=variant)
        messages += [{"role": "assistant", "content": action}, {"role": "user", "content": reply}]
