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
FORMS = ("Commands the game understands: look, inventory, go <direction>, examine <thing>, take <thing>, take <thing> "
         "from <container or supporter>, drop <thing>, put <thing> on <supporter>, insert <thing> into <container>, "
         "open <thing>, close <thing>, unlock <thing> with <key>, lock <thing> with <key>, eat <thing>.")
SYSTEM_FORMS = SYSTEM + " " + FORMS          # Stage 2's player (owner, 2026-11-05); Stage 1 and self-talk keep SYSTEM
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


# Rehearsals (the agent's, 2026-11-04, outside the frozen seeds): seed s + REHEARSAL is built from its own seed, so its
# world is new, but takes the plan of s (pool, planted kind). A rehearsal spends no registered world and decides nothing.
REHEARSAL = 900_000


def mirrored(seed: int) -> int:
    """The registered seed whose plan a rehearsal seed follows; a registered seed is its own."""
    return seed - REHEARSAL if seed >= REHEARSAL else seed


def planted(seed: int, spec: dict | None = None) -> dict | None:
    """The planted event of a Step -1 world: its kind and turn, or None. A key: read by the harness and scoring only.
    A rehearsal seed takes the kind of the seed it mirrors and draws its turn from its own seed."""
    p = (spec or worlds())["pools"]["step_minus_1"]["planted"]
    for kind in ("new_room", "rule_change"):
        if p[kind][0] <= mirrored(seed) <= p[kind][1]:
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


def ascents(turns: list[dict], start: int | None) -> list[int]:
    """Turns where a quest step is reached (owner, 2026-11-04, option (a)): the quest's remaining non-movement
    commands reach a new low for the episode. `start` is the count at the opening; a turn whose count is None (no
    winning walkthrough, or none recorded) is never an ascent and leaves the low as it was."""
    low, out = start, []
    for i, t in enumerate(turns):
        n = t.get("quest_left")
        if n is None:
            continue
        if low is not None and n < low:
            out.append(i)
        low = n if low is None else min(low, n)
    return out


def quest_left(policy, won: bool) -> int | None:
    """The non-movement commands left in the engine's walkthrough. An empty walkthrough counts only when the quest is
    won: TextWorld may return none when the quest can no longer be won, which is not progress."""
    if not policy:
        return 0 if won else None
    return sum(not c.strip().lower().startswith("go ") for c in policy)


# --- worlds (TextWorld) -------------------------------------------------------------------------------------------

def build(seed: int, out: Path, kind: str | None = None, spec: dict | None = None) -> dict:
    """Compile the world for `seed` into `out`. A new-room world is compiled twice: A, with one extra room behind a
    locked door whose key is not in the world, and B, the same with the door closed but unlocked. Returns the paths
    and the planted room's names as the compiled game holds them."""
    import textworld
    from textworld.generator import Game, World, compile_game, make_game
    from textworld.logic import Proposition, Variable

    p = params(seed, spec)
    opts = textworld.GameOptions()
    opts.seeds, opts.nb_rooms, opts.nb_objects, opts.quest_length = seed, p["rooms"], p["objects"], p["quest_length"]
    # "goal_only": the opening states the quest's last step, not every step (TextWorld's only_last_action)
    opts.grammar.only_last_action = bool(((spec or worlds())["generator"]).get("goal_only", False))
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


BANNER = set(" _|\\/$<>")


def unbanner(opening: str) -> str:
    """The opening without TextWorld's ASCII-art title (owner, 2026-11-04, Step -1 amendment 1): leading lines made only
    of the banner's characters are dropped. A line with any letter or digit is never touched."""
    lines = opening.splitlines()
    while lines and set(lines[0]) <= BANNER:
        lines.pop(0)
    return "\n".join(lines).strip()


def facts_key(facts, drop: str | None = None) -> frozenset:
    return frozenset(str(f) for f in facts if drop is None or drop not in str(f))


# --- the harness --------------------------------------------------------------------------------------------------

@dataclass
class RuleChange:
    """At the planted turn a command stops working (Step -1 amendment 2, owner, 2026-11-04): the agent's latest command
    that changed the world (not a move), or, if it has made none, the next non-move command of the engine's
    walkthrough. The turn announces it; every later use gets a fixed reply and never reaches the engine. `source`
    says which it was (a key)."""
    turn: int
    blocked: str | None = None
    fired_at: int | None = None
    source: str | None = None
    changing: list[str] = field(default_factory=list)

    def before(self, t: int, command: str) -> str | None:
        if self.blocked is not None and command.strip().lower() == self.blocked:
            return f"You cannot {self.blocked} any more: the rules here have changed."
        return None

    def after(self, t: int, command: str, changed: bool, walkthrough=()) -> str:
        """The announcement, on the planted turn only; empty otherwise (and if there is nothing to block)."""
        cmd = command.strip().lower()
        if changed and not cmd.startswith("go "):
            self.changing.append(cmd)
        if self.blocked is None and t == self.turn:
            ahead = [c.strip().lower() for c in walkthrough or () if not c.strip().lower().startswith("go ")]
            if self.changing:
                self.blocked, self.source = self.changing[-1], "played"
            elif ahead:
                self.blocked, self.source = ahead[0], "walkthrough"
            else:
                return ""
            self.fired_at = t
            return f"Something in this place shifts. From now on, you cannot {self.blocked}."
        return ""


class Engine:
    """One episode's world. For a new-room world, `switch` moves to B by replaying, silently, the commands that changed
    A's state, then checks that the two states agree apart from the door; on a mismatch it stays in A (counted)."""

    INFOS = dict(facts=True, inventory=False, location=True, score=True, admissible_commands=False,
                 policy_commands=True, won=True)

    def __init__(self, built: dict, admissible: bool = False):
        import textworld

        self._tw, self.built = textworld, built
        self.infos = textworld.EnvInfos(**{**self.INFOS, "admissible_commands": admissible})
        self.env = textworld.start(built["A"], request_infos=self.infos)
        self.state = self.env.reset()
        self.changing: list[str] = []

    def opening(self) -> str:
        return unbanner(clean(self.state.feedback))

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
        won = bool(getattr(s, "won", False))
        return {"facts": sorted(facts_key(s.facts)), "score": s.score, "room": room,
                "done": bool(s.done), "won": won, "quest_left": quest_left(s.policy_commands, won)}


# --- Stage 1's player: the script (PLAN.md, "Two stages") -------------------------------------------------------

SCRIPT_SEED, EXPLORE, LOOP, CYCLE = 700000, (8, 15), (5, 12), (2, 4)


class Script:
    """Stage 1's player (fixed by the owner, 2026-11-05): explore phases of 8-15 turns (a move through an exit not yet
    taken from this room if there is one, else an admissible command not yet tried here, else any admissible one) and
    loop phases of 5-12 turns (a cycle of 2-4 commands already used, repeated), alternating from explore, every
    choice drawn from default_rng(seed + 700000). It reads what a player sees (the room it is in) and the admissible
    commands, never the plant; and it never issues the quest's final command (read from the walkthrough), so an
    episode runs its full length. Its actions are the code's."""

    name, revision, scripted = "script-v1", None, True

    def start(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed + SCRIPT_SEED)
        self.tried: set = set()
        self.used: list[str] = []
        self.phase, self.left, self.cycle, self.i = "loop", 0, [], 0

    def _next_phase(self) -> None:
        self.phase = "explore" if self.phase == "loop" else "loop"
        lo, hi = EXPLORE if self.phase == "explore" else LOOP
        self.left = int(self.rng.integers(lo, hi + 1))
        if self.phase == "loop":
            if len(self.used) < 2:                       # nothing to loop over yet: explore again
                self.phase, self.left = "explore", int(self.rng.integers(EXPLORE[0], EXPLORE[1] + 1))
            else:
                n = min(int(self.rng.integers(CYCLE[0], CYCLE[1] + 1)), len(self.used))
                self.cycle = [str(c) for c in self.rng.choice(self.used, size=n, replace=False)]
                self.i = 0

    def act(self, engine) -> str:
        if self.left <= 0:
            self._next_phase()
        self.left -= 1
        state = engine.state
        walk = list(state.policy_commands or [])
        final = walk[0].strip().lower() if len(walk) == 1 else None
        ok = sorted({c for c in (state.admissible_commands or []) if c.strip().lower() != final})
        room = engine.keys()["room"]
        if self.phase == "loop":
            cmd = self.cycle[self.i % len(self.cycle)]
            self.i += 1
            if cmd.strip().lower() == final:             # the quest's last step: never
                cmd = str(self.rng.choice(ok)) if ok else "look"
        else:
            fresh = [c for c in ok if (room, c) not in self.tried]
            moves = [c for c in fresh if c.startswith("go ")]
            pick = moves or fresh or ok
            cmd = str(self.rng.choice(pick)) if pick else "look"
        self.tried.add((room, cmd))
        if cmd not in self.used:
            self.used.append(cmd)
        return cmd


# --- the agent ----------------------------------------------------------------------------------------------------

class Agent:
    """Qwen3 with thinking off, sampling one command a turn. Plays only; states are read later from the record.

    The key-value cache is kept between turns (`cache=True`): each prompt is compared with the tokens the cache holds,
    the cache is cut back to the longest shared start, and only the rest is read. What the model sees is the same
    text either way; dropped turns or a new episode shorten the shared start, and the restating prompt bypasses the
    cache. `sample=False` (greedy) is only for checking that cached and uncached play agree."""

    def __init__(self, model: str, device: str = "auto", dtype: str = "bfloat16", seed: int = 0,
                 cache: bool = True, sample: bool = True, nf4: bool = False, context: int = CONTEXT_TOKENS,
                 forms: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.name, self.cache, self.sample = torch, model, cache, sample
        self.load = "nf4" if nf4 else dtype                 # recorded on every episode; nf4 is for rehearsals only
        self.context = context                              # recorded too; below 16384 for rehearsals only
        self.system = SYSTEM_FORMS if forms else SYSTEM     # Stage 2's player is told the command forms
        self.tok = AutoTokenizer.from_pretrained(model)
        extra = {}
        if nf4:
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                              bnb_4bit_compute_dtype=getattr(torch, dtype))
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, dtype), device_map=device,
                                                          **extra)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self._kv, self._kv_ids = None, None                # the cache, and the tokens it holds
        self.read_tokens = 0                               # tokens the model read for prompts, for throughput
        torch.manual_seed(seed)

    def _reuse(self, ids):
        """The cache cut back to the longest start it shares with `ids` (at least one token left to read), or a new
        empty cache when nothing is shared."""
        from transformers import DynamicCache

        if self._kv is None or self._kv_ids is None:
            return DynamicCache()
        n = min(self._kv_ids.shape[1], ids.shape[1] - 1)
        same = (self._kv_ids[0, :n] == ids[0, :n]).cpu()
        shared = int(same.long().cumprod(0).sum()) if n else 0
        if shared == 0:
            return DynamicCache()
        extra = self._kv.get_seq_length() - shared
        if extra > 0:
            self._kv.crop(-extra)                          # a negative count removes tokens (transformers 4 and 5)
        return self._kv

    def _ids(self, messages):
        # return_dict: transformers 5 returns a BatchEncoding here even without it, so ask for one and take the ids.
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True)["input_ids"]

    def fit(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Drop the oldest turns (after the system prompt and the opening) until the context fits; returns the
        messages and how many turns were dropped."""
        dropped = 0
        while self._ids(messages).shape[1] > self.context - MAX_ACTION_TOKENS and len(messages) > 4:
            messages = messages[:2] + messages[4:]
            dropped += 1
        return messages, dropped

    def say(self, messages: list[dict], max_tokens: int = MAX_ACTION_TOKENS, use_cache: bool = True) -> str:
        ids = self._ids(messages).to(self.model.device)
        keep = self.cache and use_cache
        kv = self._reuse(ids) if keep else None
        self.read_tokens += ids.shape[1] - (kv.get_seq_length() if kv is not None else 0)
        sampling = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P) if self.sample else dict(do_sample=False)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_tokens, pad_token_id=self.tok.eos_token_id,
                                      past_key_values=kv, **sampling)
        if keep:
            self._kv, self._kv_ids = kv, out[:, :kv.get_seq_length()]
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return (text.splitlines() or [""])[0].strip()

    def restate(self, text: str) -> str:
        return self.say([{"role": "user", "content": RESTATE.format(text=text)}], max_tokens=64, use_cache=False)


# --- episodes -----------------------------------------------------------------------------------------------------

def play(agent, record, seed: int, games: Path, max_turns: int | None = None, kind: str | None = "auto",
         generator: dict | None = None) -> None:
    """One episode on world `seed`, appended to `record` turn by turn. Each turn is two chunks, the action (the
    model's, self) and the reply (the code's, external, elicited), then one key entry the readout never reads."""
    spec = worlds()
    if generator:                            # a candidate setting, rehearsals only: recorded on the episode
        spec = {**spec, "generator": generator}
    max_turns = max_turns or spec["episode"]["max_turns"]
    plant = planted(seed, spec) if kind == "auto" else None
    built = build(seed, games, plant["kind"] if plant else None, spec)
    scripted = getattr(agent, "scripted", False)
    engine = Engine(built, admissible=scripted)       # the script reads the admissible commands
    if scripted:
        agent.start(seed)
    rule = RuleChange(plant["turn"]) if plant and plant["kind"] == "rule_change" else None
    record.append("episode", "code", seed=seed, rehearsal=seed != mirrored(seed), pool_params=params(seed, spec),
                  model=agent.name,
                  revision=agent.revision, kv_cache=getattr(agent, "cache", None), load=getattr(agent, "load", None),
                  context=getattr(agent, "context", None), generator=generator,
                  system=None if scripted else getattr(agent, "system", SYSTEM), planted_kind=plant["kind"] if plant else None,
                  quest_left=engine.keys()["quest_left"])
    opening = engine.opening()
    record.chunk(opening, "code", "external", cause="spontaneous", seed=seed, turn=-1, slot="world")
    system = getattr(agent, "system", SYSTEM)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": opening}]
    seen = {facts_key(engine.state.facts)}
    door = built.get("door")                 # the plant's own change, left out when asking whether play was new too
    seen_play = {facts_key(engine.state.facts, door)}
    score = engine.state.score
    for t in range(max_turns):
        if scripted:                         # Stage 1: the code acts; the reading is unchanged (its slot is the player's)
            action, by, dropped = agent.act(engine), "code", 0
        else:
            messages, dropped = agent.fit(messages)
            action, by = agent.say(messages), "model"
        record.chunk(action, by, "self", seed=seed, turn=t, slot="self", dropped=dropped)
        blocked = rule.before(t, action) if rule else None
        if blocked is not None:
            reply, changed = blocked, False
        else:
            reply, changed = engine.step(action)
        note, event = {}, ""
        if plant and plant["kind"] == "new_room" and t == plant["turn"]:
            note["switched"] = engine.switch()
            if note["switched"]:
                event = f"You hear a click from the {built['door']}."
        if rule:
            event = rule.after(t, action, changed, engine.state.policy_commands)
            if rule.fired_at == t:
                note["blocked"], note["blocked_from"] = rule.blocked, rule.source
        record.chunk(reply, "code", "external", cause="elicited", seed=seed, turn=t, slot="world")
        if event:                            # the plant arrives as its own chunk (Step -1 amendment 3); the agent
            record.chunk(event, "code", "external", cause="spontaneous", seed=seed, turn=t, slot="world")
            reply += "\n\n" + event        # still reads reply and event as one message, as a player would
        keys = engine.keys()
        state = frozenset(keys["facts"])
        new = state not in seen
        seen.add(state)
        played = facts_key(engine.state.facts, door)
        if note:                             # did the agent's own action also make a new state on the planted turn?
            note["natural_new"] = played not in seen_play
        seen_play.add(played)
        record.append("key", "code", seed=seed, turn=t, new_state=new, score_change=keys["score"] - score,
                      room=keys["room"], quest_left=keys["quest_left"], planted_here=bool(note), **note)
        score = keys["score"]
        messages += [{"role": "assistant", "content": action}, {"role": "user", "content": reply}]
        if keys["done"]:
            break
    record.append("end", "code", seed=seed, turns=t + 1, done=bool(keys["done"]))


def self_talk(agent, record, seed: int, variant: str, turns: int = 30) -> None:
    """No engine: the reply slot carries the agent's own previous action, verbatim or restated by the same model.
    Recorded honestly: the reply's provenance is self and it is the model's; only its slot says world."""
    record.append("episode", "code", seed=seed, self_talk=variant, rehearsal=seed != mirrored(seed), model=agent.name,
                  revision=agent.revision,
                  kv_cache=getattr(agent, "cache", None))
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
    record.append("end", "code", seed=seed, self_talk=variant, turns=turns, done=False)
