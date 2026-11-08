"""Body readout v0.2, Step 2: the babbling gate (PLAN.md, "Babbling (v0.2 Step 2): the gate", fixed by the owner
2026-11-08 except bars 1-3's ratios, which wait on this code's 0.6B rehearsal).

The steps, in order; each is resumable and appends to the run's record. The record comes first: every turn, and the
sense vector each grafted turn was given, is appended as it happens; readings, the space, the grafts and the scores
are derived from it.

    python -m body.babble play    --run R --model M --round 0      # the base model plays round 0 (no tokens)
    python -m body.babble play    --run R --model M --heldout      # the base model plays the held-out pool
    python -m body.babble read    --run R --model M --floor F      # senses and states, graft off, every episode
    python -m body.babble fit     --run R                          # z-scores and the 64-d reply space, round 0 only
    python -m body.babble train   --run R --model M --round r      # the live and zeroed grafts, on rounds 0..r
    python -m body.babble play    --run R --model M --round r      # r >= 1: the grafted model plays, live senses
    python -m body.babble score   --run R --model M                # every arm, teacher-forced on the held-out text

A rehearsal (--rehearsal) plays mirror seeds (+900000) and decides nothing; --per-round and --heldout-size scale it.

What each arm sees is set here, in code: live (the episode's own sense tokens), zeroed (zeros in their place, in
training and test), swapped (another held-out episode's tokens at the same turn), tokens alone (ridge on b, no text),
the linear history predictor (ridge on states), the base probe (ridge on the frozen model's hidden state). The yoked
arm is not built: as drafted it trains on the same text in the same steps as live, so it differs only by seed (the
agent's finding, 2026-11-08, for the owner).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from body.machine import record_machine
from body.senses import Episode, body_vectors, fit_z, zscore
from substrate.record import Record, read

K = 2                         # sense tokens
RANK, ALPHA = 16, 32.0
SPACE = 64                    # the reply space's dimensions (owner, 2026-11-08)
LM_WEIGHT = 0.1               # (owner, 2026-11-08)
LR = 1e-4
MAX_LEN = 8192                # tokens per training window
ROUNDS, PER_ROUND = 3, 200    # (owner, 2026-11-08)
BABBLE_FROM = 10000
HELDOUT = (9500, 9599)        # (owner, 2026-11-08)
HISTORY = 10                  # the linear predictor's window of earlier chunks
LAMBDA = 10.0
DRAWS = 2000
CAP = 256                     # X_c reads chunks cut at 256 tokens, as rate's calibration
CLIP = 10.0                   # the body vector's z-scores cut at +-10 in tokens and targets (agent)
EMPTY_TRIES = 10              # samples before an empty action cuts the episode (agent)
PROPOSED = {"bar 1": ("at most", 0.90), "bar 2": ("at least", 1.10), "bar 3": ("at most", 0.90)}  # not yet fixed


# --- seeds and episodes --------------------------------------------------------------------------------------------

def config(run: Path, a=None) -> dict:
    path = run / "config.json"
    if path.exists():
        return json.loads(path.read_text())
    if a is None:
        raise SystemExit(f"{run} has no config.json: play round 0 first")
    c = {"rehearsal": bool(a.rehearsal), "per_round": a.per_round, "heldout_size": a.heldout_size,
         "rounds": ROUNDS, "model": a.model}
    run.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, indent=1))
    return c


def seeds(c: dict, round_: int | None) -> list[int]:
    from body.habitat import REHEARSAL

    off = REHEARSAL if c["rehearsal"] else 0
    if round_ is None:
        return [s + off for s in range(HELDOUT[0], HELDOUT[0] + c["heldout_size"])]
    start = BABBLE_FROM + round_ * c["per_round"]
    return [s + off for s in range(start, start + c["per_round"])]


def episodes(run: Path) -> dict:
    """seed -> episode (body.stepm1.episodes), complete ones only."""
    from body.stepm1 import episodes as eps

    return {k[0]: v for k, v in eps(read(run / "record.jsonl")).items()} if (run / "record.jsonl").exists() else {}


def turns_of(ep: dict) -> list[dict]:
    """[{"turn", "action", "reply"}] in order; the opening is ep["chunks"][0]."""
    out: dict = {}
    for c in ep["chunks"]:
        if c["turn"] < 0:
            continue
        d = out.setdefault(c["turn"], {"turn": c["turn"], "action": "", "reply": ""})
        if c["slot"] == "self":
            d["action"] = c["text"]
        else:
            d["reply"] = c["text"] if not d["reply"] else d["reply"] + "\n\n" + c["text"]
    return [out[t] for t in sorted(out)]


# --- the text as the model reads it ---------------------------------------------------------------------------------

def _piece(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def render(tok, system: str, opening: str, turns: list[dict], k: int = K, prompt: bool = False) -> dict:
    """Token ids for an episode (or a window of it), with -1 at each sense token's place. Every assistant turn is
    `<|im_start|>assistant\\n` + k sense tokens + an empty think block + the action, in training and in play alike.
    Returns ids, and per turn the sense positions, the action's token span and its last token. With `prompt`, a last
    assistant header with its sense places is added for the model to act after."""
    ids = _piece(tok, f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{opening}<|im_end|>\n")
    info = []
    head, think = _piece(tok, "<|im_start|>assistant\n"), _piece(tok, "<think>\n\n</think>\n\n")
    for d in turns + ([None] if prompt else []):
        ids += head
        sense = list(range(len(ids), len(ids) + k))
        ids += [-1] * k + think
        if d is None:
            info.append({"sense": sense})
            break
        act = _piece(tok, d["action"]) + _piece(tok, "<|im_end|>")
        start = len(ids)
        ids += act
        last = start + len(act) - 2 if len(act) > 1 else start - 1     # the action's last token (before im_end)
        info.append({"turn": d["turn"], "sense": sense, "lm": (start, start + len(act)), "last": last})
        ids += _piece(tok, f"\n<|im_start|>user\n{d['reply']}<|im_end|>\n")
    return {"ids": ids, "turns": info}


def windows(tok, system: str, opening: str, turns: list[dict], max_len: int = MAX_LEN, k: int = K) -> list[dict]:
    """The episode in windows of at most max_len tokens, split between turns; each window starts with the system
    prompt and the opening."""
    out, cur = [], []
    for d in turns:
        trial = render(tok, system, opening, cur + [d], k)
        if len(trial["ids"]) > max_len and cur:
            out.append(render(tok, system, opening, cur, k))
            cur = [d]
        else:
            cur.append(d)
    if cur:
        out.append(render(tok, system, opening, cur, k))
    return out


# --- the graft: LoRA, the projection and two heads -----------------------------------------------------------------

class Graft:
    def __init__(self, model, k: int = K):
        import torch
        from torch import nn

        from body import lora

        self.torch, self.model, self.k = torch, model, k
        hidden = model.config.hidden_size
        self.hidden = hidden
        dev = model.device
        self.lora = lora.attach(model, RANK, ALPHA)
        emb_sd = float(model.get_input_embeddings().weight.float().std())
        self.proj = nn.Linear(3, k * hidden).to(dev)
        nn.init.normal_(self.proj.weight, std=emb_sd)
        nn.init.zeros_(self.proj.bias)
        self.self_head = nn.Linear(hidden, 2).to(dev)
        self.fm_head = nn.Linear(hidden, SPACE).to(dev)
        self.opt = None

    def params(self):
        from body import lora

        return (lora.parameters(self.lora) + list(self.proj.parameters()) + list(self.self_head.parameters())
                + list(self.fm_head.parameters()))

    def optimizer(self):
        if self.opt is None:
            self.opt = self.torch.optim.AdamW(self.params(), lr=LR)
        return self.opt

    def save(self, path: Path) -> None:
        from body import lora

        self.torch.save({"lora": lora.state(self.lora), "proj": self.proj.state_dict(),
                         "self_head": self.self_head.state_dict(), "fm_head": self.fm_head.state_dict(),
                         "opt": self.opt.state_dict() if self.opt else None}, path)

    def load(self, path: Path) -> None:
        from body import lora

        sd = self.torch.load(path, map_location="cpu", weights_only=False)
        lora.load_state(self.lora, sd["lora"])
        self.proj.load_state_dict(sd["proj"])
        self.self_head.load_state_dict(sd["self_head"])
        self.fm_head.load_state_dict(sd["fm_head"])
        self.opt = None
        if sd["opt"] is not None:
            self.optimizer().load_state_dict(sd["opt"])

    def fresh(self, seed: int) -> None:
        """Back to the untrained graft (the arm's own seed), for an arm with no saved state yet."""
        from torch import nn

        g = self.torch.Generator().manual_seed(seed)
        for w in self.lora.values():
            w.A.data.copy_(self.torch.empty_like(w.A, device="cpu").uniform_(-1, 1, generator=g)
                           * (1 / math.sqrt(w.A.shape[1])))
            w.B.data.zero_()
        emb_sd = float(self.model.get_input_embeddings().weight.float().std())
        self.proj.weight.data.copy_(self.torch.randn(self.proj.weight.shape, generator=g) * emb_sd)
        self.proj.bias.data.zero_()
        for head in (self.self_head, self.fm_head):        # zero: each head starts by predicting the mean
            nn.init.zeros_(head.bias)
            nn.init.zeros_(head.weight)
        self.opt = None

    def hidden_states(self, ids: list[int], B, mode: str):
        """The last layer's hidden states (after the final norm) for one window. mode: "live" (B's rows at the
        sense places, one row per turn), "zeroed" (zeros there) or "none" (ids rendered with k = 0, graft as set)."""
        torch = self.torch
        t_ids = torch.tensor([max(i, 0) for i in ids], device=self.model.device)
        emb = self.model.get_input_embeddings()(t_ids)
        pos = [i for i, x in enumerate(ids) if x == -1]
        if pos:
            p = torch.tensor(pos, device=emb.device)
            if mode == "zeroed":
                soft = torch.zeros(len(pos), self.hidden, device=emb.device, dtype=emb.dtype)
            else:
                rows = torch.as_tensor(np.asarray(B, dtype=np.float32), device=emb.device)
                soft = self.proj(rows).view(-1, self.hidden).to(emb.dtype)
            emb = emb.index_put((p,), soft)
        out = self.model.model(inputs_embeds=emb[None])
        return out.last_hidden_state[0]


def sense_rows(Z: np.ndarray, turns: list[dict]) -> np.ndarray:
    """The rows fed at each turn's sense places: before turn t, b_{t-1}; before turn 0, zeros."""
    return np.array([Z[d["turn"] - 1] if d["turn"] > 0 else np.zeros(3) for d in turns])


# --- the grafted player -------------------------------------------------------------------------------------------

class GraftedAgent:
    """The grafted model as habitat.play's agent. It keeps its own copy of the episode, reads each new chunk with the
    graft off (the same model, `lora.off`), computes the senses as tested (body.senses), and acts with the body
    vector b_{t-1} at its sense places. Each turn's vector is appended to the record before the model acts."""

    def __init__(self, model_name: str, graft_path: Path, z: dict, unit, floor_var, layers, record,
                 device="cuda", dtype="bfloat16", seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from body.habitat import SYSTEM_FORMS
        from substrate.states import Reader

        self.torch, self.name, self.record = torch, model_name, record
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.graft = Graft(self.model)
        self.graft.load(graft_path)
        self.reader = Reader(self.model, tokenizer=self.tok, sink="first")
        self.z, self.unit, self.fv, self.layers = z, unit, floor_var, layers
        self.system, self.cache, self.load, self.context = SYSTEM_FORMS, None, f"{dtype} + graft {graft_path.name}", 16384
        torch.manual_seed(seed)

    def _read(self, text: str):
        from body import lora

        capped = self.tok.decode(self.tok(text, add_special_tokens=False)["input_ids"][:CAP])
        with lora.off(self.graft.lora):
            full = self.reader.states([text], self.layers)[0]
            cap = full if capped == text else self.reader.states([capped], self.layers)[0]
        return full, cap

    def _add(self, text: str, slot: str, turn: int, cause: str | None = None):
        full, cap = self._read(text)
        self.chunks.append({"slot": slot, "turn": turn, "text": text, **({"cause": cause} if cause else {})})
        self.senses.append(self.episode.add(full, cap, slot, turn))

    def fit(self, messages):
        return messages, 0

    def say(self, messages) -> str:
        from body.habitat import MAX_ACTION_TOKENS, TEMPERATURE, TOP_P

        torch = self.torch
        if len(messages) == 2:                             # a new episode: the opening
            self.chunks, self.senses = [], []
            self.episode = Episode(self.unit, self.fv)
            self._add(messages[1]["content"], "world", -1, "spontaneous")
            self.turns = []
        else:
            t = len(self.turns)
            action, reply = messages[-2]["content"], messages[-1]["content"]
            self._add(action, "self", t)
            self._add(reply, "world", t, "elicited")
            self.turns.append({"turn": t, "action": action, "reply": reply})
        t = len(self.turns)
        if t == 0:
            row = np.zeros(3)
        else:
            Z = zscore(body_vectors(self.chunks, self.senses), self.z, CLIP)
            row = Z[t - 1]
        b = body_vectors(self.chunks, self.senses).get(t - 1, (None, None))
        self.record.append("sense", "code", turn=t, row=[float(x) for x in row],
                           boundary=None if b[0] is None or not np.isfinite(b[0]) else float(b[0]),
                           X_c=None if b[1] is None or not np.isfinite(b[1]) else float(b[1]))
        keep = list(self.turns)
        while True:
            r = render(self.tok, self.system, self.chunks[0]["text"], keep, prompt=True)
            if len(r["ids"]) <= self.context - MAX_ACTION_TOKENS or not keep:
                break
            keep = keep[1:]
        rows = np.vstack([sense_rows(zscore(body_vectors(self.chunks, self.senses), self.z, CLIP), keep).reshape(-1, 3),
                          row[None]]) if keep else row[None]
        with torch.no_grad():
            ids = r["ids"]
            t_ids = torch.tensor([max(i, 0) for i in ids], device=self.model.device)
            emb = self.model.get_input_embeddings()(t_ids)
            pos = torch.tensor([i for i, x in enumerate(ids) if x == -1], device=emb.device)
            soft = self.graft.proj(torch.as_tensor(rows, dtype=torch.float32, device=emb.device))
            emb = emb.index_put((pos,), soft.view(-1, self.graft.hidden).to(emb.dtype))
            for tries in range(1, EMPTY_TRIES + 1):    # an empty action cannot be read: sample again, counted
                out = self.model.generate(inputs_embeds=emb[None], attention_mask=torch.ones(
                    1, len(ids), device=emb.device, dtype=torch.long), max_new_tokens=MAX_ACTION_TOKENS,
                    do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=self.tok.eos_token_id)
                text = self.tok.decode(out[0], skip_special_tokens=True).strip()
                action = (text.splitlines() or [""])[0].strip()
                if action:
                    break
        self.record.append("sense_tries", "code", turn=t, tries=tries, empty=not action)
        if not action:                                 # cut at the last full turn, as an engine failure is
            from body.habitat import Unplayable
            raise Unplayable(f"{EMPTY_TRIES} empty actions in a row")
        return action


# --- commands -----------------------------------------------------------------------------------------------------

def _layers(n: int) -> list[int]:
    return list(range(math.ceil(2 * n / 3), n))


def _units(run: Path, reader, a, record) -> tuple[np.ndarray, np.ndarray, list[int]]:
    from substrate.echo_floor import d_resumable
    from substrate.measure_d import load_sets

    layers = _layers(reader.info["layers"])
    floor = {int(k): float(v) for k, v in json.loads(Path(a.floor).read_text()).items()}
    missing = [l for l in layers if l not in floor]
    if missing:
        raise SystemExit(f"the floor file has no floor for layers {missing}")
    D = d_resumable(reader, load_sets(a.sets), layers, record, a.draft, True)
    unit = np.array([floor[l] * float(D[j]) for j, l in enumerate(layers)])
    fv = np.array([float(D[j]) ** 2 / reader.info["width"] for j in range(len(layers))])
    (run / "units.json").write_text(json.dumps({"layers": layers, "unit": unit.tolist(), "floor_var": fv.tolist()}))
    return unit, fv, layers


def cmd_play(a) -> None:
    from body.habitat import Agent, play_or_skip
    from body.step0_habitat import STAGE2

    run = Path(a.run)
    c = config(run, a)
    record = Record(run / "record.jsonl")
    done = set(episodes(run))
    if record.path.exists():
        done |= {e["seed"] for e in read(record.path) if e["kind"] == "skip"}
    round_ = None if a.heldout else a.round
    if round_ is not None and not 0 <= round_ < c["rounds"]:
        raise SystemExit(f"rounds run 0 to {c['rounds'] - 1}")
    record_machine(record, f"babble play {'--heldout' if a.heldout else f'--round {round_}'}")
    if round_ in (None, 0):                              # the base model, no tokens: round 0 and the held-out text
        agent = Agent(a.model, device=a.device, dtype=a.dtype, forms=True)
    else:
        u = json.loads((run / "units.json").read_text())
        agent = GraftedAgent(a.model, run / "grafts" / f"live-r{round_ - 1}.pt", json.loads((run / "z.json").read_text()),
                             np.array(u["unit"]), np.array(u["floor_var"]), u["layers"], record,
                             device=a.device, dtype=a.dtype, seed=round_)
    record.append("round", "code", round=round_, heldout=bool(a.heldout), player=agent.load)
    for seed in seeds(c, round_):
        if seed not in done:
            play_or_skip(agent, record, seed, run / "games", worlds_path=STAGE2, kind=None)


def _capped(tok, text: str) -> str:
    return tok.decode(tok(text, add_special_tokens=False)["input_ids"][:CAP])


def cmd_read(a) -> None:
    """Every episode's senses and states, read by the base model alone, with the tested functions themselves
    (body.stepm1.news, body.conditional.conditional); then, for grafted rounds, the senses the player was given
    against these (they should agree: reported in the record)."""
    from body.conditional import conditional
    from body.stepm1 import news
    from substrate.states import Reader

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    record_machine(record, "babble read")
    reader = Reader(a.model, device=a.device, dtype=a.dtype, sink="first")
    unit, fv, layers = _units(run, reader, a, record)
    tok = reader.tokenizer
    out = run / "senses"
    out.mkdir(exist_ok=True)
    started, n = time.time(), 0
    for seed, ep in sorted(episodes(run).items()):
        path = out / f"{seed}.npz"
        if path.exists() and len(np.load(path)["boundary"]) == len(ep["chunks"]):
            continue
        texts = [c["text"] for c in ep["chunks"]]
        slots, turns = [c["slot"] for c in ep["chunks"]], [c["turn"] for c in ep["chunks"]]
        full = reader.states(texts, layers)
        cap = full.copy()
        capped = [_capped(tok, t) for t in texts]
        diff = [i for i, (x, y) in enumerate(zip(texts, capped)) if x != y]
        if diff:
            cap[diff] = reader.states([capped[i] for i in diff], layers)
        boundary = np.nanmean(news(full, slots, unit), axis=1) if len(slots) > 1 else np.full(len(slots), np.nan)
        boundary[[i for i, s in enumerate(slots) if s == "self" or i == 0]] = np.nan
        xc = np.mean([conditional(cap[:, j].astype(float), slots, turns, float(fv[j]))["X_c"]
                      for j in range(len(layers))], axis=0)
        np.savez(path, boundary=boundary, X_c=xc, state=full.mean(axis=1).astype(np.float32))
        n += 1
    record.append("read_end", "code", step="babble", episodes=n, seconds=round(time.time() - started, 1),
                  live_against_read=live_check(run))


def _vectors(ep: dict, data) -> dict:
    senses = [{"boundary": b, "X_c": x} for b, x in zip(data["boundary"], data["X_c"])]
    return body_vectors(ep["chunks"], senses)


def live_check(run: Path) -> dict:
    """The largest difference between the body vector a grafted turn was given and the one read afterwards."""
    entries = read(run / "record.jsonl")
    live, seed = {}, None
    for e in entries:
        if e["kind"] == "episode":
            seed = e["seed"]
            live[seed] = []
        elif e["kind"] == "sense" and seed is not None:
            live[seed].append(e)
    eps, worst, n = episodes(run), {"boundary": 0.0, "X_c": 0.0}, 0
    for s, rows in live.items():
        p = run / "senses" / f"{s}.npz"
        if not rows or s not in eps or not p.exists():
            continue
        v = _vectors(eps[s], np.load(p))
        for r in rows:
            if r["turn"] == 0:
                continue
            b, x = v.get(r["turn"] - 1, (np.nan, np.nan))
            for name, mine, theirs in (("boundary", b, r["boundary"]), ("X_c", x, r["X_c"])):
                if theirs is not None and np.isfinite(mine):
                    worst[name] = max(worst[name], abs(mine - theirs))
            n += 1
    return {"turns": n, "max_abs_difference": worst}


def cmd_fit(a) -> None:
    """The z-scores of the two senses and the 64-d reply space, from round 0's episodes only."""
    run = Path(a.run)
    c = config(run)
    eps = episodes(run)
    r0 = [s for s in seeds(c, 0) if s in eps]
    vecs, replies = [], []
    for s in r0:
        data = np.load(run / "senses" / f"{s}.npz")
        vecs.append(_vectors(eps[s], data))
        replies += [data["state"][i] for i, ch in enumerate(eps[s]["chunks"])
                    if ch["slot"] == "world" and ch["turn"] >= 0 and ch.get("cause", "elicited") == "elicited"]
    z = fit_z(vecs)
    (run / "z.json").write_text(json.dumps(z))
    Y = np.vstack(replies).astype(np.float64)
    mu = Y.mean(0)
    _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    basis = Vt[:SPACE].T
    P = (Y - mu) @ basis
    np.savez(run / "space.npz", mean=mu, basis=basis, sd=P.std(0) + 1e-12)
    Record(run / "record.jsonl").append("fit", "code", episodes=len(r0), replies=len(Y), z=z, space=SPACE)


def episode_data(run: Path, ep: dict, z: dict, space) -> dict:
    """What training and scoring read for one episode: its turns, Z (turns x 3), each turn's reply target in the
    space (standardised), and each chunk's state in the space."""
    data = np.load(run / "senses" / f"{ep['info']['seed']}.npz")
    turns = turns_of(ep)
    v = _vectors(ep, data)
    Z = zscore({d["turn"]: v.get(d["turn"], (np.nan, np.nan)) for d in turns}, z, CLIP)
    P = ((data["state"].astype(np.float64) - space["mean"]) @ space["basis"]) / space["sd"]
    reply_of = {ch["turn"]: i for i, ch in enumerate(ep["chunks"])
                if ch["slot"] == "world" and ch["turn"] >= 0 and ch.get("cause", "elicited") == "elicited"}
    action_of = {ch["turn"]: i for i, ch in enumerate(ep["chunks"]) if ch["slot"] == "self"}
    T = np.vstack([P[reply_of[d["turn"]]] for d in turns]) if turns else np.zeros((0, SPACE))
    return {"turns": turns, "Z": Z, "target": T, "P": P, "reply_of": reply_of, "action_of": action_of,
            "opening": ep["chunks"][0]["text"]}


def _load(run: Path):
    space = dict(np.load(run / "space.npz"))
    return json.loads((run / "z.json").read_text()), space


def _model(a, train: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=a.device)
    model.config.use_cache = False
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
    return torch, tok, model


def _losses(torch, graft, model, win: dict, d: dict, rows, mode: str):
    """Self-prediction, forward-model and language-modelling losses for one window."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    h = graft.hidden_states(win["ids"], rows, mode)
    info = win["turns"]
    idx = [d["index"][x["turn"]] for x in info]
    Z = torch.as_tensor(d["Z"][idx], dtype=torch.float32, device=h.device)
    S = graft.self_head(h[[x["sense"][-1] for x in info]].float())
    err = (S[:, 0] - Z[:, 0]) ** 2 + Z[:, 2] * (S[:, 1] - Z[:, 1]) ** 2
    l_self = err.sum() / (len(info) + Z[:, 2].sum())
    T = torch.as_tensor(d["target"][idx], dtype=torch.float32, device=h.device)
    l_fm = F.mse_loss(graft.fm_head(h[[x["last"] for x in info]].float()), T)
    ids = torch.tensor([max(i, 0) for i in win["ids"]], device=h.device)
    pos = torch.cat([torch.arange(x["lm"][0], x["lm"][1], device=h.device) for x in info])

    def ce(hh, yy):
        return F.cross_entropy(model.lm_head(hh).float(), yy, reduction="sum")

    l_lm = sum(checkpoint(ce, h[p - 1], ids[p], use_reentrant=False) for p in pos.split(256)) / len(pos)
    return l_self, l_fm, l_lm


def cmd_train(a) -> None:
    run = Path(a.run)
    c = config(run)
    z, space = _load(run)
    torch, tok, model = _model(a, train=True)
    from body.habitat import SYSTEM_FORMS

    graft = Graft(model)
    eps = episodes(run)
    pool = [s for r in range(a.round + 1) for s in seeds(c, r) if s in eps]
    order = list(pool)
    random.Random(7919 * (a.round + 1)).shuffle(order)          # the same order for both arms
    (run / "grafts").mkdir(exist_ok=True)
    record = Record(run / "record.jsonl")
    record_machine(record, f"babble train --round {a.round}")
    for arm in ("live", "zeroed"):
        out = run / "grafts" / f"{arm}-r{a.round}.pt"
        if out.exists():
            continue
        if a.round > 0:
            graft.load(run / "grafts" / f"{arm}-r{a.round - 1}.pt")
        else:
            graft.fresh(seed=0)                                   # both arms start from the same graft
        opt = graft.optimizer()
        sums, n, started = {"self": 0.0, "fm": 0.0, "lm": 0.0}, 0, time.time()
        for s in order:
            d = episode_data(run, eps[s], z, space)
            d["index"] = {x["turn"]: i for i, x in enumerate(d["turns"])}
            for win in windows(tok, SYSTEM_FORMS, d["opening"], d["turns"]):
                rows = sense_rows(d["Z"], [d["turns"][d["index"][x["turn"]]] for x in win["turns"]])
                l_self, l_fm, l_lm = _losses(torch, graft, model, win, d, rows, "live" if arm == "live" else "zeroed")
                loss = l_self + l_fm + LM_WEIGHT * l_lm
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(graft.params(), 1.0)
                opt.step()
                for k_, v in (("self", l_self), ("fm", l_fm), ("lm", l_lm)):
                    sums[k_] += float(v.detach())
                n += 1
        graft.save(out)
        record.append("trained", "code", arm=arm, round=a.round, episodes=len(order), windows=n,
                      mean_loss={k_: v / max(n, 1) for k_, v in sums.items()}, seconds=round(time.time() - started, 1))


# --- scoring -------------------------------------------------------------------------------------------------------

def _ridge_fit(X, Y):
    from body.conditional import _ridge

    W, mu, sd = _ridge(np.asarray(X, float), np.asarray(Y, float), LAMBDA)
    return W, mu, np.where(sd < 1e-9, 1.0, sd)          # a constant column: its weight is 0, its scale 1


def _ridge_predict(fit, X):
    W, mu, sd = fit
    X = np.atleast_2d(np.asarray(X, float))
    return np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ W


def history_features(d: dict) -> np.ndarray:
    """Per turn: the mean space position of the HISTORY chunks before the action, and the action's own."""
    rows = []
    for x in d["turns"]:
        i = d["action_of"][x["turn"]]
        rows.append(np.concatenate([d["P"][max(0, i - HISTORY): i].mean(0), d["P"][i]]))
    return np.vstack(rows)


def self_error(pred: np.ndarray, Z_next: np.ndarray) -> np.ndarray:
    return (pred[:, 0] - Z_next[:, 0]) ** 2 + Z_next[:, 2] * (pred[:, 1] - Z_next[:, 1]) ** 2


def ratio(num: dict, den: dict, draws: int = DRAWS, seed: int = 0, den2: dict | None = None) -> dict:
    """sum(num) / sum(den) over episodes, with a 90% interval from resampling episodes; with den2, the denominator
    is the lower of den and den2 in each draw."""
    eps = sorted(set(num) & set(den))
    def one(pick):
        a = sum(num[e] for e in pick)
        b = sum(den[e] for e in pick)
        if den2 is not None:
            b = min(b, sum(den2[e] for e in pick))
        return a / b if b > 0 else np.nan
    point = one(eps)
    rng = np.random.default_rng(seed)
    vals = [one([eps[i] for i in rng.integers(0, len(eps), len(eps))]) for _ in range(draws)]
    lo, hi = np.nanpercentile(vals, [5, 95])
    return {"ratio": float(point), "interval_90": [float(lo), float(hi)], "episodes": len(eps)}


def cmd_score(a) -> None:
    from body import lora
    from body.habitat import SYSTEM_FORMS

    run = Path(a.run)
    c = config(run)
    z, space = _load(run)
    torch, tok, model = _model(a, train=False)
    graft = Graft(model)
    eps = episodes(run)
    last = c["rounds"] - 1
    held = [s for s in seeds(c, None) if s in eps]
    train = [s for r in range(c["rounds"]) for s in seeds(c, r) if s in eps]
    D = {s: episode_data(run, eps[s], z, space) for s in held + train}
    for s in D:
        D[s]["index"] = {x["turn"]: i for i, x in enumerate(D[s]["turns"])}
    # the swap: each turn's row from the next held-out episode (by seed, cyclic) that has that turn
    partner = {}
    for j, s in enumerate(held):
        for x in D[s]["turns"][1:]:
            for k_ in range(1, len(held)):
                o = held[(j + k_) % len(held)]
                if x["turn"] in D[o]["index"]:
                    partner[(s, x["turn"])] = o
                    break

    def run_graft(mode: str, swap: bool = False) -> dict:
        """seed -> {"self": turns x 2 predictions, "fm": turns x SPACE}, teacher-forced."""
        out = {}
        with torch.no_grad():
            for s in held:
                d = D[s]
                S = np.full((len(d["turns"]), 2), np.nan)
                Fm = np.full((len(d["turns"]), SPACE), np.nan)
                for win in windows(tok, SYSTEM_FORMS, d["opening"], d["turns"]):
                    ts = [x["turn"] for x in win["turns"]]
                    if swap:
                        rows = np.array([D[partner[(s, t)]]["Z"][D[partner[(s, t)]]["index"][t] - 1]
                                         if (s, t) in partner else (d["Z"][d["index"][t] - 1] if t > 0 else np.zeros(3))
                                         for t in ts])
                    else:
                        rows = sense_rows(d["Z"], [d["turns"][d["index"][t]] for t in ts])
                    h = graft.hidden_states(win["ids"], rows, mode)
                    idx = [d["index"][t] for t in ts]
                    S[idx] = graft.self_head(h[[x["sense"][-1] for x in win["turns"]]].float()).cpu().numpy()
                    Fm[idx] = graft.fm_head(h[[x["last"] for x in win["turns"]]].float()).cpu().numpy()
                out[s] = {"self": S, "fm": Fm}
        return out

    graft.load(run / "grafts" / f"live-r{last}.pt")
    live, swapped = run_graft("live"), run_graft("live", swap=True)
    graft.load(run / "grafts" / f"zeroed-r{last}.pt")
    zeroed = run_graft("zeroed")

    # the base probe: the frozen model's hidden state at the action's last token (graft off, no sense places)
    def base_hidden(s):
        d = D[s]
        H = np.zeros((len(d["turns"]), model.config.hidden_size), dtype=np.float32)
        with torch.no_grad(), lora.off(graft.lora):
            for win in windows(tok, SYSTEM_FORMS, d["opening"], d["turns"], k=0):
                h = graft.hidden_states(win["ids"], None, "none")
                H[[d["index"][x["turn"]] for x in win["turns"]]] = h[[x["last"] for x in win["turns"]]].float().cpu().numpy()
        return H
    probe = _ridge_fit(np.vstack([base_hidden(s) for s in train]), np.vstack([D[s]["target"] for s in train]))
    hist = _ridge_fit(np.vstack([history_features(D[s]) for s in train]), np.vstack([D[s]["target"] for s in train]))
    Xa = np.vstack([sense_rows(D[s]["Z"], D[s]["turns"]) for s in train])
    Ya = np.vstack([D[s]["Z"] for s in train])
    alone_b = _ridge_fit(Xa, Ya[:, :1])
    ok = Ya[:, 2] == 1
    alone_x = _ridge_fit(Xa[ok], Ya[ok, 1:2])

    err = {k_: {} for k_ in ("live", "zeroed", "alone", "persistence", "live_swapturns", "swapped",
                             "fm", "linear", "probe")}
    by_sense = {k_: {"boundary": 0.0, "X_c": 0.0} for k_ in ("live", "zeroed", "alone")}
    for s in held:
        d = D[s]
        rows = sense_rows(d["Z"], d["turns"])
        alone = np.hstack([_ridge_predict(alone_b, rows), _ridge_predict(alone_x, rows)])
        for name, pred in (("live", live[s]["self"]), ("zeroed", zeroed[s]["self"]), ("alone", alone),
                           ("persistence", rows[:, :2])):
            e = self_error(pred, d["Z"])
            err[name][s] = float(e.sum())
            if name in by_sense:
                by_sense[name]["boundary"] += float(((pred[:, 0] - d["Z"][:, 0]) ** 2).sum())
                by_sense[name]["X_c"] += float((d["Z"][:, 2] * (pred[:, 1] - d["Z"][:, 1]) ** 2).sum())
        sw = [i for i, x in enumerate(d["turns"]) if (s, x["turn"]) in partner]
        err["live_swapturns"][s] = float(self_error(live[s]["self"][sw], d["Z"][sw]).sum())
        err["swapped"][s] = float(self_error(swapped[s]["self"][sw], d["Z"][sw]).sum())
        T = d["target"]
        err["fm"][s] = float(((live[s]["fm"] - T) ** 2).sum())
        err["linear"][s] = float(((_ridge_predict(hist, history_features(d)) - T) ** 2).sum())
        err["probe"][s] = float(((_ridge_predict(probe, base_hidden(s)) - T) ** 2).sum())

    result = {
        "rehearsal": c["rehearsal"], "deciding": "nothing" if c["rehearsal"] else "bars 1-3, once their ratios are fixed",
        "held_out_episodes": len(held), "training_episodes": len(train),
        "bar 1, live over the lower of zeroed and tokens alone": ratio(err["live"], err["zeroed"], den2=err["alone"]),
        "bar 2, swapped over live (turns with a partner)": ratio(err["swapped"], err["live_swapturns"]),
        "bar 3, forward model over the linear history predictor": ratio(err["fm"], err["linear"]),
        "proposed ratios, not yet fixed": PROPOSED,
        "reported": {
            "live over zeroed": ratio(err["live"], err["zeroed"]),
            "live over tokens alone": ratio(err["live"], err["alone"]),
            "live over persistence": ratio(err["live"], err["persistence"]),
            "forward model over the base probe": ratio(err["fm"], err["probe"]),
            "base probe over the linear predictor": ratio(err["probe"], err["linear"]),
            "self-prediction error by sense (sums)": by_sense,
            "not built": ["the yoked arm (degenerate as drafted)", "the swap within an episode (±3 turns)",
                          "quest completion played live", "phantom-limb and rubber-hand diagnostics"],
        },
    }
    Record(run / "record.jsonl").append("score", "code", **result)
    print(json.dumps(result, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("play", "read", "fit", "train", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name != "fit":
            s.add_argument("--model", required=True)
            s.add_argument("--device", default="cuda")
            s.add_argument("--dtype", default="bfloat16")
        if name in ("play", "train"):
            s.add_argument("--round", type=int, default=0)
        if name == "play":
            s.add_argument("--heldout", action="store_true", help="the base model plays the held-out pool")
            s.add_argument("--rehearsal", action="store_true", help="mirror seeds (+900000): decides nothing")
            s.add_argument("--per-round", type=int, default=PER_ROUND)
            s.add_argument("--heldout-size", type=int, default=HELDOUT[1] - HELDOUT[0] + 1)
        if name == "read":
            s.add_argument("--floor", default="material/echo_floor_8b.json")
            s.add_argument("--sets", default="material/paraphrases.jsonl")
            s.add_argument("--draft", action="store_true")
    a = p.parse_args(argv)
    {"play": cmd_play, "read": cmd_read, "fit": cmd_fit, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
