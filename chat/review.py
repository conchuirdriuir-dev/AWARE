"""The reviewer (SPEC.md, "The reviewer"): a second call per turn, offline, in its own context. It sees only what code
gives it (the aim stack as quotes, the turn before, the turn), answers fixed questions, and quotes what it relies on;
code checks every quote. Then code builds the aim stack from its codes and adds similarities from the embedding model.
Nothing here reads the key."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from rung2.record import Recorder

from . import aims as aimstack
from . import record

FUNCTIONS = ["task", "feedback", "repair", "revision", "aside"]
RELATIONS = ["same", "narrower", "broader", "new", "back"]
SERVES = ["yes", "partly", "no"]
SHOW_CHARS = 1500                       # a long turn is shown cut, marked as cut; quotes are checked against all of it

PERSON_SCHEMA = {"type": "object", "properties": {
    "function": {"type": "string", "enum": FUNCTIONS}, "relation": {"type": "string", "enum": RELATIONS},
    "back_to": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "quote": {"type": "string"}},
    "required": ["function", "relation", "back_to", "quote"]}
MODEL_SCHEMA = {"type": "object", "properties": {
    "addresses": {"type": "string"}, "serves": {"type": "string", "enum": SERVES}, "quote": {"type": "string"}},
    "required": ["addresses", "serves", "quote"]}

HEAD = ("You are reviewing a conversation between a person and an assistant, one turn at a time. You see what the "
        "person wants so far, in their own words, and one new turn. Answer only about that turn.\n\n")
PERSON_Q = """Classify the person's new message.

function:
- task: asks, requests or instructs something; it sets or refines what they want
- feedback: acknowledges or reacts without asking for anything ("ok", "go on", "interesting")
- repair: says the assistant misunderstood an earlier message, and corrects it ("no, I meant ...")
- revision: the person changes what they want, without saying the assistant misunderstood
- aside: anything else

relation, to the most specific aim (the last one listed as active):
- same: the same thing, refined or restated
- narrower: a part or detail of it
- broader: the larger goal it serves
- new: something unrelated to the active aims
- back: returns to an earlier aim; give that aim's id in back_to (otherwise back_to is null)
If there are no aims yet, use "new".

quote: the exact words from the new message your answer rests on."""
MODEL_Q = """Judge the assistant's reply.

addresses: the id of the aim the reply addresses (one of the ids listed), or "none".
serves:
- yes: it does what that aim asks
- partly: it does some of it, or does something else as well
- no: it does something other than what the aim asks
quote: the exact words from the reply your answer rests on."""


def _show(text: str) -> str:
    return text if len(text) <= SHOW_CHARS else text[:SHOW_CHARS] + " [... cut here for length]"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’`*_]", "", text)).strip().lower()


def quote_ok(quote: str, text: str) -> bool:
    q = norm(quote)
    return bool(q) and q in norm(text)


def aims_block(stack: list[str], aims: list[dict]) -> str:
    if not aims:
        return "What the person wants so far: nothing yet.\n"
    by_id = {a["id"]: a for a in aims}
    out = ["Active aims, most general first, most specific last:"]
    out += [f"{i} (from {by_id[i]['from']}): " + " / ".join(f'"{_show(q)}"' for q in by_id[i]["quotes"])
            for i in stack] or ["(none active)"]
    earlier = [a for a in aims if a["id"] not in stack]
    if earlier:
        out.append("Earlier aims, no longer active:")
        out += [f"{a['id']} (from {a['from']}): " + " / ".join(f'"{_show(q)}"' for q in a["quotes"]) for a in earlier]
    return "\n".join(out) + "\n"


def person_prompt(stack, aims, before: dict | None, turn: dict) -> str:
    prev = f"The assistant's last reply ({before['id']}):\n\"{_show(before['text'])}\"\n\n" if before else ""
    return (HEAD + aims_block(stack, aims) + "\n" + prev +
            f"The person's new message ({turn['id']}):\n\"{_show(turn['text'])}\"\n\n" + PERSON_Q)


def model_prompt(stack, aims, before: dict | None, turn: dict) -> str:
    prev = f"The person's last message ({before['id']}):\n\"{_show(before['text'])}\"\n\n" if before else ""
    return (HEAD + aims_block(stack, aims) + "\n" + prev +
            f"The assistant's reply ({turn['id']}):\n\"{_show(turn['text'])}\"\n\n" + MODEL_Q)


def parse(reply: str) -> dict:
    try:
        out = json.loads(reply)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return round(dot / (na * nb), 4) if na and nb else 0.0


def review(session: Path, model, name: str, cfg: dict) -> Path:
    """One review of a session, written to reviews/NAME/log.jsonl: the codes, the stack after each turn, and each
    turn's similarity to the aim it is measured against."""
    out = session / "reviews" / name
    out.mkdir(parents=True, exist_ok=False)
    log = out / "log.jsonl"
    rec = Recorder(model, log, cfg.get("window", 8192))
    record.append(log, "review_start", said_by="code", **cfg, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    ts = record.turns(session)
    prefix = cfg.get("embed_prefix", "")
    vectors = dict(zip([t["id"] for t in ts], model.embed([prefix + t["text"] for t in ts]))) if ts else {}
    (out / "vectors.json").write_text(json.dumps(vectors))
    stack, aims, before = [], [], None
    for t in ts:
        if t["role"] == "person":
            raw = rec.chat("reviewer", person_prompt(stack, aims, before, t), ref=t["id"], schema=PERSON_SCHEMA,
                           max_tokens=300, temperature=0.0, seed=cfg.get("seed", 0))
            code = parse(raw)
            top = next((a for a in aims if stack and a["id"] == stack[-1]), None)
            measured = top["from"] if top else None
            new_stack = aimstack.apply(stack, aims, t, code)
            record.append(log, "code", said_by="reviewer", turn=t["id"], role="person", code=code,
                          quote_checked=quote_ok(code.get("quote", ""), t["text"]),
                          sim_to_aim=cosine(vectors[t["id"]], vectors[measured]) if measured else None,
                          measured_against=measured)
            record.append(log, "stack", said_by="code", after=t["id"], stack=new_stack,
                          aims=[dict(a, quotes=list(a["quotes"])) for a in aims])
            stack = new_stack
        else:
            raw = rec.chat("reviewer", model_prompt(stack, aims, before, t), ref=t["id"], schema=MODEL_SCHEMA,
                           max_tokens=300, temperature=0.0, seed=cfg.get("seed", 0))
            code = parse(raw)
            target = code.get("addresses") if code.get("addresses") in [a["id"] for a in aims] else None
            aim = next((a for a in aims if a["id"] == (target or (stack[-1] if stack else None))), None)
            record.append(log, "code", said_by="reviewer", turn=t["id"], role="model", code=code,
                          quote_checked=quote_ok(code.get("quote", ""), t["text"]),
                          addresses_checked=target is not None or code.get("addresses") == "none",
                          sim_to_aim=cosine(vectors[t["id"]], vectors[aim["from"]]) if aim else None,
                          measured_against=aim["from"] if aim else None,
                          sim_to_last=cosine(vectors[t["id"]], vectors[before["id"]]) if before else None)
        before = t
    record.append(log, "review_end", said_by="code", turns=len(ts))
    return log
