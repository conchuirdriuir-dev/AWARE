"""Causal authorship (PLAN.md, research track): which source made a reply happen, by intervention, and whether the model's
own account agrees. On conversations drafted for it and replies Qwen itself generated; the logged sessions cannot key this.

    python -m research.authorship --run runs/authorship-dry --material material/authorship_dry.jsonl --dry
    python -m research.authorship --run runs/authorship-qwen8b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.d_jump import MODEL
from research.gate1_pairs import clean
from substrate.record import Record

MATERIAL = "material/authorship_conversations.jsonl"
CLOSING = "What should happen next? Reply in a few sentences."
FILLER = "This part of the conversation has been left out."
QUESTION = "Where did the direction of your last reply come from? Reply with the letter only."
OPTIONS = {"goal": "the person's goal", "own": "your own earlier reply", "partner": "what the partner said",
           "none": "none of these"}
SOURCES = ("goal", "own", "partner")
KINDS = ("explicit", "frame once", "frame three times", "no frame")
MAX_NEW = 80
CHECKS = ("read twice", "null", "strong", "removal", "wording", "redundancy")
BAR = 0.8


# The conversation, as turns that can be taken away.

def turns(conv: dict, wording: int, own: str) -> list[dict]:
    out = [{"name": "goal", "role": "user", "label": "Person: ", "text": conv["goal"][wording]},
           {"name": "own", "role": "assistant", "label": "", "text": own}]
    out.append({"name": "note", "role": "user", "label": "Note: ", "text": conv["inert"]})     # before the partner (PLAN.md)
    for i, m in enumerate(conv["partner"][wording]):
        out.append({"name": f"partner{i}", "role": "user", "label": "Partner: ", "text": m})
    out.append({"name": "closing", "role": "user", "label": "", "text": CLOSING})
    return out


def render(ts: list[dict]) -> list[dict]:
    return [{"role": t["role"], "content": t["label"] + t["text"]} for t in ts]


def without(ts: list[dict], source: str, fill) -> list[dict]:
    """The turns with a source taken away: deleted if `fill` is None, otherwise each text passed through `fill`. "partner"
    is the group of its copies, "partner1" one copy."""
    hit = (lambda t: t["name"].startswith("partner")) if source == "partner" else (lambda t: t["name"] == source)
    if fill is None:
        return [t for t in ts if not hit(t)]
    return [{**t, "text": fill(t["text"])} if hit(t) else t for t in ts]


def with_note(ts: list[dict], text: str) -> list[dict]:
    return [{**t, "text": text} if t["name"] == "note" else t for t in ts]


def filler(tok, n: int) -> str:
    """Neutral text of exactly n tokens, so a replacement shifts no position. Sentences are joined by a space, so a repeat
    tokenizes as the sentence does and a cut falls on a token's edge."""
    first = tok(FILLER, add_special_tokens=False)["input_ids"]
    again = tok(" " + FILLER, add_special_tokens=False)["input_ids"]
    ids = (first + again * (n // len(again) + 1))[:n]
    for _ in range(8):
        text = tok.decode(ids, skip_special_tokens=True)
        have = len(tok(text, add_special_tokens=False)["input_ids"])
        if have == n:
            return text
        ids = ids[:n - (have - n)] if have > n else ids + again[:n - have]
    raise ValueError(f"no filler of {n} tokens")


# The measure.

def tail(hf, context: list[int], reply: list[int]) -> np.ndarray:
    """Log-probabilities of the next token at each of the reply's tokens, with the reply held fixed."""
    import torch

    with torch.no_grad():
        ids = torch.tensor([context + reply], device=hf.device)
        logits = hf(ids, logits_to_keep=len(reply) + 1, use_cache=False).logits[0, :-1]
        return logits.float().log_softmax(-1).cpu().numpy()


def influence(full: np.ndarray, other: np.ndarray) -> float:
    """Mean over the reply's tokens of KL(full || other), in nats per token."""
    return float((np.exp(full) * (full - other)).sum(-1).mean())


def top(values: dict) -> str:
    return max(SOURCES, key=lambda s: values[s])


def key(values: dict, floor_: float) -> str:
    best = top(values)
    return best if values[best] > floor_ else "none"


# The instrument: all six, or nothing below is read.

def floor(rows: list[dict]) -> float:
    return float(np.percentile([r["alt1"] for r in rows], 95))


def instrument(rows: list[dict]) -> dict:
    F = floor(rows)
    kind = lambda k: [r for r in rows if r["kind"] == k]  # noqa: E731
    reread = [abs(r["reread"][s] - r["replaced"][s]) for r in rows if "reread" in r for s in SOURCES]
    null = float(np.mean([r["alt2"] > F for r in rows]))
    strong = float(np.mean([r["replaced"]["partner"] >= 5 * F for r in kind("explicit")]))
    removal = float(np.mean([top(r["deleted"]) == top(r["replaced"]) for r in rows]))
    wording = float(np.mean([top(r["second"]) == top(r["replaced"]) for r in rows]))
    group = float(np.mean([r["replaced"]["partner"] > F for r in kind("frame three times")]))
    return {"floor": F,
            "read twice": {"value": max(reread), "bar": 0.01 * F, "pass": max(reread) <= 0.01 * F},
            "null": {"value": null, "bar": 0.1, "pass": null <= 0.1},
            "strong": {"value": strong, "bar": 0.9, "pass": strong >= 0.9},
            "removal": {"value": removal, "bar": 0.8, "pass": removal >= 0.8},
            "wording": {"value": wording, "bar": 0.8, "pass": wording >= 0.8},
            "redundancy": {"value": group, "bar": 0.9, "pass": group >= 0.9}}


def holds(checks: dict) -> bool:
    return all(checks[c]["pass"] for c in CHECKS)


# The model's own account.

def options(seed_text: str) -> dict:
    """Letter to source, the order set by a seed from the conversation's id."""
    order = np.random.default_rng(sum(seed_text.encode())).permutation(list(OPTIONS)).tolist()
    return dict(zip("ABCD", order))


def question(letters: dict) -> str:
    return QUESTION + "\n" + "\n".join(f"{k}. {OPTIONS[v]}" for k, v in letters.items())


def parse(reply: str, letters: dict) -> str | None:
    head = reply.strip()[:1].upper()
    second = reply.strip()[1:2]
    return letters[head] if head in letters and not second.isalpha() else None


def account_verdict(agree: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    boots = rng.choice(agree, (n_boot, len(agree)), replace=True).mean(1)
    low, high = np.percentile(boots, [2.5, 97.5])
    verdict = "holds" if high < BAR else "model reports well" if low >= BAR else "undecided"
    return {"agreement": float(agree.mean()), "low": float(low), "high": float(high), "verdict": verdict}


# The run.

class Bed:
    def __init__(self, reader):
        if not getattr(reader.model, "dispatched", True):
            reader.model.dispatch()
        self.hf, self.tok = reader.model._model, reader.tokenizer

    def context(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def generate(self, messages: list[dict], max_new: int = MAX_NEW) -> list[int]:
        import torch

        ids = torch.tensor([self.context(messages)], device=self.hf.device)
        with torch.no_grad():
            out = self.hf.generate(ids, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        return out[0, ids.shape[1]:].tolist()

    def text(self, ids: list[int]) -> str:
        return clean(self.tok.decode(ids, skip_special_tokens=True))

    def fill(self, text: str) -> str:
        return filler(self.tok, len(self.tok(text, add_special_tokens=False)["input_ids"]))


def wording_pass(bed: Bed, conv: dict, w: int, record: Record) -> tuple[list[dict], list[int], np.ndarray, dict]:
    own_ids = bed.generate([{"role": "user", "content": "Person: " + conv["goal"][w]}])
    own = bed.text(own_ids)
    ts = turns(conv, w, own)
    reply = bed.generate(render(ts))
    record.append("generated", "model", conversation=conv["id"], wording=w, own=own, reply=bed.text(reply),
                  own_cut_off=len(own_ids) >= MAX_NEW, reply_cut_off=len(reply) >= MAX_NEW)
    full = tail(bed.hf, bed.context(render(ts)), reply)
    replaced = {}
    for s in SOURCES:
        other = render(without(ts, s, bed.fill))
        ctx = bed.context(other)
        if len(ctx) != len(bed.context(render(ts))):
            record.append("length moved", "code", conversation=conv["id"], source=s)
        replaced[s] = influence(full, tail(bed.hf, ctx, reply))
    return ts, reply, full, replaced


def read_conversation(bed: Bed, conv: dict, record: Record, reread: bool) -> dict:
    ts, reply, full, replaced = wording_pass(bed, conv, 0, record)
    I = lambda messages: influence(full, tail(bed.hf, bed.context(messages), reply))  # noqa: E731
    row = {"id": conv["id"], "kind": conv["kind"], "replaced": replaced,
           "deleted": {s: I(render(without(ts, s, None))) for s in SOURCES},
           "alt1": I(render(with_note(ts, conv["inert_alternates"][0]))),
           "alt2": I(render(with_note(ts, conv["inert_alternates"][1]))),
           "note_replaced": I(render(without(ts, "note", bed.fill))),
           "copies": [I(render(without(ts, f"partner{i}", bed.fill))) for i in range(len(conv["partner"][0]))],
           "reply": bed.text(reply), "own": next(t["text"] for t in ts if t["name"] == "own")}
    if reread:
        row["reread"] = {s: I(render(without(ts, s, bed.fill))) for s in SOURCES}
    row["second"] = wording_pass(bed, conv, 1, record)[3]
    record.append("read", "code", conversation=row["id"], conversation_kind=row["kind"],
                  **{k: v for k, v in row.items() if k not in ("reply", "own", "id", "kind")})
    return row


def ask_account(bed: Bed, conv: dict, row: dict, F: float, record: Record) -> bool:
    letters = options(conv["id"])
    messages = render(turns(conv, 0, row["own"])) + [{"role": "assistant", "content": row["reply"]},
                                                      {"role": "user", "content": question(letters)}]
    answer = bed.text(bed.generate(messages, max_new=8))
    said, truth = parse(answer, letters), key(row["replaced"], F)
    record.append("account", "model", conversation=conv["id"], conversation_kind=conv["kind"], letters=letters, answer=answer, said=said,
                  key=truth)
    return said == truth


def finished(path: str | Path) -> dict:
    """The conversations a killed run already read, rebuilt from its record, so a crash does not lose the night."""
    path, own, rows = Path(path), {}, {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["kind"] == "generated" and r["wording"] == 0:
            own[r["conversation"]] = {"own": r["own"], "reply": r["reply"]}
        elif r["kind"] == "read":
            rows[r["conversation"]] = {"id": r["conversation"], "kind": r["conversation_kind"],
                                       **{k: r[k] for k in ("replaced", "deleted", "alt1", "alt2", "note_replaced", "copies",
                                                            "second") }, **({"reread": r["reread"]} if "reread" in r else {}),
                                       **own[r["conversation"]]}
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--material", default=MATERIAL)
    p.add_argument("--dry", action="store_true", help="plumbing and time only: outside the 40, no verdict is read")
    a = p.parse_args(argv)
    from substrate.states import Reader


    convs = [json.loads(line) for line in Path(a.material).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not a.dry and not (len(convs) == 40 and all(c.get("frozen") for c in convs)):
        raise SystemExit("the conversations are not frozen; the owner freezes them first")
    record = Record(Path(a.run) / "record.jsonl")
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    bed = Bed(reader)
    record.append("run", "code", **reader.info, material=a.material, dry=a.dry, max_new_tokens=MAX_NEW, filler=FILLER,
                  closing=CLOSING, question=QUESTION, options=OPTIONS, thinking=False, sampling="greedy")
    done = finished(record.path)
    seen, rows = set(), []
    for conv in convs:
        if conv["id"] in done:
            rows.append(done[conv["id"]])
            seen.add(conv["kind"])
            continue
        start = time.time()
        rows.append(read_conversation(bed, conv, record, reread=conv["kind"] not in seen))
        seen.add(conv["kind"])
        print(f"{conv['id']} ({conv['kind']}): {time.time() - start:.0f}s", flush=True)
    if a.dry:
        for c, r in zip(convs, rows):               # the account's plumbing too; the floor here means nothing
            ask_account(bed, c, r, 0.0, record)
            print(json.dumps(r, indent=1, ensure_ascii=False))
        return
    checks = instrument(rows)
    record.append("instrument", "code", checks=checks, holds=holds(checks))
    print(json.dumps(checks, indent=1))
    F = checks["floor"]
    copies = [{"id": r["id"], "group": r["replaced"]["partner"], "copies": r["copies"]} for r in rows
              if r["kind"] == "frame three times"]
    keys = [key(r["replaced"], F) for r in rows]
    per_kind = {k: {s: sum(key(r["replaced"], F) == s for r in rows if r["kind"] == k) for s in (*SOURCES, "none")}
                for k in KINDS}
    margins = [sorted(r["replaced"].values())[-1] - sorted(r["replaced"].values())[-2] for r in rows]
    record.append("reported", "code", copies=copies, keys=keys, per_kind=per_kind, median_margin=float(np.median(margins)))
    print("key per kind:", json.dumps(per_kind), "\ncopies:", json.dumps(copies), "\nmedian margin:", np.median(margins))
    if not holds(checks):
        print("the instrument does not hold: the model's account is not asked")
        return
    agree = np.array([ask_account(bed, c, r, F, record) for c, r in zip(convs, rows)], dtype=float)
    verdict = account_verdict(agree)
    by_kind = {k: float(agree[[r["kind"] == k for r in rows]].mean()) for k in KINDS}
    best_constant = max(keys.count(s) for s in set(keys)) / len(keys)
    record.append("account verdict", "code", **verdict, by_kind=by_kind, best_constant_answer=best_constant)
    print(json.dumps({**verdict, "by_kind": by_kind, "best_constant_answer": best_constant}, indent=1))


if __name__ == "__main__":
    main()
