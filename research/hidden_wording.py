"""Step 2's second wording (PLAN.md, "Hidden sources", "A second wording"): the same recorded replies, another question.

    python -m research.hidden_wording --run runs/hidden-wording
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research import hidden as h
from research.d_jump import MODEL
from substrate.record import Record, read

STEP2 = "runs/hidden-qwen8b/record.jsonl"
WORDING = ("Was your last reply shaped by your own reasoning, or by something you were given? Reply with the letter only.",
           {"own": "my own reasoning", "steered": "something I was given"})


def recorded(path: str | Path) -> dict:
    """Step 2's replies by conversation and condition, from its record."""
    out: dict = {}
    for r in read(path):
        if r["kind"] == "generated" and not r.get("calibration") and r["condition"] in ("G", "V"):
            out.setdefault(r["conversation"], {})[r["condition"]] = r["reply"]
    return out


def carrying(path: str | Path) -> dict:
    return {r["conversation"]: {c: r[c]["carries"] for c in ("G", "V", "P")} for r in read(path) if r["kind"] == "read"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--step2", default=STEP2)
    a = p.parse_args(argv)
    replies, carries = recorded(a.step2), carrying(a.step2)
    dirs = h.load(h.DIRECTIONS)
    goals = {g["id"]: g["goal"][0] for g in h.load(h.GOALS)}
    from substrate.states import Reader

    record = Record(Path(a.run) / "record.jsonl")
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    bed = h.Hidden(reader)
    record.append("run", "code", **reader.info, step2=a.step2, question=WORDING[0], choices=WORDING[1])
    done = {r["conversation"]: r for r in read(record.path) if r["kind"] == "read"}
    rows = []
    for i, d in enumerate(dirs):
        if d["id"] in done:
            rows.append({"id": d["id"], **{c: done[d["id"]][c] for c in ("G", "V", "P")},
                         **({"reread": done[d["id"]]["reread"]} if "reread" in done[d["id"]] else {})})
            continue
        start, goal, rep = time.time(), goals[d["id"]], replies[d["id"]]
        ctx = {"G": (h.goal_messages(goal), rep["G"]), "V": (h.steered_messages(goal, d["direction"]), rep["V"]),
               "P": (h.goal_messages(goal), rep["V"])}
        row = {"id": d["id"]}
        for c, (msgs, text) in ctx.items():
            row[c] = {**bed.account(msgs, text, WORDING), "carries": carries[d["id"]][c], "logprob": 0.0}
        if i == 0:
            row["reread"] = [bed.account(msgs, text, WORDING)["score"] for msgs, text in ctx.values()]
        record.append("read", "code", conversation=d["id"], **{k: v for k, v in row.items() if k != "id"})
        rows.append(row)
        print(f"{d['id']}: {time.time() - start:.0f}s", flush=True)
    checks = h.construction(rows, injected=False)
    record.append("construction", "code", checks=checks, holds=h.holds(checks))
    print(json.dumps(checks, indent=1))
    if not h.holds(checks):
        print("the construction does not hold: nothing below is read")
        return
    result = h.compare(rows, "P")
    vp = h.entering(rows, "V")
    reported = {"P against V": h.auc([r["P"]["score"] for r in vp], [r["V"]["score"] for r in vp]),
                "greedy steered": {c: sum(r[c]["greedy"].count("steered") for r in rows) / (2 * len(rows)) for c in "GVP"}}
    record.append("verdict", "code", result=result, reported=reported)
    print(json.dumps({"P": result, **reported}, indent=1))


if __name__ == "__main__":
    main()
