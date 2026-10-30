"""The person's marks, in review after a session (SPEC.md, "The key"). One key per turn. Nothing the meta level made
is shown: no codes, no aims, no offers, so they cannot shape the marks. Marks go to key.jsonl only."""
from __future__ import annotations

from pathlib import Path

from . import record

PERSON_KEYS = {"": None, "r": "repair"}
MODEL_KEYS = {"": None, "d": "drift"}


def mark(session: Path, read=input, write=print, again: bool = False) -> int:
    done = [m for m in record.marks(session) if m.get("when") == "review"]
    if done and not again:
        raise SystemExit(f"{session} has been marked; --again adds a second pass")
    live = {m["turn"] for m in record.marks(session) if m.get("mark") == "revision"}
    write("Read the session back. After each of your turns: r = you were correcting a misreading, Enter = not.\n"
          "After each model turn: d = it left what you asked without your moving it, Enter = not. q stops.\n")
    n = 0
    for t in record.turns(session):
        who = "you" if t["role"] == "person" else "model"
        flag = "  [you marked a revision here]" if t["id"] in live else ""
        write(f"\n{t['id']} {who}>{flag}\n{t['text']}\n")
        keys = PERSON_KEYS if t["role"] == "person" else MODEL_KEYS
        while True:
            k = read("r/Enter> " if t["role"] == "person" else "d/Enter> ").strip().lower()
            if k == "q":
                return n
            if k in keys:
                break
            write("  (not a key here)")
        if keys[k]:
            record.append(session / "key.jsonl", "mark", turn=t["id"], mark=keys[k], when="review", by="person")
            n += 1
    record.append(session / "key.jsonl", "marked", when="review", by="person", marks=n)
    return n
