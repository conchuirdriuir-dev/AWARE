"""The conversation: the person and the model alone (shadow mode, SPEC.md "Two modes").

The person types; a line starting /r marks a revision, which code strips and writes to the key only. The model gets
the conversation and nothing else: no aims, no codes, no request to report on itself.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import record

SYSTEM = "You are a helpful assistant."
REVISION = "/r"
QUIT = {"/q", "/quit"}
MAX_REPLY = 1024


def shown(rec, messages: list[dict], window: int) -> list[dict]:
    """The conversation that fits the window: the system message, then the latest turns, oldest dropped first."""
    keep = list(messages)
    while len(keep) > 2 and rec.message_tokens(keep) + MAX_REPLY > window:
        del keep[1:3 if len(keep) > 3 else 2]
    return keep


def talk(session: Path, rec, cfg: dict, read=input, write=print) -> None:
    session.mkdir(parents=True, exist_ok=True)
    if (session / "log.jsonl").exists():
        raise SystemExit(f"{session} already holds a session; each session gets its own directory")
    cfg = {**cfg, "system": SYSTEM, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "model": rec.model.name}
    (session / "session.json").write_text(json.dumps(cfg, indent=2))
    record.append(session / "log.jsonl", "start", **cfg)
    messages = [{"role": "system", "content": SYSTEM}]
    n = 0
    write(f"Session {session.name}. Start a line with {REVISION} when you change what you want; /q ends.")
    while True:
        try:
            line = read("you> ")
        except EOFError:
            break
        if line.strip() in QUIT:
            break
        text = line.strip()
        revised = text.startswith(REVISION)
        if revised:
            text = text[len(REVISION):].strip()
        if not text:
            continue
        n += 1
        record.append(session / "log.jsonl", "turn", id=f"P{n}", role="person", text=text)
        if revised:
            record.append(session / "key.jsonl", "mark", turn=f"P{n}", mark="revision", when="live", by="person")
        messages.append({"role": "user", "content": text})
        view = shown(rec, messages, cfg["window"])
        reply = rec.chat("model", view, ref=f"M{n}", max_tokens=MAX_REPLY, temperature=cfg["temperature"],
                         seed=cfg["seed"] + n).strip()
        dropped = (len(messages) - len(view)) // 2
        record.append(session / "log.jsonl", "turn", id=f"M{n}", role="model", text=reply,
                      shown_from=f"P{dropped + 1}")
        messages.append({"role": "assistant", "content": reply})
        write(f"\nmodel> {reply}\n")
    record.append(session / "log.jsonl", "end", turns=n)
