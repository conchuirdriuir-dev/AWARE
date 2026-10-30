"""A session on disk: the conversation's log, the person's key, and reviews of it, each in its own file.

    DIR/session.json            how the session was run
    DIR/log.jsonl               every turn and model call, appended as it happens
    DIR/key.jsonl               the person's marks: read by the scorer only, never by the meta level
    DIR/reviews/NAME/log.jsonl  one review of the session: the reviewer's calls, its codes, the aim stack, similarities
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path: Path, kind: str, **data) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **data}, ensure_ascii=False) + "\n")


def turns(session: Path) -> list[dict]:
    """The conversation as it happened: person and model turns in order, nothing from the key."""
    return [e for e in read_jsonl(session / "log.jsonl") if e["kind"] == "turn"]


def marks(session: Path) -> list[dict]:
    """The person's marks. Only the scorer and the marking tool call this."""
    return read_jsonl(session / "key.jsonl")
