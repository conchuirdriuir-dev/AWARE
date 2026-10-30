"""The record: one append-only JSONL file per run. Every entry says who it comes from (the person, the model or the
code) and is never rewritten; anything else (states, scores, reports) is derived from it and can be rebuilt.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SOURCES = ("person", "model", "code")


class Record:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = len(read(self.path)) if self.path.exists() else 0

    def append(self, kind: str, by: str, **fields) -> dict:
        if by not in SOURCES:
            raise ValueError(f"an entry comes from one of {SOURCES}, not {by!r}")
        entry = {"seq": self.seq, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "by": by, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.seq += 1
        return entry


def read(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
