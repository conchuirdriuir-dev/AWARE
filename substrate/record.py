"""The record: one append-only JSONL file per run. Every entry says who it comes from (the person, the model or the
code) and is never rewritten; anything else (states, scores, reports) is derived from it and can be rebuilt.

A context chunk also carries its provenance (the Entity tab: logged at the source): self, if the trajectory produced
it, or external. Provenance follows origin, not channel: the model's own output read back is still self. An external
chunk also carries its arrival cause (DECISIONS.md, from the biologist review): spontaneous, or elicited by the
entity's preceding action (a tool result, a probe's answer). Biology calls the elicited kind reafference: external
content whose arrival the entity caused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SOURCES = ("person", "model", "code")
PROVENANCE = ("self", "external")
CAUSES = ("spontaneous", "elicited")


class Record:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = len(read(self.path)) if self.path.exists() else 0

    def append(self, kind: str, by: str, **fields) -> dict:
        if by not in SOURCES:
            raise ValueError(f"an entry comes from one of {SOURCES}, not {by!r}")
        if "provenance" in fields and fields["provenance"] not in PROVENANCE:
            raise ValueError(f"provenance is one of {PROVENANCE}, not {fields['provenance']!r}")
        if "cause" in fields and fields["cause"] not in CAUSES:
            raise ValueError(f"arrival cause is one of {CAUSES}, not {fields['cause']!r}")
        entry = {"seq": self.seq, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "by": by, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.seq += 1
        return entry

    def chunk(self, text: str, by: str, provenance: str, cause: str | None = None, **fields) -> dict:
        """A piece of context, tagged at creation. Re-reading a chunk cites its seq rather than re-tagging it. An
        external chunk must say whether its arrival was spontaneous or elicited."""
        if provenance == "external" and cause is None:
            raise ValueError("an external chunk needs its arrival cause: spontaneous or elicited")
        if cause is not None:
            fields["cause"] = cause
        return self.append("chunk", by, text=text, provenance=provenance, **fields)


def read(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
