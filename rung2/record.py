"""The run's log: every model call in full, appended as one JSON line, before anything uses its reply."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class TooLong(Exception):
    """A prompt and its reply would not fit the window: the arm is over budget, which the test forbids."""


class Recorder:
    def __init__(self, model, log_path: Path, window: int = 8192):
        self.model = model
        self.path = Path(log_path)
        self.window = window
        self._counts: dict[str, int] = {}

    def event(self, kind: str, **data) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **data}, ensure_ascii=False) + "\n")

    def tokens(self, text: str) -> int:
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self._counts:
            self._counts[key] = self.model.tokens(text) if text else 0
        return self._counts[key]

    def chat(self, role: str, prompt: str, *, ref: str = "", schema: dict | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, seed: int = 0) -> str:
        n = self.tokens(prompt)
        if n + max_tokens > self.window:
            self.event("too_long", role=role, ref=ref, prompt_tokens=n, max_tokens=max_tokens)
            raise TooLong(f"{role} {ref}: {n} + {max_tokens} tokens exceed the {self.window}-token window")
        t0 = time.time()
        reply = self.model.chat(prompt, schema=schema, max_tokens=max_tokens, temperature=temperature, seed=seed)
        self.event("call", role=role, ref=ref, model=self.model.name, seed=seed, temperature=temperature,
                   prompt_tokens=n, seconds=round(time.time() - t0, 3), prompt=prompt, reply=reply)
        return reply
