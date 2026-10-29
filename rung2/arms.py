"""The six arms (SPEC.md, "Arms"): each turns a question asked after chapter N into a view of at most VIEW tokens,
and the same answer prompt is used for every arm but 0.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import builder, store
from .retrieval import Index
from .text import fit, fit_end

VIEW = 6000
ARMS = "0ABCDE"

ANSWER = ("Below is material from a novel, as of the end of chapter {n}. Answer the question using only this "
          "material. If the material does not say, answer \"Not stated in the material.\"\n\n"
          "MATERIAL:\n{view}\n\nQUESTION: {question}\nANSWER (one or two sentences):")
CONTINUE = ("Below is material from a novel, as of the end of chapter {n}. Write the next scene, about 400 words, "
            "continuing from the end of chapter {n}. Stay consistent with everything in the material.\n\n"
            "MATERIAL:\n{view}\n\nQUESTION: {question}\nTHE NEXT SCENE:")
GUESS = ("Answer this question about a novel. You have not been given its text; give your best guess in one or two "
         "sentences.\n\nQUESTION: {question}\nANSWER:")


class Arms:
    def __init__(self, conn, run_dir: Path, count, embed, evidence: dict[str, list[str]] | None = None):
        self.conn, self.run, self.count, self.embed = conn, Path(run_dir), count, embed
        self.evidence = evidence or {}
        self._index = None
        self._summaries = None

    def _texts(self, n: int) -> list[str]:
        return [r["text"] for r in store.passages(self.conn, n)]

    def _recent(self, n: int, budget: int) -> list[str]:
        return fit_end(self._texts(n), budget, self.count)

    def view(self, arm: str, q: dict) -> str:
        n = q["after"]
        if arm == "0":
            return ""
        if arm == "A":
            return "\n\n".join(self._recent(n, VIEW))
        if arm == "B":
            if self._index is None:
                self._index = Index.load(self.run, store.passages(self.conn))
            recent = self._recent(n, 1000)
            by_id = {r["id"]: r for r in store.passages(self.conn, n)}
            ranked = [by_id[i] for i in self._index.rank(q["question"], self.embed([q["question"]])[0], n)
                      if i in by_id and by_id[i]["text"] not in recent]
            found = fit([f"[ch. {r['chapter']}] {r['text']}" for r in ranked], VIEW - 1000 - 50, self.count)
            return ("== PASSAGES MATCHING THE QUESTION ==\n" + "\n\n".join(found) +
                    f"\n\n== THE END OF CHAPTER {n} ==\n" + "\n\n".join(recent))
        if arm == "C":
            if self._summaries is None:
                self._summaries = json.loads((self.run / "summaries.json").read_text())
            summary = self._summaries[str(n)]
            room = VIEW - self.count(summary) - 50
            return (f"== SUMMARY OF CHAPTERS 1 TO {n} ==\n{summary}\n\n== THE END OF CHAPTER {n} ==\n" +
                    "\n\n".join(self._recent(n, room)))
        if arm == "D":
            return builder.build_view(self.conn, q["question"], n, self.count)
        if arm == "E":
            quotes = [store.norm(x) for x in self.evidence.get(q["id"], [])]
            hits = [r for r in store.passages(self.conn) if any(x and x in store.norm(r["text"]) for x in quotes)]
            return "\n\n".join(fit([f"[ch. {r['chapter']}] {r['text']}" for r in hits], VIEW, self.count))
        raise ValueError(f"unknown arm {arm!r}")

    def prompt(self, arm: str, q: dict, view: str) -> str:
        if arm == "0":
            return GUESS.format(question=q["question"])
        template = CONTINUE if q["type"] == 6 else ANSWER
        return template.format(n=q["after"], view=view, question=q["question"])
