"""Arm B's index: BM25 and embedding similarity over passages, merged by reciprocal rank fusion."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


class Index:
    def __init__(self, ids: list[str], chapters: list[int], texts: list[str], vectors: np.ndarray):
        self.ids, self.chapters, self.vectors = ids, chapters, vectors
        self.docs = [Counter(words(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - k + 0.5) / (k + 0.5)) for w, k in df.items()}

    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        q = words(query)
        return [sum(self.idf.get(w, 0) * d[w] * (k1 + 1) / (d[w] + k1 * (1 - b + b * L / self.avg))
                    for w in q if w in d) for d, L in zip(self.docs, self.lengths)]

    def rank(self, query: str, qvec: list[float], upto: int, k: int = 60) -> list[str]:
        """Passage ids up to chapter `upto`, best first."""
        allowed = [i for i, c in enumerate(self.chapters) if c <= upto]
        lex = self.bm25(query)
        sem = self.vectors @ np.asarray(qvec, dtype=float)
        score = Counter()
        for scores in (lex, sem):
            for r, i in enumerate(sorted(allowed, key=lambda i: -scores[i])):
                score[i] += 1 / (k + r + 1)
        return [self.ids[i] for i, _ in score.most_common()]

    def save(self, path: Path) -> None:
        np.save(path / "vectors.npy", self.vectors)
        (path / "vector_ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, path: Path, rows) -> "Index":
        ids = json.loads((path / "vector_ids.json").read_text())
        by_id = {r["id"]: r for r in rows}
        return cls(ids, [by_id[i]["chapter"] for i in ids], [by_id[i]["text"] for i in ids],
                   np.load(path / "vectors.npy"))


def build(rows, embed) -> Index:
    texts = [r["text"] for r in rows]
    vectors = []
    for i in range(0, len(texts), 16):
        vectors.extend(embed(texts[i:i + 16]))
    v = np.asarray(vectors, dtype=float)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return Index([r["id"] for r in rows], [r["chapter"] for r in rows], texts, v)
