"""The story as chapters and passages, and fitting text to a token budget.

Budgets add the token counts of the items plus one per separator: each item is counted once (counts are cached), and
the sum is within a few tokens of counting the joined text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

Count = Callable[[str], int]


def load_chapters(story_dir: Path) -> list[tuple[int, str]]:
    """Chapter files named with their number (ch01.md, 02.txt, ...), in order."""
    chapters = []
    for p in Path(story_dir).iterdir():
        m = re.search(r"(\d+)", p.stem)
        if p.suffix in (".md", ".txt") and m:
            chapters.append((int(m.group(1)), p.read_text(encoding="utf-8").strip()))
    chapters.sort()
    if [n for n, _ in chapters] != list(range(1, len(chapters) + 1)):
        raise ValueError(f"chapters must be numbered 1 to {len(chapters)}: found {[n for n, _ in chapters]}")
    return chapters


HEADING = re.compile(r"^\s*(chapter|CHAPTER|Chapter)\s+([0-9IVXLC]+|[A-Za-z-]+)\b.*$", re.M)


def split_book(text: str, limit: int | None = None) -> list[str]:
    """A plain-text book (Project Gutenberg's header and footer removed) cut at its CHAPTER headings."""
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    body = text[start.end() if start else 0:end.start() if end else None]
    heads = list(HEADING.finditer(body))
    chapters = [body[h.end():(heads[i + 1].start() if i + 1 < len(heads) else None)].strip()
                for i, h in enumerate(heads)]
    chapters = [c for c in chapters if len(c.split()) > 200]          # a table of contents is not a chapter
    return chapters[:limit] if limit else chapters


def _sentences(paragraph: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s]


SCENE_BREAK = re.compile(r"^\s*(?:-{3,}|\*(?:\s*\*){2,}|#)\s*$")     # ---, * * *, #: a scene ends here


def chunk(chapter: int, text: str, count: Count, target: int = 400) -> list[dict]:
    """Passages of about `target` tokens, split at paragraphs, or at sentences inside a long paragraph. A scene break
    always ends a passage, so no passage spans two scenes; the break itself is not text."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if SCENE_BREAK.match(para):
            units.append(None)
        elif para:
            units.extend(_sentences(para) if count(para) > target else [para])
    groups, current, size = [], [], 0
    for u in units:
        if u is None:
            if current:
                groups.append(current)
            current, size = [], 0
            continue
        n = count(u) + 1
        if current and size + n > target:
            groups.append(current)
            current, size = [], 0
        current.append(u)
        size += n
    if current:
        groups.append(current)
    out = []
    for i, g in enumerate(groups, 1):
        t = "\n\n".join(g)
        out.append({"id": f"c{chapter:02d}-p{i:03d}", "chapter": chapter, "seq": i, "text": t, "tokens": count(t)})
    return out


def fit(items: list[str], budget: int, count: Count) -> list[str]:
    """The leading items that fit the budget."""
    out, size = [], 0
    for item in items:
        n = count(item) + 1
        if size + n > budget:
            break
        out.append(item)
        size += n
    return out


def fit_end(items: list[str], budget: int, count: Count) -> list[str]:
    """The trailing items that fit the budget, in their original order."""
    return list(reversed(fit(list(reversed(items)), budget, count)))
