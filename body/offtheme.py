"""Step −1C's off-structure theme: TextWorld room grammar with new templates and only `house`'s words.

The owner's choice (2026-11-05, proposals/staged-schedule.md §4a): an off-structure room differs from an on-structure
one in its sentence templates only. Topic is held by using no word that `house_room.twg` does not already use. The theme
(`material/text_grammars/offhouse_room.twg`) was written once by the agent, before any reading, and is checked here in
code:

1. Same words: every word of its template text occurs in `house_room.twg`'s template text.
2. Other templates: no word 4-gram of its template text occurs in `house_room.twg`'s, placeholders counted as tokens
   and n-grams made only of placeholders ignored.
3. Usable: it defines every symbol `basic_room.twg` defines (the set TextWorld's room description needs).

Template text is a grammar line's right-hand side with Inform7 code (`[...]`) removed; `#symbol#` and `(slot)` become
placeholder tokens. Nothing here reads a state or a reply.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

THEME = "offhouse"
THEME_DIR = Path(__file__).resolve().parent.parent / "material" / "text_grammars"
N = 4

_SYMBOL = re.compile(r"#[^#\s]+#")
_SLOT = re.compile(r"\([^()\s]+\)")
_I7 = re.compile(r"\[[^\]]*\]")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def textworld_grammar_dir() -> Path:
    import textworld.generator.data as data
    return Path(data.TEXT_GRAMMARS_PATH)


def rules(path: os.PathLike) -> dict[str, list[str]]:
    """Symbol -> its alternatives, from a .twg file (comments and blank lines skipped)."""
    out: dict[str, list[str]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        sym, rhs = line.split(":", 1)
        out.setdefault(sym.strip(), []).extend(rhs.split(";"))
    return out


def tokens(alt: str) -> list[str]:
    """One alternative's template text as tokens: words lowercased, placeholders as '<s>' and '<x>'."""
    alt = _I7.sub(" ", alt)
    alt = _SYMBOL.sub(" <s> ", alt)
    alt = _SLOT.sub(" <x> ", alt)
    out = []
    for piece in alt.lower().split():
        if piece in ("<s>", "<x>"):
            out.append(piece)
        else:
            out.extend(_WORD.findall(piece))
    return out


def _texts(r: dict[str, list[str]], skip=("i7_", "inform7", "ordinary_adj")) -> list[list[str]]:
    return [tokens(a) for s, alts in r.items() if not s.startswith(skip) for a in alts]


def words(r) -> set[str]:
    return {t for ts in _texts(r) for t in ts if not t.startswith("<")}


def ngrams(r, n: int = N) -> set[tuple[str, ...]]:
    out = set()
    for ts in _texts(r):
        for i in range(len(ts) - n + 1):
            g = tuple(ts[i:i + n])
            if not all(t.startswith("<") for t in g):
                out.add(g)
    return out


def check(theme_path: os.PathLike | None = None) -> dict:
    """The three checks of the module docstring; `ok` is True only if all hold."""
    house = rules(textworld_grammar_dir() / "house_room.twg")
    basic = rules(textworld_grammar_dir() / "basic_room.twg")
    theme = rules(theme_path or THEME_DIR / f"{THEME}_room.twg")
    new_words = sorted(words(theme) - words(house))
    shared = sorted(ngrams(theme) & ngrams(house))
    missing = sorted(set(basic) - set(theme))
    return {"words_not_in_house": new_words, "shared_4grams": [" ".join(g) for g in shared],
            "missing_symbols": missing, "theme_words": len(words(theme)), "house_words": len(words(house)),
            "ok": not new_words and not shared and not missing}


def grammar(rng=None):
    """The theme as a TextWorld Grammar (its own directory as the knowledge base's grammar path)."""
    from textworld.generator.data import KnowledgeBase
    from textworld.generator.text_grammar import Grammar
    kb = KnowledgeBase.load(grammar_path=str(THEME_DIR))
    return Grammar(options={"theme": THEME}, rng=rng, kb=kb)


TOLERANCE = 0.10
TRIES = 50
TOKENIZER = "Qwen/Qwen3-8B"     # every Qwen3 size shares it (checked: the 0.6B's vocabulary and ids are the same)
_tok = None


def qwen_tokens(text: str) -> int:
    """Tokens of `text` under the reader's tokenizer, no special tokens (as substrate/echo_floor.py counts them)."""
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    return len(_tok(text, add_special_tokens=False)["input_ids"])


def words_count(text: str) -> int:
    return len(text.split())


def length(desc: str, count=qwen_tokens) -> int:
    """A description's length before rendering: its text with Inform7 code removed, counted by `count` (the reader's
    tokens by default, the owner's bar of 2026-11-05). Both themes share the Inform7 snippets (object lists, open or
    closed), so these are left out of both sides alike; rendered token lengths are reported at scoring.

    Why tokens and not words (measured 2026-11-05 on 180 rooms of 30 generated worlds, no reading): of pairs matched
    within 10% in words, 18% fell outside 10% in tokens, the off-structure side longer twice as often as shorter.
    The match rate is the same either way (88%)."""
    return count(" ".join(_I7.sub(" ", desc).split()))


def describe(room, game, seed: int, count=qwen_tokens) -> dict:
    """The planted room's off-structure description, matched in length to its on-structure (house) one.

    Draws up to TRIES expansions of the theme, the k-th from RandomState(seed * TRIES + k), and keeps the first whose
    length is within TOLERANCE of the house description's (agent's choice of rule, 2026-11-05; domain-free: it reads
    lengths only). If none is, `desc` is None and the episode is counted as unmatched, never loosened."""
    import numpy as np
    from textworld.generator.text_generation import assign_description_to_room
    target = length(game.infos[room.id].desc, count)
    for k in range(TRIES):
        desc = assign_description_to_room(room, game, grammar(rng=np.random.RandomState(seed * TRIES + k)))
        n = length(desc, count)
        if abs(n - target) <= TOLERANCE * target:
            return {"desc": desc, "draw": k, "length": n, "house_length": target}
    return {"desc": None, "draw": None, "length": None, "house_length": target}


if __name__ == "__main__":
    import json
    print(json.dumps(check(), indent=1))
