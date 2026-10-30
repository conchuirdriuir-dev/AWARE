"""Cell symbols (docs/proposals/CELL_SYMBOLS_v0_1.md; KERNEL §5.3): a populated cell of a target's held rules, made a
symbol that later targets may cite.

A target's held rules split its candidates into cells, one per sign vector (kernel §4); the cell table can code any
Boolean combination of them for that target, but only there (kernel §3, limit 1). A cell symbol makes one combination
citable: it holds at a candidate tuple where each of its positive rules holds and each negative one does not. It is
proposed for every populated cell (CS-2 required a positive sign; it had no witness and was dropped, CS-10), defined
by the shortest set of its literals with the same extension on the training evidence, two shortest sets that tie both
kept, as rivals are (CS-3, CS-11), and priced by a code fixed given the rules already held (CS-5): which earlier
target, then a uniform choice among the 3^k - 1 non-empty literal sets. Nothing here reads a name.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

MAX_RULES_MINIMISED = 10      # beyond this many rules a cell keeps its full literal set (the subsets are too many)


def literal_bits(k: int, positive: bool = False) -> float:
    """log2 of the number of literal sets over k rules, each rule absent, positive or negative: any but the empty set
    (CS-10), or, with `positive`, those with at least one positive (CS-2, kept to reproduce its witness run)."""
    return math.log2(3 ** k - 2 ** k) if positive else math.log2(3 ** k - 1)


def structure_bits(sources: int, k: int, positive: bool = False) -> float:
    """S(Z): the source among `sources` targets formed so far, then the literal set over its k held rules (CS-5)."""
    return math.log2(max(sources, 1)) + literal_bits(k, positive)


def extension(cols: list[np.ndarray], literals) -> np.ndarray:
    """Where a definition holds: every positive literal's rule holds and no negative one's does."""
    out = np.ones(len(cols[0]), dtype=bool)
    for i, sign in literals:
        out &= cols[i] if sign else ~cols[i]
    return out


def proposals(cols: list[np.ndarray], positive: bool = False, minimise: bool = True) -> list[list[tuple]]:
    """For each populated cell of these rule columns (bool, one per held rule): its shortest
    equivalent definitions, each a tuple of (rule index, sign). A definition of one positive literal is the rule
    itself, which the rule's own symbol already names, and is not proposed. `positive=True` restores CS-2's restriction
    and `minimise=False` lifts CS-3's shortening, for their witness runs (step note §4, §8)."""
    if not cols:
        return []
    k = len(cols)
    bits = np.stack(cols).astype(np.int64)
    keys = np.zeros(len(cols[0]), dtype=np.int64)
    for row in bits:
        keys = keys * 2 + row
    out = []
    for key in np.unique(keys):
        signs = [(int(key) >> (k - 1 - i)) & 1 for i in range(k)]
        if positive and not any(signs):
            continue
        full = tuple((i, bool(s)) for i, s in enumerate(signs))
        target = keys == key
        defs = [full]
        if minimise and k <= MAX_RULES_MINIMISED:
            for size in range(1, k + 1):
                found = [d for d in combinations(full, size) if (any(sg for _, sg in d) or not positive)
                         and np.array_equal(extension(cols, d), target)]
                if found:
                    defs = found
                    break
        if len(defs) == 1 and len(defs[0]) == 1 and defs[0][0][1]:
            continue                                     # one positive literal: the rule itself
        out.append(defs)
    return out
