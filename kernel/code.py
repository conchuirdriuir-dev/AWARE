"""Code lengths (KERNEL §4, §8.2). KT coding of binary outcomes per cell, and the structure codes for a pattern (KD-2):
the descriptions code (the default) and the prefix code.

Two data codes (SMALLEST_AGENT §9ak, SA-37). **cells**: each cell of the held rules' sign vectors coded by KT on its own
(KERNEL §4). **ctw**: context-tree weighting over the held rules in admission order: the cells are the leaves of a
binary tree whose node at depth d is the prefix of the first d rules' signs, and every node's code is the even
mixture of KT on its own counts and the product of its children's codes, so a cell with little evidence is coded by
the coarser cells above it, and a rule's rate carries into a combination of rules the evidence has not shown. It is
a prefix code fixed before the data with no constant beyond KT's ½, at most one bit per node longer than the best
pruning of the tree, and never worse than the cells code by more than the tree's size."""
from __future__ import annotations

from functools import lru_cache
from itertools import permutations
import math

import numpy as np


@lru_cache(maxsize=None)
def kt_bits(n: int, k: int) -> float:
    """Krichevsky-Trofimov code length of k ones in n binary outcomes; kt_bits(0, 0) = 0."""
    if not 0 <= k <= n:
        raise ValueError(f"invalid KT counts n={n} k={k}")
    if n == 0:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 0.5) - math.lgamma(n - k + 0.5) + math.log(math.pi)) / math.log(2)


def cell_ids(columns: list[np.ndarray], rows: int) -> np.ndarray:
    """One cell per distinct feature vector: the held rules' columns read as bits."""
    ids = np.zeros(rows, dtype=np.int64)
    for c in columns:
        ids = ids * 2 + c
    return ids


def counts(cells: np.ndarray, y: np.ndarray) -> dict:
    """{cell: (n, k)} over the cells that occur."""
    if len(cells) and cells.max() < 1 << 20:
        n = np.bincount(cells)
        k = np.bincount(cells, weights=y).astype(np.int64)
        nz = np.flatnonzero(n)
        return {int(c): (int(n[c]), int(k[c])) for c in nz}
    u, inv = np.unique(cells, return_inverse=True)
    n = np.bincount(inv, minlength=len(u))
    k = np.bincount(inv, weights=y, minlength=len(u)).astype(np.int64)
    return {int(c): (int(a), int(b)) for c, a, b in zip(u, n, k)}


def data_bits(cells: np.ndarray, y: np.ndarray) -> float:
    """D(R | H_R) = sum over cells of KT(n_c, k_c)."""
    return float(sum(kt_bits(n, k) for n, k in counts(cells, y).values()))


def split_bits(cells: np.ndarray, column: np.ndarray, y: np.ndarray) -> float:
    """D(R | H_R plus one rule with this column), without building the joint cells twice."""
    return data_bits(cells * 2 + column, y)


def continued_bits(train: dict, cells: np.ndarray, y: np.ndarray, depth: int | None = None) -> float:
    """Code length of held-out outcomes coded sequentially, each cell's counts starting from training's. The code
    is fixed by the training data alone, so it is a valid prefix code for the test data (KERNEL §8.2). With `depth`
    (the number of held rules), the ctw code: the tree with the test counts added, less the tree on training alone."""
    got = counts(cells, y)
    if depth is not None:
        both = dict(train)
        for c, (n, k) in got.items():
            n0, k0 = both.get(c, (0, 0))
            both[c] = (n0 + n, k0 + k)
        return CTW(both, depth).bits() - CTW(train, depth).bits()
    out = 0.0
    for c, (n, k) in got.items():
        n0, k0 = train.get(c, (0, 0))
        out += kt_bits(n0 + n, k0 + k) - kt_bits(n0, k0)
    return out


DATA_CODES = ("cells", "ctw")
DEFAULT_DATA_CODE = "ctw"           # adopted 2026-10-14 (SA-37); "cells" is KERNEL §4 as first built


class CTW:
    """The context-tree code over the held rules in admission order (module docstring): the leaves are the cells
    (their ids read as the rules' signs, the first rule the most significant bit), node (d, p) is the prefix p of the
    first d signs, with the summed counts of the leaves below it."""

    def __init__(self, leaves: dict, depth: int):
        self.depth = depth
        self.levels: list[dict] = [dict() for _ in range(depth + 1)]
        for c, (n, k) in leaves.items():
            for d in range(depth + 1):
                q = c >> (depth - d)
                n0, k0 = self.levels[d].get(q, (0, 0))
                self.levels[d][q] = (n0 + n, k0 + k)

    @staticmethod
    def mix(a: float, b: float) -> float:
        """-log2(½ 2^-a + ½ 2^-b), computed stably."""
        m = min(a, b)
        return m + 1.0 - math.log2(1.0 + 2.0 ** (-abs(a - b)))

    def bits(self, leaf_bits: dict | None = None) -> float:
        """The tree's code: with `leaf_bits`, those leaves' own KT replaced (a further split below them)."""
        val = {q: (leaf_bits[q] if leaf_bits is not None and q in leaf_bits else kt_bits(n, k))
               for q, (n, k) in self.levels[self.depth].items()}
        for d in range(self.depth - 1, -1, -1):
            nxt = {}
            for q, (n, k) in self.levels[d].items():
                kids = val.get(2 * q, 0.0) + val.get(2 * q + 1, 0.0)
                nxt[q] = self.mix(kt_bits(n, k), kids)
            val = nxt
        return val.get(0, 0.0)

    def predictive(self, cell: int) -> float:
        """P(the target holds) at a candidate in `cell`: the sequential code's next-symbol probability."""
        base = self.bits()
        n, k = self.levels[self.depth].get(cell, (0, 0))
        leaves = dict(self.levels[self.depth])
        leaves[cell] = (n + 1, k + 1)
        return 2.0 ** -(CTW(leaves, self.depth).bits() - base)


def target_bits(cols: list, y: np.ndarray, kind: str = "cells") -> float:
    """D(R | H_R) under the declared data code, from the held rules' columns in admission order."""
    cells = cell_ids(cols, len(y))
    if kind == "cells":
        return data_bits(cells, y)
    if kind == "ctw":
        return CTW(counts(cells, y), len(cols)).bits()
    raise ValueError(f"unknown data code {kind!r}")


def continued_cell_bits(train: dict, cell: int, n: int, k: int, depth: int | None = None) -> float:
    """Code length of n more outcomes, k of them holding, at a candidate in `cell`, continued from the training
    counts: KT at the cell, or with `depth` the ctw code."""
    n0, k0 = train.get(int(cell), (0, 0))
    if depth is None:
        return kt_bits(n0 + n, k0 + k) - kt_bits(n0, k0)
    more = dict(train)
    more[int(cell)] = (n0 + n, k0 + k)
    return CTW(more, depth).bits() - CTW(train, depth).bits()


def predictive(train: dict, cell: int, depth: int | None = None) -> float:
    """P(the target holds) at a candidate in `cell` from the training counts: the cell's KT estimate, (k + ½) / (n + 1),
    or with `depth` the ctw code's."""
    if depth is None:
        n, k = train.get(int(cell), (0, 0))
        return (k + 0.5) / (n + 1.0)
    return CTW(train, depth).predictive(int(cell))


def pattern_bits(atoms, arity: int, vocabulary: int) -> float:
    """S(p): a prefix code for a pattern over a vocabulary of `vocabulary` symbols (KERNEL §4). First the number u of
    variables beyond the target's exported ones, in unary (u + 1 bits). Then per atom: one bit to continue,
    log2 |vocabulary| for its symbol, and log2(arity + u) per argument to name its variable (a 0-ary atom has none).
    One bit to stop. The length depends only on the pattern, not on the order its atoms were found in."""
    names = arity + len({v for _, args in atoms for v in args if v >= arity})
    bits = (names - arity + 1) + 1.0
    for _, args in atoms:
        bits += 1.0 + math.log2(vocabulary) + (len(args) * math.log2(names) if args else 0.0)
    return bits



STRUCTURE_CODES = ("prefix", "descriptions")


def ml_bits(n, k) -> np.ndarray:
    """n H(k / n): the code length of k ones in n outcomes under their own best rate, fitted after the fact. No code
    does better on those outcomes, so it is a floor on KT; it is jointly concave in (n, k) (KERNEL §6.1)."""
    n, k = np.asarray(n, dtype=float), np.asarray(k, dtype=float)
    out = np.zeros(np.broadcast(n, k).shape)
    safe_n = np.where(n > 0, n, 1.0)
    for part in (k, n - k):
        pos = part > 0
        out -= np.where(pos, part * np.log2(np.where(pos, part, 1.0) / safe_n), 0.0)
    return out


def step_bits(atoms: int, vocabulary: int, max_atoms: int, kind: str = "descriptions") -> float:
    """A floor on S(d) - S(c) for every refinement d of a pattern c of `atoms` atoms, d having at most `max_atoms`
    (KERNEL §6.1). Under `prefix` an atom adds at least a continue bit and a symbol. Under `descriptions` each
    description of d, with its new atoms deleted, is a description of c no longer than it by at least
    1 + log2 |vocabulary| per deleted atom, and at most k'!/k! descriptions of d delete to the same one; so
    S(d) >= S(c) + sum over j = k+1 .. k' of (1 + log2 |vocabulary| - log2 j), least over the k' allowed."""
    if atoms >= max_atoms:
        return math.inf
    per = 1.0 + math.log2(max(vocabulary, 1))
    if kind == "prefix":
        return per
    if kind != "descriptions":
        raise ValueError(f"unknown structure code {kind!r}")
    total, least = 0.0, math.inf
    for j in range(atoms + 1, max_atoms + 1):
        total += per - math.log2(j)
        least = min(least, total)
    return least


def description_bits(atoms, arity: int, vocabulary: int) -> float:
    """S(p) as -log2 of the summed probability of every description of p (KD-2's tightened alternative). A description
    is an order of the atoms with the non-exported variables named in order of first use. It is coded as: the number
    of atoms k in unary (k + 1 bits); per atom its symbol (log2 |vocabulary|); per argument one choice among the
    variables in use, the exported ones included, and one new variable (log2(in use + 1); nothing when the only choice
    is a new one). Over all descriptions these lengths satisfy Kraft with equality, so summing over a pattern's
    distinct descriptions is a valid code for the pattern. It is fixed before data, does not depend on the order the
    atoms are given in, and charges a pattern once rather than once per way of writing it."""
    key = (tuple(sorted((s, tuple(a)) for s, a in atoms)), arity, vocabulary)
    return _description_bits(*key)


@lru_cache(maxsize=1 << 18)
def _description_bits(atoms: tuple, arity: int, vocabulary: int) -> float:
    k = len(atoms)
    head = (k + 1) + k * math.log2(vocabulary)
    seen, total = set(), 0.0
    for order in permutations(atoms):
        names, used, bits, seq = {}, arity, head, []
        for sym, args in order:
            out = []
            for v in args:
                if used:
                    bits += math.log2(used + 1)
                if v < arity:
                    out.append(v)
                else:
                    if v not in names:
                        names[v] = used
                        used += 1
                    out.append(names[v])
            seq.append((sym, tuple(out)))
        seq = tuple(seq)
        if seq not in seen:
            seen.add(seq)
            total += 2.0 ** -bits
    return -math.log2(total)


def structure_bits(atoms, arity: int, vocabulary: int, kind: str = "descriptions") -> float:
    """S(p) under the declared structure code (KD-2): `descriptions` (the kernel's default) or `prefix`."""
    if kind == "prefix":
        return pattern_bits(atoms, arity, vocabulary)
    if kind == "descriptions":
        return description_bits(atoms, arity, vocabulary)
    raise ValueError(f"unknown structure code {kind!r}")


class KT:
    """Vectorized KT code lengths over arrays of counts, from tables of math.lgamma, so each term equals kt_bits."""

    def __init__(self, n_max: int):
        self.lg1 = np.array([math.lgamma(n + 1) for n in range(n_max + 1)])
        self.lgh = np.array([math.lgamma(k + 0.5) for k in range(n_max + 1)])

    def bits(self, n: np.ndarray, k: np.ndarray) -> np.ndarray:
        """kt_bits(n, k) elementwise; kt_bits(0, 0) = 0 falls out of the formula."""
        return (self.lg1[n] - self.lgh[k] - self.lgh[n - k] + math.log(math.pi)) / math.log(2)


class Cells:
    """The held rules' cells for one round, compacted, with each cell's counts: what scoring a candidate rule needs.
    `split_bits(rows)` is D after adding a rule that holds at `rows`, in time proportional to len(rows). With `depth`
    (the number of held rules) the ctw code: the new rule as the deepest split (every node above re-mixing) or as the
    first split (the root hedging once, the old tree beneath each side), whichever codes shorter (`place` "best";
    "last" keeps the deepest split only)."""

    def __init__(self, cells: np.ndarray, y: np.ndarray, kt: KT, depth: int | None = None, place: str = "best"):
        self.ids, self.inv = np.unique(cells, return_inverse=True)
        self.y, self.kt, self.place = y, kt, place
        self.n = np.bincount(self.inv, minlength=len(self.ids))
        self.k = np.bincount(self.inv, weights=y, minlength=len(self.ids)).astype(np.int64)
        self.tree = CTW({int(c): (int(n), int(k)) for c, n, k in zip(self.ids, self.n, self.k)}, depth) \
            if depth is not None else None
        self.own = self.kt.bits(self.n, self.k)
        self.bits = self.tree.bits() if self.tree is not None else float(self.own.sum())

    def split_bits(self, rows: np.ndarray) -> float:
        """D after adding a rule that holds at `rows`: under the cells code the split of every cell; under the tree
        code the better of the rule as the deepest split and as the first (SMALLEST_AGENT §9av, SA-45), unless
        `place` is "last"."""
        return self.split_both(rows)[0]

    def split_both(self, rows: np.ndarray) -> tuple[float, str]:
        """(D after the rule, where it stands: "last" or "first")."""
        c = self.inv[rows]
        n1 = np.bincount(c, minlength=len(self.ids))
        k1 = np.bincount(c, weights=self.y[rows], minlength=len(self.ids)).astype(np.int64)
        split = self.kt.bits(n1, k1) + self.kt.bits(self.n - n1, self.k - k1)
        if self.tree is None:
            return float(split.sum()), "last"
        leaf = {int(cid): CTW.mix(float(a), float(b)) for cid, a, b in zip(self.ids, self.own, split)}
        last = self.tree.bits(leaf)
        if self.place == "last" or self.tree.depth == 0:
            return last, "last"
        inside = {int(cid): (int(a), int(b)) for cid, a, b in zip(self.ids, n1, k1) if a}
        outside = {int(cid): (int(a), int(b)) for cid, a, b in zip(self.ids, self.n - n1, self.k - k1) if a}
        first = CTW.mix(kt_bits(int(self.n.sum()), int(self.k.sum())),
                        CTW(inside, self.tree.depth).bits() + CTW(outside, self.tree.depth).bits())
        return (first, "first") if first < last - 1e-9 else (last, "last")

    def bound_bits(self, rows: np.ndarray) -> float:
        """D after a rule holding exactly at the rows of `rows` where the target holds: the search's optimistic bound."""
        return self.split_bits(rows[self.y[rows] == 1])

    def floor_bits(self, rows: np.ndarray) -> float:
        """A floor on D after any rule holding at a subset of `rows`, which is every refinement's (KERNEL §6.1). In each
        cell the subset takes a of the P positives among the rows and b of the Q negatives; the ML cost of the split
        is concave in (a, b), so over the box it is least at a corner: none, the positives, the negatives, or all."""
        c, yr = self.inv[rows], self.y[rows]
        P = np.bincount(c, weights=yr, minlength=len(self.ids))
        Q = np.bincount(c, weights=1 - yr, minlength=len(self.ids))
        n, k = self.n, self.k
        corners = [ml_bits(a + b, a) + ml_bits(n - a - b, k - a) for a, b in ((0, 0), (P, 0), (0, Q), (P, Q))]
        return float(np.minimum.reduce(corners).sum())
