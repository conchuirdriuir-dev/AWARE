"""Patterns (KERNEL §3): conjunctions of atoms over variables, with the target's arguments as exported variables.

A pattern is a tuple of atoms (symbol, (v, ...)); variables are integers, and 0 .. arity-1 are the exported ones,
aligned with the target's arguments. A pattern holds at a candidate tuple when some assignment of its other variables
makes every atom a fact, with the exported variables sent to the tuple. An exported variable no atom mentions is free:
the pattern holds at every value of it.

Evaluation is incremental. The assignments satisfying a pattern in a structure are kept, and a child pattern (one
atom more) extends its parent's assignments instead of starting again.
"""
from __future__ import annotations

from itertools import permutations, product

import numpy as np



class Assignments:
    """The assignments satisfying a pattern: one row per assignment, a column per bound variable (`vars`), and the
    structure each row lies in. `None` stands for the empty pattern, which every assignment satisfies."""
    __slots__ = ("vars", "data", "struct")

    def __init__(self, vars_: tuple, data: np.ndarray, struct: np.ndarray):
        self.vars, self.data, self.struct = vars_, data, struct

    def __len__(self) -> int:
        return len(self.struct)


def _key(cols: list[np.ndarray], base: int) -> np.ndarray:
    k = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        k = k * base + c
    return k


def _unique_rows(rows: np.ndarray, base: int) -> np.ndarray:
    """np.unique(rows, axis=0) for entity ids below `base`, in the same order: each row packed into one integer key
    (much faster than a row-wise sort), unless the key would overflow."""
    if base ** rows.shape[1] >= 2 ** 63:
        return np.unique(rows, axis=0)
    _, first = np.unique(_key(list(rows.T), base), return_index=True)
    return rows[first]


def extend(c, asg: Assignments | None, atom, nvars: int = 0) -> Assignments:
    """The assignments that also satisfy `atom`: a join of the rows with the atom's facts, on the variables they
    share, or on the structure when they share none (facts never join two structures)."""
    sym, args = atom
    if not args:                                    # a 0-ary atom: a fact of the structure, which keeps or drops rows
        held = c.nullary.get(sym, ())
        if asg is None:
            st = np.array(sorted(held), dtype=np.int64)
            return Assignments((), np.zeros((len(st), 0), dtype=np.int64), st)
        keep = np.isin(asg.struct, np.fromiter(held, dtype=np.int64, count=len(held)))
        return Assignments(asg.vars, asg.data[keep], asg.struct[keep])
    got = c.facts.get(sym)
    empty = Assignments(tuple(dict.fromkeys(args if asg is None else asg.vars + tuple(args))),
                        np.zeros((0, 0), dtype=np.int64), np.zeros(0, dtype=np.int64))
    if got is None or (asg is not None and not len(asg)):
        return empty
    F, Fs = got
    first = {}
    keep = np.ones(len(F), dtype=bool)
    for k, v in enumerate(args):                    # a variable repeated in the atom: its positions agree
        if v in first:
            keep &= F[:, k] == F[:, first[v]]
        else:
            first[v] = k
    F, Fs = F[keep], Fs[keep]
    if asg is None:
        vars_ = tuple(first)
        return Assignments(vars_, F[:, [first[v] for v in vars_]], Fs)
    shared = [v for v in first if v in asg.vars]
    base = max(c.entity_count, c.size) + 1
    if shared:
        ka = _key([asg.data[:, asg.vars.index(v)] for v in shared], base)
        kf = _key([F[:, first[v]] for v in shared], base)
    else:
        ka, kf = asg.struct, Fs
    order = np.argsort(kf, kind="stable")
    kf = kf[order]
    lo = np.searchsorted(kf, ka, side="left")
    hi = np.searchsorted(kf, ka, side="right")
    counts = hi - lo
    if not counts.any():
        return empty
    ia = np.repeat(np.arange(len(ka)), counts)
    starts = np.repeat(lo - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    jf = order[np.arange(len(ia)) + starts]
    new = [v for v in first if v not in asg.vars]
    data = np.concatenate([asg.data[ia]] + [F[jf][:, [first[v]]] for v in new], axis=1) if new else asg.data[ia]
    return Assignments(asg.vars + tuple(new), data, asg.struct[ia])


def assignments(c, atoms, arity: int = 0) -> Assignments | None:
    out = None
    for atom in atoms:
        out = extend(c, out, atom)
        if not len(out):
            break
    return out


def mentioned(atoms, arity: int) -> tuple[int, ...]:
    return tuple(sorted({v for _, args in atoms for v in args if v < arity}))


def support(c, asg: Assignments | None, atoms, arity: int, domain) -> np.ndarray:
    """The rows (Corpus.rows order) where the pattern holds, sorted: its column, stored by where it is true. Cheaper
    than the column for a pattern that holds at few candidates, and what the search caches (KERNEL §6)."""
    r = c.rows(arity, domain)
    total = int(r["start"][-1])
    if asg is None:
        return np.arange(total, dtype=np.int64)
    if not len(asg):
        return np.zeros(0, dtype=np.int64)
    if arity == 0:
        return np.unique(asg.struct).astype(np.int64)
    bound = [v for v in range(arity) if v in asg.vars]
    free = [v for v in range(arity) if v not in asg.vars]
    if bound:
        sub = _unique_rows(asg.data[:, [asg.vars.index(v) for v in bound]], c.entity_count)
        st = c.owner[sub[:, 0]]
    else:
        sub, st = np.zeros((0, 0), dtype=np.int64), np.unique(asg.struct)
    n = r["size"][st]
    idx = {v: r["index"][sub[:, k]] for k, v in enumerate(bound)}
    ok = np.ones(len(st), dtype=bool)
    for v in bound:
        ok &= idx[v] >= 0
    st, n = st[ok], n[ok]
    idx = {v: a[ok] for v, a in idx.items()}
    base = r["start"][st]
    for v in bound:
        base = base + idx[v] * n ** (arity - 1 - v)
    parts = []
    for fill in product(range(int(n.max()) if len(n) else 0), repeat=len(free)):
        row, valid = base.copy(), np.ones(len(base), dtype=bool)
        for v, j in zip(free, fill):
            row += j * n ** (arity - 1 - v)
            valid &= j < n
        parts.append(row[valid])
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


def column(c, asg: Assignments | None, atoms, arity: int, domain) -> np.ndarray:
    """The pattern's value at every candidate of the corpus (Corpus.rows order)."""
    col = np.zeros(int(c.rows(arity, domain)["start"][-1]), dtype=bool)
    col[support(c, asg, atoms, arity, domain)] = True
    return col


def canonical(atoms, arity: int) -> tuple:
    """A key equal for two patterns exactly when they differ only in the names of their non-exported variables and
    the order of their atoms."""
    inner = sorted({v for _, args in atoms for v in args if v >= arity})
    best = None
    for perm in permutations(range(arity, arity + len(inner))):
        m = dict(zip(inner, perm))
        key = tuple(sorted((sym, tuple(m.get(v, v) for v in args)) for sym, args in atoms))
        if best is None or key < best:
            best = key
    return best if best is not None else ()


def refinements(atoms, arity: int, vocabulary: dict) -> list[tuple]:
    """Every pattern with one atom more (KERNEL §6). The new atom's arguments are variables already in use or one new
    one, and it shares at least one variable with the variables in use, the exported ones included. Where no variable
    is in use yet (a 0-ary target's pattern, empty or of 0-ary atoms only), any atom may start it; a 0-ary atom
    mentions no variable and may join any pattern."""
    used = max([arity] + [v + 1 for _, args in atoms for v in args])
    present = set(atoms)
    out = []
    for sym, k in vocabulary.items():
        for args in _arg_tuples(k, used):
            if args and used and min(args) >= used:
                continue                       # not connected to anything in use
            atom = (sym, args)
            if atom not in present:
                out.append(tuple(atoms) + (atom,))
    return out


def bridges(atoms, arity: int, vocabulary: dict, functional) -> list[tuple]:
    """Every pattern with two atoms more, taken as one step (KERNEL §6, bridge refinements): a binary atom that leads
    from a variable in use to one new variable, determinately (`functional(symbol, i, j)`: each entity at i has one
    partner at j on the evidence), and a unary atom on the new variable. So `next(u, t) ∧ w:by(u)` is one step from a
    pattern on t, where the stepping stone `next(u, t)` alone would have to win a place in the beam first."""
    used = max([arity] + [v + 1 for _, args in atoms for v in args])
    present = set(atoms)
    unary = [sym for sym, k in vocabulary.items() if k == 1]
    out = []
    for sym, k in vocabulary.items():
        if k != 2:
            continue
        for e in range(used):
            for args, (i, j) in (((e, used), (0, 1)), ((used, e), (1, 0))):
                link = (sym, args)
                if link in present or not functional(sym, i, j):
                    continue
                out.extend(tuple(atoms) + (link, (u, (used,))) for u in unary)
    return out


def _arg_tuples(k: int, used: int):
    """Argument tuples over the variables in use, introducing new variables in order (so each new one is named once)."""
    def rec(prefix, nxt):
        if len(prefix) == k:
            yield tuple(prefix)
            return
        for v in range(nxt + 1):
            yield from rec(prefix + [v], nxt + 1 if v == nxt else nxt)
    yield from rec([], used)
