"""Count symbols (docs/proposals/COUNT_SYMBOLS_v0_1.md; KERNEL §5.4): the number of witnesses of a held rule's internal
variable, made a symbol: "exactly k" or "at least k".

For a held rule with pattern φ, exported variables 0 .. arity-1 and an internal variable y, n(ā) is the number of
distinct values of y over the assignments of φ that send the exported variables to ā. A count symbol (rule, y, cmp, k)
holds at ā when n(ā) = k, or n(ā) >= k. "= 0" is the rule's own cell and ">= 0" always holds, so k >= 1; ">= 1" is the
rule itself and is not proposed. Definitions are expanded to primitive atoms (`inline`), so a count symbol can be
defined on any evidence before the rules that cite it, with no dependence on symbols a later pass re-forms.

S(Z) (CT-3) is fixed given the rules already held: which rule (log2 of the rules held so far), which of its internal
variables, one bit for the comparator, and Elias's gamma code for k. Nothing here reads a name.
"""
from __future__ import annotations

import math

import numpy as np

from kernel.pattern import assignments


def gamma_bits(k: int) -> int:
    """Elias gamma: 2 floor(log2 k) + 1 bits for a positive integer k."""
    return 2 * (k.bit_length() - 1) + 1


def structure_bits(rules_so_far: int, internal: int, k: int, k_code: str = "gamma", k_max: int = 1) -> float:
    """S(Z): the source rule, the counted variable, the comparator, then k (`k_code` "uniform": log2 of `k_max`, the
    witness run of CT-3's alternative)."""
    kb = gamma_bits(k) if k_code == "gamma" else math.log2(max(k_max, 1))
    return math.log2(max(rules_so_far, 1)) + math.log2(max(internal, 1)) + 1 + kb


def internal_vars(atoms, arity: int) -> list[int]:
    return sorted({v for _, args in atoms for v in args if v >= arity})


def inline(atoms, defs: dict) -> tuple | None:
    """The atoms with every defined symbol in `defs` ({symbol: (atoms, arity)}) replaced by its definition, the
    definition's exported variables sent to the atom's arguments and its own variables made fresh; recursively. Returns
    None when an atom cites a symbol in `defs` mapped to None (a symbol that is not a conjunction, such as a cell)."""
    out, top = [], max([v for _, args in atoms for v in args], default=-1) + 1

    def expand(sym, args):
        nonlocal top
        if sym not in defs:
            out.append((sym, tuple(args)))
            return True
        d = defs[sym]
        if d is None:
            return False
        body, arity = d
        ren = {i: args[i] for i in range(arity)}
        for _, bargs in body:
            for v in bargs:
                if v not in ren:
                    ren[v] = top
                    top += 1
        return all(expand(s, tuple(ren[v] for v in a)) for s, a in body)

    for sym, args in atoms:
        if not expand(sym, args):
            return None
    return tuple(out)


def counts_of(c, atoms, arity: int, var: int) -> tuple[np.ndarray, np.ndarray]:
    """(tuples (m, arity), n per tuple): the number of distinct values of `var` over the pattern's assignments, per
    exported tuple, wherever the pattern holds."""
    asg = assignments(c, atoms)
    if asg is None or not len(asg) or any(v not in asg.vars for v in range(arity)) or var not in asg.vars:
        return np.zeros((0, arity), dtype=np.int64), np.zeros(0, dtype=np.int64)
    if arity == 0:                                  # no exported tuple: one count per structure, keyed by the structure
        pairs = np.unique(np.stack([asg.struct, asg.data[:, asg.vars.index(var)]], axis=1), axis=0)
        tuples, n = np.unique(pairs[:, :1], axis=0, return_counts=True)
        return tuples, n
    cols = [asg.vars.index(v) for v in range(arity)] + [asg.vars.index(var)]
    pairs = np.unique(asg.data[:, cols], axis=0)
    tuples, n = np.unique(pairs[:, :arity], axis=0, return_counts=True)
    return tuples, n


def extension(tuples: np.ndarray, n: np.ndarray, cmp: str, k: int) -> np.ndarray:
    """The tuples where the count symbol holds."""
    return tuples[n == k] if cmp == "=" else tuples[n >= k]


def proposals(tuples: np.ndarray, n: np.ndarray, comparators=("=", ">="), copies: bool = True) -> list[tuple[str, int]]:
    """(cmp, k) for every k >= 1 the evidence shows (CT-2), "= k" before ">= k"; ">= 1" (the rule itself) is left out,
    as is a proposal whose extension equals an earlier one's (equivalent forms are one hypothesis, N16): the earlier,
    shorter one is kept. A rule whose witnesses always number the same k still gives "= k", a copy of the rule that
    a later pass can cite as one atom before the rule itself is formed (CT-13); `copies=False` withholds it, as CT-9
    first did (its witness, count symbols step note §8.4)."""
    out, seen = [], set()
    ks = sorted({int(x) for x in n if x >= 1})
    if len(ks) == 1 and not copies:     # "= k" equal to its rule: proposed (CT-13), withheld only by CT-9's witness
        return []
    for k in ks:
        for cmp in comparators:
            if cmp == ">=" and k == 1:
                continue
            ext = extension(tuples, n, cmp, k)
            if not len(ext):
                continue
            key = ext.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((cmp, k))
    return out
