"""Naming (KERNEL §5.2): a symbol for a piece several held rules share, admitted when it shortens the structure code.

A piece of a rule is a connected sub-conjunction of two or more of its atoms, not the whole rule. Its interface is the
variables it shares with the rest of the rule or with the target's arguments; its other variables are its own. A
symbol N defined by the piece, exporting the interface, can replace it in the rule exactly: the rule holds where it
held, so the data code D does not change, and only S does. Over a target's held rules the change in total code is

    ΔL = Σ_rules S(r | V) − [ S(N's definition | V) + Σ_rules S(r′ | V + 1) ]

where r′ is the rule rewritten to cite N (or r itself, re-priced: one more symbol makes every atom dearer), and V is
the vocabulary's size. N is admitted when ΔL clears the margin: this is the naming law of abstraction as invariance
§3.6, u · (L_V(I) − L(ref)) > L(def), with the vocabulary's growth charged. Nothing here reads a name: pieces are
grouped up to renaming, a rule with two occurrences of one piece is left as it is, and a tie between two different
pieces at the top admits neither.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

from kernel import code
from kernel.pattern import canonical

TIE_BITS = 1e-9


@dataclass
class Invention:
    atoms: tuple                 # the definition: atoms over variables, 0 .. arity-1 exported
    arity: int
    structure_bits: float        # S of the definition, at the vocabulary before it
    gain_bits: float             # ΔL when admitted
    rewrites: dict               # rule index -> its atoms citing the new symbol (under the placeholder name)


def _connected(atoms) -> bool:
    groups = [set(a) for _, a in atoms]
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i] & groups[j]:
                    groups[i] |= groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    return len(groups) == 1 and all(a for _, a in atoms)


def pieces(atoms, arity: int):
    """(indices, interface, own variables) for each connected piece of 2 .. len(atoms) - 1 atoms with an interface."""
    n = len(atoms)
    for size in range(2, n):
        for idx in combinations(range(n), size):
            part = [atoms[i] for i in idx]
            if not _connected(part):
                continue
            seen = list(dict.fromkeys(v for _, a in part for v in a))
            outside = set(range(arity)) | {v for i in range(n) if i not in idx for v in atoms[i][1]}
            iface = tuple(v for v in seen if v in outside)
            if iface:
                yield idx, iface, tuple(v for v in seen if v not in outside)


def _definition(atoms, idx, iface, own, order) -> tuple:
    """The piece over the symbol's own variables: the interface in `order` as 0 .. k-1, its own variables after."""
    m = {v: j for j, v in enumerate(order)}
    m.update({v: len(iface) + j for j, v in enumerate(own)})
    return tuple((s, tuple(m[v] for v in a)) for s, a in (atoms[i] for i in idx))


def candidates(rules: list, arity: int, frozen=frozenset()) -> dict:
    """{class key: [(rule index, piece indices, alignments)]}: occurrences grouped up to renaming. An alignment is an
    order of the rule's interface variables that sends the piece to the class's definition (key); a symmetric piece
    has several. Rules with two occurrences of one class are left out of it, and so are `frozen` rules (a rule held
    with rivals: rewriting one pattern of a rival set would part it from the others)."""
    found: dict = {}
    for r, atoms in enumerate(rules):
        if r in frozen:
            continue
        for idx, iface, own in pieces(atoms, arity):
            keyed = [(canonical(_definition(atoms, idx, iface, own, p), len(iface)), p) for p in permutations(iface)]
            key = min(k for k, _ in keyed)
            found.setdefault(key, {}).setdefault(r, []).append((idx, [p for k, p in keyed if k == key]))
    out = {}
    for key, per_rule in found.items():
        occ = [(r, got[0][0], got[0][1]) for r, got in sorted(per_rule.items()) if len(got) == 1]
        if len(occ) >= 2:
            out[key] = occ
    return out


def best(rules: list, arity: int, vocabulary: int, kind: str = "descriptions", frozen=frozenset()) -> tuple:
    """(the Invention with the largest ΔL, or None; how many classes tied at the top with it). The new symbol is
    written as the placeholder "@" in the rewrites; the caller names it."""
    before = sum(code.structure_bits(a, arity, vocabulary, kind) for a in rules)
    scored = []
    for key, occ in candidates(rules, arity, frozen).items():
        k = len(occ[0][2][0])
        def_bits = code.structure_bits(key, k, vocabulary, kind)
        best_total, best_rw, best_sigma = None, None, None
        for sigma in permutations(range(k)):          # the order of N's arguments: it changes the rules' S only
            rw, total = {}, def_bits
            for r, idx, aligns in occ:
                rest = tuple(a for i, a in enumerate(rules[r]) if i not in idx)
                options = [rest + (("@", tuple(p[s] for s in sigma)),) for p in aligns]
                bits = [code.structure_bits(o, arity, vocabulary + 1, kind) for o in options]
                j = min(range(len(options)), key=lambda i: bits[i])
                rw[r], total = options[j], total + bits[j]
            total += sum(code.structure_bits(a, arity, vocabulary + 1, kind) for i, a in enumerate(rules) if i not in rw)
            if best_total is None or total < best_total - TIE_BITS:
                best_total, best_rw, best_sigma = total, rw, sigma
        # N's argument i is the definition's variable sigma[i]: rename it to i (S of the definition does not change,
        # since every exported variable is in use from the start)
        back = {v: i for i, v in enumerate(best_sigma)}
        defined = tuple((s, tuple(back.get(v, v) for v in a)) for s, a in key)
        scored.append((before - best_total, Invention(defined, k, def_bits, before - best_total, best_rw)))
    if not scored:
        return None, 0
    top = max(g for g, _ in scored)
    tied = [inv for g, inv in scored if g >= top - TIE_BITS]
    return tied[0], len(tied)
