"""Structures (KERNEL §2): a finite set of facts, each a symbol applied to a tuple of entities.

Entities and symbols are opaque strings. A structure is immutable; `with_facts` returns a new one. Lookups go through
an index built once per structure: by symbol, and by (symbol, argument position, entity).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np


class Structure:
    __slots__ = ("facts", "_by_pos", "_entities")

    def __init__(self, facts: dict):
        self.facts = {s: frozenset(tuple(t) for t in ts) for s, ts in facts.items()}
        by_pos = defaultdict(list)
        for s, ts in self.facts.items():
            for t in ts:
                for i, e in enumerate(t):
                    by_pos[(s, i, e)].append(t)
        self._by_pos = dict(by_pos)
        self._entities: dict = {}

    def with_facts(self, extra: dict) -> "Structure":
        return Structure(dict(self.facts, **extra))

    def arity(self, symbol: str) -> int | None:
        ts = self.facts.get(symbol)
        return len(next(iter(ts))) if ts else None

    def lookup(self, symbol: str, bound: list[tuple[int, str]]):
        """The facts of `symbol` agreeing with every (position, entity) in `bound`, from the smallest index list."""
        if not bound:
            return self.facts.get(symbol, ())
        return min((self._by_pos.get((symbol, i, e), ()) for i, e in bound), key=len)

    def entities(self, symbols) -> tuple[str, ...]:
        """The entities occurring in facts of `symbols` (a channel's universe), sorted."""
        key = frozenset(symbols)
        if key not in self._entities:
            self._entities[key] = tuple(sorted({e for s in key for t in self.facts.get(s, ()) for e in t}))
        return self._entities[key]

    def candidates(self, arity: int, domain) -> list[tuple]:
        """Every tuple of the domain's entities: the closed-world candidates of a target (KERNEL §4)."""
        return list(product(self.entities(domain), repeat=arity))


def rename(s: Structure, entities: dict, symbols: dict) -> Structure:
    """The same structure with entities and symbols renamed (KERNEL §8.1, renaming)."""
    return Structure({symbols.get(k, k): {tuple(entities.get(e, e) for e in t) for t in ts}
                      for k, ts in s.facts.items()})


class Corpus:
    """Many structures as one, for evaluation. Entities are integers; entity e of structure i is its own integer, so a
    fact never mentions two structures, and every fact records its structure. A symbol's facts are an integer array.
    Candidates for a target are laid out structure by structure, each in `Structure.candidates` order."""

    def __init__(self, structures: list[Structure]):
        self.size = len(structures)
        self._at: dict = {}                  # (symbol, position) -> mask over entity ids: who occurs there
        self._meets: dict = {}               # pair of positions, or a position and a domain -> share an entity
        ids: dict = {}
        owner, self.names, rank = [], [], []   # per entity id: its structure, its name, its place in sorted names
        self.facts: dict = {}                # symbol -> (array (m, arity) of entity ids, array (m,) of structures)
        self.nullary: dict = {}              # symbol -> the structures where a 0-ary fact holds
        grouped: dict = {}
        for i, st in enumerate(structures):
            first = len(owner)
            for sym, ts in st.facts.items():
                if () in ts:
                    self.nullary.setdefault(sym, set()).add(i)
                for t in ts:
                    if not t:
                        continue
                    row = []
                    for e in t:
                        k = ids.get(e)
                        if k is None:
                            k = ids[e] = len(owner)
                            owner.append(i)
                            self.names.append(e)
                        row.append(k)
                    grouped.setdefault(sym, []).append(row)
            here = self.names[first:]
            place = [0] * len(here)
            for j, x in enumerate(sorted(range(len(here)), key=here.__getitem__)):
                place[x] = j
            rank += place
            ids.clear()
        self.owner = np.array(owner, dtype=np.int64)
        self._rank = np.array(rank, dtype=np.int64)
        for sym, rows in grouped.items():
            self._set(sym, np.array(sorted(rows), dtype=np.int64))
        self._given = dict(self.facts)       # the structures' own facts: what a channel's universe is read from
        self._rows: dict = {}
        self._functional: dict = {}

    def _set(self, symbol: str, arr: np.ndarray) -> None:
        self._forget(symbol)
        if len(arr):
            self.facts[symbol] = (arr, self.owner[arr[:, 0]])
        else:
            self.facts.pop(symbol, None)

    def functional(self, symbol: str, i: int, j: int) -> bool:
        """Whether, in every fact of a binary symbol, the entity at position i has exactly one partner at position j:
        a determinate step from i to j (FOIL), read off the evidence, never supplied."""
        key = (symbol, i, j)
        if key not in self._functional:
            got = self.facts.get(symbol)
            self._functional[key] = bool(got is not None and got[0].shape[1] == 2
                                         and len(np.unique(got[0][:, i])) == len(got[0]))
        return self._functional[key]

    def add(self, symbol: str, arr: np.ndarray) -> None:
        """Facts (an integer array over this corpus's entities) for a symbol defined on it."""
        self._functional = {k: v for k, v in self._functional.items() if k[0] != symbol}
        if symbol in self.facts:
            arr = np.concatenate([self.facts[symbol][0], arr])
        self._set(symbol, np.unique(arr, axis=0) if len(arr) else arr)

    def add_nullary(self, symbol: str, structures) -> None:
        """A 0-ary symbol defined on this corpus: the structures where it holds."""
        self.nullary.setdefault(symbol, set()).update(int(i) for i in structures)

    def remove(self, symbol: str) -> None:
        self._forget(symbol)
        self.facts.pop(symbol, None)
        self.nullary.pop(symbol, None)

    def _forget(self, symbol: str) -> None:
        if self._at:
            self._at = {k: v for k, v in self._at.items() if k[0] != symbol}
            self._meets = {k: v for k, v in self._meets.items() if k[0][0] != symbol and k[1][0] != symbol}

    def _mask(self, pos: tuple) -> np.ndarray:
        got = self._at.get(pos)
        if got is None:
            got = np.zeros(self.entity_count, dtype=bool)
            f = self.facts.get(pos[0])
            if f is not None:
                got[f[0][:, pos[1]]] = True
            self._at[pos] = got
        return got

    def meets(self, a: tuple, b: tuple) -> bool:
        """Whether some entity occurs both at position a and at position b, each a (symbol, argument index): a
        necessary condition for a variable at both to have a binding. Read off the evidence, cached."""
        key = (a, b) if a <= b else (b, a)
        got = self._meets.get(key)
        if got is None:
            got = self._meets[key] = bool((self._mask(a) & self._mask(b)).any())
        return got

    def meets_domain(self, pos: tuple, arity: int, domain) -> bool:
        """Whether some entity at `pos` is in the domain of a target's candidates: necessary for an exported variable."""
        key = (pos, ("", arity, frozenset(domain)))
        got = self._meets.get(key)
        if got is None:
            got = self._meets[key] = bool((self._mask(pos) & (self.rows(arity, domain)["index"] >= 0)).any())
        return got

    @property
    def entity_count(self) -> int:
        return len(self.owner)

    def vocabulary(self) -> dict:
        """Every symbol with facts here, and its arity; 0-ary symbols included."""
        return {**{k: 0 for k in self.nullary}, **{k: v[0].shape[1] for k, v in self.facts.items()}}

    def rows(self, arity: int, domain) -> dict:
        """For a target's candidates: `index` (entity id -> its place in its structure's domain, or -1), `size` (per
        structure its domain size), `start` (per structure its first row, and the total at the end)."""
        key = (arity, frozenset(domain))
        if key not in self._rows:
            # each structure's domain entities in Structure.entities order (sorted names), read from the given facts
            got = [self._given[sym][0].ravel() for sym in key[1] if sym in self._given]
            ents = np.unique(np.concatenate(got)) if got else np.zeros(0, dtype=np.int64)
            ents = ents[np.lexsort((self._rank[ents], self.owner[ents]))]
            st = self.owner[ents]
            size = np.bincount(st, minlength=self.size).astype(np.int64)
            index = np.full(self.entity_count, -1, dtype=np.int64)
            index[ents] = np.arange(len(ents)) - (np.cumsum(size) - size)[st]
            start = np.concatenate([[0], np.cumsum(size ** arity)])
            self._rows[key] = {"index": index, "size": size, "start": start, "entities": ents}
        return self._rows[key]

    def tuples_of(self, arity: int, domain, rows: np.ndarray) -> np.ndarray:
        """The entity-id tuples (m, arity) of candidate rows: the inverse of `row_of`."""
        r = self.rows(arity, domain)
        rows = np.asarray(rows, dtype=np.int64)
        st = np.searchsorted(r["start"], rows, side="right") - 1
        n, off = r["size"][st], rows - r["start"][st]
        first = (np.cumsum(r["size"]) - r["size"])[st]
        out = np.zeros((len(rows), arity), dtype=np.int64)
        for k in range(arity - 1, -1, -1):
            out[:, k] = r["entities"][first + off % n]
            off = off // n
        return out

    def row_of(self, arity: int, domain, cols: list[np.ndarray]) -> np.ndarray:
        """The row of each candidate tuple given as entity-id columns (-1 where an entity is outside the domain)."""
        r = self.rows(arity, domain)
        idx = [r["index"][c] for c in cols]
        st = self.owner[cols[0]]
        n = r["size"][st]
        row = r["start"][st].copy()
        ok = np.ones(len(st), dtype=bool)
        for k, ix in enumerate(idx):
            row += ix * n ** (arity - 1 - k)
            ok &= ix >= 0
        return np.where(ok, row, -1)
