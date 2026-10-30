"""Formation, admission and retirement (KERNEL §4-§6), and coding held-out evidence with what was formed (§8.2).

`learn` forms rules for each target in turn. A round searches patterns by refinement, scores each by its gain over
the rules already held, and admits the best when its gain clears the margin. Patterns whose gains tie are rivals:
held together if they have the same value at every candidate on the evidence, and an ambiguity that stops formation
for the target if they do not. Nothing in the search orders by a name, so renaming entities and symbols changes no
number and no decision (§8.1), unless a frontier is cut (reported).

Search (§6). Every child of the frontier is scored for admission. Which children are refined further is decided by
an optimistic bound: by default `bound="safe"`, a bound on every refinement's gain that never drops the best rule
(KD-9); `bound="positives"` is v0.1's heuristic, the gain if a refinement kept exactly the positives. A
child whose column equals its parent's (a stepping stone, such as `next(v, t)` on the way to `w:to(v)`) is refined
further when its bound is high enough, so rules that need such atoms are reachable.

The frontier is expanded in an order (`Search.order`: by bound, shuffled, or a proposer's), and before a frontier
pattern is expanded its safe bound is checked again against the best gain found so far in the round
(`Search.recheck_bound`, architecture §2.8 (j), C-2d): a pattern none of whose refinements can reach the best is
dropped. This changes what a round costs and never what it decides: a dropped pattern's refinements are all below the
top, so the top and its ties are those of the full round; and when the top is ambiguous (tied patterns with different
columns, KD-11), the round is re-run without dropping, so the next-best decision is the full round's too. An order
that finds the best sooner drops more. The record of what each round scored (`Search.record`, `Model.searches`) is
what search rules are formed over (C-2d).

An admitted rule whose pattern binds every exported variable, and has no rival, becomes a symbol: its facts are added
to the evidence, and rules for later targets can cite it as one atom (§3, recursion).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math

import numpy as np

from kernel import cells as cellcode, code, counts as countcode, naming
from kernel.pattern import (assignments, bridges, canonical, column as pattern_column, extend, mentioned,
                                 refinements, support)
from kernel.structure import Corpus, Structure

MARGIN_BITS = 2.0          # A1 REDUCED: strict G > 2 bits
TIE_BITS = 1e-9


@dataclass(frozen=True)
class Target:
    symbol: str
    arity: int
    domain: tuple[str, ...]                # the channel whose entities give the candidate tuples
    body: tuple[str, ...] | None = None    # symbols a rule may cite; None: every symbol but the target
    hide: tuple[str, ...] = ()             # symbols no rule may cite, directly or through a defined symbol


@dataclass(frozen=True)
class Search:
    max_atoms: int = 3
    deep_symbols: frozenset = frozenset()  # architecture §4.4.51 (C-3 reopened for one job): with it set, a pattern of
                                           # more than `deep_after` atoms is scored and carried only if one of its
                                           # atoms cites one of these symbols; shorter patterns are searched as without
                                           # it. Empty: the search as it was
    deep_after: int = 2
    beam: int | None = 8                   # None: no beam; with bound="safe" and no frontier cut, exact
    budget: int | None = 200_000           # KD-9: the beam applies only once a round has evaluated this many patterns,
                                           # so a search that fits runs without one; None: the beam always applies
                                           # (v0.1, and every recorded run)
    max_rules: int = 16
    frontier_cap: int = 128                # a frontier tied beyond this is cut, and the cut is reported
    cache_bytes: int = 4 << 30             # the column cache's budget per target (ColumnCache)
    bridges: bool | str = "on"             # bridge steps (a determinate link and a unary atom at once): "on", "off",
                                           # or "stall": only when a round's plain search admits nothing (True: "on")
    structure_code: str = "descriptions"   # S(h) (KD-2, adopted): "descriptions" (code.description_bits) or "prefix"
    prune_joins: bool = True               # skip a child whose new atom puts a variable at a position that shares no
                                           # entity with one it already holds (Corpus.meets): its support is empty, so
                                           # it would be discarded unscored; changes time, never results (U5 Arm A)
    carry_constant: bool = True            # carry a pattern that holds at every candidate into the next depth; False
                                           # scores it but does not carry it (U5 Arm A: such atoms tie at the top
                                           # bound for a 0-ary target and take the beam)
    invent: bool = True                    # after a target's formation, name pieces its held rules share when that
                                           # shortens S, and re-form with them (KERNEL §5.2; KD-10, adopted)
    reform: int = 2                        # with invent: re-form the target with its named pieces in Σ, at most this
                                           # many times, keeping a re-formation only when the total code is shorter
    ties: str = "skip"                     # tied patterns with different columns: "skip", admit none of them and go
                                           # on with the best pattern below them (KD-11, adopted), or "stop" formation
                                           # (AMBIGUOUS), as recorded runs did
    cell_witness: str = ""                 # ablations for the step note's witnesses (§4, §8): "positive" restores
                                           # CS-2's positive sign (dropped, CS-10), "full" keeps full cells (no CS-3),
                                           # "one" holds any cited cell in one pass (no CS-4); "" is the adopted mechanism
    cells: bool = True                     # offer each target's populated cells to later targets as symbols, in a
                                           # second pass held when the total code is shorter by the margin (KERNEL
                                           # §5.3; CS-1 to CS-13: the default since CS-13); False as recorded runs did
    counts: bool = False                   # offer counts over each held rule's internal variables ("exactly k", "at
                                           # least k") to the target and the targets after it as symbols, in a second
                                           # pass held when the total code is shorter by the margin (KERNEL §5.4;
                                           # CT-1 to CT-8); opt-in until measured. With `cells` since CT-10
    count_witness: str = ""                # ablations for the step note's witnesses (§4): "eq" proposes = only, "ge"
                                           # >= only, "uniform" codes k uniformly up to the largest proposed, "later"
                                           # offers a target's counts to later targets only, "nocopies" withholds
                                           # "= k" where it equals its rule (CT-9's second clause, revised by CT-13);
                                           # "" is the adopted mechanism
    count_relations: int = 0               # with counts: also propose counts over patterns of up to this many atoms of
                                           # the target's primitive symbols, anchored at an exported variable, priced
                                           # S(pattern) + log2(internal variables) + 1 + gamma(k) (SA-46, opt-in; 0: off)
    chain: bool = False                    # the chain rule (CA-10, architecture §3.4 finding 1): each target hides every
                                           # target after it in the order given, so the sum of the targets' codes is a
                                           # code of the evidence; off: any target may cite any other, as recorded runs
    bound: str = "safe"                    # which children to refine further (KD-9, adopted): "safe", a bound on every
                                           # refinement's gain that never drops the best rule (KERNEL §6.1), or
                                           # "positives", v0.1's heuristic (the gain if a refinement kept exactly the
                                           # positives)
    recheck_bound: bool = True             # before a frontier pattern is expanded, its safe bound is checked again against
                                           # the best gain found so far in the round, and it is dropped when no refinement
                                           # of it can reach that (architecture §2.8 (j), C-2d). Exact: the same rules,
                                           # rivals and code as without it (a round whose top is ambiguous is re-run
                                           # without dropping, so KD-11's next-best is exact too); safe bound only.
                                           # False: the search as it was, C-3's first arm
    order: str = "bound"                   # the order the frontier is expanded in, once the beam and the cap have kept
                                           # it: "bound" (highest safe bound first: the exact baseline), "shuffled" (by
                                           # `order_seed`: C-3's control and tie guard), or "proposer" (`proposer.rank`)
    order_seed: int = 0
    proposer: object | None = None         # with order="proposer": an object whose rank(target, depth, held, cands)
                                           # returns one key per (atoms, bound) in cands, the highest expanded first
                                           # (C-2d's search rules). It orders and never drops: the bound decides that
    record: bool = False                   # keep what every round scored in Model.searches (C-2d's record of searches)
    tree_place: str = "best"               # under "ctw": a candidate scored as the deepest split and as the first, and
                                           # placed where its gain is larger (SMALLEST_AGENT §9av, SA-45); "last":
                                           # the deepest split only, as §9an measured
    data_code: str = field(default_factory=lambda: code.DEFAULT_DATA_CODE)   # D (code.DATA_CODES): "cells", KT per
                                           # cell (KERNEL §4), or "ctw", the context-tree mixture over the rules'
                                           # prefix tree (SMALLEST_AGENT §9ak, SA-37); the default is the module
                                           # constant, which tests/conftest.py sets from DARA_DATA_CODE to run every
                                           # test under the other code


@dataclass
class Rule:
    target: str
    atoms: tuple
    rivals: list
    gain_bits: float
    structure_bits: float
    symbol: str | None = None             # the symbol it defines, if it binds every exported variable
    depth: int = 1


@dataclass
class Model:
    targets: list
    rules: dict = field(default_factory=dict)          # target -> [Rule], in tree order (admission order, a rule
                                                       #   placed first under the tree code standing first; SA-45)
    train_counts: dict = field(default_factory=dict)   # target -> {cell: (n, k)}
    train_bits: dict = field(default_factory=dict)     # target -> D on the training evidence
    stops: dict = field(default_factory=dict)          # target -> why formation stopped
    compute: dict = field(default_factory=dict)        # target -> patterns evaluated
    truncated: dict = field(default_factory=dict)      # target -> rounds whose frontier was cut
    cache: dict = field(default_factory=dict)          # target -> column cache hits, misses (joins) and bytes
    bridged: dict = field(default_factory=dict)        # target -> rules admitted only after a stalled round bridged
    pruned: dict = field(default_factory=dict)         # target -> patterns counted in compute but skipped unjoined
    beam_cuts: dict = field(default_factory=dict)      # target -> rounds whose beam dropped a child that could win
    dropped: dict = field(default_factory=dict)        # target -> frontier patterns dropped by the re-checked bound
    reruns: dict = field(default_factory=dict)         # target -> rounds re-run without dropping (an ambiguous top)
    searches: dict = field(default_factory=dict)       # target -> [round]: with Search.record, every pattern each round
                                                       #   scored, with its gain, bound, parent and whether it ended
                                                       #   within the margin of the top or admitted (C-2d's record)
    proposed: dict = field(default_factory=dict)       # target -> the searched code and each proposal's, and which won
    inventions: dict = field(default_factory=dict)     # target -> [{symbol, atoms, arity, structure_bits, gain_bits,
                                                       #   uses, depth}]: symbols named from its rules' pieces (§5.2)
    naming_stops: dict = field(default_factory=dict)   # target -> why naming stopped
    reformed: dict = field(default_factory=dict)       # target -> [{before, after, kept, dropped}] per re-formation
    cells: dict = field(default_factory=dict)          # source target -> [{symbol, arity, domain, definitions,
                                                       #   structure_bits, members, depth}]: cell symbols (§5.3)
    cells_record: dict = field(default_factory=dict)   # the two passes' totals, what was offered, dropped and held
    counts: dict = field(default_factory=dict)         # source target -> [{symbol, arity, domain, atoms, var, cmp,
                                                       #   k, structure_bits, members, depth}]: count symbols (§5.4)
    counts_record: dict = field(default_factory=dict)  # the two passes' totals, what was offered, dropped and held
    below: dict = field(default_factory=dict)          # target -> {gain, bits}: the best candidate of the round that
                                                       #   stopped formation and its S (reported, never admitted)
    priced_at: dict = field(default_factory=dict)      # target -> {size, cells}: the vocabulary its rules are priced
                                                       #   at, and the cell symbols in it
    data_code: str = "cells"                           # the data code D was computed under (code.DATA_CODES)


def _corpus(evidence) -> Corpus:
    return evidence if isinstance(evidence, Corpus) else Corpus(list(evidence))


# --- evidence for one target ------------------------------------------------------------------------------------------

def outcomes(evidence, t: Target) -> np.ndarray:
    c = _corpus(evidence)
    if t.arity == 0:
        held = c.nullary.get(t.symbol, set())
        return np.array([i in held for i in range(c.size)], dtype=np.int64)
    y = np.zeros(int(c.rows(t.arity, t.domain)["start"][-1]), dtype=np.int64)
    got = c.facts.get(t.symbol)
    if got is not None:
        rows = c.row_of(t.arity, t.domain, [got[0][:, k] for k in range(t.arity)])
        y[rows[rows >= 0]] = 1
    return y


def column(evidence, t: Target, atoms) -> np.ndarray:
    c = _corpus(evidence)
    return pattern_column(c, assignments(c, atoms), atoms, t.arity, t.domain)


def _vocabulary(c: Corpus, t: Target, excluded: set, model: Model) -> dict:
    """The symbols a rule for `t` may cite: its body (every symbol if none is declared) less the excluded ones, and
    every defined symbol whose definition cites only symbols it may cite. So a symbol hidden from a target stays hidden
    when another target's rule has folded it into a symbol of its own."""
    syms = c.vocabulary()
    defs = _definitions(model)
    allowed = {k for k in syms if k not in defs and (t.body is None or k in t.body) and k not in excluded}
    for sym, atoms in defs.items():                     # in the order they were formed: definitions cite earlier ones
        if sym in syms and sym not in excluded and all(a in allowed for a, _ in atoms):
            allowed.add(sym)
    return {k: a for k, a in syms.items() if k in allowed}


# --- one round of search ----------------------------------------------------------------------------------------------

class ColumnCache:
    """Each pattern's support (the rows where it holds), kept across a target's rounds. A pattern's support does not
    change when rules are admitted, only its gain does, so later rounds rescore cached patterns by counting and join
    only patterns they have not met (KERNEL §6). Holding the cache changes no result, only time. Supports are stored
    as int32 rows; past `cap_bytes` new patterns are scored but not stored."""

    def __init__(self, cap_bytes: int):
        self.cap, self.bytes, self.store = cap_bytes, 0, {}
        self.hits = self.misses = 0

    def get(self, key):
        got = self.store.get(key)
        if got is not None:
            self.hits += 1
        return got

    def put(self, key, rows: np.ndarray):
        self.misses += 1
        digest = hashlib.blake2b(rows.tobytes(), digest_size=16).digest() if len(rows) else None
        entry = (rows.astype(np.int32), digest)
        if self.bytes + entry[0].nbytes <= self.cap:
            self.store[key] = entry
            self.bytes += entry[0].nbytes
        return entry


def _bridge_mode(search: Search) -> str:
    mode = {True: "on", False: "off"}.get(search.bridges, search.bridges)
    if mode not in ("off", "on", "stall"):
        raise ValueError(f"bridges is off, on or stall, not {search.bridges!r}")
    return mode


def _round(c: Corpus, t: Target, vocab: dict, cells: code.Cells, search: Search, cache: ColumnCache,
           with_bridges: bool = False, out: dict | None = None, held: int = 0):
    """Every pattern reached, scored: [(gain, S, column digest, atoms, rows, parent's bound, parent's gain, own
    bound)], rows None for a pattern that cannot tie the top; patterns evaluated (the pruned ones included, so the
    count does not depend on pruning; a dropped frontier pattern's children are never reached, so they are not
    counted); whether a frontier was cut; patterns pruned; whether the beam dropped a child whose bound could still
    win. `out`, when given, receives "dropped" (frontier patterns dropped by the re-checked bound). A frontier
    pattern's assignments are computed only when one of its children is not in the cache."""
    frontier = [[(), None, None, None, None]]          # atoms, assignments, places, bound, gain
    scored, evaluated, keys, best, cut, pruned = [], 0, set(), MARGIN_BITS, False, 0
    top, beamed, safe = -math.inf, False, search.bound == "safe"
    dropped, recheck = 0, safe and search.recheck_bound
    if search.bound not in ("positives", "safe"):
        raise ValueError(f"unknown search bound {search.bound!r}")
    if search.order not in ("bound", "shuffled", "proposer"):
        raise ValueError(f"unknown frontier order {search.order!r}")
    for depth in range(search.max_atoms):
        children = []
        for entry in frontier:
            atoms = entry[0]
            if recheck and entry[3] is not None and entry[3] < best - TIE_BITS:
                dropped += 1                   # no refinement of it can reach the best found since it was kept
                continue
            steps = refinements(atoms, t.arity, vocab)
            if with_bridges and len(atoms) + 2 <= search.max_atoms:
                steps += bridges(atoms, t.arity, vocab, c.functional)
            for child in steps:
                key = canonical(child, t.arity)
                if key in keys:
                    continue
                keys.add(key)
                if (search.deep_symbols and len(child) > search.deep_after
                        and not any(a[0] in search.deep_symbols for a in child)):
                    continue                   # §4.4.51: beyond `deep_after` atoms, only patterns citing a deep symbol
                evaluated += 1
                if search.prune_joins:
                    if entry[2] is None:
                        entry[2] = _places(atoms)
                    if not _joins(c, t, entry[2], child[len(atoms):]):
                        pruned += 1
                        continue
                got = cache.get(key)
                if got is None:
                    if atoms and entry[1] is None:
                        entry[1] = assignments(c, atoms)
                    asg = entry[1]
                    for atom in child[len(atoms):]:
                        asg = extend(c, asg, atom)
                    rows = support(c, asg, child, t.arity, t.domain) if len(asg) else np.zeros(0, dtype=np.int64)
                    got = cache.put(key, rows)
                rows, digest = got
                if not len(rows):
                    continue
                s_bits = code.structure_bits(child, t.arity, len(vocab), search.structure_code)
                gain = cells.bits - cells.split_bits(rows) - s_bits
                best, top = max(best, gain), max(top, gain)
                # only patterns tied at the top need their rows (_decide); the top only rises, so a pattern already
                # below it never ties, and holding every pattern's rows grows with the round (one array each)
                b = None
                if len(child) < search.max_atoms and (search.carry_constant or len(rows) < len(cells.y)):
                    if safe:
                        b = cells.bits - cells.floor_bits(rows) - s_bits - code.step_bits(
                            len(child), len(vocab), search.max_atoms, search.structure_code)
                    else:
                        b = cells.bits - cells.bound_bits(rows) - s_bits
                    children.append((b, key, child, gain))
                scored.append((gain, s_bits, digest, child, rows if gain >= top - TIE_BITS else None,
                               entry[3], entry[4], b))
            entry[1] = None                    # its children are scored: its assignments are not needed again
        # safe: a child whose bound can still tie the best is kept, so rivals tied at the top are found too
        live = sorted(((b, ch, g) for b, _, ch, g in children if (b >= best - TIE_BITS if safe else b > best)),
                      key=lambda x: -x[0])
        over = search.budget is None or evaluated >= search.budget
        if search.beam is not None and over and len(live) > search.beam:
            floor = live[search.beam - 1][0] - TIE_BITS
            kept = [x for x in live if x[0] >= floor]
            beamed |= len(kept) < len(live)
            live = kept
        if len(live) > search.frontier_cap:
            live, cut = live[:search.frontier_cap], True
        live = _order(live, search, t, depth + 1, held)      # what is kept is decided above; only the order changes
        frontier = [[ch, None, None, b, g] for b, ch, g in live]
        if not frontier:
            break
    if out is not None:
        out["dropped"] = dropped
    return scored, evaluated, cut, pruned, beamed


def _order(live: list, search: Search, t: Target, depth: int, held: int) -> list:
    """The order the kept frontier is expanded in (C-2d). "bound": as kept, highest bound first. "shuffled": a seeded
    permutation (the control). "proposer": `search.proposer.rank(target, depth, held, [(atoms, bound)])` gives one
    key per pattern, the highest first; a stable sort, so ties keep the bound's order."""
    if search.order == "bound" or len(live) < 2:
        return live
    if search.order == "shuffled":                  # a permutation from the seed, the depth and each position: reads
        def key(i):                                 # no name, so renaming changes nothing here either
            return hashlib.blake2b(f"{search.order_seed}:{depth}:{i}".encode(), digest_size=8).digest()
        return [x for _, x in sorted(enumerate(live), key=lambda z: key(z[0]))]
    keys = search.proposer.rank(t.symbol, depth, held, [(ch, b) for b, ch, _ in live])
    if len(keys) != len(live):
        raise ValueError("the proposer must return one key per pattern")
    return [x for _, x in sorted(zip(keys, live), key=lambda z: -z[0])]


def _search(c: Corpus, t: Target, vocab: dict, cells: code.Cells, search: Search, cache: ColumnCache,
            with_bridges: bool, model: "Model", y: np.ndarray, held: int):
    """One round, accounted in the model, exact: with the bound re-checked (dropping), a round whose top is ambiguous
    is re-run without dropping, so that KD-11's next-best decision reads every pattern the full round would."""
    out: dict = {}
    scored, n, cut, skip, beamed = _round(c, t, vocab, cells, search, cache, with_bridges, out, held)
    model.compute[t.symbol] += n
    model.pruned[t.symbol] += skip
    model.truncated[t.symbol] += cut
    model.beam_cuts[t.symbol] += beamed
    model.dropped[t.symbol] += out.get("dropped", 0)
    if out.get("dropped") and search.ties == "skip" and _decide(scored, t.arity, y, False, cells.inv)[1] == "AMBIGUOUS":
        model.reruns[t.symbol] += 1
        scored, n, cut, skip, beamed = _round(c, t, vocab, cells, replace(search, recheck_bound=False), cache,
                                              with_bridges, None, held)
        model.compute[t.symbol] += n
        model.pruned[t.symbol] += skip
        model.truncated[t.symbol] += cut
        model.beam_cuts[t.symbol] += beamed
    return scored


def _record(model: "Model", t: Target, scored: list, held: int, got, why) -> None:
    """One round of the record of searches (C-2d): every pattern scored, with what it came to."""
    top = max((g for g, *_ in scored), default=-math.inf)
    admitted = {canonical(p, t.arity) for p in got[0]} if got is not None else set()
    model.searches.setdefault(t.symbol, []).append({
        "held": held, "top": top, "decision": why or "ADMITTED",
        "patterns": [{"atoms": x[3], "depth": len(x[3]), "gain": float(x[0]), "bits": float(x[1]),
                      "bound": None if x[7] is None else float(x[7]),
                      "parent_bound": None if x[5] is None else float(x[5]),
                      "parent_gain": None if x[6] is None else float(x[6]),
                      "within": bool(x[0] >= top - MARGIN_BITS),
                      "admitted": canonical(x[3], t.arity) in admitted} for x in scored]})


def _places(atoms) -> dict:
    """Each variable of a pattern, and the (symbol, position) places it occupies."""
    out: dict = {}
    for sym, args in atoms:
        for i, v in enumerate(args):
            out.setdefault(v, []).append((sym, i))
    return out


def _joins(c: Corpus, t: Target, places: dict, new) -> bool:
    """False only when the new atoms cannot hold anywhere with the pattern: a variable put at a place that shares no
    entity with a place it already holds, or an exported variable at a place with no entity in the target's domain.
    Each is a necessary condition for a binding, so a child that fails has empty support (KERNEL §6, U5 Arm A)."""
    seen = {v: list(ps) for v, ps in places.items()}
    for sym, args in new:
        for i, v in enumerate(args):
            pos = (sym, i)
            if v < t.arity and not c.meets_domain(pos, t.arity, t.domain):
                return False
            for q in seen.get(v, ()):
                if not c.meets(pos, q):
                    return False
            seen.setdefault(v, []).append(pos)
    return True


def _same_partition(rows_a, rows_b, cells: np.ndarray | None, n: int) -> bool:
    """Whether two columns split every held cell alike: within each cell the rows they mark are the same or exactly
    complementary (a word and its antonym on the places they are said of). With no cells given, the whole candidate
    set is the one cell. Reads counts only."""
    a = np.zeros(n, dtype=bool); a[rows_a] = True
    b = np.zeros(n, dtype=bool); b[rows_b] = True
    groups = cells if cells is not None else np.zeros(n, dtype=np.int64)
    for c in np.unique(groups):
        here = groups == c
        xa, xb = a[here], b[here]
        if not (np.array_equal(xa, xb) or np.array_equal(xa, ~xb)):
            return False
    return True


def _decide(scored, arity: int, y: np.ndarray | None = None, skip: bool = False, cells: np.ndarray | None = None):
    """(its least-S patterns, gain, S) or (None, reason).

    Tied patterns with different columns are an ambiguity, except columns that split every held cell alike (the
    same rows, or exactly complementary ones, within each cell): they make the same cells and say the same thing
    (docs/PROBE_SYNTHESIS_v0_1.md I-5; a word and its antonym are one hypothesis on the places they are said of),
    so one is kept: of two exactly complementary columns the one whose marked candidates have the higher rate of the
    target, otherwise the one of least S. The choice reads counts only, never a name."""
    if not scored:
        return None, "NO_HYPOTHESIS"
    top = max(g for g, *_ in scored)
    tied = [x for x in scored if x[0] >= top - TIE_BITS]
    digests = sorted({x[2] for x in tied})
    if len(digests) == 2 and y is not None:
        a, b = (next(x[4] for x in tied if x[2] == d) for d in digests)
        if a is not None and b is not None and len(a) + len(b) == len(y) and not np.intersect1d(a, b).size:
            ra, rb = float(y[a].mean()), float(y[b].mean())
            if ra != rb:
                keep = digests[0] if ra > rb else digests[1]
                tied, digests = [x for x in tied if x[2] == keep], [keep]
    if len(digests) > 1 and y is not None:
        first = {d: next(x for x in tied if x[2] == d) for d in digests}
        if all(first[d][4] is not None for d in digests):
            ref = first[digests[0]][4]
            if all(_same_partition(ref, first[d][4], cells, len(y)) for d in digests[1:]):
                keep = min(digests, key=lambda d: (first[d][1], d))          # the least S, then the digest
                tied, digests = [x for x in tied if x[2] == keep], [keep]
    if not top > MARGIN_BITS:                        # first: a tie among patterns that do not pay is no ambiguity
        return None, "BELOW_MARGIN"
    if len(digests) > 1:
        if skip:                                     # the tied set is passed over; the next best is decided alike
            rest = [x for x in scored if x[0] < top - TIE_BITS]
            got, why = _decide(rest, arity, y, skip, cells)
            return (got, why) if got is not None else (None, "AMBIGUOUS")
        return None, "AMBIGUOUS"
    s_min = min(x[1] for x in tied)
    pats = sorted({canonical(x[3], arity) for x in tied if x[1] <= s_min + TIE_BITS})
    return (pats, top, s_min), None


# --- formation --------------------------------------------------------------------------------------------------------

def _define(c: Corpus, t: Target, atoms, name: str) -> None:
    _define_symbol(c, atoms, t.arity, name)


def _define_symbol(c: Corpus, atoms, arity: int, name: str) -> None:
    asg = assignments(c, atoms)
    if arity == 0:
        c.add_nullary(name, range(c.size) if asg is None else np.unique(asg.struct))
    elif asg is not None and len(asg):
        c.add(name, asg.data[:, [asg.vars.index(v) for v in range(arity)]])


def _definitions(model: Model) -> dict:
    """{symbol: its atoms}, in the order they were formed: per target, its inventions, then its rules' symbols."""
    out = {}
    for t in model.targets:
        for z in model.counts.get(t.symbol, []):
            out[z["symbol"]] = z["atoms"]
        for inv in model.inventions.get(t.symbol, []):
            out[inv["symbol"]] = inv["atoms"]
        for r in model.rules.get(t.symbol, []):
            if r.symbol:
                out[r.symbol] = r.atoms
        for z in model.cells.get(t.symbol, []):
            out[z["symbol"]] = _cell_atoms(z)
    return out


def _cell_atoms(z: dict) -> tuple:
    """Every atom a cell symbol's definitions cite: what decides whether a target may cite it."""
    return tuple(a for d in z["definitions"] for lit in d for p in lit["rivals"] for a in p)


def _cell_rows(c: Corpus, t: Target, z: dict) -> np.ndarray:
    """The candidate rows of the source target where a cell symbol holds: every definition (rivals: AND)."""
    held = None
    for d in z["definitions"]:
        here = None
        for lit in d:
            col = rival_column(c, t, lit["rivals"])[0]
            col = col if lit["sign"] else ~col
            here = col if here is None else here & col
        held = here if held is None else held & here
    return np.flatnonzero(held)


def _define_cell(c: Corpus, t: Target, z: dict) -> None:
    rows = _cell_rows(c, t, z)
    if t.arity == 0:
        c.add_nullary(z["symbol"], rows)
    elif len(rows):
        c.add(z["symbol"], c.tuples_of(t.arity, t.domain, rows))


def _offer_cells(c: Corpus, t: Target, rules: list, cols: list, model: Model, search: Search) -> None:
    """Pass 2 (§5.3): each populated cell of the target's held rules, defined on the evidence for later targets."""
    made = model.cells.setdefault(t.symbol, [])
    sources = [x.symbol for x in model.targets].index(t.symbol) + 1
    bools = [np.asarray(col).astype(bool) for col in cols]
    positive = search.cell_witness == "positive"
    for defs in cellcode.proposals(bools, positive, search.cell_witness != "full"):
        z = {"symbol": f"#{t.symbol}.c{len(made)}", "arity": t.arity, "domain": list(t.domain),
             "definitions": [[{"sign": sign, "rivals": rules[i].rivals} for i, sign in d] for d in defs],
             "structure_bits": cellcode.structure_bits(sources, len(rules), positive),
             "positive": any(sign for d in defs for _, sign in d)}
        z["depth"] = 1 + max([_depth(model, sym) for sym, _ in _cell_atoms(z)], default=0)
        _define_cell(c, t, z)
        z["members"] = int(len(_cell_rows(c, t, z)))
        made.append(z)


def _derived_from(model: Model, target: str) -> set:
    return {r.symbol for r in model.rules.get(target, []) if r.symbol}


def _depth(model: Model, sym: str) -> int:
    for rules in model.rules.values():
        for r in rules:
            if r.symbol == sym:
                return r.depth
    for invs in model.inventions.values():
        for inv in invs:
            if inv["symbol"] == sym:
                return inv["depth"]
    for zs in list(model.cells.values()) + list(model.counts.values()):
        for z in zs:
            if z["symbol"] == sym:
                return z["depth"]
    return 0


def _name(c: Corpus, t: Target, rules: list, search: Search, model: Model) -> None:
    """Name pieces the target's held rules share while the total code shortens by more than the margin (§5.2). Each
    naming rewrites the rules that contain the piece to cite the new symbol (the same column, so D is unchanged) and
    re-prices every rule of the target at the vocabulary with one more symbol."""
    made = model.inventions.setdefault(t.symbol, [])
    kind = search.structure_code
    while True:
        vocab = _vocabulary(c, t, {t.symbol, *t.hide} | _derived_from(model, t.symbol), model)
        frozen = {i for i, r in enumerate(rules) if len(r.rivals) > 1}
        inv, tied = naming.best([r.atoms for r in rules], t.arity, len(vocab), kind, frozen)
        if inv is None:
            model.naming_stops[t.symbol] = "NO_SHARED_PIECE"
            return
        if tied > 1:
            model.naming_stops[t.symbol] = "AMBIGUOUS"
            return
        if not inv.gain_bits > MARGIN_BITS:
            model.naming_stops[t.symbol] = "BELOW_MARGIN"
            return
        name = f"#{t.symbol}.n{len(made)}"
        depth = 1 + max([_depth(model, sym) for sym, _ in inv.atoms], default=0)
        made.append({"symbol": name, "atoms": inv.atoms, "arity": inv.arity, "structure_bits": inv.structure_bits,
                     "gain_bits": inv.gain_bits, "uses": len(inv.rewrites), "depth": depth,
                     "vocabulary": len(vocab)})
        _define_symbol(c, inv.atoms, inv.arity, name)
        for i, atoms in inv.rewrites.items():
            r = rules[i]
            r.atoms = tuple((name if sym == "@" else sym, a) for sym, a in atoms)
            r.rivals = [r.atoms]
            r.depth = 1 + max([_depth(model, sym) for sym, _ in r.atoms], default=0)
        for r in rules:
            r.structure_bits = min(code.structure_bits(p, t.arity, len(vocab) + 1, kind) for p in r.rivals)


def _form(c: Corpus, t: Target, y: np.ndarray, model: Model, search: Search, start: list) -> tuple[list, list]:
    """One target's formation (see `learn`): the rules in `start` held first, then rounds of search until nothing
    clears the margin, then retirement. Its rules are model.rules[t] (returned with their columns)."""
    cols: list[np.ndarray] = []
    rules: list[Rule] = []
    model.rules[t.symbol] = rules
    kt, cache, mode = code.KT(len(y)), ColumnCache(search.cache_bytes), _bridge_mode(search)
    for held in start:
        vocab = _vocabulary(c, t, {t.symbol, *t.hide} | _derived_from(model, t.symbol), model)
        atoms = tuple((sym, tuple(a)) for sym, a in held.atoms)
        if any(vocab.get(sym) != len(a) for sym, a in atoms):
            continue
        col = column(c, t, atoms).astype(np.int64)
        s_bits = code.structure_bits(atoms, t.arity, len(vocab), search.structure_code)
        gain = code.target_bits(cols, y, search.data_code) - code.target_bits(cols + [col], y, search.data_code) - s_bits
        rule = Rule(t.symbol, atoms, [atoms], gain, s_bits,
                    depth=1 + max([_depth(model, sym) for sym, _ in atoms], default=0))
        if held.symbol:
            rule.symbol = held.symbol
            _define(c, t, atoms, rule.symbol)
        rules.append(rule)
        cols.append(col)
    while len(rules) < search.max_rules:
        vocab = _vocabulary(c, t, {t.symbol, *t.hide} | _derived_from(model, t.symbol), model)
        cells = code.Cells(code.cell_ids(cols, len(y)), y, kt, len(cols) if search.data_code == "ctw" else None,
                           search.tree_place)
        scored = _search(c, t, vocab, cells, search, cache, mode == "on", model, y, len(rules))
        got, why = _decide(scored, t.arity, y, search.ties == "skip", cells.inv)
        if got is None and why != "AMBIGUOUS" and mode == "stall":      # the plain search stalled: bridge
            scored = _search(c, t, vocab, cells, search, cache, True, model, y, len(rules))
            got, why = _decide(scored, t.arity, y, search.ties == "skip", cells.inv)
            model.bridged[t.symbol] += got is not None
        if search.record:
            _record(model, t, scored, len(rules), got, why)
        if got is None:
            model.stops[t.symbol] = why
            if scored:
                g, b, *_ = max(scored, key=lambda x: x[0])
                model.below[t.symbol] = {"gain": float(g), "bits": float(b)}
            break
        pats, gain, s_bits = got
        atoms = pats[0]
        col = rival_column(c, t, pats)[0].astype(np.int64)
        rule = Rule(t.symbol, atoms, pats, gain, s_bits,
                    depth=1 + max([_depth(model, sym) for sym, _ in atoms], default=0))
        if len(mentioned(atoms, t.arity)) == t.arity and len(pats) == 1:
            k, taken = len(rules), {r.symbol for r in rules}
            while f"#{t.symbol}/{k}" in taken:                  # a held rule (`start`) may have the name
                k += 1
            rule.symbol = f"#{t.symbol}/{k}"
            _define(c, t, atoms, rule.symbol)
        if cells.split_both(np.flatnonzero(col))[1] == "first":     # SA-45: the rule stands where its gain is larger
            rules.insert(0, rule)
            cols.insert(0, col)
        else:
            rules.append(rule)
            cols.append(col)
    else:
        model.stops[t.symbol] = "MAX_RULES"
    model.cache[t.symbol] = {"hits": cache.hits, "joins": cache.misses, "bytes": cache.bytes}
    defined = {r.symbol for r in rules if r.symbol}
    _retire(rules, cols, y, search.data_code)
    for sym in defined - {r.symbol for r in rules}:
        c.remove(sym)
    return rules, cols


def chained(targets: list[Target]) -> list[Target]:
    """The targets with each hiding every target after it (the chain rule, CA-10). Each target's code is then
    conditioned only on what is given and on the targets before it, so the codes sum to one code of the evidence."""
    syms = [t.symbol for t in targets]
    return [replace(t, hide=tuple(dict.fromkeys(t.hide + tuple(syms[i + 1:])))) for i, t in enumerate(targets)]


def learn(evidence, targets: list[Target], search: Search = Search(), proposals: dict | None = None,
          start: dict | None = None) -> tuple[Model, Corpus]:
    """Form rules for each target in turn (see `_learn`). With `search.cells` and more than one target, two passes
    (KERNEL §5.3): pass 1 as `_learn`; pass 2 again, with each target's populated cells offered to the targets after it
    as symbols. Cell symbols no rule cites leave Σ, and the targets are re-priced. Pass 2 is held when the total code
    is shorter than pass 1's by more than the margin; otherwise pass 1 is held, exactly. `cells_record` reports both."""
    c = Corpus(list(evidence)) if not isinstance(evidence, Corpus) else evidence
    if search.chain:
        targets = chained(targets)
    if search.counts:
        return _learn_counts(c, targets, search, proposals, start)
    return _learn_cells(c, targets, search, proposals, start)


def _learn_cells(c: Corpus, targets, search: Search, proposals, start) -> tuple[Model, Corpus]:
    """`learn` without count symbols: one pass, or with `search.cells` and more than one target the two passes of
    cell symbols (KERNEL §5.3)."""
    if not search.cells or len(targets) < 2:
        return _learn(c, targets, search, proposals, start, offer=False)
    first, _ = _learn(c, targets, search, proposals, start, offer=False)
    first_bits = _model_bits(first)
    for sym in _symbols_of(first):
        c.remove(sym)
    second, _ = _learn(c, targets, search, proposals, start, offer=True)
    offered = [z["symbol"] for zs in second.cells.values() for z in zs]
    dropped = _drop_uncited_cells(c, second, search)
    second_bits = _model_bits(second)
    held = "second" if second_bits < first_bits - MARGIN_BITS or search.cell_witness == "one" else "first"
    record = {"first_bits": first_bits, "second_bits": second_bits, "offered": offered, "dropped": dropped,
              "held": held}
    if held == "first":
        for sym in _symbols_of(second):
            c.remove(sym)
        apply(first, c)
        first.cells_record = record
        return first, c
    second.cells_record = record
    return second, c


def _learn_counts(c: Corpus, targets, search: Search, proposals, start) -> tuple[Model, Corpus]:
    """Two passes for count symbols (KERNEL §5.4): pass 1 as `_learn`; its held rules' counts are proposed; pass 2
    forms every target again with each target's counts in Σ for it and the targets after it (CT-4). Counts no rule
    cites leave Σ and the targets are re-priced; pass 2 is held when the total code is shorter than pass 1's by more
    than the margin. `counts_record` reports both.

    With cell symbols on (CT-10), pass 1 is the best code without counts, the cell symbols' own two passes, and pass 2
    offers both, each target's cells to the targets after it and its counts to it and them; counts are proposed only
    from rules that do not cite a cell symbol (a cell is not a conjunction, so it cannot be expanded)."""
    first, _ = _learn_cells(c, targets, search, proposals, start)
    first_bits = _model_bits(first)
    props = _count_proposals(c, first, search)
    for sym in _symbols_of(first):
        c.remove(sym)
    both = search.cells and len(targets) > 1
    second, _ = _learn(c, targets, search, proposals, start, offer=both, counts=props)
    offered = [z["symbol"] for zs in list(second.counts.values()) + list(second.cells.values()) for z in zs]
    dropped = _drop_uncited_cells(c, second, search)
    second_bits = _model_bits(second)
    held = "second" if second_bits < first_bits - MARGIN_BITS else "first"
    record = {"first_bits": first_bits, "second_bits": second_bits, "offered": offered, "dropped": dropped,
              "held": held, "proposed": sum(len(v) for v in props.values()), "with_cells": both}
    if held == "first":
        for sym in _symbols_of(second):
            c.remove(sym)
        apply(first, c)
        first.counts_record = record
        return first, c
    second.counts_record = record
    second.cells_record = first.cells_record
    return second, c


def _count_proposals(c: Corpus, model: Model, search: Search) -> dict:
    """{target: [count symbol]} from pass 1's held rules (CT-2): every rule with one pattern that binds the target's
    arguments, every internal variable of it, every k the training evidence shows. Definitions are expanded to
    primitives, so they hold on any evidence before what cites them; a rule citing a cell symbol is not expanded."""
    defmap = {}
    for t in model.targets:
        for inv in model.inventions.get(t.symbol, []):
            defmap[inv["symbol"]] = (inv["atoms"], inv["arity"])
        for r in model.rules.get(t.symbol, []):
            if r.symbol:
                defmap[r.symbol] = (r.atoms, t.arity)
        for z in model.cells.get(t.symbol, []):
            defmap[z["symbol"]] = None
    comps = {"eq": ("=",), "ge": (">=",)}.get(search.count_witness, ("=", ">="))
    out, held, k_max = {}, 0, 1
    for t in model.targets:
        made = out.setdefault(t.symbol, [])
        rules = model.rules.get(t.symbol, [])
        held += len(rules)
        for r, pat in ((r, pat) for r in rules for pat in r.rivals):
            # every pattern of a held rule's rival set is held, and counting rewrites none of them, so each is a
            # source; which one is part of the description (log2 of the rivals; step note §8)
            atoms = countcode.inline(pat, defmap)
            if atoms is None:
                continue
            # a count over the exported variables the rule mentions: one it leaves free changes no count, so the
            # symbol is over the mentioned ones, renumbered 0 .. m-1, and a rule cites it with them (step note §8)
            ms = sorted(mentioned(pat, t.arity))
            ren = {v: i for i, v in enumerate(ms)}
            top = len(ms)
            for v in sorted({v for _, args in atoms for v in args} - set(ms)):
                ren[v], top = top, top + 1
            atoms = tuple((sym, tuple(ren[v] for v in args)) for sym, args in atoms)
            internal = countcode.internal_vars(pat, t.arity)
            for var in internal:
                tuples, n = countcode.counts_of(c, atoms, len(ms), ren[var])
                for cmp, k in countcode.proposals(tuples, n, comps, search.count_witness != "nocopies"):
                    made.append({"arity": len(ms), "over": [list(t.domain), ms], "atoms": atoms, "var": ren[var],
                                 "cmp": cmp, "k": k, "rules_so_far": held, "internal": len(internal),
                                 "rivals": len(r.rivals), "depth": 1,
                                 "_key": (len(ms), countcode.extension(tuples, n, cmp, k).tobytes())})
                    k_max = max(k_max, k)
    if search.count_relations:
        for t in model.targets:
            out.setdefault(t.symbol, []).extend(_relation_counts(c, model, t, search, comps))
    kc = "uniform" if search.count_witness == "uniform" else "gamma"
    for t in model.targets:
        zs = out.get(t.symbol, [])
        for z in zs:
            if z.get("relation"):
                continue                        # priced by its own description (SA-46)
            z["structure_bits"] = countcode.structure_bits(z["rules_so_far"], z["internal"], z["k"], kc, k_max) \
                + math.log2(z["rivals"])
        # CT-9: proposals with equal extensions on training are one hypothesis (N16), across the target's rules and
        # rivals; the one with the shortest description is kept, the first of those on a tie
        best: dict = {}
        for i, z in enumerate(zs):
            key = z.pop("_key")
            if key not in best or z["structure_bits"] < zs[best[key]]["structure_bits"]:
                best[key] = i
        kept = [zs[i] for i in sorted(best.values())]
        for j, z in enumerate(kept):
            z["symbol"] = f"#{t.symbol}.k{j}"       # .k: counts; .n: named pieces (§5.2), never the same name
        out[t.symbol] = kept
    return out


def _relation_counts(c: Corpus, model: Model, t: Target, search: Search, comps) -> list:
    """Count symbols over patterns of relations, not only a held rule's (SA-46; SMALLEST_AGENT §9aw, §9ax): patterns
    of up to `search.count_relations` atoms of the primitive symbols the target may cite, anchored at one exported
    variable (renumbered 0); every internal variable counted, every k the evidence shows. Each is priced by its own
    description, S(pattern) at the target's vocabulary + log2(internal variables) + 1 + gamma(k). Nothing reads a
    name: symbols are taken in sorted order and only their arities are used."""
    vocab = _vocabulary(c, t, {t.symbol, *t.hide} | _derived_from(model, t.symbol), model)
    prims = sorted((sym, a) for sym, a in vocab.items() if sym in c._given and a >= 1)
    size = len(vocab)
    pats = []
    for sym, a in prims:
        if a < 2:
            continue
        for pos in range(a):
            args = [0 if i == pos else 1 + i - (i > pos) for i in range(a)]
            pats.append(((sym, tuple(args)),))
    if search.count_relations >= 2:
        deeper = []
        for (first,) in pats:
            used = sorted({v for v in first[1] if v != 0})
            top = max(first[1]) + 1
            for share in used:
                for sym, a in prims:
                    if a == 1:
                        deeper.append((first, (sym, (share,))))
                        continue
                    for pos in range(a):
                        args, nxt = [], top
                        for i in range(a):
                            if i == pos:
                                args.append(share)
                            else:
                                args.append(nxt)
                                nxt += 1
                        deeper.append((first, (sym, tuple(args))))
        pats += deeper
    out = []
    for atoms in pats:
        internal = countcode.internal_vars(atoms, 1)
        for var in internal:
            tuples, n = countcode.counts_of(c, atoms, 1, var)
            if not len(n):
                continue
            for cmp, k in countcode.proposals(tuples, n, comps, search.count_witness != "nocopies"):
                bits = code.description_bits(atoms, 1, size) + math.log2(len(internal)) + 1 + countcode.gamma_bits(k)
                out.append({"arity": 1, "over": [list(t.domain), [0]], "atoms": atoms, "var": var, "cmp": cmp,
                            "k": k, "internal": len(internal), "depth": 1, "relation": True,
                            "structure_bits": bits,
                            "_key": (1, countcode.extension(tuples, n, cmp, k).tobytes())})
    return out


def _define_count(c: Corpus, z: dict) -> None:
    tuples, n = countcode.counts_of(c, z["atoms"], z["arity"], z["var"])
    ext = countcode.extension(tuples, n, z["cmp"], z["k"])
    if z["arity"] == 0:
        c.add_nullary(z["symbol"], ext[:, 0] if len(ext) else [])
    elif len(ext):
        c.add(z["symbol"], ext)


def _count_members(c: Corpus, z: dict) -> int:
    tuples, n = countcode.counts_of(c, z["atoms"], z["arity"], z["var"])
    return len(countcode.extension(tuples, n, z["cmp"], z["k"]))


def _symbols_of(model: Model) -> list:
    return list(_definitions(model))


def _model_bits(model: Model) -> float:
    """The total code of a model: every target's D, its rules' and named pieces' S, and every cell symbol's S."""
    return sum(model.train_bits[t.symbol] + sum(r.structure_bits for r in model.rules[t.symbol])
               + sum(i["structure_bits"] for i in model.inventions.get(t.symbol, [])) for t in model.targets) \
        + sum(z["structure_bits"] for zs in list(model.cells.values()) + list(model.counts.values()) for z in zs)


def _drop_uncited_cells(c: Corpus, model: Model, search: Search) -> list:
    """Cell symbols no rule cites, directly or through a named piece or a cited cell, leave Σ; every target that had
    them in its vocabulary is re-priced at the smaller one (rules and named pieces). Returns what was dropped."""
    defs = _definitions(model)
    cellsyms = {z["symbol"] for zs in list(model.cells.values()) + list(model.counts.values()) for z in zs}
    keep, todo = set(), [sym for rs in model.rules.values() for r in rs for p in r.rivals for sym, _ in p]
    todo += [sym for invs in model.inventions.values() for i in invs for sym, _ in i["atoms"]]
    while todo:
        sym = todo.pop()
        if sym in cellsyms and sym not in keep:
            keep.add(sym)
            todo += [s for s, _ in defs[sym]]
    dropped = sorted(cellsyms - keep)
    for sym in dropped:
        c.remove(sym)
    for src in model.cells:
        model.cells[src] = [z for z in model.cells[src] if z["symbol"] in keep]
    for src in model.counts:
        model.counts[src] = [z for z in model.counts[src] if z["symbol"] in keep]
    for t in model.targets:
        at = model.priced_at.get(t.symbol)
        d = len(set(at["cells"]) & set(dropped)) if at else 0
        if not d:
            continue
        at["size"] -= d
        for r in model.rules[t.symbol]:
            r.structure_bits = min(code.structure_bits(p, t.arity, at["size"], search.structure_code)
                                   for p in r.rivals)
        for i in model.inventions.get(t.symbol, []):
            i["vocabulary"] -= d
            i["structure_bits"] = code.structure_bits(i["atoms"], i["arity"], i["vocabulary"], search.structure_code)
    return dropped


def _learn(c: Corpus, targets: list[Target], search: Search, proposals: dict | None, start: dict | None,
           offer: bool, counts: dict | None = None) -> tuple[Model, Corpus]:
    """Form rules for each target in turn. Returns the model and the evidence (a Corpus) with every defined symbol.

    `proposals`: {target: [rule set, ...]}, each rule set a list of patterns (atom tuples), from anywhere: an earlier
    formation, another run, another partition. After the search, each is scored by the same code (S at the target's
    vocabulary, D by its cells, retired at the margin like a searched set), and the shortest of the searched set and
    the proposals is held (soundness principle 5: any search may propose; the code decides). A proposal citing a
    symbol the target may not cite is refused, and the refusal is recorded.

    `start`: {target: [Rule, ...]}, rules already held (an earlier formation's, in admission order): continuing
    formation (W1 G2, revision within a trial). They are held first, each priced on this evidence and keeping the
    symbol it defined, then the search continues from them by the one law; at the end every rule, held or new, is
    retired at the margin like any. A held rule citing a symbol the target may not cite here is dropped."""
    model = Model(targets=list(targets), data_code=search.data_code)
    later = search.count_witness == "later"
    for t in targets:
        if counts and not later:                    # a target's counts are offered to it and the targets after it
            for z in counts.get(t.symbol, []):
                _define_count(c, z)
                model.counts.setdefault(t.symbol, []).append(dict(z, members=int(_count_members(c, z))))
        y = outcomes(c, t)
        base = _vocabulary(c, t, {t.symbol, *t.hide} | _derived_from(model, t.symbol), model)
        model.compute[t.symbol], model.truncated[t.symbol], model.bridged[t.symbol] = 0, 0, 0
        model.pruned[t.symbol], model.beam_cuts[t.symbol] = 0, 0
        model.dropped[t.symbol], model.reruns[t.symbol] = 0, 0
        rules, cols = _form(c, t, y, model, search, (start or {}).get(t.symbol, []))
        if proposals and proposals.get(t.symbol):
            rules, cols = _proposed(c, t, y, base, rules, cols, proposals[t.symbol], search, model)
            model.rules[t.symbol] = rules
        if search.invent:
            _name(c, t, rules, search, model)
            rules, cols = _reform(c, t, y, model, search, rules, cols)
        cells = code.cell_ids(cols, len(y))
        model.train_counts[t.symbol] = code.counts(cells, y)
        model.train_bits[t.symbol] = code.target_bits(cols, y, search.data_code)
        vocab = _vocabulary(c, t, {t.symbol, *t.hide} | _derived_from(model, t.symbol), model)
        offered = {z["symbol"] for zs in list(model.cells.values()) + list(model.counts.values()) for z in zs}
        model.priced_at[t.symbol] = {"size": len(vocab), "cells": sorted(offered & set(vocab))}
        if offer and t is not targets[-1]:
            _offer_cells(c, t, rules, cols, model, search)
        if counts and later and t is not targets[-1]:   # the "later" witness: offered to the targets after it only
            for z in counts.get(t.symbol, []):
                _define_count(c, z)
                model.counts.setdefault(t.symbol, []).append(dict(z, members=int(_count_members(c, z))))
    return model, c


def _target_bits(model: Model, t: Target, rules, cols, y) -> float:
    """The target's share of L: D, its rules' S, and the S of the pieces named from its rules (§5.2)."""
    return _total(rules, cols, y, model.data_code) + sum(i["structure_bits"] for i in model.inventions.get(t.symbol, []))


def _cited(model: Model, t: Target, rules) -> set:
    """The target's named pieces its rules cite, directly or through another of its named pieces."""
    defs = {i["symbol"]: i["atoms"] for i in model.inventions.get(t.symbol, [])}
    out, todo = set(), [sym for r in rules for p in r.rivals for sym, _ in p if sym in defs]
    while todo:
        sym = todo.pop()
        if sym not in out:
            out.add(sym)
            todo += [s for s, _ in defs[sym] if s in defs]
    return out


def _reform(c: Corpus, t: Target, y: np.ndarray, model: Model, search: Search, rules, cols) -> tuple[list, list]:
    """Re-form the target with its named pieces in Σ (§5.2): its rules are formed again from nothing, so a named piece
    is one atom to the search and rules beyond max_atoms in primitives come within reach; then naming runs again, and
    named pieces no rule cites leave Σ. The re-formation is kept when the target's total code is shorter, and the
    earlier rules are restored otherwise. At most `search.reform` times, while each one names something new."""
    log = model.reformed.setdefault(t.symbol, [])
    for _ in range(search.reform):
        made = model.inventions.get(t.symbol, [])
        if not made:
            return rules, cols
        before = _target_bits(model, t, rules, cols, y)
        held = (rules, cols, [dict(i) for i in made], model.stops.get(t.symbol), model.naming_stops.get(t.symbol))
        names_before = {i["symbol"] for i in made}
        for r in rules:
            if r.symbol:
                c.remove(r.symbol)
        new_rules, new_cols = _form(c, t, y, model, search, [])
        _name(c, t, new_rules, search, model)
        keep = _cited(model, t, new_rules)
        dropped = [i["symbol"] for i in model.inventions[t.symbol] if i["symbol"] not in keep]
        for sym in dropped:
            c.remove(sym)
        model.inventions[t.symbol] = [i for i in model.inventions[t.symbol] if i["symbol"] in keep]
        if dropped:                            # a smaller Σ: the rules cost less
            vocab = _vocabulary(c, t, {t.symbol, *t.hide} | _derived_from(model, t.symbol), model)
            for r in new_rules:
                r.structure_bits = min(code.structure_bits(p, t.arity, len(vocab), search.structure_code)
                                       for p in r.rivals)
        after = _target_bits(model, t, new_rules, new_cols, y)
        kept = after < before - TIE_BITS
        log.append({"before": before, "after": after, "kept": kept, "dropped": dropped})
        if not kept:                           # restore: the earlier rules, their symbols and the earlier named pieces
            for r in new_rules:
                if r.symbol:
                    c.remove(r.symbol)
            for i in model.inventions[t.symbol]:
                if i["symbol"] not in names_before:
                    c.remove(i["symbol"])
            rules, cols, invs, stop, nstop = held
            for i in invs:
                if i["symbol"] in dropped:
                    _define_symbol(c, i["atoms"], i["arity"], i["symbol"])
            model.inventions[t.symbol] = invs
            model.rules[t.symbol] = rules
            model.stops[t.symbol], model.naming_stops[t.symbol] = stop, nstop
            for r in rules:
                if r.symbol:
                    _define(c, t, r.atoms, r.symbol)
            return rules, cols
        rules, cols = new_rules, new_cols
        if not {i["symbol"] for i in model.inventions[t.symbol]} - names_before:
            return rules, cols                 # nothing new was named: another re-formation would find the same
    return rules, cols


def _total(rules, cols, y, kind: str = "cells") -> float:
    return code.target_bits(cols, y, kind) + sum(r.structure_bits for r in rules)


def _proposed(c: Corpus, t: Target, y, base: dict, rules, cols, offered, search: Search, model: Model):
    """The shortest of the searched rules and each offered rule set (see `learn`)."""
    kind = search.data_code
    record = {"searched_bits": _total(rules, cols, y, kind), "proposals": [], "held": "searched"}
    best, best_bits = (rules, cols), record["searched_bits"]
    for k, pats in enumerate(offered):
        pats = [tuple((s, tuple(a)) for s, a in p) for p in pats]
        bad = sorted({s for p in pats for s, a in p if base.get(s) != len(a)})
        if bad:
            record["proposals"].append({"refused": bad})
            continue
        prules, pcols, done = [], [], code.target_bits([], y, kind)
        for p in pats:
            col = column(c, t, p).astype(np.int64)
            s_bits = code.structure_bits(p, t.arity, len(base), search.structure_code)
            after = code.target_bits(pcols + [col], y, kind)
            prules.append(Rule(t.symbol, p, [p], done - after - s_bits, s_bits))
            pcols.append(col)
            done = after
        _retire(prules, pcols, y, kind)
        bits = _total(prules, pcols, y, kind)
        record["proposals"].append({"bits": bits, "rules": len(prules)})
        if bits < best_bits - TIE_BITS:
            best, best_bits, record["held"] = (prules, pcols), bits, k
    model.proposed[t.symbol] = record
    if record["held"] == "searched":
        return rules, cols
    for r in rules:
        if r.symbol:
            c.remove(r.symbol)
    prules, pcols = best
    for i, r in enumerate(prules):
        if len(mentioned(r.atoms, t.arity)) == t.arity:
            r.symbol = f"#{t.symbol}/{i}"
            _define(c, t, r.atoms, r.symbol)
    return prules, pcols


def _retire(rules: list, cols: list, y: np.ndarray, kind: str = "cells") -> None:
    """Once formation stops, drop any rule that no longer clears the margin given all the others (KERNEL §5), the
    weakest first, until every held rule does. A rule another rule cites is kept."""
    cited = {sym for r in rules for sym, _ in r.atoms}
    while True:
        full = code.target_bits(cols, y, kind)
        worst, worst_g = None, None
        for i, r in enumerate(rules):
            if r.symbol and r.symbol in cited:
                continue
            rest = cols[:i] + cols[i + 1:]
            g = code.target_bits(rest, y, kind) - full - r.structure_bits
            if g <= MARGIN_BITS and (worst_g is None or g < worst_g):
                worst, worst_g = i, g
        if worst is None:
            return
        rules.pop(worst)
        cols.pop(worst)


# --- using what was formed --------------------------------------------------------------------------------------------

def apply(model: Model, evidence) -> Corpus:
    """New evidence with every defined symbol added, in the order the symbols were formed."""
    c = Corpus(list(evidence)) if not isinstance(evidence, Corpus) else evidence
    for t in model.targets:
        for z in model.counts.get(t.symbol, []):
            _define_count(c, z)
        for inv in model.inventions.get(t.symbol, []):
            _define_symbol(c, inv["atoms"], inv["arity"], inv["symbol"])
        for r in model.rules.get(t.symbol, []):
            if r.symbol:
                _define(c, t, r.atoms, r.symbol)
        for z in model.cells.get(t.symbol, []):
            _define_cell(c, t, z)
    return c


def rival_column(evidence, t: Target, rivals) -> tuple[np.ndarray, np.ndarray]:
    """A rule's column where its rivals agree, and where they do not. Rivals agree everywhere on the evidence that
    formed them; on new evidence the column takes the value all of them give (AND), a choice that reads no name, and
    the disagreements are returned for prediction to abstain on."""
    cols = [column(evidence, t, p) for p in rivals]
    both = np.logical_and.reduce(cols)
    return both, np.logical_or.reduce(cols) & ~both


def cells_for(model: Model, c: Corpus, t: Target) -> tuple[np.ndarray, np.ndarray]:
    n = len(outcomes(c, t))
    cols, split = [], np.zeros(n, dtype=bool)
    for r in model.rules.get(t.symbol, []):
        col, d = rival_column(c, t, r.rivals)
        cols.append(col.astype(np.int64))
        split |= d
    return code.cell_ids(cols, n), split


def heldout_bits(model: Model, evidence, t: Target) -> float:
    """Code length of the target on new evidence, rules frozen, KT counts continued from training (§8.2)."""
    c = apply(model, evidence)
    return code.continued_bits(model.train_counts[t.symbol], cells_for(model, c, t)[0], outcomes(c, t), _depth_of(model, t))


def _depth_of(model: Model, t: Target) -> int | None:
    """The number of held rules, the ctw tree's depth; None under the cells code."""
    return len(model.rules.get(t.symbol, [])) if model.data_code == "ctw" else None


def predict(model: Model, structures: list[Structure], t: Target) -> list[np.ndarray]:
    """Per structure, at each candidate: 1 (holds), 0 (does not) or -1 (abstain: an unseen cell, an even split, or
    rivals that disagree), from the training counts of its cell."""
    c = apply(model, structures)
    cells, split = cells_for(model, c, t)
    tc, depth = model.train_counts[t.symbol], _depth_of(model, t)
    starts = c.rows(t.arity, t.domain)["start"]
    p1 = {int(u): code.predictive(tc, int(u), depth) for u in np.unique(cells)}
    out = []
    for i in range(len(structures)):
        part = []
        for r in range(starts[i], starts[i + 1]):
            p = p1[int(cells[r])]
            part.append(-1 if split[r] or abs(p - 0.5) < 1e-12 else int(p > 0.5))
        out.append(np.array(part, dtype=np.int64))
    return out


def summary(model: Model) -> dict:
    return {t.symbol: {"rules": [{"atoms": [[s, list(a)] for s, a in r.atoms], "rivals": len(r.rivals),
                                  "gain_bits": round(r.gain_bits, 3), "structure_bits": round(r.structure_bits, 3),
                                  "symbol": r.symbol, "depth": r.depth} for r in model.rules[t.symbol]],
                       "train_bits": round(model.train_bits[t.symbol], 3), "stop": model.stops.get(t.symbol),
                       "patterns_evaluated": model.compute[t.symbol],
                       "frontier_cuts": model.truncated[t.symbol],
                       "bridged_rules": model.bridged.get(t.symbol, 0),
                       "beam_cuts": model.beam_cuts.get(t.symbol, 0),
                       "frontier_dropped": model.dropped.get(t.symbol, 0),
                       "exact_reruns": model.reruns.get(t.symbol, 0),
                       "inventions": [{"symbol": i["symbol"], "atoms": [[s, list(a)] for s, a in i["atoms"]],
                                       "arity": i["arity"], "gain_bits": round(i["gain_bits"], 3),
                                       "structure_bits": round(i["structure_bits"], 3), "uses": i["uses"],
                                       "depth": i["depth"]} for i in model.inventions.get(t.symbol, [])],
                       "reformed": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}
                                    for e in model.reformed.get(t.symbol, [])],
                       "cells": [{"symbol": z["symbol"], "arity": z["arity"], "members": z["members"],
                                  "structure_bits": round(z["structure_bits"], 3), "depth": z["depth"],
                                  "definitions": [[{"sign": lit["sign"],
                                                    "atoms": [[s_, list(a)] for s_, a in lit["rivals"][0]]}
                                                   for lit in d] for d in z["definitions"]]}
                                 for z in model.cells.get(t.symbol, [])],
                       "counts": [{"symbol": z["symbol"], "arity": z["arity"], "cmp": z["cmp"], "k": z["k"],
                                   "var": z["var"], "atoms": [[s_, list(a)] for s_, a in z["atoms"]],
                                   "members": z["members"], "structure_bits": round(z["structure_bits"], 3)}
                                  for z in model.counts.get(t.symbol, [])]}
            for t in model.targets}


def bits_each(model: Model, structures, t: Target) -> np.ndarray:
    """Per structure, the code length of the target under the frozen model: each candidate coded by its cell's
    training estimate, (k + 1/2) / (n + 1). This is D(c | u) for a speaker scoring utterances (KERNEL §8.3)."""
    c = apply(model, structures)
    cells, _ = cells_for(model, c, t)
    y = outcomes(c, t)
    tc, depth = model.train_counts[t.symbol], _depth_of(model, t)
    u, inv = np.unique(cells, return_inverse=True)
    p1 = np.array([code.predictive(tc, int(x), depth) for x in u], dtype=float)[inv]
    bits = -np.log2(np.where(y == 1, p1, 1.0 - p1))
    starts = c.rows(t.arity, t.domain)["start"]
    # Empty candidate domains have zero cost, including between nonempty
    # structures. reduceat treats repeated boundaries as one row and rejects a
    # trailing boundary at len(bits). Keep the original summation for nonempty
    # segments and explicitly leave empty ones at zero.
    out = np.zeros(c.size)
    nonempty = np.diff(starts) > 0
    if len(bits):
        out[nonempty] = np.add.reduceat(bits, starts[:-1][nonempty])
    return out


# --- saving what was formed -------------------------------------------------------------------------------------------

MODEL_FORMAT = "DARA-KERNEL-MODEL/0.1"


def to_record(model: Model) -> dict:
    """The model as plain data: targets, rules (with rivals and the symbols they define), each target's training
    counts per cell, and what formation reported. Everything `apply`, `predict`, `heldout_bits` and `bits_each` need;
    nothing about the evidence it was formed on."""
    atoms = lambda a: [[s, list(v)] for s, v in a]                                    # noqa: E731
    return {"format": MODEL_FORMAT,
            "targets": [{"symbol": t.symbol, "arity": t.arity, "domain": list(t.domain),
                         "body": None if t.body is None else list(t.body), "hide": list(t.hide)}
                        for t in model.targets],
            "rules": {k: [{"atoms": atoms(r.atoms), "rivals": [atoms(p) for p in r.rivals], "gain_bits": r.gain_bits,
                           "structure_bits": r.structure_bits, "symbol": r.symbol, "depth": r.depth} for r in rs]
                      for k, rs in model.rules.items()},
            "train_counts": {k: [[c, n, kk] for c, (n, kk) in sorted(v.items())]
                             for k, v in model.train_counts.items()},
            "train_bits": model.train_bits, "stops": model.stops, "compute": model.compute,
            "truncated": model.truncated, "cache": model.cache, "bridged": model.bridged,
            "beam_cuts": model.beam_cuts, "dropped": model.dropped, "reruns": model.reruns,
            "inventions": {k: [dict(i, atoms=atoms(i["atoms"])) for i in v] for k, v in model.inventions.items()},
            "reformed": model.reformed,
            "cells": {k: [dict(z, definitions=[[{"sign": lit["sign"], "rivals": [atoms(p) for p in lit["rivals"]]}
                                                for lit in d] for d in z["definitions"]]) for z in v]
                      for k, v in model.cells.items()},
            "cells_record": model.cells_record,
            "data_code": model.data_code,
            "counts": {k: [dict(z, atoms=atoms(z["atoms"])) for z in v] for k, v in model.counts.items()},
            "counts_record": model.counts_record}


def from_record(rec: dict) -> Model:
    if rec.get("format") != MODEL_FORMAT:
        raise ValueError(f"not a {MODEL_FORMAT} record")
    atoms = lambda a: tuple((s, tuple(v)) for s, v in a)                              # noqa: E731
    m = Model(targets=[Target(t["symbol"], t["arity"], tuple(t["domain"]),
                              None if t["body"] is None else tuple(t["body"]), tuple(t["hide"]))
                       for t in rec["targets"]])
    m.rules = {k: [Rule(k, atoms(r["atoms"]), [atoms(p) for p in r["rivals"]], r["gain_bits"], r["structure_bits"],
                        r["symbol"], r["depth"]) for r in rs] for k, rs in rec["rules"].items()}
    m.train_counts = {k: {int(c): (int(n), int(kk)) for c, n, kk in v} for k, v in rec["train_counts"].items()}
    m.train_bits, m.stops, m.compute = dict(rec["train_bits"]), dict(rec["stops"]), dict(rec["compute"])
    m.truncated, m.cache = dict(rec.get("truncated", {})), dict(rec.get("cache", {}))
    m.bridged, m.beam_cuts = dict(rec.get("bridged", {})), dict(rec.get("beam_cuts", {}))
    m.dropped, m.reruns = dict(rec.get("dropped", {})), dict(rec.get("reruns", {}))
    m.inventions = {k: [dict(i, atoms=atoms(i["atoms"])) for i in v] for k, v in rec.get("inventions", {}).items()}
    m.reformed = {k: [dict(e) for e in v] for k, v in rec.get("reformed", {}).items()}
    m.cells = {k: [dict(z, definitions=[[{"sign": lit["sign"], "rivals": [atoms(p) for p in lit["rivals"]]}
                                         for lit in d] for d in z["definitions"]]) for z in v]
               for k, v in rec.get("cells", {}).items()}
    m.cells_record = dict(rec.get("cells_record", {}))
    m.data_code = rec.get("data_code", "cells")
    m.counts = {k: [dict(z, atoms=atoms(z["atoms"])) for z in v] for k, v in rec.get("counts", {}).items()}
    m.counts_record = dict(rec.get("counts_record", {}))
    return m


def save(model: Model, path) -> None:
    import json
    with open(path, "w") as f:
        json.dump(to_record(model), f, indent=1)


def load(path) -> Model:
    import json
    with open(path) as f:
        return from_record(json.load(f))
