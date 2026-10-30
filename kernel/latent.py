"""Latent symbols: a class nobody named, admitted when positing it shortens the code (docs/RECURSIVE_ABSTRACTION_v0_1.md).

Every other kernel symbol is observed or defined by a pattern over observed facts. A latent class Z is assigned: a set
of identities (the names of entities that persist across structures, such as a conversation's entity ids or a world's
objects) of which Z holds, in every structure where they occur. Its cost is the code of the assignment; its value is
what the rules citing it save across every target (GOAL_DIRECTED_KERNEL_DIRECTION §4):

    G(Z) = code(targets | without Z) - [code(targets | with Z) + L(assignment of Z)]

where code(targets | ...) is the kernel's two-part code, D plus every held rule's S, from `learn`, and L is the KT
code of which of the N identities are members. Z is admitted when G(Z) > MARGIN_BITS, as any rule is.

The assignment is searched by alternation, a declared heuristic: seed it per target where the target is concentrated
(the identities whose evidence holds more of it than its overall rate) and where the rules without Z leave the most
surprise (more of it than those rules expect), form rules with
Z, reassign every identity to whichever side codes the evidence shorter with those rules held fixed, and repeat until
an assignment recurs. Then a polish: every single flip judged with the rules re-formed, the best kept, until none
shortens the code (an identity that a held rule already covers, by memory, looks the same on either side while those
rules are held fixed). Every proposal is judged by the same fixed code, so the search can be wrong only by missing the
best class, never by admitting one that does not pay. Nothing reads a name: identities enter as opaque strings, in
sorted order, and seeds and reassignment read only counts and code lengths.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from kernel import code
from kernel import learn as kl
from kernel.structure import Structure


@dataclass(frozen=True)
class Latent:
    name: str = "Z"
    domain: tuple[str, ...] = ("ent",)     # the channel whose entities are identities; Z(e) holds for members
    max_rounds: int = 12                   # alternation rounds per seed
    max_seeds: int = 16
    polish_sweeps: int = 4                 # after alternation: single flips judged with the rules re-formed (0: none)
    polish_candidates: int = 8             # per sweep, only the identities whose flip costs least with the rules held
                                           # (ties at the cut all included); 0: every identity
    warm: bool = True                      # during the search, each kernel run continues from the rules the last one
                                           # held (learn's `start`) instead of searching from nothing; the class kept
                                           # is judged by one fresh run, so admission reads the same code either way
    settle: int = 8                        # reassignment passes with the rules held (cheap) before the next kernel
                                           # run, until no identity moves; 1: one pass per run, as v0.1


@dataclass
class Invention:
    members: frozenset
    identities: tuple
    gain_bits: float                       # G(Z): baseline code minus code with Z and its assignment
    admitted: bool
    baseline_bits: float
    bits: float                            # code with Z, its assignment included
    assignment_bits: float
    model: kl.Model | None                 # the model formed with Z (the best class's), or the baseline's
    baseline: kl.Model
    trace: list = field(default_factory=list)    # per seed and round: members, bits
    learns: int = 0                        # kernel runs the search cost (compute, reported)


def identities(structures: list[Structure], domain) -> tuple:
    return tuple(sorted({e for s in structures for e in s.entities(domain)}))


def with_class(structures: list[Structure], members, name: str, domain) -> list[Structure]:
    """The structures with Z(e) for every entity of the domain whose identity is a member."""
    members = set(members)
    return [s.with_facts({name: {(e,) for e in s.entities(domain) if e in members}}) for s in structures]


def assignment_bits(n: int, k: int) -> float:
    """L(Z): which k of n identities are members, KT-coded (a prefix code fixed before the data)."""
    return code.kt_bits(n, k)


def code_bits(model: kl.Model) -> float:
    """The two-part code of every target: D plus every held rule's S, the S of every piece named (KERNEL §5.2) and of
    every cell symbol held (§5.3)."""
    return kl._model_bits(model)


class _Recoder:
    """The targets' data code with the rules held as formed, for Z reassigned one identity at a time. Only the
    structures where that identity occurs are recoded (the corpus never joins structures, so no other row changes),
    and the code changes only in the cells their rows move between: the same numbers as recoding everything."""

    def __init__(self, model: kl.Model, structures, members, latent: Latent, where: dict):
        self.model, self.structures, self.latent, self.where = model, structures, latent, where
        self.members = frozenset(members)
        c = kl.apply(model, with_class(structures, self.members, latent.name, latent.domain))
        self.cells, self.y, self.start, self.counts = {}, {}, {}, {}
        for t in model.targets:
            self.cells[t.symbol] = kl.cells_for(model, c, t)[0]
            self.y[t.symbol] = kl.outcomes(c, t)
            self.start[t.symbol] = c.rows(t.arity, t.domain)["start"]
            self.counts[t.symbol] = code.counts(self.cells[t.symbol], self.y[t.symbol])

    def flip_bits(self, e) -> float:
        """How much the data code changes when e changes side (negative: shorter)."""
        idx = self.where.get(e, [])
        if not idx:
            return 0.0
        sub = with_class([self.structures[i] for i in idx], self.members ^ {e}, self.latent.name, self.latent.domain)
        c = kl.apply(self.model, sub)
        delta = 0.0
        for t in self.model.targets:
            st = self.start[t.symbol]
            rows = np.concatenate([np.arange(st[i], st[i + 1]) for i in idx])
            old, y = self.cells[t.symbol][rows], self.y[t.symbol][rows]
            new = kl.cells_for(self.model, c, t)[0]
            if np.array_equal(old, new):
                continue
            moved: dict = {}
            for cell, sign in ((old, -1), (new, 1)):
                for x, (n, k) in code.counts(cell, y).items():
                    a, b = moved.get(x, (0, 0))
                    moved[x] = (a + sign * n, b + sign * k)
            counts = self.counts[t.symbol]
            for x, (dn, dk) in moved.items():
                if dn or dk:
                    n, k = counts.get(x, (0, 0))
                    delta += code.kt_bits(n + dn, k + dk) - code.kt_bits(n, k)
        return delta


def _boundary(model, structures, members, ids, latent: Latent, where: dict) -> list:
    """The identities the polish re-forms rules for: those whose flip costs least with the rules held fixed, the
    cheapest `polish_candidates` and any tied with the last (a cut no name decides). Every identity if 0."""
    if not latent.polish_candidates or latent.polish_candidates >= len(ids):
        return list(ids)
    rec = _Recoder(model, structures, members, latent, where)
    cost = {e: rec.flip_bits(e) + assignment_bits(len(ids), len(members ^ {e})) for e in ids}
    cut = sorted(cost.values())[latent.polish_candidates - 1]
    return [e for e in ids if cost[e] <= cut + kl.TIE_BITS]


def _says_nothing_new(structures, members, ids, latent: Latent, targets=()) -> bool:
    """A class of every identity or of none divides nothing, and a class whose facts are exactly an observed unary
    symbol's (on the domain's entities, in every structure) is that symbol under a new name (I-5): neither is a
    hypothesis the evidence can credit, whatever a search finds with it. The targets being coded are excepted
    (SMALLEST_AGENT §9w, SA-28): a rule for a target may not cite it, so a class equal to a target's extension says
    what no rule can, that the same identities hold it in every structure."""
    if not members or len(members) == len(ids):
        return True
    held = {(i, e) for i, s in enumerate(structures) for e in s.entities(latent.domain) if e in members}
    coded = {t.symbol for t in targets}
    syms = {sym for s in structures for sym, ts in s.facts.items()
            if ts and len(next(iter(ts))) == 1 and sym not in coded}
    for sym in syms:
        got = {(i, e) for i, s in enumerate(structures) for (e,) in s.facts.get(sym, ())
               if e in s.entities(latent.domain)}
        if got == held:
            return True
    return False


def _where(structures, domain) -> dict:
    """Per identity, the structures where it occurs."""
    out: dict = {}
    for i, s in enumerate(structures):
        for e in s.entities(domain):
            out.setdefault(e, []).append(i)
    return out


def _residuals(model: kl.Model, structures, ids: tuple, domain, rules: bool = True) -> dict:
    """Per target, per identity: how much more of the target its evidence holds than expected, sum(y - p): expected
    by the model's rules, or (rules=False) by the target's overall rate. A candidate that is no identity (a frame)
    credits the identities one fact away from it in its structure (the things in its roles). A candidate with no
    identity in its roles or one hop from them (a 0-ary target, or a fixture's whose neighbours are fixtures and
    places) credits the identities the structure *involves*: those standing in more facts there than the least
    involved identity does, which are the things the step did something to; a step that involves none credits none.
    (Seeding only, a declared heuristic: the code decides every class the search then proposes.)"""
    c = kl.apply(model, structures)
    index = {e: k for k, e in enumerate(ids)}
    out = {}
    for t in model.targets:
        cells, _ = kl.cells_for(model, c, t)
        y = kl.outcomes(c, t)
        if rules:
            tc = model.train_counts[t.symbol]
            u, inv = np.unique(cells, return_inverse=True)
            nk = np.array([tc.get(int(x), (0, 0)) for x in u], dtype=float).reshape(-1, 2)
            r = y - ((nk[:, 1] + 0.5) / (nk[:, 0] + 1.0))[inv]
        else:
            r = y - y.mean() if len(y) else y.astype(float)
        start = c.rows(t.arity, t.domain)["start"]
        res = np.zeros(len(ids))
        for i, s in enumerate(structures):
            here = [index[e] for e in s.entities(domain) if e in index]
            near: dict = {}                             # one hop along any fact: a frame to the things in its roles
            count = {k: 0 for k in here}                # facts each identity stands in here: its involvement
            for ts in s.facts.values():
                for tup in ts:
                    for a in tup:
                        near.setdefault(a, set()).update(index[b] for b in tup if b != a and b in index)
                        if a in index and index[a] in count:
                            count[index[a]] += 1
            least = min(count.values()) if count else 0
            involved = {k for k, c in count.items() if c > least}
            for j, cand in enumerate(s.candidates(t.arity, t.domain)):
                own = {index[e] for e in cand if e in index}
                linked = (own or set().union(*(near.get(e, set()) for e in cand))) if cand else set()
                for k in (linked or involved):
                    res[k] += r[start[i] + j]
        out[t.symbol] = res
    return out


def _start(model: kl.Model | None, latent: Latent) -> dict | None:
    """The rules a warm kernel run continues from: the last run's (learn drops any citing a symbol not in Σ now)."""
    return {t: list(rs) for t, rs in model.rules.items()} if (latent.warm and model is not None) else None


def _settle(model: kl.Model, structures, members: frozenset, ids, latent: Latent, where: dict) -> frozenset:
    """Reassign every identity to the side that codes its evidence shorter with `model`'s rules held, and again on the
    result, until no identity moves or `latent.settle` passes: the cheap half of the alternation, run to a fixed
    point before the next kernel run pays for re-forming the rules."""
    seen = {members}
    for _ in range(max(1, latent.settle)):
        nxt, rec = set(members), _Recoder(model, structures, members, latent, where)
        here = assignment_bits(len(ids), len(members))
        for e in ids:                                   # each identity to the side that codes shorter, others fixed
            there = rec.flip_bits(e) + assignment_bits(len(ids), len(members ^ {e}))
            inside, outside = (here, there) if e in members else (there, here)
            (nxt.add if inside < outside else nxt.discard)(e)
        nxt = frozenset(nxt)
        if nxt in seen:                                 # a fixed point, or a cycle the next kernel run must break
            return nxt
        seen.add(nxt)
        members = nxt
    return members


def _alternate(job, seen: set | None = None):
    """One seed's alternation (see the module docstring): returns its best (bits, members, model), its trace, and the
    kernel runs it cost. `seen`, shared across seeds run in one process, stops a path another seed already took."""
    structures, targets, search, latent, ids, where, n_seed, members = job
    seen = set() if seen is None else seen
    best, trace, learns, model = None, [], 0, None
    for rnd in range(latent.max_rounds):
        if members in seen:
            break
        seen.add(members)
        ev = with_class(structures, members, latent.name, latent.domain)
        model, _ = kl.learn(ev, targets, search, start=_start(model, latent))
        learns += 1
        bits = code_bits(model) + assignment_bits(len(ids), len(members))
        trace.append({"seed": n_seed, "round": rnd, "members": sorted(members), "bits": round(bits, 3)})
        if (best is None or bits < best[0] - 1e-9) and not _says_nothing_new(structures, members, ids, latent, targets):
            best = (bits, members, model)
        if not any(latent.name == sym for rs in model.rules.values() for r in rs for p in r.rivals for sym, _ in p):
            break                                       # no rule cites Z: reassigning it changes nothing
        members = _settle(model, structures, members, ids, latent, where)
    return best, trace, learns


def invent(structures: list[Structure], targets: list[kl.Target], search: kl.Search = kl.Search(),
           latent: Latent = Latent(), seeds: list | None = None, pmap=None) -> Invention:
    """Search for one latent class over the domain's identities; admit it if it pays (module docstring). `seeds`, if
    given, replaces the seeds computed from the evidence (revision re-fits a class from its own members). `pmap`, if
    given, maps the seeds' alternations (a process pool's `map`, supplied by the caller: the kernel imports none); the
    result is the same, taken in seed order."""
    structures = list(structures)
    ids = identities(structures, latent.domain)
    base, _ = kl.learn(structures, targets, search)
    base_bits = code_bits(base)
    learns = 1
    seeds = [frozenset(x) for x in seeds] if seeds is not None else []
    for rules in (() if seeds else (False, True)):     # where each target is concentrated; then where the rules without Z fall short
        for res in _residuals(base, structures, ids, latent.domain, rules).values():
            s = frozenset(e for e, r in zip(ids, res) if r > 0)
            if s and len(s) < len(ids) and s not in seeds:
                seeds.append(s)
    where = _where(structures, latent.domain)
    jobs = [(structures, targets, search, latent, ids, where, n, seed) for n, seed in enumerate(seeds[:latent.max_seeds])]
    if pmap is not None and len(jobs) > 1:
        runs = list(pmap(_alternate, jobs))
    else:
        seen: set = set()                               # an assignment any seed reached continues as it did there
        runs = [_alternate(job, seen) for job in jobs]
    best, trace = None, []
    for got, steps, n in runs:                          # in seed order: the first of equal codes is kept
        trace += steps
        learns += n
        if got is not None and (best is None or got[0] < best[0] - 1e-9):
            best = got
    if best is None:
        return Invention(frozenset(), ids, 0.0, False, base_bits, base_bits, 0.0, None, base, trace, learns)
    for sweep in range(latent.polish_sweeps):          # an identity a held rule already covers (memory) looks no
        bits, members, model = best                     # better on either side with the rules fixed; re-forming
        tried = []                                      # the rules after the flip shows whether it belongs
        for e in _boundary(model, structures, members, ids, latent, where):
            flip = members ^ {e}
            if _says_nothing_new(structures, flip, ids, latent, targets):
                continue
            m, _ = kl.learn(with_class(structures, flip, latent.name, latent.domain), targets, search,
                            start=_start(model, latent))
            learns += 1
            tried.append((code_bits(m) + assignment_bits(len(ids), len(flip)), flip, m))
        if not tried:
            break
        b = min(t[0] for t in tried)
        winners = [t for t in tried if t[0] <= b + kl.TIE_BITS]
        if b >= bits - 1e-9 or len(winners) > 1:        # nothing shorter, or a tie only a name could break: stop
            break
        best = winners[0]
        trace.append({"seed": "polish", "round": sweep, "members": sorted(best[1]), "bits": round(best[0], 3)})
    bits, members, model = best
    a = assignment_bits(len(ids), len(members))
    if latent.warm:                                     # judged by a fresh run: the same code as the baseline's
        model, _ = kl.learn(with_class(structures, members, latent.name, latent.domain), targets, search)
        learns += 1
        bits = code_bits(model) + a
        trace.append({"seed": "fresh", "round": 0, "members": sorted(members), "bits": round(bits, 3)})
    gain = base_bits - bits
    return Invention(members, ids, gain, gain > kl.MARGIN_BITS, base_bits, bits, a, model, base, trace, learns)


def place(inv: Invention, structures: list[Structure], identity: str, latent: Latent = Latent()) -> bool:
    """Whether a new identity belongs to Z: the side under which the admitted model codes its evidence shorter (the
    rules and the training counts held fixed). With no admitted class, never."""
    if not inv.admitted:
        return False
    seen = [s for s in structures if identity in s.entities(latent.domain)]

    def bits(members):
        ev = with_class(seen, members, latent.name, latent.domain)
        return sum(float(kl.bits_each(inv.model, ev, t).sum()) for t in inv.model.targets)
    return bits({identity}) < bits(set())



@dataclass
class Classes:
    classes: list                          # the admitted classes' members, in the order they now stand
    names: list                            # their symbols in the evidence: {latent.name}1, {latent.name}2, ...
    baseline_bits: float                   # the code with no class
    bits: float                            # the code with every class, their assignments included
    model: kl.Model                        # the model formed with every class
    evidence: list                         # the structures with every class
    attempts: list = field(default_factory=list)    # every invention tried, admitted or not
    revisions: list = field(default_factory=list)   # what revision did: retired and re-fitted classes
    learns: int = 0

    @property
    def gain_bits(self) -> float:
        return self.baseline_bits - self.bits


def _with_classes(structures, classes, latent: Latent):
    names = [f"{latent.name}{k}" for k in range(1, len(classes) + 1)]
    for name, members in zip(names, classes):
        structures = with_class(structures, members, name, latent.domain)
    return structures, names


def code_with(structures, classes, targets, search, latent: Latent) -> tuple[float, kl.Model]:
    """The code of every target with these classes in the evidence, each class's assignment included."""
    ev, _ = _with_classes(structures, classes, latent)
    model, _ = kl.learn(ev, targets, search)
    n = len(identities(structures, latent.domain))
    return code_bits(model) + sum(assignment_bits(n, len(m)) for m in classes), model


def revise(structures, classes, targets, search, latent: Latent, passes: int = 3, pmap=None) -> tuple[list, list, int]:
    """Classes are hypotheses like rules, and are revised like them. Retirement: a class whose removal costs no more
    than the margin, given every other class, is dropped, the weakest first (KERNEL §5). Re-fitting: each class is
    searched again from its own members with the others held, and replaced if the whole code shortens. A class
    found early is so judged against the classes found after it. Returns the classes, a log, and kernel runs."""
    classes, log, learns = [frozenset(c) for c in classes], [], 0
    cur, _ = code_with(structures, classes, targets, search, latent)
    learns += 1
    for _ in range(passes):
        changed = False
        while classes:
            costs = []
            for k in range(len(classes)):
                b, _ = code_with(structures, classes[:k] + classes[k + 1:], targets, search, latent)
                learns += 1
                costs.append((b - cur, k, b))
            cost, k, b = min(costs)
            if cost > kl.MARGIN_BITS:
                break
            log.append({"retired": sorted(classes[k]), "cost_bits": round(cost, 3)})
            classes, cur, changed = classes[:k] + classes[k + 1:], b, True
        for k in range(len(classes)):
            others = classes[:k] + classes[k + 1:]
            ev, _ = _with_classes(structures, others, latent)
            lat = replace(latent, name=f"{latent.name}{len(others) + 1}")
            inv = invent(ev, targets, search, lat, seeds=[classes[k]], pmap=pmap)
            learns += inv.learns
            if not inv.admitted or inv.members == classes[k]:
                continue
            b, _ = code_with(structures, others + [inv.members], targets, search, latent)
            learns += 1
            if b < cur - 1e-9:
                log.append({"refitted": sorted(classes[k]), "to": sorted(inv.members), "saving_bits": round(cur - b, 3)})
                classes, cur, changed = others + [inv.members], b, True
        if not changed:
            break
    return classes, log, learns



def invent_all(structures: list[Structure], targets: list[kl.Target], search: kl.Search = kl.Search(),
               latent: Latent = Latent(), max_classes: int = 8, revision: bool = True, pmap=None,
               start=()) -> Classes:
    """Recursion: invent a class, add it to the evidence as an ordinary symbol, and invent the next with it in the
    vocabulary (so the next class is judged against, and may be cited together with, the classes before it), until
    one does not pay; after each admission, revise every class held (`revise`). `start`: classes already held (from
    earlier evidence), revised first and then extended. Classes are named `{latent.name}1`, `{latent.name}2`, ... in
    the order they stand."""
    structures = list(structures)
    classes, attempts, log, learns = [frozenset(c) for c in start], [], [], 0
    base, _ = kl.learn(structures, targets, search)
    learns += 1
    if classes:
        classes, log, n = revise(structures, classes, targets, search, latent, pmap=pmap)
        learns += n
    while len(classes) < max_classes:
        ev, _ = _with_classes(structures, classes, latent)
        lat = replace(latent, name=f"{latent.name}{len(classes) + 1}")
        inv = invent(ev, targets, search, lat, pmap=pmap)
        attempts.append(inv)
        learns += inv.learns
        if not inv.admitted:
            break
        classes.append(inv.members)
        if revision:
            classes, more, n = revise(structures, classes, targets, search, latent, pmap=pmap)
            log += more
            learns += n
    bits, model = code_with(structures, classes, targets, search, latent)
    ev, names = _with_classes(structures, classes, latent)
    return Classes(classes, names, code_bits(base), bits, model, ev, attempts, log, learns + 1)


def cited(model: kl.Model, names) -> dict:
    """Per target, the classes (of `names`) its rules cite, directly or through a symbol a rule defined."""
    names = set(names)
    defs = kl._definitions(model)                       # rules' symbols and named pieces

    def classes(atoms, seen=()):
        out = set()
        for s, _ in atoms:
            if s in names:
                out.add(s)
            elif s in defs and s not in seen:
                out |= classes(defs[s], seen + (s,))
        return out
    return {t: set().union(*(classes(r.atoms) for r in rs)) if rs else set() for t, rs in model.rules.items()}


# --- reference: a latent relation as a partial function (K4 stage B) -------------------------------------------------
#
# docs/proposals/K4_LEARNED_REFERENCE_v0_1.md §3, K4-3 and K4-4; SYMMETRIES_AS_PATTERNS §3 (a class with a constraint).
# A latent binary symbol same(t, x) whose facts are derived from an assignment: a partial function from source symbols
# (the word forms, unary symbols over token entities) to referents (the entities of a declared domain, persistent across
# structures by their names). In every structure, a token whose word form the assignment maps to an entity that occurs
# there gets same(token, entity). Nothing else crosses structures. The caller declares which symbols are sources and
# which domain holds the referents (the one declaration SYMMETRIES §6 leaves on the horizon); the kernel reads neither
# as a name.
#
# Code of the assignment (SY-2's sequential partition code under the constraint that a class holds at most one
# referent, which makes it order-free): per source symbol one bit, named or not, plus log2(number of referents) if
# named. Admission by the one law: the targets' two-part code with same, plus the assignment, against without.
#
# Search: alternation (K4-4). Seeds from co-occurrence: a referent is *involved* in a structure where it stands in a
# target tuple that holds and the model without same predicted it would not (or abstained): the surprising facts;
# a source's lift for a referent is its rate among those structures over its overall rate; each referent seeds the
# source of highest lift. Then: form rules with same, reassign each source to the referent (or none) under which the
# targets code shortest with the rules and counts held fixed, the assignment's own code included, ties to none; re-form;
# repeat until the assignment recurs. The shortest code over seeds and rounds is kept.


@dataclass(frozen=True)
class Reference:
    name: str = "same"                      # the latent relation's symbol
    sources: tuple = ()                     # unary symbols over the token-like entities (the word forms), declared
    domain: tuple = ("ent",)                # the symbols whose entities are the referents, declared
    max_rounds: int = 8
    extra_seeds: tuple = ()                 # further seed assignments ({source: referent}), apparatus for a demonstration


@dataclass
class Linking:
    assignment: dict                        # source symbol -> referent (entities not mapped are absent)
    referents: tuple
    gain_bits: float
    admitted: bool
    base_bits: float                        # D + S without the relation
    bits: float                             # D + S with the relation + the assignment's code
    assignment_bits: float
    model: kl.Model | None
    base: kl.Model
    trace: list
    learns: int


def _token_forms(structures: list[Structure], sources) -> list[dict]:
    """Per structure: {token entity: its source symbol}."""
    out = []
    for s in structures:
        d = {}
        for sym in sources:
            for (t,) in s.facts.get(sym, ()):
                d[t] = sym
        out.append(d)
    return out


def with_relation(structures: list[Structure], assignment: dict, ref: Reference, forms: list[dict] | None = None) -> list[Structure]:
    """The structures with same(t, e) for every token whose word form the assignment maps to an entity present."""
    forms = forms if forms is not None else _token_forms(structures, ref.sources)
    out = []
    for s, f in zip(structures, forms):
        present = set(s.entities(ref.domain))
        facts = {(t, assignment[sym]) for t, sym in f.items() if sym in assignment and assignment[sym] in present}
        out.append(s.with_facts({ref.name: facts}))
    return out


def reference_bits(n_sources: int, n_named: int, n_referents: int) -> float:
    """L(assignment): one bit per source (named or not) plus log2(referents) for each named one. SY-2's sequential
    partition code under the one-referent-per-class constraint, which makes it independent of the order."""
    return n_sources + n_named * (np.log2(n_referents) if n_referents > 1 else 0.0)


def _code(model: kl.Model, structures, forms, assignment, ref: Reference, n_sources: int, n_ref: int) -> float:
    """The reassignment's score: D under the frozen model (each candidate by its cell's training estimate, the plug-in
    code, as `place` scores a new identity) on the structures as the assignment derives them, plus the assignment's
    code. A relative measure with the rules and counts held fixed; admission uses the prequential code (invent_reference)."""
    ev = with_relation(structures, assignment, ref, forms)
    d = sum(float(kl.bits_each(model, ev, t).sum()) for t in model.targets)
    return d + reference_bits(n_sources, len(assignment), n_ref)


def _involvement_seed(base: kl.Model, structures, targets, ref: Reference, referents) -> dict:
    """K4-4: for each referent, the source whose presence has the highest lift where the referent is involved (stands
    in a surprising fact: a target tuple that holds where the model without the relation predicted not, or abstained)."""
    forms = _token_forms(structures, ref.sources)
    present = [set(f.values()) for f in forms]
    n = len(structures)
    overall = {s: sum(1 for p in present if s in p) / n for s in ref.sources}
    involved = {e: set() for e in referents}
    for t in targets:
        preds = kl.predict(base, structures, t)
        for i, (s, pred) in enumerate(zip(structures, preds)):
            cands = s.candidates(t.arity, t.domain)
            held = s.facts.get(t.symbol, frozenset())
            for c, p in zip(cands, pred):
                if c in held and p != 1:
                    for e in c:
                        if e in involved:
                            involved[e].add(i)
    seed, claimed = {}, {}
    for e in sorted(referents, key=lambda e: -len(involved[e])):
        idx = involved[e]
        if not idx:
            continue
        best, best_lift = None, 1.0
        for s in ref.sources:
            if overall[s] == 0:
                continue
            rate = sum(1 for i in idx if s in present[i]) / len(idx)
            lift = rate / overall[s]
            if lift > best_lift + 1e-9:
                best, best_lift = s, lift
            elif best is not None and abs(lift - best_lift) <= 1e-9:
                best = None                                    # a tie: this referent seeds nothing
                best_lift = max(best_lift, lift)
        if best is not None and best not in claimed:
            seed[best] = e
            claimed[best] = e
    return seed


def invent_reference(structures: list[Structure], targets: list[kl.Target], ref: Reference,
                     search: kl.Search = kl.Search()) -> Linking:
    """Search for a reference relation (module section above); admit it if it pays by the margin."""
    structures = list(structures)
    referents = identities(structures, ref.domain)
    n_src, n_ref = len(ref.sources), len(referents)
    forms = _token_forms(structures, ref.sources)
    base, _ = kl.learn(structures, targets, search)
    base_bits = code_bits(base)
    learns = 1
    seeds = [_involvement_seed(base, structures, targets, ref, referents)] + [dict(s) for s in ref.extra_seeds]
    best, trace = None, []
    for n_seed, seed in enumerate(seeds):
        assignment = dict(seed)
        seen = []
        for rnd in range(ref.max_rounds):
            if not assignment:
                trace.append({"seed": n_seed, "round": rnd, "named": 0, "note": "empty assignment: nothing to form on"})
                break
            model, _ = kl.learn(with_relation(structures, assignment, ref, forms), targets, search)
            learns += 1
            # the admission total is the prequential two-part code, as for classes (code_bits), plus the assignment;
            # the plug-in score of _code serves only the reassignment below, where the rules and counts are held fixed
            total = code_bits(model) + reference_bits(n_src, len(assignment), n_ref)
            trace.append({"seed": n_seed, "round": rnd, "named": len(assignment), "bits": round(total, 3),
                          "rules": sum(len(model.rules[t.symbol]) for t in model.targets)})
            if best is None or total < best[0] - 1e-9:
                best = (total, dict(assignment), model)
            key = tuple(sorted(assignment.items()))
            if key in seen:
                break
            seen.append(key)
            # reassign each source, rules and counts fixed, the assignment's code included; a tie leaves it unnamed
            for src in sorted(ref.sources):
                options = []
                for e in (None,) + tuple(referents):
                    trial = {k: v for k, v in assignment.items() if k != src}
                    if e is not None:
                        trial[src] = e
                    options.append((_code(model, structures, forms, trial, ref, n_src, n_ref), e))
                least = min(o[0] for o in options)
                winners = [e for b, e in options if b <= least + kl.TIE_BITS]
                choice = None if (None in winners or len(winners) > 1) else winners[0]
                if choice is None:
                    assignment.pop(src, None)
                else:
                    assignment[src] = choice
    if best is None:
        return Linking({}, referents, 0.0, False, base_bits, base_bits, reference_bits(n_src, 0, n_ref), None, base, trace, learns)
    total, assignment, model = best
    gain = base_bits - total
    return Linking(assignment, referents, gain, gain > kl.MARGIN_BITS, base_bits, total,
                   reference_bits(n_src, len(assignment), n_ref), model, base, trace, learns)


# --- several latent domains, and rival assignments (SMALLEST_AGENT §9ao, SA-39) -----------------------------------------

def invent_many(structures: list[Structure], targets: list[kl.Target], search: kl.Search = kl.Search(),
                latents: list[Latent] = (), start: dict | None = None) -> dict:
    """Classes over several domains (cells and places, say): each domain's classes invented with the others' in the
    evidence, the domains taken in every order, and the order whose total code (every class's assignment and the
    model) is shortest kept; a tie keeps the first order. `start`: {domain index: classes already held}. Returns
    {"classes": {i: [members]}, "bits", "model", "evidence", "order"}."""
    from itertools import permutations
    structures = list(structures)
    best = None
    for order in permutations(range(len(latents))):
        classes = {i: [frozenset(c) for c in (start or {}).get(i, [])] for i in range(len(latents))}
        for i in order:
            evid = structures
            for j in range(len(latents)):
                if j != i and classes[j]:
                    evid, _ = _with_classes(evid, classes[j], latents[j])
            classes[i] = list(invent_all(evid, targets, search, latents[i], start=classes[i]).classes)
        bits, model, evid = code_many(structures, classes, targets, search, latents)
        if best is None or bits < best["bits"] - 1e-9:
            best = {"classes": classes, "bits": bits, "model": model, "evidence": evid, "order": order}
    return best


def code_many(structures, classes: dict, targets, search, latents) -> tuple[float, kl.Model, list]:
    """The code of every target with these classes over their domains, each assignment included."""
    evid, bits = list(structures), 0.0
    for j, lat in enumerate(latents):
        if classes.get(j):
            evid, _ = _with_classes(evid, classes[j], lat)
            n = len(identities(structures, lat.domain))
            bits += sum(assignment_bits(n, len(c)) for c in classes[j])
    model, _ = kl.learn(evid, targets, search)
    return bits + code_bits(model), model, evid


def rivals_of(structures, classes: dict, targets, search, latents, identity: str, domain: int,
              margin: float = kl.MARGIN_BITS) -> list[dict]:
    """Where a new identity of one domain may belong: each held class of that domain with the identity added, and
    none of them, priced by `code_many` with the identity counted among the domain's identities; every assignment
    within the margin of the best is a rival, held together as tied rules are (KERNEL §5). Returns [{"classes",
    "bits"}], the best first; one entry when the past decides."""
    n_extra = 0 if identity in identities(structures, latents[domain].domain) else 1
    options = [dict(classes)]
    for k, c in enumerate(classes.get(domain, [])):
        alt = dict(classes)
        alt[domain] = [cc | {identity} if kk == k else cc for kk, cc in enumerate(classes[domain])]
        options.append(alt)
    priced = []
    for opt in options:
        bits, model, evid = code_many(structures, opt, targets, search, latents)
        if n_extra:                                     # the new identity counts in the domain's assignment code
            n = len(identities(structures, latents[domain].domain))
            for c in opt.get(domain, []):
                bits += assignment_bits(n + 1, len(c)) - assignment_bits(n, len(c))
        priced.append({"classes": opt, "bits": bits, "model": model})
    priced.sort(key=lambda x: x["bits"])
    return [x for x in priced if x["bits"] <= priced[0]["bits"] + margin]
