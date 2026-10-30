"""DARA's kernel tests (tests/unit/test_kernel.py at DARA main 95cea64), imports renamed. Left in DARA: six tests of its
Q1 teacher (dara.teacher), which AWARE does not use, and two that need `modules` and `latent`, which come with later
stages (PLAN.md, "Porting DARA, stage by stage").

The relational kernel's guarantees (DARA's docs/KERNEL_v0_1.md §8.1, stage K0) and its first language check (§9, K1)."""
from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path
import random

import numpy as np
import pytest

from kernel import code, learn as K
from kernel.pattern import canonical
from kernel.structure import Corpus, Structure, rename


def _toy(seed=0, n=60, noise=0.0):
    """T(x) holds when x points at something marked: T(x) <- R(x, y) & Q(y). Q2(x) is unrelated."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        ents = [f"e{i}.{j}" for j in range(5)]
        R = {(a, b) for a in ents for b in ents if a != b and r.random() < .3}
        Q = {(a,) for a in ents if r.random() < .5}
        Q2 = {(a,) for a in ents if r.random() < .5}
        T = {(x,) for (x, y) in R if (y,) in Q}
        T = {(x,) for x in ents if ((x,) in T) != (r.random() < noise)}
        out.append(Structure({"R": R, "Q": Q, "Q2": Q2, "T": T}))
    return out


TOY = K.Target("T", 1, ("R", "Q", "Q2"))


@pytest.mark.invariant
def test_the_kernel_finds_a_relation_through_an_unnamed_entity():
    m, _ = K.learn(_toy(), [TOY])
    (r,) = m.rules["T"]
    assert canonical(r.atoms, 1) == canonical((("R", (0, 1)), ("Q", (1,))), 1)
    assert r.symbol and r.gain_bits > 100


@pytest.mark.invariant
def test_nothing_is_admitted_when_the_target_ignores_the_structure():
    r = random.Random(3)
    data = [Structure(dict(s.facts, T={(e,) for e in s.entities(("R", "Q", "Q2")) if r.random() < .4}))
            for s in _toy(seed=4)]
    m, _ = K.learn(data, [TOY])
    assert m.rules["T"] == [] and m.stops["T"] == "BELOW_MARGIN"         # ties that do not pay are no ambiguity


@pytest.mark.invariant
def test_renaming_entities_and_symbols_changes_no_number_and_no_decision():
    data = _toy(seed=1, noise=0.1)
    syms = {"R": "zz", "Q": "a", "Q2": "mm", "T": "b"}
    r = random.Random(9)
    ents = sorted({e for s in data for e in s.entities(("R", "Q", "Q2", "T"))})
    perm = ents[:]
    r.shuffle(perm)
    renamed = [rename(s, dict(zip(ents, perm)), syms) for s in data]
    m1, _ = K.learn(data, [TOY])
    m2, _ = K.learn(renamed, [K.Target("b", 1, ("zz", "a", "mm"))])
    assert m1.train_bits["T"] == m2.train_bits["b"]
    back = {v: k for k, v in syms.items()}
    for a, b in zip(m1.rules["T"], m2.rules["b"]):
        assert a.gain_bits == b.gain_bits and a.structure_bits == b.structure_bits
        assert {canonical(p, 1) for p in a.rivals} == \
            {canonical(tuple((back.get(s, s), args) for s, args in p), 1) for p in b.rivals}
    assert len(m1.rules["T"]) == len(m2.rules["b"])


@pytest.mark.invariant
def test_the_same_evidence_gives_the_same_model():
    a, b = (K.summary(K.learn(_toy(seed=2, noise=.1), [TOY])[0]) for _ in range(2))
    assert a == b


@pytest.mark.invariant
def test_the_kernel_imports_nothing_outside_itself():
    root = Path(K.__file__).parent
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for n in names:
                top = n.split(".")[0]
                assert top in {"__future__", "numpy", "math", "hashlib", "itertools", "functools", "collections",
                               "dataclasses", "json"} or n.startswith("kernel"), (path.name, n)


@pytest.mark.invariant
def test_a_rule_for_a_later_target_cites_an_earlier_rules_symbol():
    """U(x) <- T*(x) & P(x), where T* is T's clean pattern R(x, y) & Q(y) and the observed T is noisy. With at most two
    atoms per pattern, U can be stated only through the symbol T's rule defined."""
    r = random.Random(7)
    data = []
    for s in _toy(seed=5, n=120, noise=0.15):
        ents = s.entities(("R", "Q", "Q2"))
        clean = {(x,) for (x, y) in s.facts["R"] if (y,) in s.facts["Q"]}
        P = {(e,) for e in ents if r.random() < .5}
        data.append(s.with_facts({"P": P, "U": clean & P}))
    m, _ = K.learn(data, [TOY, K.Target("U", 1, ("R", "Q", "Q2", "P"))], K.Search(max_atoms=2))
    sym = m.rules["T"][0].symbol
    assert sym and canonical(m.rules["T"][0].atoms, 1) == canonical((("R", (0, 1)), ("Q", (1,))), 1)
    first = m.rules["U"][0]
    assert {s for s, _ in first.atoms} == {sym, "P"} and first.depth == 2


@pytest.mark.invariant
def test_heldout_code_is_kt_coded_one_outcome_at_a_time():
    r = random.Random(1)
    cells = np.array([r.randrange(3) for _ in range(200)])
    y = np.array([int(r.random() < .3 + .2 * c) for c in cells])
    train = code.counts(cells[:120], y[:120])
    seq, cnt = 0.0, dict(train)
    for c, v in zip(cells[120:], y[120:]):
        n, k = cnt.get(int(c), (0, 0))
        p1 = (k + .5) / (n + 1)
        seq -= math.log2(p1 if v else 1 - p1)
        cnt[int(c)] = (n + 1, k + v)
    assert code.continued_bits(train, cells[120:], y[120:]) == pytest.approx(seq, abs=1e-9)


@pytest.mark.invariant
def test_the_pattern_code_is_a_prefix_code_and_ignores_atom_order():
    atoms = (("R", (0, 1)), ("Q", (1,)), ("S", (2, 0)))
    assert code.pattern_bits(atoms, 1, 8) == code.pattern_bits(atoms[::-1], 1, 8)
    # Kraft: summing 2^-S over every pattern of up to two atoms over a 2-symbol vocabulary (arities 1, 2)
    vocab, total = {"A": 1, "B": 2}, 0.0
    from kernel.pattern import refinements
    frontier = [()]
    for _ in range(2):
        nxt = []
        for p in frontier:
            for c in refinements(p, 1, vocab):
                nxt.append(c)
                total += 2 ** -code.pattern_bits(c, 1, len(vocab))
        frontier = nxt
    assert total < 1



@pytest.mark.invariant
def test_the_description_code_is_valid_order_free_and_tighter():
    """KD-2's alternative: Kraft over pattern classes (canonical forms) of up to three atoms, arities 0, 1 and 2."""
    from kernel.pattern import refinements
    vocab = {"A": 1, "B": 2, "C": 0}
    for arity in (0, 1):
        classes, frontier = set(), [()]
        for _ in range(3):
            nxt = []
            for p in frontier:
                for c in refinements(p, arity, vocab):
                    nxt.append(c)
                    classes.add(canonical(c, arity))
            frontier = nxt
        assert sum(2 ** -code.description_bits(c, arity, len(vocab)) for c in classes) <= 1
    atoms = (("R", (0, 1)), ("Q", (1,)), ("S", (2, 0)))
    assert code.description_bits(atoms, 1, 8) == code.description_bits(atoms[::-1], 1, 8)
    assert code.description_bits(atoms, 1, 8) == code.description_bits((("R", (0, 2)), ("Q", (2,)), ("S", (1, 0))), 1, 8)
    two = (("U", (0, 1)), ("V", (0, 1)))
    assert code.description_bits(two, 0, 2) == pytest.approx(8.1699, abs=1e-4)
    assert code.pattern_bits(two, 0, 2) == 12


@pytest.mark.invariant
def test_the_description_code_finds_the_same_rule_through_an_unnamed_entity():
    m, _ = K.learn(_toy(), [TOY], K.Search(structure_code="descriptions"))
    (r,) = m.rules["T"]
    assert canonical(r.atoms, 1) == canonical((("R", (0, 1)), ("Q", (1,))), 1)
    assert r.structure_bits < code.pattern_bits(r.atoms, 1, 3)
    with pytest.raises(ValueError, match="structure code"):
        code.structure_bits(r.atoms, 1, 3, "manifest")

# --- theory: the margin, the safe bound, what a cell table says (KERNEL §5.1, §6.1, §3) ------------------------------

def _all_patterns(vocab: dict, arity: int, max_atoms: int):
    """Every connected pattern of 1 .. max_atoms atoms, once each up to renaming and order."""
    from kernel.pattern import refinements
    seen, frontier = set(), [()]
    for _ in range(max_atoms):
        nxt = []
        for p in frontier:
            for c in refinements(p, arity, vocab):
                if canonical(c, arity) not in seen:
                    seen.add(canonical(c, arity))
                    nxt.append(c)
        yield from nxt
        frontier = nxt


def _one_round(data, t: K.Target, held: list, search: K.Search):
    c = Corpus(list(data))
    y = K.outcomes(c, t)
    vocab = K._vocabulary(c, t, {t.symbol}, K.Model(targets=[t]))
    cols = [K.column(c, t, a).astype(np.int64) for a in held]
    cells = code.Cells(code.cell_ids(cols, len(y)), y, code.KT(len(y)))
    return c, y, vocab, cells, K._round(c, t, vocab, cells, search, K.ColumnCache(1 << 30))


@pytest.mark.invariant
def test_the_refinement_floor_is_below_every_subset_of_the_rows():
    """§6.1: no rule holding at a subset of a child's rows codes the target in fewer bits than Cells.floor_bits;
    exhaustive on small cases, sampled on larger ones."""
    from itertools import product
    r = random.Random(11)
    for trial in range(300):
        n = r.randrange(1, 9) if trial < 200 else r.randrange(20, 80)
        y = np.array([int(r.random() < r.random()) for _ in range(n)])
        cells = code.Cells(np.array([r.randrange(3) for _ in range(n)]), y, code.KT(n))
        rows = np.flatnonzero([r.random() < .6 for _ in range(n)])
        floor = cells.floor_bits(rows)
        subsets = product((0, 1), repeat=len(rows)) if n < 9 else \
            ([int(r.random() < .5) for _ in rows] for _ in range(200))
        for bits in subsets:
            assert cells.split_bits(rows[np.array(bits, dtype=bool)]) >= floor - 1e-9


@pytest.mark.invariant
def test_one_more_atom_costs_at_least_the_step_under_both_structure_codes():
    """§6.1: S(d) - S(c) >= code.step_bits for every refinement d of c, up to four atoms, under both codes."""
    vocab = {"A": 1, "B": 2, "C": 2}
    pats = list(_all_patterns(vocab, 1, 4))
    from kernel.pattern import refinements
    for kind in code.STRUCTURE_CODES:
        for c in (p for p in pats if len(p) < 4):
            floor = code.step_bits(len(c), len(vocab), 4, kind)
            s_c = code.structure_bits(c, 1, len(vocab), kind)
            frontier = refinements(c, 1, vocab)
            while frontier:
                nxt = []
                for d in frontier:
                    assert code.structure_bits(d, 1, len(vocab), kind) - s_c >= floor - 1e-9, (kind, c, d)
                    if len(d) < 4:
                        nxt += refinements(d, 1, vocab)
                frontier = nxt


@pytest.mark.invariant
def test_the_safe_bound_finds_the_best_pattern_a_full_enumeration_finds():
    """§6.1: with no beam, pruning by the safe bound loses nothing. On the toy world, with no rule held and with one,
    the best gain the search reports equals the best over every pattern of up to three atoms."""
    data = _toy(seed=6, n=30, noise=.1)
    for held in ([], [(("Q2", (0,)),)]):
        c, y, vocab, cells, (scored, evaluated, cut, pruned, beamed) = _one_round(
            data, TOY, held, K.Search(beam=None, bound="safe", bridges="off"))
        brute = max(cells.bits - cells.split_bits(np.flatnonzero(K.column(c, TOY, a)))
                    - code.structure_bits(a, 1, len(vocab)) for a in _all_patterns(vocab, 1, 3))
        assert not cut and not beamed
        assert max(g for g, *_ in scored) == pytest.approx(brute, abs=1e-9)
        assert evaluated < sum(1 for _ in _all_patterns(vocab, 1, 3))


@pytest.mark.invariant
def test_the_safe_bound_changes_no_rule_on_the_toy_world():
    a = K.summary(K.learn(_toy(seed=2, noise=.1), [TOY])[0])["T"]
    b = K.summary(K.learn(_toy(seed=2, noise=.1), [TOY], K.Search(bound="safe"))[0])["T"]
    assert a["rules"] == b["rules"] and a["train_bits"] == b["train_bits"]


def test_a_target_unrelated_to_the_structure_clears_the_margin_at_most_2_to_the_minus_m_of_the_time():
    """§5.1: if the target's outcomes follow the baseline code (KT on one cell: a rate drawn from Beta(1/2, 1/2), then
    independent draws), the chance that any pattern has G > m is at most 2^-m. The exact search's best gain over every
    pattern of up to two atoms is sampled; its tail must sit under the bound."""
    r = random.Random(17)
    base = _toy(seed=8, n=20)
    trials, best = 300, []
    for _ in range(trials):
        theta = r.betavariate(.5, .5)
        data = [Structure(dict(s.facts, T={(e,) for e in s.entities(("R", "Q", "Q2")) if r.random() < theta}))
                for s in base]
        *_, (scored, *_rest) = _one_round(data, TOY, [], K.Search(max_atoms=2, beam=None, bound="safe"))
        best.append(max([g for g, *_ in scored], default=-math.inf))
    for m in (0.0, 1.0, 2.0):
        rate, bound = sum(g > m for g in best) / trials, min(1.0, 2.0 ** -m)
        assert rate <= bound + 3 * math.sqrt(bound * (1 - bound) / trials) + 1e-12, (m, rate)


@pytest.mark.invariant
def test_a_cell_table_says_no_such_thing_exists():
    """§3: T(x) <- not exists y. R(x, y) & Q(y) is not a pattern, but the cell table of the pattern R(x, y) & Q(y)
    reads it exactly: the target holds in the cell where the rule does not."""
    def negated(seed):
        out = []
        for s in _toy(seed=seed):
            hit = {(x,) for (x, y) in s.facts["R"] if (y,) in s.facts["Q"]}
            out.append(s.with_facts({"T": {(e,) for e in s.entities(("R", "Q", "Q2")) if (e,) not in hit}}))
        return out
    m, _ = K.learn(negated(0), [TOY])
    assert [canonical(r.atoms, 1) for r in m.rules["T"]] == [canonical((("R", (0, 1)), ("Q", (1,))), 1)]
    test = negated(1)
    for s, pred in zip(test, K.predict(m, test, TOY)):
        assert list(pred) == [int(c in s.facts["T"]) for c in s.candidates(1, ("R", "Q", "Q2"))]


# --- naming: a symbol for a piece held rules share (KERNEL §5.2) ------------------------------------------------------

K2_RULES = [(("at0", (0, 1)),), (("at0", (0, 2)), ("names", (3, 0)), ("place", (1,)))] + \
    [(("names", (2, 0)), ("next", (2, 3)), ("next", (3, 4)), (w, (4,))) for w in ("w:the", "w:to", "w:from")] + \
    [(("names", (2, 1)), ("next", (3, 2)), ("next", (4, 3)), ("w:to", (4,)))]


@pytest.mark.invariant
def test_naming_finds_the_piece_k2s_rules_share_and_prices_it_by_the_naming_law():
    from kernel import naming
    inv, tied = naming.best(K2_RULES, 2, 37)
    assert tied == 1 and inv.arity == 2 and len(inv.atoms) == 3 and sorted(inv.rewrites) == [2, 3, 4]
    assert sorted(s for s, _ in inv.atoms) == ["names", "next", "next"]
    for i, atoms in inv.rewrites.items():                       # the piece replaced by one atom citing the symbol
        assert len(atoms) == 2 and ("@" in {s for s, _ in atoms}) and K2_RULES[i][3] in atoms
    before = sum(code.description_bits(r, 2, 37) for r in K2_RULES)
    after = inv.structure_bits + sum(code.description_bits(inv.rewrites.get(i, r), 2, 38)
                                     for i, r in enumerate(K2_RULES))
    assert inv.gain_bits == pytest.approx(before - after) and inv.gain_bits > 20


@pytest.mark.invariant
def test_naming_reads_no_name():
    from kernel import naming
    sym = {"at0": "q", "names": "zz", "next": "a", "place": "m", "w:the": "k", "w:to": "b", "w:from": "c"}
    renamed = [tuple((sym[s], a) for s, a in r) for r in K2_RULES]
    a, b = naming.best(K2_RULES, 2, 37), naming.best(renamed, 2, 37)
    assert a[1] == b[1] and a[0].gain_bits == pytest.approx(b[0].gain_bits) and sorted(a[0].rewrites) == \
        sorted(b[0].rewrites)


@pytest.mark.invariant
def test_naming_leaves_rules_with_rivals_alone_and_needs_two_uses():
    from kernel import naming
    assert naming.best(K2_RULES, 2, 37, frozen={2, 3, 4})[0] is None or \
        sorted(naming.best(K2_RULES, 2, 37, frozen={2, 3, 4})[0].rewrites) != [2, 3, 4]
    assert naming.best(K2_RULES[:3], 2, 37)[0] is None            # one rule holds the piece: nothing to share


def _markers(seed, n=80, marks="ABCD"):
    """T(x) holds when x reaches, by R then S, something with one of the marks: four rules sharing R(x, y) ∧ S(y, z)."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        e = [f"e{i}.{j}" for j in range(7)]
        R = {(a, b) for a in e for b in e if a != b and r.random() < .15}
        S = {(a, b) for a in e for b in e if a != b and r.random() < .15}
        mark = {a: r.choice(marks) if r.random() < .35 else "-" for a in e}
        facts = {m: {(a,) for a in e if mark[a] == m} for m in marks}
        T = {(x,) for (x, y) in R for (y2, z) in S if y2 == y and mark[z] != "-"}
        out.append(Structure(dict(facts, R=R, S=S, T=T)))
    return out


@pytest.mark.invariant
def test_a_named_piece_shortens_the_code_and_changes_no_reading():
    """§5.2: four rules share R(x, y) ∧ S(y, z); named once, the total code is shorter by the naming's gain, and the
    model reads new evidence exactly as without it."""
    t = K.Target("T", 1, ("R", "S", "A", "B", "C", "D"))
    plain, _ = K.learn(_markers(0), [t], K.Search(max_atoms=3, invent=False))
    named, c = K.learn(_markers(0), [t], K.Search(max_atoms=3))                 # the default names (KD-10)
    (inv,) = named.inventions["T"]
    later = K.Target("U", 1, ("R", "S", "A", "B", "C", "D"))
    assert inv["symbol"] in K._vocabulary(c, later, {"U"}, named)                 # a later target may cite it
    assert inv["uses"] == 4 and inv["arity"] == 2 and sorted(s for s, _ in inv["atoms"]) == ["R", "S"]
    assert all(any(s == inv["symbol"] for s, _ in r.atoms) and len(r.atoms) == 2 for r in named.rules["T"])
    total = lambda m: m.train_bits["T"] + sum(r.structure_bits for r in m.rules["T"]) + \
        sum(i["structure_bits"] for i in m.inventions.get("T", []))                       # noqa: E731
    assert total(plain) - total(named) == pytest.approx(inv["gain_bits"]) and inv["gain_bits"] > 2
    test = _markers(1)
    assert K.heldout_bits(named, test, t) == K.heldout_bits(plain, test, t)
    assert all((a == b).all() for a, b in zip(K.predict(named, test, t), K.predict(plain, test, t)))
    assert K.summary(K.from_record(K.to_record(named))) == K.summary(named)


def _beyond(seed, n=120):
    """T(x): x reaches by R then S something marked A-D, or something both E and F: a rule of four atoms."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        e = [f"e{i}.{j}" for j in range(7)]
        R = {(a, b) for a in e for b in e if a != b and r.random() < .15}
        S = {(a, b) for a in e for b in e if a != b and r.random() < .15}
        mark = {a: r.choice("ABCD") if r.random() < .3 else "-" for a in e}
        E_, F = {(a,) for a in e if r.random() < .4}, {(a,) for a in e if r.random() < .4}
        facts = {m: {(a,) for a in e if mark[a] == m} for m in "ABCD"}
        T = {(x,) for (x, y) in R for (y2, z) in S if y2 == y and (mark[z] != "-" or ((z,) in E_ and (z,) in F))}
        out.append(Structure(dict(facts, R=R, S=S, E=E_, F=F, T=T)))
    return out


@pytest.mark.invariant
def test_re_forming_with_a_named_piece_reaches_a_rule_beyond_max_atoms():
    """§5.2: with three atoms the rule R(x, y) ∧ S(y, z) ∧ E(z) ∧ F(z) is out of reach. Named, R ∧ S is one atom, and
    re-forming finds it: the held-out code of a four-atom search, and a shorter total code than either."""
    t = K.Target("T", 1, ("R", "S", "A", "B", "C", "D", "E", "F"))
    total = lambda m: m.train_bits["T"] + sum(r.structure_bits for r in m.rules["T"]) + \
        sum(i["structure_bits"] for i in m.inventions.get("T", []))                       # noqa: E731
    named, _ = K.learn(_beyond(0), [t], K.Search(max_atoms=3, invent=True, reform=0))
    reformed, _ = K.learn(_beyond(0), [t], K.Search(max_atoms=3, invent=True))
    wide, _ = K.learn(_beyond(0), [t], K.Search(max_atoms=4, invent=False))
    (sym,) = [i["symbol"] for i in reformed.inventions["T"]]
    assert any(sorted(s for s, _ in r.atoms) == sorted([sym, "E", "F"]) for r in reformed.rules["T"])
    assert not any({"E", "F"} <= {s for s, _ in r.atoms} for r in named.rules["T"])
    assert reformed.reformed["T"][0]["kept"] and total(reformed) < total(wide) < total(named)
    test = _beyond(1)
    assert K.heldout_bits(reformed, test, t) == pytest.approx(K.heldout_bits(wide, test, t))
    assert K.heldout_bits(reformed, test, t) < K.heldout_bits(named, test, t) - 20


@pytest.mark.invariant
def test_a_re_formation_that_codes_longer_is_undone_exactly():
    """§5.2: a re-formation is kept only when it codes shorter; on the markers world it ties (86.70 bits), so the
    earlier rules, their symbols and the named pieces are restored."""
    t = K.Target("T", 1, ("R", "S", "A", "B", "C", "D"))
    once, _ = K.learn(_markers(0), [t], K.Search(max_atoms=3, invent=True, reform=0))
    again, c = K.learn(_markers(0), [t], K.Search(max_atoms=3, invent=True))
    (log,) = again.reformed["T"]
    assert not log["kept"] and log["after"] >= log["before"] - 1e-9
    assert [r.atoms for r in again.rules["T"]] == [r.atoms for r in once.rules["T"]]
    assert again.inventions["T"] == once.inventions["T"]
    test = _markers(1)
    assert K.heldout_bits(again, test, t) <= K.heldout_bits(once, test, t) + 1e-9
    assert all(i["symbol"] in c.vocabulary() for i in again.inventions["T"])


# --- cell symbols: negation inside a symbol (KERNEL §5.3; docs/proposals/CELL_SYMBOLS_v0_1.md) -----------------------

@pytest.mark.invariant
def test_candidate_rows_go_back_to_their_tuples():
    c = Corpus(_toy(seed=3, n=6))
    rows = np.arange(int(c.rows(2, ("R", "Q"))["start"][-1]))
    tup = c.tuples_of(2, ("R", "Q"), rows)
    assert (c.row_of(2, ("R", "Q"), [tup[:, 0], tup[:, 1]]) == rows).all()


@pytest.mark.invariant
def test_a_cell_is_defined_by_its_shortest_equivalent_literals_and_priced_by_them():
    from kernel import cells
    person = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    holds = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    got = cells.proposals([person, holds])
    # cells: (person, holds) is "holds" alone, the rule itself, so not proposed; (person, not holds) needs both
    # literals; (not person, not holds) is "not person" alone (CS-10: no positive sign needed)
    assert sorted(got) == sorted([[((0, True), (1, False))], [((0, False),)]])
    assert cells.proposals([person, holds], positive=True) == [[((0, True), (1, False))]]
    assert cells.literal_bits(2) == pytest.approx(math.log2(3 ** 2 - 1))
    assert cells.literal_bits(2, positive=True) == pytest.approx(math.log2(3 ** 2 - 2 ** 2))
    assert cells.structure_bits(1, 2) == cells.literal_bits(2)
    assert cells.structure_bits(4, 2) == pytest.approx(2 + cells.literal_bits(2))


def _free(seed, n=100, cause=True, use=True):
    """A(x): x is a P holding nothing (some of them); B(x): x reaches by R something that is a P holding nothing. A is
    hidden from B, so B can state "holding nothing" only through a cell of A's rules. cause=False: A ignores H.
    use=False: B ignores what it reaches."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        e = [f"e{i}.{j}" for j in range(8)]
        P = {(a,) for a in e if r.random() < .6}
        H = {(a, b) for (a,) in sorted(P) for b in e if b != a and r.random() < .12}   # sorted: the draws must not
        R = {(a, b) for a in e for b in e if a != b and r.random() < .15}               # follow the hash seed
        free = {a for (a,) in P if not any(h[0] == a for h in H)}
        A = {(a,) for (a,) in sorted(P) if (a in free or not cause) and r.random() < .8}
        B = {(x,) for (x, z) in R if z in free} if use else {(a,) for a in e if r.random() < .3}
        out.append(Structure({"P": P, "H": H, "R": R, "A": A, "B": B}))
    return out


FREE_TARGETS = [K.Target("A", 1, ("P", "H", "R")), K.Target("B", 1, ("P", "H", "R"), hide=("A",))]


@pytest.mark.invariant
def test_a_cell_symbol_lets_a_later_target_say_holding_nothing():
    """§5.3: B needs "a P holding nothing", which no pattern states. Offered A's populated cells, B cites the cell
    P ∧ ¬H; the total code is shorter by more than the margin and B's held-out code falls to almost nothing."""
    ta, tb = FREE_TARGETS
    plain, _ = K.learn(_free(0), [ta, tb], K.Search(max_atoms=3, cells=False))
    cells, _ = K.learn(_free(0), [ta, tb], K.Search(max_atoms=3))                # the default offers cells (CS-13)
    rec = cells.cells_record
    assert rec["held"] == "second" and rec["second_bits"] < rec["first_bits"] - 2
    (z,) = cells.cells["A"]
    assert [sorted((lit["sign"], tuple(s for s, _ in lit["rivals"][0])) for lit in d) for d in z["definitions"]] == \
        [[(False, ("H",)), (True, ("P",))]]
    assert any(z["symbol"] in {s for s, _ in r.atoms} for r in cells.rules["B"])
    test = _free(1)
    assert K.heldout_bits(cells, test, tb) < K.heldout_bits(plain, test, tb) - 100
    assert K.summary(K.from_record(K.to_record(cells))) == K.summary(cells)
    assert K.heldout_bits(K.from_record(K.to_record(cells)), test, tb) == K.heldout_bits(cells, test, tb)


@pytest.mark.invariant
def test_no_cell_symbol_without_its_cause_or_its_use():
    """§5.3, C1 and C2: with A blind to H no cell of A's rules says "holding nothing" to B; with B blind to it, a cell
    is offered but not held."""
    ta, tb = FREE_TARGETS
    off, _ = K.learn(_free(0, cause=False), [ta, tb], K.Search(max_atoms=3, cells=True))
    assert not any(".c" in s for r in off.rules["B"] for p in r.rivals for s, _ in p)
    unused, _ = K.learn(_free(0, use=False), [ta, tb], K.Search(max_atoms=3, cells=True))
    assert unused.cells_record["held"] == "first" and not unused.cells.get("A")


@pytest.mark.invariant
def test_cell_symbols_read_no_name():
    ta, tb = FREE_TARGETS
    data = _free(0)
    syms = {"P": "q", "H": "zz", "R": "b", "A": "m", "B": "a"}
    r = random.Random(4)
    ents = sorted({e for st in data for ts in st.facts.values() for t in ts for e in t})
    perm = ents[:]
    r.shuffle(perm)
    renamed = [rename(st, dict(zip(ents, perm)), syms) for st in data]
    t2 = [K.Target("m", 1, ("q", "zz", "b")), K.Target("a", 1, ("q", "zz", "b"), hide=("m",))]
    m1, _ = K.learn(data, [ta, tb], K.Search(max_atoms=3, cells=True))
    m2, _ = K.learn(renamed, t2, K.Search(max_atoms=3, cells=True))
    assert m1.cells_record["first_bits"] == pytest.approx(m2.cells_record["first_bits"])
    assert m1.cells_record["second_bits"] == pytest.approx(m2.cells_record["second_bits"])
    assert m1.cells_record["held"] == m2.cells_record["held"]


# --- K1: teacher study Q1 --------------------------------------------------------------------------------------------







# --- K5: speaking toward a setpoint -------------------------------------------------------------------------------------





# --- saving and the column cache ----------------------------------------------------------------------------------------



@pytest.mark.invariant
def test_the_column_cache_changes_time_not_results():
    data = _toy(seed=6, noise=.1)
    big, _ = K.learn(data, [TOY], K.Search(max_atoms=3))
    none, _ = K.learn(data, [TOY], K.Search(max_atoms=3, cache_bytes=0))
    assert K.summary(big) == K.summary(none)
    assert big.cache["T"]["hits"] > 0 and none.cache["T"]["bytes"] == 0
    assert big.cache["T"]["joins"] < none.cache["T"]["joins"]


@pytest.mark.invariant
def test_a_hidden_symbol_stays_hidden_inside_another_targets_symbol():
    r = random.Random(7)
    data = []
    for s in _toy(seed=5, n=120, noise=0.15):
        ents = s.entities(("R", "Q", "Q2"))
        clean = {(x,) for (x, y) in s.facts["R"] if (y,) in s.facts["Q"]}
        P = {(e,) for e in ents if r.random() < .5}
        data.append(s.with_facts({"P": P, "U": clean & P}))
    seen = K.Target("U", 1, ("R", "Q", "Q2", "P"))
    hidden = K.Target("U", 1, ("R", "Q", "Q2", "P"), hide=("Q",))
    m1, _ = K.learn(data, [TOY, seen], K.Search(max_atoms=2))
    m2, _ = K.learn(data, [TOY, hidden], K.Search(max_atoms=2))
    sym = m1.rules["T"][0].symbol
    assert any(s == sym for r in m1.rules["U"] for s, _ in r.atoms)
    assert not any(s in (sym, "Q") for r in m2.rules["U"] for s, _ in r.atoms)


# --- bridge refinements -------------------------------------------------------------------------------------------------

@pytest.mark.invariant
def test_a_bridge_steps_through_a_determinate_link_onto_a_unary_atom():
    from kernel.pattern import bridges
    from kernel.structure import Corpus
    s = Structure({"next": {("t0", "t1"), ("t1", "t2")}, "names": {("t0", "a"), ("t2", "b")},
                   "many": {("a", "t0"), ("a", "t1")}, "w:by": {("t1",)}, "w:x": {("t0",)}})
    c = Corpus([s])
    assert c.functional("next", 0, 1) and c.functional("next", 1, 0)
    assert c.functional("many", 1, 0) and not c.functional("many", 0, 1)
    vocab = {"next": 2, "names": 2, "many": 2, "w:by": 1, "w:x": 1}
    got = bridges((("names", (1, 0)),), 1, vocab, c.functional)
    assert (("names", (1, 0)), ("next", (2, 1)), ("w:by", (2,))) in got     # the word before the name
    assert not any(atoms[1] == ("many", (0, 2)) for atoms in got)          # 'many' is not determinate from 0
    assert all(len(atoms) == 3 and atoms[2][1] == (2,) for atoms in got)


@pytest.mark.invariant
def test_bridge_modes_agree_where_plain_search_suffices_and_keep_the_guarantees():
    data = _toy(seed=8, noise=.1)
    off, _ = K.learn(data, [TOY], K.Search(max_atoms=3, bridges="off"))
    stall, _ = K.learn(data, [TOY], K.Search(max_atoms=3, bridges="stall"))
    b, _ = K.learn(data, [TOY], K.Search(max_atoms=3))                        # the default: bridges on
    first = lambda m: K.summary(m)["T"]["rules"][0]["atoms"]                  # noqa: E731
    assert first(off) == first(stall) == first(b)
    assert stall.compute["T"] > off.compute["T"] and stall.bridged["T"] == 0   # it bridged once, at the end, in vain
    with pytest.raises(ValueError):
        K.learn(data, [TOY], K.Search(bridges="sometimes"))
    syms = {"R": "zz", "Q": "a", "Q2": "mm", "T": "b"}
    ents = sorted({e for s in data for e in s.entities(("R", "Q", "Q2", "T"))})
    perm = ents[::-1]
    c, _ = K.learn([rename(s, dict(zip(ents, perm)), syms) for s in data],
                   [K.Target("b", 1, ("zz", "a", "mm"))], K.Search(max_atoms=3, bridges=True))
    assert c.train_bits["b"] == b.train_bits["T"]


# --- 0-ary targets and atoms (W1-11) ---------------------------------------------------------------------------------

@pytest.mark.invariant
def test_pruning_unjoinable_atoms_changes_time_not_results():
    """A second channel on entities of its own (U5 Arm A): its atoms cannot join the target's variables, and the
    search skips them unevaluated. Rules, gains, counts and the patterns-evaluated figure are unchanged."""
    def other(data, seed):
        r = random.Random(seed)
        out = []
        for i, st in enumerate(data):
            ents = [f"o{i}.{j}" for j in range(4)]
            out.append(st.with_facts({"S": {(a, b) for a in ents for b in ents if a != b and r.random() < .3},
                                      "P": {(a,) for a in ents if r.random() < .5}}))
        return out
    cases = [(other(_toy(seed=6, noise=.1), 1), TOY),
             (other(_scenes(), 2), K.Target("Y", 0, ("Q", "Q2"), body=("Q", "Q2", "T", "S", "P")))]
    for data, t in cases:
        on, _ = K.learn(data, [t], K.Search(max_atoms=3))
        off, _ = K.learn(data, [t], K.Search(max_atoms=3, prune_joins=False))
        assert K.summary(on) == K.summary(off)
        assert on.compute == off.compute and on.train_bits == off.train_bits
        assert on.pruned[t.symbol] > 0 and off.pruned[t.symbol] == 0




@pytest.mark.invariant
def test_a_proposed_rule_set_is_held_only_when_the_code_prefers_it():
    data = _toy(seed=6, noise=.1)
    full, _ = K.learn(data, [TOY], K.Search(max_atoms=3))
    found = [r.atoms for r in full.rules["T"]]
    short, _ = K.learn(data, [TOY], K.Search(max_atoms=1))            # cannot reach R(x, y) & Q(y)
    held, _ = K.learn(data, [TOY], K.Search(max_atoms=1), proposals={"T": [found]})
    assert held.proposed["T"]["held"] == 0 and [r.atoms for r in held.rules["T"]] == found
    assert held.train_bits["T"] == full.train_bits["T"] < short.train_bits["T"]
    again, _ = K.learn(data, [TOY], K.Search(max_atoms=3), proposals={"T": [[(("Q2", (0,)),)]]})
    assert again.proposed["T"]["held"] == "searched" and K.summary(again) == K.summary(full)
    bad, _ = K.learn(data, [TOY], K.Search(max_atoms=1), proposals={"T": [[(("Z", (0,)),)]]})
    assert bad.proposed["T"]["proposals"] == [{"refused": ["Z"]}]
    assert K.heldout_bits(held, _toy(seed=7, noise=.1), TOY) == K.heldout_bits(full, _toy(seed=7, noise=.1), TOY)


def _scenes(seed=0, n=240):
    """Y() holds when some entity is both Q and Q2. P1() is P0() flipped by a toggle T(x) (the cells give the XOR)."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        ents = [f"e{i}.{j}" for j in range(4)]
        Q = {(a,) for a in ents if r.random() < .3}
        Q2 = {(a,) for a in ents if r.random() < .3}
        T = {(ents[0],)} if r.random() < .3 else set()
        p0 = r.random() < .5
        out.append(Structure({"Q": Q, "Q2": Q2, "T": T, "Y": {()} if Q & Q2 else set(),
                              "P0": {()} if p0 else set(), "P1": {()} if p0 != bool(T) else set()}))
    return out


@pytest.mark.invariant
def test_a_0_ary_target_forms_a_rule_with_variables_of_its_own():
    t = K.Target("Y", 0, ("Q", "Q2"), body=("Q", "Q2", "T"))
    m, _ = K.learn(_scenes(), [t])
    (r,) = m.rules["Y"]
    assert canonical(r.atoms, 0) == canonical((("Q", (0,)), ("Q2", (0,))), 0)
    assert K.heldout_bits(m, _scenes(seed=1, n=60), t) < 5


@pytest.mark.invariant
def test_a_0_ary_atom_is_cited_and_a_0_ary_rule_defines_a_symbol():
    t = K.Target("P1", 0, ("Q",), body=("P0", "T", "Q"))
    m, c = K.learn(_scenes(), [t])
    got = {canonical(r.atoms, 0) for r in m.rules["P1"]}
    assert canonical((("P0", ()),), 0) in got and canonical((("T", (0,)),), 0) in got
    assert K.heldout_bits(m, _scenes(seed=1, n=60), t) < 5                  # the cells give the XOR
    rule = next(r for r in m.rules["P1"] if r.atoms == (("P0", ()),))
    assert rule.symbol and c.nullary[rule.symbol] == c.nullary["P0"]        # defined, 0-ary, where P0 holds
    assert code.pattern_bits((("P0", ()),), 0, 4) == 1 + 1 + 1 + 2          # u=0 (1), stop, continue, symbol (2)


@pytest.mark.invariant
def test_renaming_changes_no_number_for_0_ary_targets_and_atoms():
    data = _scenes(seed=3)
    syms = {"Q": "k", "Q2": "b", "T": "zz", "Y": "w", "P0": "aa", "P1": "c"}
    ents = sorted({e for s in data for e in s.entities(("Q", "Q2", "T"))})
    renamed = [rename(s, dict(zip(ents, ents[::-1])), syms) for s in data]
    for sym, dom, body in (("Y", ("Q", "Q2"), ("Q", "Q2", "T")), ("P1", ("Q",), ("P0", "T", "Q"))):
        m1, _ = K.learn(data, [K.Target(sym, 0, dom, body=body)])
        m2, _ = K.learn(renamed, [K.Target(syms[sym], 0, tuple(syms[d] for d in dom),
                                           body=tuple(syms[b] for b in body))])
        assert m1.train_bits[sym] == m2.train_bits[syms[sym]]
        assert [r.gain_bits for r in m1.rules[sym]] == [r.gain_bits for r in m2.rules[syms[sym]]]




# --- the frontier re-checked and ordered (architecture §2.8 (j), C-2d) ---------------------------------------------

def _same_model(a: K.Model, b: K.Model, t: str) -> bool:
    sa, sb = K.summary(a)[t], K.summary(b)[t]
    return sa["rules"] == sb["rules"] and sa["train_bits"] == sb["train_bits"] and a.stops == b.stops


@pytest.mark.invariant
def test_the_rechecked_bound_drops_frontier_patterns_and_changes_nothing():
    """C-2d: before a frontier pattern is expanded its safe bound is checked against the best found so far. Dropping
    what cannot reach it changes no rule, no code and no stop, and evaluates no more patterns; on these worlds it
    drops some."""
    dropped = 0
    for seed, noise in ((2, .1), (5, .2), (7, .0), (1, .3)):
        data = _toy(seed=seed, noise=noise)
        was = K.learn(data, [TOY], K.Search(recheck_bound=False))[0]
        now = K.learn(data, [TOY])[0]
        assert _same_model(was, now, "T")
        assert now.compute["T"] <= was.compute["T"] and was.dropped["T"] == 0
        dropped += now.dropped["T"]
    assert dropped > 0


@pytest.mark.invariant
def test_the_frontier_order_changes_no_rule_only_the_cost():
    """C-2d, the tie guard: the order the frontier is expanded in (by bound, shuffled, or a proposer's worst order)
    changes nothing but the patterns evaluated."""
    data = _toy(seed=2, noise=.1)
    base = K.learn(data, [TOY])[0]
    for seed in range(3):
        sh = K.learn(data, [TOY], K.Search(order="shuffled", order_seed=seed))[0]
        assert _same_model(base, sh, "T")

    class Worst:                                    # the lowest bound first: the best is found last
        def rank(self, target, depth, held, cands):
            return [-b for _, b in cands]

    class Best:                                     # the proposer that knows the answer: R first
        def rank(self, target, depth, held, cands):
            return [1.0 if any(sym == "R" for sym, _ in atoms) else 0.0 for atoms, _ in cands]

    worst = K.learn(data, [TOY], K.Search(order="proposer", proposer=Worst()))[0]
    best = K.learn(data, [TOY], K.Search(order="proposer", proposer=Best()))[0]
    assert _same_model(base, worst, "T") and _same_model(base, best, "T")
    assert best.compute["T"] <= base.compute["T"] <= worst.compute["T"]


@pytest.mark.invariant
def test_a_round_whose_top_is_ambiguous_is_decided_as_the_full_round_decides():
    """KD-11's next-best after an ambiguous top needs every pattern below the top, so such a round is re-run without
    dropping. P∧Q and R∧S mark different rows with the same counts on T (an exact tie, different columns: an
    ambiguity at the top), the singles say nothing alone, and U's bound falls below the tied top once it is found,
    so the round drops U and must re-run to decide among the rest as the full round does."""
    data = []
    for i in range(60):
        e = [f"e{i}.{j}" for j in range(8)]
        T = {e[0], e[1], e[2], e[3]}
        P, Q = {e[0], e[1], e[4], e[5]}, {e[0], e[1], e[6], e[7]}        # P∧Q = {e0, e1}: all positive
        R, S = {e[2], e[3], e[4], e[6]}, {e[2], e[3], e[5], e[7]}        # R∧S = {e2, e3}: the same counts, other rows
        U = {e[4]}
        data.append(Structure({k: {(x,) for x in v} for k, v in dict(P=P, Q=Q, R=R, S=S, U=U, T=T).items()}))
    tgt = K.Target("T", 1, ("P", "Q", "R", "S", "U"))
    search = K.Search(max_atoms=2, invent=False)
    was = K.learn(data, [tgt], replace(search, recheck_bound=False))[0]
    now = K.learn(data, [tgt], search)[0]
    assert _same_model(was, now, "T") and now.stops["T"] == "AMBIGUOUS"
    assert now.dropped["T"] >= 1 and now.reruns["T"] >= 1 and was.reruns["T"] == 0


@pytest.mark.invariant
def test_the_record_of_searches_says_what_each_pattern_came_to():
    """C-2d's record: with Search.record every round lists what it scored, each pattern with its gain, its parent's
    bound and gain, whether it ended within the margin of the top, and whether it was admitted; the admitted rule is
    marked in exactly one round."""
    m = K.learn(_toy(seed=2, noise=.1), [TOY], K.Search(record=True))[0]
    rounds = m.searches["T"]
    assert rounds and all(r["patterns"] for r in rounds)
    (rule,) = m.rules["T"]
    marked = [p for r in rounds for p in r["patterns"] if p["admitted"]]
    assert len(marked) >= 1 and all(canonical(p["atoms"], 1) == canonical(rule.atoms, 1) for p in marked)
    for r in rounds:
        top = max(p["gain"] for p in r["patterns"])
        assert r["top"] == pytest.approx(top)
        assert all(p["within"] == (p["gain"] >= top - K.MARGIN_BITS) for p in r["patterns"])
        assert all((p["depth"] == 1) == (p["parent_bound"] is None) for p in r["patterns"])
    assert rounds[-1]["decision"] in ("BELOW_MARGIN", "NO_HYPOTHESIS", "AMBIGUOUS", "MAX_RULES")
