"""Count symbols (docs/proposals/COUNT_SYMBOLS_v0_1.md; KERNEL §5.4): a count over a held rule's witnesses."""
from __future__ import annotations

import random

import pytest

from kernel import counts as C, learn as K
from kernel.structure import Structure, rename


def _world(seed=0, n=80, k=2, cmp="=", cause=True):
    """T(x) holds when x points at exactly k marked things (= k), or at least k (>= k). With the cause off, T holds at
    random among things that point at one or more."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        ents = [f"e{i}.{j}" for j in range(7)]
        Q = {(a,) for a in ents if r.random() < .5}
        R = {(a, b) for a in ents for b in ents if a != b and r.random() < .3}
        cnt = {x: len({b for (a, b) in R if a == x and (b,) in Q}) for x in ents}
        if cause:
            T = {(x,) for x in ents if (cnt[x] == k if cmp == "=" else cnt[x] >= k)}
        else:
            T = {(x,) for x in ents if cnt[x] >= 1 and r.random() < .4}
        out.append(Structure({"R": R, "Q": Q, "T": T}))
    return out


T = K.Target("T", 1, ("R", "Q"))
S = K.Search(max_atoms=3)


@pytest.mark.invariant
def test_gamma_codes_the_positive_integers_and_prices_a_count_symbol():
    assert [C.gamma_bits(k) for k in (1, 2, 3, 4, 7, 8)] == [1, 3, 3, 5, 5, 7]
    assert sum(2.0 ** -C.gamma_bits(k) for k in range(1, 4096)) < 1               # a prefix code: Kraft holds
    assert C.structure_bits(4, 2, 2) == pytest.approx(2 + 1 + 1 + 3)


@pytest.mark.invariant
def test_a_definition_is_expanded_to_primitives_with_fresh_variables():
    defs = {"N": ((("R", (0, 2)), ("Q", (2,))), 1), "Z": None}
    got = C.inline((("N", (0,)), ("R", (0, 1))), defs)
    assert got == (("R", (0, 2)), ("Q", (2,)), ("R", (0, 1)))
    assert C.inline((("Z", (0,)),), defs) is None                                  # a cell is not a conjunction


@pytest.mark.invariant
def test_a_count_symbol_says_exactly_two_where_no_pattern_can():
    plain, _ = K.learn(_world(), [T], S)
    assert [r.atoms for r in plain.rules["T"]] == [(("Q", (1,)), ("R", (0, 1)))]   # the source: x points at a mark
    with_counts, _ = K.learn(_world(), [T], K.Search(max_atoms=3, counts=True))
    test = _world(seed=7)
    assert with_counts.counts_record["held"] == "second"
    held = {(z["cmp"], z["k"]): z["symbol"] for z in with_counts.counts["T"]}
    assert ("=", 2) in held                    # other counts may be held beside it when they code the same cells
    assert any(sym == held[("=", 2)] for r in with_counts.rules["T"] for sym, _ in r.atoms)
    assert K.heldout_bits(with_counts, test, T) < K.heldout_bits(plain, test, T) - 20
    assert all((p == y).all() or p.min() < 0 for p, y in zip(K.predict(with_counts, test, T), _truths(test)))


def _truths(structures):
    import numpy as np
    return [np.array([int(c in s.facts.get("T", ())) for c in s.candidates(1, T.domain)]) for s in structures]


@pytest.mark.invariant
def test_at_least_k_is_proposed_and_held_where_it_pays():
    m, _ = K.learn(_world(cmp=">="), [T], K.Search(max_atoms=3, counts=True))
    assert m.counts_record["held"] == "second"
    assert (">=", 2) in {(z["cmp"], z["k"]) for z in m.counts["T"]}          # {= 1, >= 2} codes the same cells as
    assert K.heldout_bits(m, _world(seed=7, cmp=">="), T) < 5                 # {the rule, >= 2}; either is the code


@pytest.mark.invariant
def test_no_count_symbol_is_held_when_the_count_does_not_decide():
    m, _ = K.learn(_world(cause=False), [T], K.Search(max_atoms=3, counts=True))
    assert m.counts_record["held"] == "first" and not any(m.counts.values())
    plain, _ = K.learn(_world(cause=False), [T], S)
    assert K.summary(m)["T"]["rules"] == K.summary(plain)["T"]["rules"]            # pass 1 held exactly


@pytest.mark.invariant
def test_renaming_changes_no_count_and_no_bit():
    data = _world(seed=2)
    r = random.Random(5)
    ents = sorted({e for s in data for e in s.entities(("R", "Q", "T"))})
    emap = {e: f"z{r.random():.10f}" for e in ents}
    smap = {"R": "q1", "Q": "q2", "T": "q3"}
    renamed = [rename(s, emap, smap) for s in data]
    a, _ = K.learn(data, [T], K.Search(max_atoms=3, counts=True))
    b, _ = K.learn(renamed, [K.Target("q3", 1, ("q1", "q2"))], K.Search(max_atoms=3, counts=True))
    assert a.counts_record["first_bits"] == pytest.approx(b.counts_record["first_bits"])
    assert a.counts_record["second_bits"] == pytest.approx(b.counts_record["second_bits"])
    assert [(z["cmp"], z["k"], z["members"]) for z in a.counts["T"]] == \
        [(z["cmp"], z["k"], z["members"]) for z in b.counts["q3"]]


@pytest.mark.invariant
def test_a_model_with_count_symbols_saves_and_reads_the_same(tmp_path):
    m, _ = K.learn(_world(), [T], K.Search(max_atoms=3, counts=True))
    K.save(m, tmp_path / "m.json")
    back = K.load(tmp_path / "m.json")
    test = _world(seed=7)
    assert K.heldout_bits(back, test, T) == pytest.approx(K.heldout_bits(m, test, T))


@pytest.mark.invariant
def test_a_rule_whose_witnesses_always_number_the_same_gives_its_copy():
    import numpy as np
    assert C.proposals(np.array([[1], [2], [3]]), np.array([1, 1, 1])) == [("=", 1)]   # a copy of the rule (CT-13)
    assert C.proposals(np.array([[1], [2], [3]]), np.array([1, 1, 1]), copies=False) == []      # CT-9's witness
    assert C.proposals(np.array([[1], [2], [3]]), np.array([1, 2, 2])) == [("=", 1), ("=", 2)]   # ">= 2" equals "= 2"


@pytest.mark.invariant
def test_counts_and_cells_run_together_and_cells_still_say_holding_nothing():
    """CT-10: the cell symbols' witness world with counts on as well; the cell symbol is still held and cited."""
    from test_kernel import FREE_TARGETS, _free
    ta, tb = FREE_TARGETS
    cells, _ = K.learn(_free(0), [ta, tb], K.Search(max_atoms=3, cells=True))
    both, _ = K.learn(_free(0), [ta, tb], K.Search(max_atoms=3, cells=True, counts=True))
    cited = lambda m: {s for r in m.rules["B"] for p in r.rivals for s, _ in p if ".c" in s}    # noqa: E731
    assert cited(cells) and cited(both)
    assert both.counts_record["with_cells"]
    test = _free(1)
    assert K.heldout_bits(both, test, tb) <= K.heldout_bits(cells, test, tb) + 1


@pytest.mark.invariant
def test_count_symbols_and_named_pieces_never_share_a_name():
    # a target's counts are defined before its pieces are named, in one Corpus, so equal names would merge their facts
    m, _ = K.learn(_world(), [T], K.Search(max_atoms=3, counts=True))
    counts = {z["symbol"] for zs in m.counts.values() for z in zs}
    assert counts and all(s.startswith("#T.k") for s in counts)
    assert not counts & {i["symbol"] for i in m.inventions.get("T", [])}


def _many(seed, depends=True, names=("dom", "r", "t"), prefix="e"):
    """Entities each with 1 to 5 r-related entities; t holds where there are at least 3 (or at random)."""
    import random
    from kernel.structure import Structure
    dom, r, t = names
    rng, out = random.Random(seed), []
    for s in range(8):
        f = {dom: set(), r: set(), t: set()}
        for i in range(6):
            x = f"{prefix}{s}x{i}"
            f[dom].add((x,))
            n = rng.randint(1, 5)
            for j in range(n):
                f[r].add((x, f"{x}y{j}"))
            if (n >= 3) if depends else rng.random() < 0.5:
                f[t].add((x,))
        out.append(Structure(f))
    return out


def _count_rules(structures, depth, t="t", dom="dom"):
    from kernel import learn as K
    m, _ = K.learn(structures, [K.Target(t, 1, (dom,))], K.Search(max_atoms=2, counts=True, count_relations=depth))
    return [r for r in m.rules[t] if any(sym.startswith("#") for sym, _ in r.atoms)], m


@pytest.mark.invariant
def test_a_count_over_relations_forms_where_it_pays_and_not_otherwise():
    held, _ = _count_rules(_many(0), 0)
    assert not held                                   # without the option: no held rule binds r, so nothing counts it
    held, m = _count_rules(_many(0), 1)
    assert held and m.counts_record["held"] == "second"
    z = next(z for z in m.counts["t"] if z["symbol"] == held[0].atoms[0][0])
    assert z["relation"] and z["k"] == 3 and z["cmp"] == ">="
    held, m = _count_rules(_many(0, depends=False), 1)
    assert not held


@pytest.mark.invariant
def test_renaming_changes_no_count_over_relations():
    _, a = _count_rules(_many(0), 2)
    _, b = _count_rules(_many(0, names=("thing", "link", "goal"), prefix="q"), 2, t="goal", dom="thing")
    assert [round(r.gain_bits, 6) for r in a.rules["t"]] == [round(r.gain_bits, 6) for r in b.rules["goal"]]
    assert [(z["k"], z["cmp"], z["var"]) for z in a.counts["t"]] == [(z["k"], z["cmp"], z["var"]) for z in b.counts["goal"]]
