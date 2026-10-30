"""The context-tree data code (KERNEL §4 as amended; SMALLEST_AGENT §9ak, SA-37): a prefix code over the held rules'
cells that is never far from the cells code, agrees with it when there are no rules, continues sequentially, and lets
a cell the evidence has not shown inherit the rate of the cells above it."""
import math

import numpy as np
import pytest

from kernel import code, learn as K
from kernel.structure import Structure, rename


@pytest.mark.invariant
def test_no_rules_the_two_codes_agree_and_one_rule_costs_at_most_the_tree():
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.3).astype(np.int64)
    assert code.target_bits([], y, "ctw") == pytest.approx(code.target_bits([], y, "cells"))
    cols = [(rng.random(200) < 0.5).astype(np.int64) for _ in range(3)]
    cells, ctw = code.target_bits(cols, y, "cells"), code.target_bits(cols, y, "ctw")
    nodes = sum(len(level) for level in code.CTW(code.counts(code.cell_ids(cols, 200), y), 3).levels)
    assert ctw <= cells + nodes                                 # at most one bit per node over the best pruning
    assert ctw <= code.target_bits([], y, "ctw") + nodes        # and never far from the root alone


@pytest.mark.invariant
def test_the_code_is_sequential_and_an_unseen_cell_inherits_its_parents_rate():
    train = {0b00: (80, 80), 0b10: (80, 0)}                     # the first rule separates on from off; the second never seen
    tree = code.CTW(train, 2)
    p = code.predictive(train, 0b01, 2)                          # a candidate in the first rule's on-cell, the second rule holding
    assert 0.6 < p < 0.95 and code.predictive(train, 0b01) == 0.5   # the cells code knows nothing of it
    assert code.predictive(train, 0b11, 2) < 0.4
    whole = dict(train)
    whole[0b01] = (5, 5)
    assert code.continued_cell_bits(train, 0b01, 5, 5, 2) == pytest.approx(code.CTW(whole, 2).bits() - tree.bits())
    assert code.predictive(train, 0b01, 2) == pytest.approx(2.0 ** -code.continued_cell_bits(train, 0b01, 1, 1, 2))


def ring(prefix="c", sym=("site", "next", "on")):
    site, nxt, on = sym
    sites = [f"{prefix}{i}" for i in range(8)]
    return [Structure({site: {(s,) for s in sites}, nxt: {(sites[i], sites[(i + 1) % 8]) for i in range(8)},
                       on: {(s,) for s in sites[:4]}, "Z": {(s,) for s in sites[:4]}}) for _ in range(20)]


@pytest.mark.invariant
def test_formation_under_the_tree_code_reads_no_name_and_records_its_code():
    search = K.Search(max_atoms=2, data_code="ctw")
    a, _ = K.learn(ring(), [K.Target("on", 1, ("site",))], search)
    b, _ = K.learn(ring("k", ("cell", "beside", "lit")), [K.Target("lit", 1, ("cell",))], search)
    assert a.data_code == "ctw" and a.train_bits["on"] == pytest.approx(b.train_bits["lit"])
    assert [r.gain_bits for r in a.rules["on"]] == pytest.approx([r.gain_bits for r in b.rules["lit"]])
    assert K.from_record(K.to_record(a)).data_code == "ctw"
    plain, _ = K.learn(ring(), [K.Target("on", 1, ("site",))], K.Search(max_atoms=2, data_code="cells"))
    assert a.train_bits["on"] == pytest.approx(plain.train_bits["on"] + 1.0, abs=1e-6)   # one rule: the one mixing bit


def _thin_lift_world(seed, n_ids=32, steps=30, lift=0.15, effect=0.2):
    """Four attributes that each raise the target's rate, and a thin lift z over every cell they make."""
    import random
    r = random.Random(seed)
    ids = [f"e{i}" for i in range(n_ids)]
    attrs = {e: tuple(r.random() < 0.5 for _ in range(4)) for e in ids}
    zs = set(ids[: n_ids // 2])
    out = []
    for _ in range(steps):
        facts = {"ent": {(e,) for e in ids}, "a": set(), "b": set(), "c": set(), "d": set(), "z": {(e,) for e in zs}, "on": set()}
        for e in ids:
            for sym, v in zip("abcd", attrs[e]):
                if v:
                    facts[sym].add((e,))
            p = 0.08 + effect * sum(attrs[e]) + (lift if e in zs else 0.0)
            if r.random() < min(p, 0.97):
                facts["on"].add((e,))
        out.append(Structure(facts))
    return out


@pytest.mark.invariant
def test_a_thin_rule_refused_as_the_deepest_split_is_admitted_where_it_stands_best():
    """SA-45's witness (SMALLEST_AGENT §9av, §9ay): on fragmented evidence the lift z, spread thinly over the cells the
    attribute rules make, fails the margin as the deepest split and under the cell table, and is admitted when the
    search may place it first or last; the code with it is shorter than either."""
    ev = _thin_lift_world(2)
    t = [K.Target("on", 1, ("ent",))]
    runs = {key: K.learn(ev, t, K.Search(max_atoms=1, data_code=dc, tree_place=place, invent=False, cells=False))[0]
            for key, dc, place in (("cells", "cells", "best"), ("last", "ctw", "last"), ("best", "ctw", "best"))}
    names = {k: [r.atoms[0][0] for r in m.rules["on"]] for k, m in runs.items()}
    assert "z" not in names["last"] and "z" not in names["cells"] and "z" in names["best"]
    assert runs["best"].train_bits["on"] < runs["last"].train_bits["on"]
    assert runs["best"].data_code == "ctw"
