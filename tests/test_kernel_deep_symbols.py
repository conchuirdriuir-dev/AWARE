"""The reopened depth, restricted to patterns citing a deep symbol (architecture §4.4.51)."""
from __future__ import annotations

import random

import pytest

from kernel import learn as K
from kernel.pattern import canonical
from kernel.structure import Structure

pytestmark = pytest.mark.invariant


def _three(seed=0, n=60, mark="K"):
    """T(x) <- R(x, y) & Q(y) & <mark>(y): a three-atom rule; Q2 is unrelated noise."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        ents = [f"e{i}.{j}" for j in range(6)]
        R = {(a, b) for a in ents for b in ents if a != b and r.random() < .3}
        facts = {s: {(a,) for a in ents if r.random() < .5} for s in ("Q", "Q2", "K", "M")}
        T = {(x,) for (x, y) in R if (y,) in facts["Q"] and (y,) in facts[mark]}
        out.append(Structure({"R": R, **facts, "T": T}))
    return out


TARGET = K.Target("T", 1, ("R", "Q", "Q2", "K", "M"))


def rules(model):
    return sorted(canonical(r.atoms, 1) for r in model.rules["T"])


def test_an_empty_deep_set_is_the_search_as_it_was():
    data = _three(mark="M")
    plain, _ = K.learn(data, [TARGET], K.Search(max_atoms=3))
    empty, _ = K.learn(data, [TARGET], K.Search(max_atoms=3, deep_symbols=frozenset()))
    assert rules(plain) == rules(empty)


def test_a_rule_beyond_two_atoms_must_cite_a_deep_symbol():
    data = _three(mark="M")                                  # the true rule cites M, not the deep symbol K
    plain, _ = K.learn(data, [TARGET], K.Search(max_atoms=3))
    deep, _ = K.learn(data, [TARGET], K.Search(max_atoms=3, deep_symbols=frozenset({"K"})))
    assert any(len(r) == 3 for r in rules(plain))            # unrestricted, the three-atom rule is found
    assert all(len(r) <= 2 or any(a[0] == "K" for a in r) for r in rules(deep))


def test_a_true_rule_citing_the_deep_symbol_is_still_found():
    data = _three(mark="K")
    plain, _ = K.learn(data, [TARGET], K.Search(max_atoms=3))
    deep, _ = K.learn(data, [TARGET], K.Search(max_atoms=3, deep_symbols=frozenset({"K"})))
    assert rules(deep) == rules(plain)
    assert any(len(r) == 3 and any(a[0] == "K" for a in r) for r in rules(deep))


def test_depth_two_is_unchanged_by_the_field():
    data = _three(mark="K")
    two, _ = K.learn(data, [TARGET], K.Search(max_atoms=2))
    two_deep, _ = K.learn(data, [TARGET], K.Search(max_atoms=2, deep_symbols=frozenset({"K"})))
    assert rules(two) == rules(two_deep)
