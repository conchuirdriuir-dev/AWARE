"""Latent classes (docs/RECURSIVE_ABSTRACTION_v0_1.md): the first witness, at its smallest."""
import math
import sys
from pathlib import Path

import pytest

from kernel import latent as lt
from kernel.structure import Structure

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_witness_probe as probe  # noqa: E402


@pytest.mark.invariant
def test_a_hidden_kind_is_posited_once_and_only_where_there_is_one():
    structures, kind, ids, _ = probe.life(0, 16, 300, "coupled")
    inv = lt.invent(structures, probe.targets(), probe.SEARCH, probe.LATENT)
    assert inv.admitted and probe.same_partition(inv.members, kind, ids)
    assert not any(r.atoms[0][0].startswith("id:") for rs in inv.model.rules.values() for r in rs)
    cited = {sym for rs in inv.model.rules.values() for r in rs for sym, _ in r.atoms}
    assert "Z" in cited and len(inv.model.rules["a:w1"]) == 1

    off, _, _, _ = probe.life(0, 16, 300, "off")
    assert not lt.invent(off, probe.targets(), probe.SEARCH, probe.LATENT).admitted


@pytest.mark.invariant
def test_a_new_identity_is_placed_by_one_sentence():
    structures, kind, ids, _ = probe.life(0, 16, 300, "coupled")
    inv = lt.invent(structures, probe.targets(), probe.SEARCH, probe.LATENT)
    said = lambda w: Structure({"ent": {("q",), ("p0",)}, "id:q": {("q",)}, "id:p0": {("p0",)},  # noqa: E731
                                f"a:{w}": {("q",)}})
    kind_side = set(inv.members) == kind
    assert lt.place(inv, [said("w1")], "q", probe.LATENT) == kind_side
    assert lt.place(inv, [said("w5")], "q", probe.LATENT) != kind_side


@pytest.mark.invariant
def test_renaming_changes_no_bit_and_no_partition():
    assert probe.run(1, 16, 300, "coupled", check_renaming=True)["renaming_same"]


def test_the_assignment_code_is_symmetric_and_prefix():
    assert lt.assignment_bits(16, 4) == pytest.approx(lt.assignment_bits(16, 12))
    assert sum(2 ** -lt.assignment_bits(6, k) * math.comb(6, k) for k in range(7)) <= 1 + 1e-9


@pytest.mark.invariant
def test_invention_repeats_only_while_a_class_pays():
    """One hidden kind: the first class is it, the second does not pay, and the evidence carries the first."""
    structures, kind, ids, _ = probe.life(0, 16, 300, "coupled")
    got = lt.invent_all(structures, probe.targets(), probe.SEARCH, probe.LATENT)
    assert [i.admitted for i in got.attempts] == [True, False] and len(got.classes) == 1
    assert probe.same_partition(got.classes[0], kind, ids) and got.gain_bits > 100
    assert all("Z1" in s.facts for s in got.evidence) and not any("Z2" in s.facts for s in got.evidence)


def _rang_world(seed: int = 0, n: int = 400, cause: bool = True):
    """Six objects in every step, each with a fixed shape; one is acted on; `rang()` holds when it is of a hidden kind
    (90%; a coin with the cause off). Only which object was acted on varies from step to step."""
    import random
    from kernel import learn as kl
    r = random.Random(seed)
    objs, kind = [f"o{i}" for i in range(6)], {"o0", "o3", "o4"}
    shape = {o: ("round" if i % 2 else "square") for i, o in enumerate(objs)}
    out = []
    for _ in range(n):
        o = r.choice(objs)
        rang = (r.random() < 0.9) == (o in kind) if cause else r.random() < 0.5
        f = {"object": {(x,) for x in objs}, "act": {("a",)}, "obj": {("a", o)}, "rang": {()} if rang else set()}
        for x in objs:
            f.setdefault(shape[x], set()).add((x,))
            f[f"id:{x}"] = {(x,)}
        out.append(Structure(f))
    return out, kind, objs, [kl.Target("rang", 0, ())]


@pytest.mark.invariant
def test_a_0_ary_effect_seeds_the_class_through_what_varies():
    """Every object is in every step, so the step itself credits none; the object acted on is the one that varies."""
    from kernel import learn as kl
    st, kind, objs, targets = _rang_world()
    base, _ = kl.learn(st, targets, kl.Search(max_atoms=3))
    seeds = [{e for e, x in zip(objs, res) if x > 0} for res in lt._residuals(base, st, tuple(objs), ("object",), False).values()]
    assert kind in seeds
    lat = lt.Latent(name="Z", domain=("object",))
    inv = lt.invent(st, targets, kl.Search(max_atoms=3), lat)
    assert inv.admitted and probe.same_partition(inv.members, kind, objs)
    off, _, _, _ = _rang_world(cause=False)
    assert not lt.invent(off, targets, kl.Search(max_atoms=3), lat).admitted


@pytest.mark.invariant
def test_a_class_equal_to_a_targets_extension_is_a_hypothesis():
    """SMALLEST_AGENT §9w (SA-28): on the ring's infancy the form Z = {c0..c3} is the target `on`'s own extension; a
    rule for `on` may not cite `on`, so the class says what no rule can and is admitted (I-5 excepts the targets)."""
    from kernel import learn as kl
    sites = [f"c{i}" for i in range(8)]
    st = [Structure({"site": {(s,) for s in sites}, "next": {(sites[i], sites[(i + 1) % 8]) for i in range(8)},
                     "on": {(s,) for s in sites[:4]}}) for _ in range(20)]
    inv = lt.invent(st, [kl.Target("on", 1, ("site",))], kl.Search(max_atoms=2), lt.Latent(name="Z", domain=("site",)))
    assert inv.admitted and inv.members == frozenset(sites[:4]) and inv.gain_bits > 100
    assert [r.atoms for r in inv.model.rules["on"]] == [(("Z", (0,)),)]
