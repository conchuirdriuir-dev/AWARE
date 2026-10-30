"""The first witness for latent classes (docs/RECURSIVE_ABSTRACTION_v0_1.md §4): a hidden kind the kernel must posit.

A world of persistent identities (p0..p15), half of them of a hidden kind. Each structure mentions two of them; each
mention says at most one word. Members are mostly said with the kind's words (w1-w3), others with the other words
(w4-w6). No word co-occurs with another about the same thing in a structure, and identity is supplied (`id:pN`), so
without a latent symbol the kernel can only learn word by word and identity by identity: memory.

    coupled    the kind decides which words are said (90%)
    off        a coin decides, the same rates, nothing to find (control)

Reported per world and seed: the baseline code, the class found and its gain, whether it is admitted, whether it
equals the hidden kind (up to complement), the search's kernel runs; transfer: a new member, q, seen once, placed by
that one sentence, then its later words coded with and without the class; renaming: every name and symbol renamed,
the same bits and the same partition. Optional: an exhaustive sweep over every class of a smaller cast, to check
that the alternation finds the shortest.

    PYTHONPATH=src python scripts/latent_witness_probe.py [--seeds 3] [--ids 16] [--structures 300] [--exhaustive]
"""
from __future__ import annotations

import argparse
import itertools
import random
import time

import numpy as np

from kernel import learn as kl
from kernel import latent as lt
from kernel.structure import Structure, rename

KIND, OTHER = ("w1", "w2", "w3"), ("w4", "w5", "w6")
WORDS = KIND + OTHER
SEARCH = kl.Search(max_atoms=2, beam=8, max_rules=16, bound="positives", budget=None, invent=False, ties="stop",
                   cells=False)
LATENT = lt.Latent(name="Z", domain=("ent",))


def targets(words=WORDS) -> list[kl.Target]:
    syms = [f"a:{w}" for w in words]
    return [kl.Target(s, 1, ("ent",), hide=tuple(x for x in syms if x != s)) for s in syms]


def mention(r: random.Random, member: bool, world: str) -> str | None:
    if r.random() >= 0.7:
        return None
    kind = (r.random() < 0.9) == member if world == "coupled" else r.random() < 0.5
    return r.choice(KIND if kind else OTHER)


def structure(r: random.Random, people: list[str], kind: set, world: str) -> Structure:
    facts: dict = {"ent": set()}
    for p in people:
        facts["ent"].add((p,))
        facts.setdefault(f"id:{p}", set()).add((p,))
        w = mention(r, p in kind, world)
        if w:
            facts.setdefault(f"a:{w}", set()).add((p,))
    return Structure(facts)


def life(seed: int, n_ids: int, n_structures: int, world: str):
    r = random.Random(seed)
    ids = [f"p{i}" for i in range(n_ids)]
    kind = set(r.sample(ids, n_ids // 2))
    return [structure(r, r.sample(ids, 2), kind, world) for _ in range(n_structures)], kind, ids, r


def same_partition(a, b, ids) -> bool:
    return set(a) == set(b) or set(a) == set(ids) - set(b)


def transfer(inv: lt.Invention, r: random.Random, ids, kind, world: str) -> dict:
    """q, a new member, is said with a kind word once (with someone known), then appears in 20 more structures."""
    first = Structure({"ent": {("q",), (ids[0],)}, "id:q": {("q",)}, f"id:{ids[0]}": {(ids[0],)},
                       "a:w1": {("q",)}})
    later = [structure(r, ["q", r.choice(ids)], kind | {"q"}, world) for _ in range(20)]
    placed = lt.place(inv, [first], "q", LATENT)

    def q_bits(model, members):
        ev = lt.with_class(later, members, LATENT.name, LATENT.domain) if model is not inv.baseline else later
        c = kl.apply(model, ev)
        out = 0.0
        for t in model.targets:
            cells, _ = kl.cells_for(model, c, t)
            y = kl.outcomes(c, t)
            tc = model.train_counts[t.symbol]
            rows = [c.rows(1, t.domain)["start"][i] + list(s.entities(t.domain)).index("q")
                    for i, s in enumerate(later)]
            for row in rows:
                n, k = tc.get(int(cells[row]), (0, 0))
                p1 = (k + 0.5) / (n + 1.0)
                out += -np.log2(p1 if y[row] else 1 - p1)
        return out
    got = {"placed": placed, "baseline_bits": q_bits(inv.baseline, set())}
    if inv.admitted:
        inside = inv.members | {"q"} if set(inv.members) == kind else inv.members      # q on the kind's side
        got["placed_bits"] = q_bits(inv.model, inv.members | ({"q"} if placed else set()))
        got["truth_bits"] = q_bits(inv.model, inside)
    return got


def renamed(structures, ids):
    ents = {e: f"x{(7 * i + 3) % len(ids)}" for i, e in enumerate(ids)}
    syms = {f"id:{e}": f"id:{v}" for e, v in ents.items()} | {f"a:{w}": f"a:v{i}" for i, w in enumerate(WORDS)}
    syms["ent"] = "thing"
    return [rename(s, ents, syms) for s in structures], ents, syms


def run(seed: int, n_ids: int, n_structures: int, world: str, check_renaming: bool) -> dict:
    structures, kind, ids, r = life(seed, n_ids, n_structures, world)
    t0 = time.time()
    inv = lt.invent(structures, targets(), SEARCH, LATENT)
    out = {"world": world, "seed": seed, "baseline_bits": inv.baseline_bits, "bits": inv.bits,
           "gain_bits": inv.gain_bits, "admitted": inv.admitted, "members": sorted(inv.members),
           "is_kind": same_partition(inv.members, kind, ids), "learns": inv.learns, "seconds": time.time() - t0,
           "memory_rules": sum(len(rs) for rs in inv.baseline.rules.values())}
    if inv.admitted:
        out["rules"] = {t.symbol: [" ∧ ".join(f"{s}{a}" for s, a in r.atoms) for r in inv.model.rules[t.symbol]]
                        for t in inv.model.targets}
    if world == "coupled":
        out["transfer"] = transfer(inv, r, ids, kind, world)
    if check_renaming:
        rs, ents, syms = renamed(structures, ids)
        tg = [kl.Target(syms[t.symbol], 1, ("thing",), hide=tuple(syms[h] for h in t.hide)) for t in targets()]
        rinv = lt.invent(rs, tg, SEARCH, lt.Latent(name="Z", domain=("thing",)))
        back = {v: k for k, v in ents.items()}
        out["renaming_same"] = (abs(rinv.gain_bits - inv.gain_bits) < 1e-6 and rinv.admitted == inv.admitted
                                and same_partition({back[e] for e in rinv.members}, inv.members, ids))
    return out


def exhaustive(seed: int, n_ids: int, n_structures: int) -> dict:
    structures, kind, ids, _ = life(seed, n_ids, n_structures, "coupled")
    inv = lt.invent(structures, targets(), SEARCH, LATENT)
    best = None
    for k in range(len(ids) + 1):
        for members in itertools.combinations(ids, k):
            m, _ = kl.learn(lt.with_class(structures, members, "Z", ("ent",)), targets(), SEARCH)
            bits = lt.code_bits(m) + lt.assignment_bits(len(ids), k)
            if best is None or bits < best[0]:
                best = (bits, set(members))
    return {"seed": seed, "ids": n_ids, "alternation_bits": inv.bits, "exhaustive_bits": best[0],
            "alternation_members": sorted(inv.members), "exhaustive_members": sorted(best[1]),
            "kind": sorted(kind), "learns": inv.learns, "classes": 2 ** n_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ids", type=int, default=16)
    ap.add_argument("--structures", default="300,1200", help="life lengths, comma-separated (300: memory does not "
                                                            "pay; 1200: it does)")
    ap.add_argument("--exhaustive", action="store_true", help="also sweep every class of 8 identities")
    ap.add_argument("--no-renaming", action="store_true")
    args = ap.parse_args()
    sizes = [int(x) for x in args.structures.split(",")]
    for world, n, seed in itertools.product(("coupled", "off"), sizes, range(args.seeds)):
        o = run(seed, args.ids, n, world, not args.no_renaming and n == sizes[0])
        print(f"{world:8s} {n:5d} structures, seed {seed}: baseline {o['baseline_bits']:.1f} bits "
              f"({o['memory_rules']} rules), with Z {o['bits']:.1f}; "
              f"gain {o['gain_bits']:+.1f}; admitted {o['admitted']}; Z is the kind: {o['is_kind']}; "
              f"{o['learns']} kernel runs, {o['seconds']:.0f} s"
              + (f"; renaming identical: {o['renaming_same']}" if "renaming_same" in o else ""))
        if o.get("rules") and seed == 0:
            for sym, rs in o["rules"].items():
                print(f"    {sym}: {'; '.join(rs) or 'none'}")
        if "transfer" in o:
            t = o["transfer"]
            print(f"    transfer: q placed in Z from one sentence: {t['placed']}; q's 20 later sentences "
                  f"{t.get('placed_bits', float('nan')):.1f} bits with Z, {t['baseline_bits']:.1f} without")
    if args.exhaustive:
        for seed in range(args.seeds):
            e = exhaustive(seed, 8, sizes[0])
            print(f"exhaustive seed {seed} (8 identities, {e['classes']} classes): alternation "
                  f"{e['alternation_bits']:.1f} bits in {e['learns']} runs, exhaustive best {e['exhaustive_bits']:.1f}; "
                  f"same class: {set(e['alternation_members']) == set(e['exhaustive_members'])}")


if __name__ == "__main__":
    main()
