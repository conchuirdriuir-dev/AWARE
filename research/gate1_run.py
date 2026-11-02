"""Gate 1, run (PLAN.md, research track): reads the states once, into a cache, then reads the verdict from them. No corpus
text enters the record or the cache, only references and mean states.

    python -m research.gate1_run --run runs/gate1-qwen8b --corpus ~/dara-session-transcripts/sessions
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from research import gate1
from research.d_jump import MODEL
from research.gate1_pairs import PAIRS
from substrate.geometry import spread
from substrate.measure_d import load_sets, measure
from substrate.record import Record

LAYERS = [12, 24, 28, 32]
REVIEW = "material/negation_review.jsonl"
NEGATION_WORD = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't", re.I)
SETS = "material/paraphrases.jsonl"


def rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def subsets(pairs: list[dict], review: list[dict]) -> dict:
    """Which pairs each line of the report keeps: as a boolean mask over the pairs."""
    reading = {r["pair"]: r["reading"] for r in review}
    wrong = np.array([reading.get(i) == "wrong" for i in range(len(pairs))])
    borderline = np.array([reading.get(i) == "borderline" for i in range(len(pairs))])
    negated = np.array([bool(NEGATION_WORD.search(p["opposite"])) for p in pairs])
    return {"all": np.ones(len(pairs), bool), "without wrong": ~wrong, "without wrong or borderline": ~wrong & ~borderline,
            "post hoc: negation word": negated, "post hoc: no negation word": ~negated}


def verdicts(X, Y, groups, D: dict, model=gate1.FREE) -> dict:
    """Both scopes at each layer: arrays are pairs x layers x width. Errors in units of D."""
    out = {}
    for scope, score in (("global", gate1.held_out_by_topic), ("per topic", gate1.leave_one_out_per_topic)):
        layers = []
        for j, x in enumerate(LAYERS):
            e = score(X[:, j], Y[:, j], groups, model)
            layers.append({"layer": x, "verdict": gate1.decide(e, D[x]),
                           "as_first_written": gate1.decide(e, D[x], nesting=False),
                           **{h: (e[h] / D[x] if h != "lam" else e[h]) for h in e}})
        out[scope] = {"layers": layers, "verdict": gate1.scope_verdict([r["verdict"] for r in layers]),
                      "as_first_written": gate1.scope_verdict([r["as_first_written"] for r in layers])}
    return out


def void_against(result: dict, shuffled: dict) -> dict:
    """A form that wins a scope on shuffled pairs has its win there voided."""
    return {s: ("void" if result[s]["verdict"] != "undecided" and result[s]["verdict"] == shuffled[s]["verdict"]
                else result[s]["verdict"]) for s in result}


def midpoint_gap(X, Y, u, b) -> float:
    return float(abs(((X + Y) / 2 @ u).mean() - b))


def centres(X, Y, Ysh, groups, par, sessions, d: float) -> dict:
    """The fitted centre's distance from each candidate, in D: X, Y, Ysh, par, sessions at one layer. Not judged."""
    def about(form, param, Xs=None, Yss=None) -> dict | None:
        ds = [gate1.distance_from_centre(form, param, s) for s in sessions]
        if ds[0] is None:
            return None
        return {"paraphrase mean": gate1.distance_from_centre(form, param, par) / d,
                "mean of sessions": gate1.distance_from_centre(form, param, sessions.mean(0)) / d,
                "one session median": float(np.median(ds)) / d, "one session min": min(ds) / d, "one session max": max(ds) / d}

    out = {}
    fit, shf = gate1.fit_free(X, Y), gate1.fit_free(X, Ysh)
    u, b = fit["mirror"]
    us, _ = shf["mirror"]
    out["mirror, global"] = {**about("mirror", fit["mirror"]), "b_over_D": b / d, "normal_vs_shuffled_cos": float(abs(u @ us)),
                             "share_of_x_minus_y_along_u": float(((X - Y) @ u).var() + ((X - Y) @ u).mean() ** 2)
                             / float(((X - Y) ** 2).sum(1).mean())}
    lam, v = fit["contraction"]
    lams, vs = shf["contraction"]
    centre, cs = gate1.contraction_centre(lam, v), gate1.contraction_centre(lams, vs)
    row = about("contraction", fit["contraction"]) or {"no finite centre": True}
    if centre is not None and cs is not None:
        row["shuffled centre"] = float(np.linalg.norm(centre - cs)) / d
    out["contraction, global"] = {**row, "lam": lam, "lam_shuffled": lams}
    per = {"mirror": [], "contraction": []}
    none = 0
    for g in np.unique(groups):
        w = groups == g
        f, fs = gate1.fit_free(X[w], Y[w]), gate1.fit_free(X[w], Ysh[w])
        row = about("mirror", f["mirror"])
        row["shuffled midpoint"] = midpoint_gap(X[w], Ysh[w], *f["mirror"]) / d
        per["mirror"].append(row)
        row = about("contraction", f["contraction"])
        if row is None:
            none += 1
            continue
        c, cs = gate1.contraction_centre(*f["contraction"]), gate1.contraction_centre(*fs["contraction"])
        if cs is not None:
            row["shuffled centre"] = float(np.linalg.norm(c - cs)) / d
        row["lam"] = f["contraction"][0]
        per["contraction"].append(row)
    for form, rs in per.items():
        out[f"{form}, per topic (median)"] = {k: float(np.median([r[k] for r in rs if k in r])) for k in rs[0]} if rs else {}
    out["contraction, per topic (median)"]["topics_without_a_finite_centre"] = none
    return out


def read_states(a, record: Record, cache: Path) -> None:
    from research.origin_candidates import earliest_externals
    from substrate.echo_floor import corpus_paths
    from substrate.states import Reader

    pairs = rows(PAIRS)
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    report = measure(reader, load_sets(SETS), LAYERS, record)
    S = np.load(record.path.parent / "states.npy").astype(np.float32)
    X = reader.states([p["statement"] for p in pairs], LAYERS)
    Y = reader.states([p["opposite"] for p in pairs], LAYERS)
    five, _, refs = earliest_externals(reader, corpus_paths(a.corpus), LAYERS)
    record.append("earliest externals", "code", refs=refs)
    tok = reader.tokenizer
    count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    lengths = {"statements": float(np.median([count(p["statement"]) for p in pairs])),
               "opposites": float(np.median([count(p["opposite"]) for p in pairs])),
               "paraphrase texts": float(np.median([count(t) for s in load_sets(SETS) for t in s["texts"]]))}
    record.append("token lengths, median", "code", **lengths)
    np.savez(cache, X=X, Y=Y, S=S, sessions=five.astype(np.float32))
    record.append("states read", "code", cache=str(cache), layers=LAYERS, D={r["layer"]: r["D"] for r in report["layers"]})


def print_table(name: str, result: dict) -> None:
    for scope, r in result.items():
        print(f"\n{name} / {scope}: verdict {r['verdict']}  (as first written: {r['as_first_written']})")
        print(f"  {'layer':>5} {'identity':>9} {'additive':>9} {'contract':>9} {'mirror':>9} {'lam':>6}  verdict")
        for x in r["layers"]:
            print(f"  {x['layer']:>5} {x['identity']:>9.3f} {x['additive']:>9.3f} {x['contraction']:>9.3f} "
                  f"{x['mirror']:>9.3f} {x['lam']:>6.2f}  {x['verdict']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--corpus", required=True)
    a = p.parse_args(argv)
    record = Record(Path(a.run) / "record.jsonl")
    cache = Path(a.run) / "gate1_states.npz"
    if not cache.exists():
        read_states(a, record, cache)
    D = next(r for r in reversed(rows(record.path)) if r["kind"] == "states read")["D"]
    D = {int(k): v for k, v in D.items()}
    A = np.load(cache)
    X, Y, S, sessions = (A[k].astype(np.float64) for k in ("X", "Y", "S", "sessions"))

    pairs, review = rows(PAIRS), rows(REVIEW)
    names = sorted({p["topic"] for p in pairs})
    groups = np.array([names.index(p["topic"]) for p in pairs])
    report = {}
    keep = subsets(pairs, review)
    for name, mask in keep.items():
        report[name] = verdicts(X[mask], Y[mask], groups[mask], D)
    perm = gate1.shuffle_opposites(groups)
    Ysh = Y[perm]
    report["shuffled control"] = verdicts(X, Ysh, groups, D)
    Xp, Yp, gp = paraphrase_pairs_all(S)
    report["paraphrase control"] = verdicts(Xp, Yp, gp, D)
    par = S.reshape(-1, *S.shape[2:]).mean(0)                   # layers x width
    report["fixed at paraphrase mean"] = {}
    for scope, score in (("global", gate1.held_out_by_topic), ("per topic", gate1.leave_one_out_per_topic)):
        layers = []
        for j, x in enumerate(LAYERS):
            e = score(X[:, j], Y[:, j], groups, gate1.fixed_centre(par[j]))
            v = gate1.decide(e, D[x], nesting=False)            # its contraction has no free offset, so it nests nothing
            layers.append({"layer": x, "verdict": v, "as_first_written": v,
                           **{h: (e[h] / D[x] if h != "lam" else e[h]) for h in e}})
        report["fixed at paraphrase mean"][scope] = {"layers": layers, "verdict": gate1.scope_verdict([r["verdict"] for r in layers]),
                                                     "as_first_written": gate1.scope_verdict([r["verdict"] for r in layers])}
    voided = {k: void_against(report[k], report["shuffled control"]) for k in keep}
    for name, result in report.items():
        print_table(name, result)
    print("\nafter the shuffled control's void rule:", json.dumps(voided, indent=1))

    where = {}
    for j, x in enumerate(LAYERS):
        where[x] = centres(X[:, j], Y[:, j], Ysh[:, j], groups, par[j], sessions[:, j], D[x])
    print("\ncentres, in D:", json.dumps(where, indent=1))

    beside = []
    for j, x in enumerate(LAYERS):
        topic = [X[groups == g, j] for g in np.unique(groups)]
        opp = [Y[groups == g, j] for g in np.unique(groups)]
        beside.append({"layer": x, "D": D[x], "statement_to_opposite": float(np.sqrt(((X[:, j] - Y[:, j]) ** 2).sum(1).mean())) / D[x],
                       "statements_within_topic": spread(topic) / D[x], "opposites_within_topic": spread(opp) / D[x]})
    print("\nbeside, in D:", json.dumps(beside, indent=1))
    record.append("gate 1", "code", report=report, void_rule_applied=voided, centres=where, beside=beside, pairs=len(pairs),
                  subset_sizes={k: int(v.sum()) for k, v in keep.items()}, shuffle_seed=0)


def paraphrase_pairs_all(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S is sets x texts x layers x width: each layer's paraphrase pairs, stacked to pairs x layers x width."""
    made = [gate1.paraphrase_pairs(S[:, :, j]) for j in range(S.shape[2])]
    return np.stack([m[0] for m in made], 1), np.stack([m[1] for m in made], 1), made[0][2]


if __name__ == "__main__":
    main()
