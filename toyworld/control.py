"""Stage 1: controllability learned from interventions (toyworld/SPEC.md, "Stage 1").

Each day an anomaly was present is one structure for DARA's kernel (`kernel/`): what was done after that day
(`did:<action>`, by `world.action_symbol`) and after the day before (`did:<action>@-1`), and whether the anomaly fell by
the next day (`fell:<anomaly>`). Giving the delay as evidence lets the kernel decide by MDL which lag explains a fall: an
order works the next day, a head-office report that names only the problem two days later. A fall is never credited to
a day other than the one whose action could have caused it. For each anomaly the kernel forms one-atom rules over the
days it was open, admitted only on MDL gain above its 2-bit margin. The anomalies are measured on what the system's
actions cannot write: the reports for stockout and the till, the bank for reconciliation, never the system's own
reconciliation.
"""
from __future__ import annotations

from kernel import learn as K
from kernel.structure import Structure

from .world import action_symbol

ANOMALIES = ("stockout", "till_minimum", "reconciliation")
LAGS = 2            # an action may show its effect the next day or the one after (the late-levers condition delays
                    # orders by one)
FALL = 0.75         # a fall: below three quarters of the previous day's value
OPEN = {"stockout": 3, "till_minimum": 0.0, "reconciliation": 0.0}   # present: above this value
MIN_TRIED = 3       # tries while present before "no effect" is read rather than "untested"


def series(s: dict, anomaly: str) -> list[float]:
    if anomaly == "stockout":
        return [r["oat_milk_unfilled_requests"] for r in s["reports"]]
    if anomaly == "till_minimum":
        return [r["sub5_raise_rate"] for r in s["reports"]]
    return [b["shortfall"] for b in s["bank"]]          # the bank, which the system's actions cannot write


def action_of(symbol: str) -> str:
    """The action a lagged symbol names: did:set_order_qty@-1 -> did:set_order_qty."""
    return symbol.split("@")[0]


def days(states: list[dict]) -> list[dict]:
    """One row per day with an observed next day: what was done after it and the days before, and per anomaly whether
    it was present and fell by the next day."""
    rows = []
    for s in states:
        done = {a["day"]: sorted({x for x in (action_symbol(y) for y in a["actions"]) if x}) for a in s["actions"]}
        for t in sorted(done):
            did = [f"{d}@-{k}" if k else d for k in range(LAGS) for d in done.get(t - k, [])]
            row = {"run": s["run"], "day": t, "did": sorted(set(did)), "open": [], "fell": []}
            for anomaly in ANOMALIES:
                xs = series(s, anomaly)
                if t < len(xs) and xs[t - 1] > OPEN[anomaly]:
                    row["open"].append(anomaly)
                    if xs[t] < FALL * xs[t - 1]:
                        row["fell"].append(anomaly)
            rows.append(row)
    return rows


def structures(rows: list[dict]) -> list[Structure]:
    out = []
    for r in rows:
        facts = {"day": {("d",)}}
        facts.update({d: {("d",)} for d in r["did"]})
        facts.update({f"open:{x}": {("d",)} for x in r["open"]})
        facts.update({f"fell:{x}": {("d",)} for x in r["fell"]})
        out.append(Structure(facts))
    return out


def learn(states: list[dict]) -> dict:
    """The learned map, with the counts behind every entry, so an offer can cite them."""
    rows = days(states)
    body = tuple(sorted({d for r in rows for d in r["did"]}))
    targets = [K.Target(f"fell:{x}", 1, (f"open:{x}",), body=body) for x in ANOMALIES]
    model, _ = K.learn(structures(rows), targets, K.Search(max_atoms=1, invent=False, cells=False))
    out = {"evidence_days": len(rows), "lags": LAGS, "fall": FALL, "anomalies": {}}
    for x in ANOMALIES:
        open_rows = [r for r in rows if x in r["open"]]
        tallies = {}
        for d in body:
            w = [r for r in open_rows if d in r["did"]]
            wo = [r for r in open_rows if d not in r["did"]]
            if w:
                tallies[d] = {"tried": len(w), "fell": sum(x in r["fell"] for r in w),
                              "without": len(wo), "fell_without": sum(x in r["fell"] for r in wo)}
        moves, rules = [], []
        for rule in model.rules[f"fell:{x}"]:
            for atoms in rule.rivals or [rule.atoms]:
                d = atoms[0][0]
                c = tallies.get(d)
                if c is None:
                    continue
                up = c["fell"] / c["tried"] > (c["fell_without"] / c["without"] if c["without"] else 0.0)
                rules.append({"symbol": d, "action": action_of(d), "gain_bits": round(rule.gain_bits, 2),
                              "raises_the_fall": up})
                if up and action_of(d) not in moves:
                    moves.append(action_of(d))
        out["anomalies"][x] = {"open_days": len(open_rows), "fell_days": sum(x in r["fell"] for r in open_rows),
                               "moves": moves, "rules": rules, "tallies": tallies,
                               "stop": model.stops.get(f"fell:{x}")}
    return out


def tried(control: dict, anomaly: str, action: str | None = None) -> int:
    """How many times an action (any, if none is named) was applied the day the anomaly was present."""
    ts = control["anomalies"].get(anomaly, {}).get("tallies", {})
    return sum(c["tried"] for d, c in ts.items() if "@" not in d and (action is None or d == action))


def describe(control: dict) -> str:
    lines = [f"Learned from {control['evidence_days']} days (a fall: below {control['fall']} of the previous day's value; "
             f"an action's effect may show up to {control['lags']} days later, @-1 marking the day before)"]
    for x, c in control["anomalies"].items():
        moved = ", ".join(c["moves"]) or "nothing"
        lines.append(f"  {x}: open on {c['open_days']} days, fell on {c['fell_days']}; moved by {moved}")
        for d, t in sorted(c["tallies"].items()):
            lines.append(f"    {d}: fell {t['fell']}/{t['tried']} with it, {t['fell_without']}/{t['without']} without")
    return "\n".join(lines)
