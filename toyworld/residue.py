"""Stage 2: kinds of failure, learned from residue (toyworld/SPEC.md, "Stage 2").

Residue is where the record contradicts what the learned map (toyworld/control.py) expects. Two types of row:
  mover    a learned mover was applied while its anomaly was present; target `still_present`: the anomaly is still
           present at the lag the mover's rule cites
  neglect  an anomaly with a learned mover was present two days running; target `unaddressed`: none of its movers
           was proposed that day (one the exploration design withheld counts as proposed: that was the design's doing)
A row says what happened in the record's own terms: the anomaly, the actions applied (as `action_symbol` names them),
whether the anomaly fell, whether the register held a goal for it. Never the world's terms, never an action's numbers.

Kinds are learned in two tiers, each admitted only on MDL gain above DARA's 2-bit margin:
  tier 1  rules over the rows (`kernel.learn`), one per target, at most two atoms
  tier 2  a latent class over kinds of action (`kernel.latent`), on the mover rows; provisional (DECISIONS.md,
          2026-10-30): built with the kernel's one-step test before the lookahead question was settled
and every admission is checked against the same rows with their targets shuffled (the control against reading noise).
"""
from __future__ import annotations

import json
import random

from kernel import latent as lt
from kernel import learn as K
from kernel.structure import Structure

from . import control as ctl
from .world import action_symbol

TARGET = {"mover": "still_present", "neglect": "unaddressed"}
LABEL = {"stockout": "unfilled requests", "till_minimum": "raise rate", "reconciliation": "bank shortfall"}
SHUFFLES = 20


def movers(cmap: dict) -> dict[tuple[str, str], int]:
    """(anomaly, action) -> the lag, in days, at which the admitted rule says the action moves the anomaly."""
    out = {}
    for a, c in cmap["anomalies"].items():
        for r in c["rules"]:
            if r["raises_the_fall"]:
                k = 2 if "@-1" in r["symbol"] else 1
                out[(a, r["action"])] = min(k, out.get((a, r["action"]), k))
    return out


def _goal(s: dict, day: int, anomaly: str) -> bool:
    regs = [r["register"] for r in s.get("registers", []) if r["day"] == day]
    return bool(regs) and any(e.get("direction") == "goal" and e.get("anomaly") == anomaly for e in regs[0])


def rows(states: list[dict], cmap: dict) -> list[dict]:
    """Every residue row the record can show now (a mover row only once its lag has been observed)."""
    mv, out = movers(cmap), []
    by_anomaly = {}
    for (a, d) in mv:
        by_anomaly.setdefault(a, []).append(d)
    for s in states:
        if s["arm"] == "belief":                    # it may not act: its days are the arm's rule, not residue
            continue
        for act in s["actions"]:
            t = act["day"]
            did = sorted({x for x in (action_symbol(y) for y in act["actions"]) if x})
            withheld = {action_symbol(y) for y in act.get("dropped", []) if y.get("why") == "withheld"}
            base = {"run": s["run"], "arm": s["arm"], "day": t, "did": did}
            for (a, d), k in sorted(mv.items()):
                xs = ctl.series(s, a)
                if d in did and xs[t - 1] > ctl.OPEN[a] and t + k - 1 < len(xs):
                    after = xs[t + k - 1]
                    used = [y for y in act["actions"] if action_symbol(y) == d]
                    out.append({**base, "type": "mover", "anomaly": a, "action": d, "lag": k,
                                "params": params(used[0]) if used else "",
                                "fell": after < ctl.FALL * xs[t - 1], "goal": _goal(s, t, a),
                                "target": after > ctl.OPEN[a], "before": xs[t - 1], "after": after})
            for a, ds in sorted(by_anomaly.items()):
                xs = ctl.series(s, a)
                if t >= 2 and xs[t - 1] > ctl.OPEN[a] and xs[t - 2] > ctl.OPEN[a]:
                    out.append({**base, "type": "neglect", "anomaly": a, "action": None, "lag": 0,
                                "fell": False, "goal": _goal(s, t, a),
                                "target": not any(d in did or d in withheld for d in ds),   # a withheld proposal
                                "before": xs[t - 2], "after": xs[t - 1]})                     # was addressed
    return out


def facts(r: dict) -> set[str]:
    """What a row says, as the tier-1 rules may cite it."""
    f = {f"anom:{r['anomaly']}"} | ({"fell"} if r["fell"] else set()) | ({"goal"} if r["goal"] else set())
    if r["type"] == "mover":
        f |= set(r["did"])
    return f


def _tier1(rs: list[dict], labels: list[bool], target: str) -> list[dict]:
    st = []
    for r, y in zip(rs, labels):
        f = {"row": {("x",)}, **{k: {("x",)} for k in facts(r)}}
        if y:
            f[target] = {("x",)}
        st.append(Structure(f))
    body = tuple(sorted({k for s in st for k in s.facts} - {"row", target}))
    t = K.Target(target, 1, ("row",), body=body)
    m, _ = K.learn(st, [t], K.Search(max_atoms=2, invent=False, cells=False))
    rate = sum(labels) / len(labels) if labels else 0.0
    kinds = []
    for rule in m.rules[target]:
        need = {a[0] for a in rule.atoms}
        inside = [y for r, y in zip(rs, labels) if need <= facts(r)]
        if inside and sum(inside) / len(inside) > rate:              # a failure kind: the target more often inside it
            kinds.append({"atoms": sorted(need), "gain_bits": round(rule.gain_bits, 2),
                          "rate_inside": round(sum(inside) / len(inside), 3), "rate_overall": round(rate, 3),
                          "rows": len(inside)})
    return kinds


def _tier2(rs: list[dict], labels: list[bool]) -> dict | None:
    st = []
    for r, y in zip(rs, labels):
        e = r["action"].replace("did:", "")
        f = {"act": {(e,)}, f"id:{e}": {(e,)}, f"anom:{r['anomaly']}": {(e,)}}
        if r["fell"]:
            f["fell"] = {(e,)}
        if y:
            f["still"] = {(e,)}
        st.append(Structure(f))
    if len({next(iter(s.facts["act"]))[0] for s in st}) < 2:
        return None                                                     # one kind of action: no class to posit
    inv = lt.invent(st, [K.Target("still", 1, ("act",))], K.Search(max_atoms=2), lt.Latent(name="Z", domain=("act",)))
    if not inv.admitted:
        return None
    return {"members": sorted(f"did:{m}" for m in inv.members), "gain_bits": round(inv.gain_bits, 2)}


def learn(states: list[dict], cmap: dict, shuffles: int = SHUFFLES) -> dict:
    """The kinds, each with the rows behind it, and the shuffled control's count of admissions."""
    rs = rows(states, cmap)
    out = {"rows": len(rs), "kinds": [], "control": {"shuffles": shuffles, "tier1_admitted": 0, "tier2_admitted": 0}}
    for typ, target in TARGET.items():
        sub = [r for r in rs if r["type"] == typ]
        if not sub:
            continue
        y = [r["target"] for r in sub]
        for k in _tier1(sub, y, target):
            members = [r for r in sub if set(k["atoms"]) <= facts(r) and r["target"]]
            out["kinds"].append({"id": f"K{len(out['kinds']) + 1}", "tier": 1, "type": typ, "target": target, **k,
                                 "examples": [brief(r) for r in members[:6]], "members": len(members)})
        if typ == "mover":
            z = _tier2(sub, y)
            if z:
                members = [r for r in sub if r["action"] in z["members"] and r["target"]]
                out["kinds"].append({"id": f"K{len(out['kinds']) + 1}", "tier": 2, "type": typ, "target": target,
                                     **z, "examples": [brief(r) for r in members[:6]], "members": len(members)})
        for seed in range(shuffles):
            z_ = y[:]
            random.Random(seed).shuffle(z_)
            out["control"]["tier1_admitted"] += bool(_tier1(sub, z_, target))
            if typ == "mover":
                out["control"]["tier2_admitted"] += bool(_tier2(sub, z_))
    return out


def matches(kind: dict, r: dict) -> bool:
    if kind["type"] != r["type"] or not r["target"]:
        return False
    if kind["tier"] == 1:
        return set(kind["atoms"]) <= facts(r)
    return r["action"] in kind["members"]


def brief(r: dict) -> str:
    level = f"{LABEL[r['anomaly']]} {r['before']} -> {r['after']}"
    if r["type"] == "mover":
        return f"{r['run']} day {r['day']}: {r['action']} applied for {r['anomaly']}, still present ({level})"
    return f"{r['run']} day {r['day']}: {r['anomaly']} present two days running, no mover applied ({level})"


def readable(symbol: str) -> str:
    """An action name as the model knows actions: did:report_to_head_office:reconciliation:open ->
    report_to_head_office with problem "reconciliation" and status "open"."""
    parts = symbol.replace("did:", "").split(":")
    if parts[0] == "report_to_head_office" and len(parts) == 3:
        return f'report_to_head_office with problem "{parts[1]}" and status "{parts[2]}"'
    if parts[0] == "set_log_filter" and len(parts) == 2:
        return f'set_log_filter with rule "{"exclude refunds" if parts[1] == "refunds" else "none"}"'
    return parts[0]


def note(kind: dict, rs: list[dict], fixes: list[str] = ()) -> str:
    """What the model is told when the owner accepts a kind's offer: its own record, nothing more."""
    days = ", ".join(str(r["day"]) for r in rs)
    if kind["type"] == "mover":
        a, d = rs[0]["anomaly"], rs[0]["action"]
        levels = "; ".join(f"day {r['day']}: {r.get('params') or readable(d)}, {LABEL[a]} {r['before']} -> {r['after']}"
                           for r in rs)
        return f"{a} was still present after {readable(d)} on day(s) {days} ({levels})"
    a = rs[0]["anomaly"]
    levels = "; ".join(f"day {r['day']}: {r['after']}" for r in rs)
    moved = " or ".join(readable(f) for f in fixes) or "the actions that move it"
    return (f"{a} was present on day(s) {days} and {moved}, which moved it in exploration, was not applied "
            f"({LABEL[a]}: {levels})")


FIELDS = {"set_order_qty": ("qty",), "send_memo": ("text",), "change_till_setting": ("key", "value"),
          "set_log_filter": ("rule",), "report_to_head_office": ("problem", "status", "cause")}


def params(a: dict) -> str:
    """The model's own action as it wrote it, for a note: its type's fields only (stage 2's rerun, RESULTS.md)."""
    return ", ".join(f"{k} {json.dumps(a[k])}" for k in FIELDS.get(a.get("type"), ()) if a.get(k) not in (None, ""))


def describe(learned: dict) -> str:
    lines = [f"Residue: {learned['rows']} rows. Kinds admitted: {len(learned['kinds'])}."]
    for k in learned["kinds"]:
        what = " & ".join(k["atoms"]) if k["tier"] == 1 else "a class of actions: " + ", ".join(k["members"])
        lines.append(f"  {k['id']} (tier {k['tier']}, {k['gain_bits']} bits): {k['target']} <- {what}; "
                     f"{k['members']} member rows")
        lines += [f"    {e}" for e in k["examples"][:3]]
    c = learned["control"]
    lines.append(f"Shuffled control ({c['shuffles']} shuffles per type): tier 1 admitted a kind in {c['tier1_admitted']}, "
                 f"tier 2 in {c['tier2_admitted']}.")
    return "\n".join(lines)
