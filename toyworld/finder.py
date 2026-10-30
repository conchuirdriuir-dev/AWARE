"""The finder: regularities in the reports, admitted under DARA's law (DARA CORE, "The law, precisely"). It is code, not
a model; rung 2's keeper is a different part, a model call.

DARA's kernel reads relational structures of opaque symbols and never what a number means, so it cannot state
"sold + unfilled stays at 40". What is taken from DARA is the law, not the kernel: a posit is admitted only when it
makes the record shorter by more than m = 2 bits against the reference, rivals within the margin stay rivals, and a
held posit is retired when its record codes worse than the reference by more than the margin. The posit family below
is declared here (toyworld/INVARIANT.md); its prior S is fixed before the data.

Posits. For a target field y of the daily report and another field x, with s = +1 or -1: y_t = a + s * x_t + r_t, the
constant a unknown. The reference codes y around its own level: y_t = a + r_t.

Codes, prequential (no fitted parameter is paid for, because each day is coded from the days before it):
  day 1:   the first value of y - s * x (the reference: of y), signed Elias gamma
  day t:   the residual from the mean of y - s * x over days 1 .. t-1, rounded, signed Elias gamma
  S(h):    log2 of the target fields + log2 of the other fields + 1 bit for s

The finder reads only the reports, which the model also sees, never the key. What it holds is labelled inferred, by
the finder.
"""
from __future__ import annotations

import math

MARGIN_BITS = 2.0      # DARA's m: admitted only on a saving of more than 2 bits
FIELDS = ["oat_milk_ordered", "oat_milk_delivered", "oat_milk_sold", "oat_milk_unfilled_requests",
          "sub5_card_charges", "sub5_raised_to_5", "complaints_inconsistent_minimum",
          "nightly_reconciliation_discrepancy"]


def gamma_bits(n: int) -> int:
    """Elias gamma for a positive integer."""
    return 2 * (n.bit_length() - 1) + 1


def signed_bits(z: int) -> int:
    """Signed Elias gamma: 0, -1, 1, -2, 2 ... sent to 1, 2, 3, 4, 5 ..."""
    return gamma_bits(2 * z + 1 if z >= 0 else -2 * z)


def prequential_bits(values: list[float]) -> float:
    bits, seen = 0.0, []
    for v in values:
        pred = sum(seen) / len(seen) if seen else 0.0
        bits += signed_bits(round(v - pred))
        seen.append(v)
    return bits


def prior_bits() -> float:
    return math.log2(len(FIELDS)) + math.log2(len(FIELDS) - 1) + 1


def series(s: dict, field: str) -> list[float]:
    return [float(r[field]) for r in s["reports"]]


def candidates(s: dict) -> list[dict]:
    """Every posit of the family, scored on the reports so far."""
    out = []
    for y in FIELDS:
        ys = series(s, y)
        reference = prequential_bits(ys)
        for x in FIELDS:
            if x == y:
                continue
            xs = series(s, x)
            for sign in (-1, 1):
                combo = [a - sign * b for a, b in zip(ys, xs)]
                gain = reference - prequential_bits(combo) - prior_bits()
                out.append({"y": y, "x": x, "sign": sign, "gain": round(gain, 2),
                            "level": round(sum(combo) / len(combo), 1), "values": combo})
    return out


def phrase(p: dict) -> str:
    """The posit as the model reads it: s = -1 is y + x; s = +1 is y - x."""
    if p["sign"] == -1:
        return f"{p['y']} + {p['x']} stays at about {p['level']:g}"
    if p["level"] == 0:
        return f"{p['y']} equals {p['x']}"
    return f"{p['y']} - {p['x']} stays at about {p['level']:g}"


def key(p: dict) -> tuple:
    return (p["y"], p["x"], p["sign"])


def _cycle(held: list[dict], p: dict) -> bool:
    """Whether coding p's target from its other field would close a loop among the held posits: the record is coded
    once, so a field cannot be coded from a field that is itself coded from it."""
    from_ = {h["y"]: h["x"] for h in held}
    x, seen = p["x"], set()
    while x in from_ and x not in seen:
        if from_[x] == p["y"]:
            return True
        seen.add(x)
        x = from_[x]
    return x == p["y"]


def update(s: dict) -> list[dict]:
    """Re-score every posit on the reports so far; admit, retire or keep. Returns the changes, for the record.
    Posits are admitted largest saving first, at most one per target and never closing a loop (a field coded from a
    field coded from it); candidates within the margin of a held posit are its rivals. A held posit is retired when the reference codes its target better by more than
    the margin, or when a rival beats it by more than the margin."""
    held = s.setdefault("finder", {"held": [], "rivals": {}})
    day = len(s["reports"])
    scored = {key(p): p for p in candidates(s)}
    changes = []
    by_target: dict[str, list[dict]] = {}
    for p in scored.values():
        by_target.setdefault(p["y"], []).append(p)
    still = []
    for h in held["held"]:
        now = scored[key(h)]
        others = [o for o in held["held"] if o is not h]
        best = max((p for p in by_target[h["y"]] if not _cycle(others, p)), key=lambda p: p["gain"])
        if now["gain"] < -MARGIN_BITS or best["gain"] > now["gain"] + MARGIN_BITS:
            changes.append({"change": "retired", "day": day, "posit": phrase(h), "gain": now["gain"],
                            "y": h["y"], "x": h["x"], "sign": h["sign"]})
        else:
            still.append({**h, **{k: now[k] for k in ("gain", "level", "values")}})
    held["held"] = still
    for p in sorted(scored.values(), key=lambda p: -p["gain"]):     # the largest saving first
        if p["gain"] <= MARGIN_BITS:
            break
        if p["y"] in {h["y"] for h in held["held"]} or _cycle(held["held"], p):
            continue
        h = {**p, "since_day": day, "said_by": "finder"}
        held["held"].append(h)
        changes.append({"change": "admitted", "day": day, "posit": phrase(h), "gain": h["gain"],
                        "y": h["y"], "x": h["x"], "sign": h["sign"], "values": h["values"]})
    held["rivals"] = {}
    for h in held["held"]:
        others = [o for o in held["held"] if o is not h]
        rivals = [p for p in by_target[h["y"]] if key(p) != key(h) and p["gain"] > MARGIN_BITS
                  and p["gain"] >= h["gain"] - MARGIN_BITS and not _cycle(others, p)]
        held["rivals"][h["y"]] = [phrase(p) for p in sorted(rivals, key=lambda p: -p["gain"])]
    return changes


def view_lines(s: dict) -> list[str]:
    """What the finder arm's model is shown: each held posit, its daily values, its saving, its rivals."""
    k = s.get("finder") or {}
    out = []
    for h in k.get("held", []):
        vals = ", ".join(f"{v:g}" for v in h["values"])
        line = (f"Finder (code; inferred, not stated): {phrase(h)} (days 1-{len(h['values'])}: {vals}; "
                f"shortens the record by {h['gain']:g} bits)")
        rivals = k.get("rivals", {}).get(h["y"])
        if rivals:
            line += "; fits as well: " + "; ".join(rivals)
        out.append(line)
    return out
