"""What a run's store holds and how the keeper fared, printed for checking by hand (SPEC.md, "Before trusting the
store"): every entity, the latest states, the knowledge rows, the loops, and the records the keeper's replies lost.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import store


def show(run: Path, upto: int | None = None) -> str:
    conn = store.open_store(run / "store.sqlite")
    n = upto or conn.execute("SELECT MAX(chapter) FROM passage").fetchone()[0] or 0
    out = []

    calls, applied, unparsed, seconds, loop_events = 0, [], [], 0.0, []
    finishes = Counter()
    log = run / "log.jsonl"
    for line in log.read_text(encoding="utf-8").splitlines() if log.exists() else []:
        e = json.loads(line)
        if e["kind"] == "call" and e["role"] == "keeper":
            calls += 1
            seconds += e["seconds"]
            finishes[e.get("finish") or "not recorded"] += 1
        elif e["kind"] == "keeper_applied":
            applied.append(e)
        elif e["kind"] == "keeper_unparsed":
            unparsed.append(e)
        elif e["kind"] in ("loops_applied", "loops_unparsed"):
            loop_events.append(e)
    written, why, made, names = Counter(), Counter(), 0, Counter()
    per = {}
    for e in applied:
        written.update(e["written"])
        why.update(f"{d['kind']}: {d['why'].split(' ')[0] if d['why'].startswith('unresolved') else d['why']}"
                   for d in e["dropped"])
        made += len(e.get("created", []))
        names.update(d["why"].split(" ", 2)[2].split(":")[0] for d in e["dropped"]
                     if d["why"].startswith(("unresolved entity", "ambiguous entity")))
        c = per.setdefault(int(e["ref"][1:3]), Counter())
        c["passages"] += 1
        c["written"] += sum(e["written"].values())
        c["dropped"] += len(e["dropped"])
    for e in unparsed:
        c = per.setdefault(int(e["ref"][1:3]), Counter())
        c["passages"] += 1
        c["unparsed"] += 1
    kept, lost = sum(written.values()), sum(why.values())
    out += [f"KEEPER: {calls} calls, {seconds / max(calls, 1):.1f} s each, {len(unparsed)} replies not JSON",
            "  how replies ended: " + ", ".join(f"{k} {v}" for k, v in finishes.most_common())
            + ("   ('length' = cut off at the token limit)" if finishes.get("length") else ""),
            f"  records written: {kept}; dropped: {lost} ({100 * lost / max(kept + lost, 1):.0f}%); "
            f"entities created from names in the passage: {made}",
            "  written by kind: " + ", ".join(f"{k} {v}" for k, v in written.items() if v)]
    out += [f"  dropped: {k} x{v}" for k, v in why.most_common()]
    out += ["  (event_participant and event_place: the event is kept, only its link to that entity is lost)"]
    if names:
        out += ["  names that matched no entity, most common first: "
                + ", ".join(f"{n} x{c}" for n, c in names.most_common(15))]
    out += ["", "  by chapter:  passages  not JSON  written  dropped"]
    out += [f"  ch. {ch:<9} {c['passages']:>8}  {c['unparsed']:>8}  {c['written']:>7}  {c['dropped']:>7}"
            for ch, c in sorted(per.items())]
    lw, ld = Counter(), Counter()
    for e in loop_events:
        if e["kind"] == "loops_unparsed":
            ld[f"not JSON (ended: {e.get('finish')})"] += 1
            continue
        lw.update(e["written"])
        ld.update(d["why"].split(" ")[0] + " entity" if d["why"].startswith("unresolved") else
                  "no such open loop" if d["why"].startswith("no open loop") else d["why"] for d in e["dropped"])
    if loop_events:
        out += ["", f"LOOP PASS: {len(loop_events)} chapters; opened {lw['opened']}, advanced {lw['advanced']}, "
                    f"resolved {lw['resolved']}"] + [f"  dropped: {k} x{v}" for k, v in ld.most_common()]
    for e in unparsed[:3]:
        out += ["", f"  NOT JSON: {e['ref']} ({e.get('error', '')}; ended: {e.get('finish')}; {e.get('length')} chars)",
                "    starts: " + (e.get("head") or "").replace("\n", " ")[:200],
                "    ends:   " + (e.get("tail") or "").replace("\n", " ")[-200:]]

    out += ["", f"ENTITIES ({conn.execute('SELECT COUNT(*) FROM entity').fetchone()[0]})"]
    for r in conn.execute("SELECT * FROM entity ORDER BY kind, name"):
        aliases = json.loads(r["aliases"])
        out.append(f"  {r['id']:<28} {r['kind']:<9} {r['name']}" + (f"  (also: {', '.join(aliases)})" if aliases else ""))

    out += ["", f"STATES, latest as of chapter {n}"]
    for r in conn.execute("SELECT s.*, e.name FROM state s JOIN entity e ON e.id = s.entity_id WHERE s.from_chapter <= ? "
                          "AND s.id = (SELECT MAX(id) FROM state WHERE entity_id = s.entity_id AND attribute = s.attribute "
                          "AND from_chapter <= ?) ORDER BY e.name, s.attribute", (n, n)):
        changed = " (changed)" if r["supersedes"] else ""
        out.append(f"  {r['name']} | {r['attribute']}: {r['value']}  [ch. {r['from_chapter']}{changed}]")

    out += ["", "KNOWLEDGE"]
    for r in conn.execute("SELECT k.*, e.name FROM knowledge k JOIN entity e ON e.id = k.character_id ORDER BY k.id"):
        flags = ("; false" if r["true_in_story"] == 0 else "") + ("; later revised" if r["corrected_by"] else "")
        out.append(f"  ch. {r['from_chapter']}: {r['name']} {r['stance']} that {r['proposition']}{flags}")

    out += ["", "RULES"] + [f"  ch. {r['chapter']}: {r['statement']}" for r in conn.execute("SELECT * FROM rule")]

    out += ["", "LOOPS"]
    for r in conn.execute("SELECT l.*, (SELECT COUNT(*) FROM loop_touch WHERE loop_id = l.id) AS touches FROM loop l"):
        state = f"closed ch. {r['closed_chapter']}" if r["closed_chapter"] else "open"
        out.append(f"  {r['id']}: {r['description']}  [opened ch. {r['opened_chapter']}, {state}, {r['touches']} touches]")

    out += ["", "CHAPTER LINES"] + [f"  {r['chapter']}: {r['line']}"
                                    for r in conn.execute("SELECT * FROM chapter_line ORDER BY chapter")]
    out += ["", f"EVENTS: {conn.execute('SELECT COUNT(*) FROM event').fetchone()[0]} "
                f"({conn.execute('SELECT COUNT(*) FROM event WHERE happened = 0').fetchone()[0]} planned or cancelled)"]
    return "\n".join(out) + "\n"
