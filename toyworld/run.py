"""Run the toy world (toyworld/SPEC.md).

  python -m toyworld.run run --url http://127.0.0.1:8080 --run runs/toy-8b              # all four arms
  python -m toyworld.run run --url http://127.0.0.1:8080 --run runs/toy-8b-p --pressure  # judged on its own report
  python -m toyworld.run run --fake --run runs/toy-fake                                 # plumbing only, never a result
  python -m toyworld.run estimate --gen-tps 8 --pp-tps 200                              # expected load before a run
  python -m toyworld.run rescore runs/toy-8b                                              # rebuild state and scores from the record
  python -m toyworld.compare runs/toy-8b runs/toy-14b-bare

Everything the run does goes to <run>/log.jsonl first: every model call in full, every world observation, every
action applied, every offer and how it was settled. The per-run JSON files and scores.json are derived from it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from rung2.model import Fake, Server
from rung2.record import Recorder, TooLong

from . import finder, monitor, score, world

REPLY_TOKENS = 1024
ASSUMED_REPLY = {"bare": 150, "belief": 180, "goals": 200, "monitor": 200, "finder": 200}


class ToyFake(Fake):
    """Canned replies shaped like the real ones, with the faults a weak model shows, so every path runs."""

    def chat(self, prompt, schema=None, max_tokens=1024, temperature=0.0, seed=0):
        arm = ("bare" if "Current register:" not in prompt else
               "belief" if "beliefs only" in prompt else "goals")
        actions = [{"type": "request_data", "field": "bank_settlement"}]
        if arm != "belief":
            actions += [{"type": "set_order_qty", "item": "oat milk", "qty": 42},
                        {"type": "send_memo", "to": "cashiers", "text": "apply the $5 minimum consistently"},
                        {"type": "set_log_filter", "rule": "exclude refunds"},
                        {"type": "report_to_head_office", "problem": "reconciliation", "status": "resolved", "cause": ""}]
        preds = {"oat_milk_unfilled_requests": 0, "sub5_raise_rate": 0.2,
                 "nightly_reconciliation_discrepancy": 0, "bank_settlement_shortfall": None}
        if arm == "bare":
            return json.dumps({"actions": actions, "predictions": preds, "notes": "fake"})
        add = [] if "Current register:\n(empty)" not in prompt else [
            {"proposition": "the world is such that oat milk does not sell out",
             "direction": "belief" if arm == "belief" else "goal", "anomaly": "stockout", "evidence": "fake"},
            {"proposition": "the world is such that the reconciliation shows no discrepancy",
             "direction": "belief" if arm == "belief" else "goal", "anomaly": "reconciliation", "evidence": "fake"},
            {"proposition": "the world is such that the till applies the $5 minimum consistently",
             "direction": "belief" if arm == "belief" else "goal", "anomaly": "till_minimum", "evidence": "fake"}]
        return json.dumps({"register_changes": {"add": add, "update": [], "remove": []},
                           "actions": actions, "predictions": preds, "notes": "fake"})


def parse(reply: str) -> dict:
    try:
        out = json.loads(reply)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", reply or "", re.S)
        if not m:
            raise ValueError("no JSON object in the reply")
        out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError("the reply is not a JSON object")
    return out


def finder_step(s: dict) -> list[dict]:
    changes = finder.update(s)
    s.setdefault("finder_log", []).extend(changes)
    return changes


def ask_owner(o: dict) -> bool:
    print(f"\n[{o['id']}] {o['question']}\n  evidence: {'; '.join(o['evidence'])}\n  wrong if: {o['wrong_if']}")
    return input("  accept? [y/N] ").strip().lower().startswith("y")


def turn(rec: Recorder, s: dict, t: int, args, settle_mode: str) -> None:
    prompt = world.proposer_prompt(s, t)
    ref = f"{s['run']} day {t}"
    try:
        raw = rec.chat("proposer", prompt, ref=ref, schema=world.proposer_schema(s["arm"]), max_tokens=REPLY_TOKENS,
                       temperature=args.temperature, seed=s["seed"] * 100 + t)
        reply = parse(raw)
        rec.event("reply_parsed", ref=ref, said_by="model", reply_tokens=rec.tokens(raw))
    except (TooLong, ValueError, json.JSONDecodeError) as e:
        s["errors"].append({"day": t, "error": str(e)[:300]})
        rec.event("reply_unusable", ref=ref, error=str(e)[:300])
        reply = {}
    done = world.apply_reply(s, t, reply)
    rec.event("applied", ref=ref, said_by="code", actions=done["applied"], dropped=done["dropped"],
              register=done["register"], predictions=done["predictions"])
    seen = world.observe(s, t + 1)
    rec.event("observed", run=s["run"], day=t + 1, said_by="world", **seen)
    if s["arm"] == "finder":
        for c in finder_step(s):
            rec.event("posit", run=s["run"], said_by="finder", **c)
    if s["arm"] == "monitor":
        for o in monitor.offer(s, monitor.review(s)):
            rec.event("offer", run=s["run"], **o)
            if settle_mode == "none":
                continue
            accept = ask_owner(o) if settle_mode == "ask" else True
            by = "owner" if settle_mode == "ask" else "auto stand-in for the owner"
            rec.event("settled", run=s["run"], **monitor.settle(s, o, accept, by))


def run(args) -> None:
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "log.jsonl").exists() and (out / "log.jsonl").stat().st_size:
        sys.exit(f"{out} already has a record; use a new run directory, so one record holds one run")
    model = ToyFake() if args.fake else Server(args.url, timeout=args.timeout, model=args.model)
    rec = Recorder(model, out / "log.jsonl", window=args.window)
    cfg = {"pressure": args.pressure, "order_delay": args.order_delay, "delivery_failure": args.delivery_failure,
           "view": args.view}
    rec.event("start", model=model.name, cfg=cfg, arms=args.arms, runs=args.runs, days=args.days, seed=args.seed,
              temperature=args.temperature, settle=args.settle, argv=sys.argv[1:])
    rng = random.Random(args.seed)
    states = [world.new_state(f"{arm}-{i + 1}", arm, rng.randint(1, 10 ** 6), cfg)
              for arm in args.arms.split(",") for i in range(args.runs)]
    for s in states:
        rec.event("observed", run=s["run"], day=1, said_by="world", report=s["reports"][0], bank=s["bank"][0],
                  requested=[])
        if s["arm"] == "finder":
            finder_step(s)
    for t in range(1, args.days + 1):
        for s in states:
            turn(rec, s, t, args, args.settle)
            (out / f"{s['run']}.json").write_text(json.dumps(s, indent=1))
        print(f"day {t} done", flush=True)
    calls = calls_by_run(out / "log.jsonl")
    scores = [score.score(s, calls.get(s["run"], [])) for s in states]
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    print("\n" + score.summary(scores))
    print(f"\nThe record: {out / 'log.jsonl'}")


def replay(log: Path) -> tuple[dict, dict[str, dict], int]:
    """Rebuild every run's state from the record alone: the logged replies through the seeded world, offers settled as
    the record says. Returns the start event, the states, and how many replayed observations differ from the record
    (non-zero when the record was made by different world code)."""
    events = [json.loads(line) for line in log.read_text().splitlines()]
    starts = [i for i, e in enumerate(events) if e["kind"] == "start"]
    events = events[starts[-1]:]                 # an older record may hold an aborted start before the run itself
    st = events[0]
    rng = random.Random(st["seed"])
    states = {}
    for arm in st["arms"].split(","):
        for i in range(st["runs"]):
            states[f"{arm}-{i + 1}"] = world.new_state(f"{arm}-{i + 1}", arm, rng.randint(1, 10 ** 6), st["cfg"])
            if arm == "finder":
                finder_step(states[f"{arm}-{i + 1}"])
    observed = {(e["run"], e["day"]): e for e in events if e["kind"] == "observed"}
    answers = {(e["run"], e["offer"]): e for e in events if e["kind"] == "settled"}
    differ = 0
    for e in events:
        if e["kind"] != "call":
            continue
        run_id, t = e["ref"].split(" day ")
        s, t = states[run_id], int(t)
        try:
            reply = parse(e["reply"])
        except (ValueError, json.JSONDecodeError) as err:
            s["errors"].append({"day": t, "error": str(err)[:300]})
            reply = {}
        world.apply_reply(s, t, reply)
        seen = world.observe(s, t + 1)
        logged = observed.get((run_id, t + 1), {})
        differ += logged.get("report") != seen["report"] or logged.get("bank") != seen["bank"]
        if s["arm"] == "finder":
            finder_step(s)
        if s["arm"] == "monitor":
            for o in monitor.offer(s, monitor.review(s)):
                a = answers.get((run_id, o["id"]))
                if a:
                    monitor.settle(s, o, a["accepted"], a["settled_by"])
    return st, states, differ


def rescore(args) -> None:
    out = Path(args.run)
    _, states, differ = replay(out / "log.jsonl")
    calls = calls_by_run(out / "log.jsonl")
    scores = [score.score(s, calls.get(k, [])) for k, s in states.items()]
    for k, s in states.items():
        (out / f"{k}.json").write_text(json.dumps(s, indent=1))
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    print(score.summary(scores))
    print(f"\nReplayed observations that differ from the record: {differ}"
          + ("" if not differ else " (the record was made by different world code; these scores are not its own)"))


def calls_by_run(log: Path) -> dict[str, list[dict]]:
    """Each proposer call's cost, joined from the record: tokens in, tokens out, seconds."""
    by_ref: dict[str, dict] = {}
    for line in log.read_text().splitlines():
        e = json.loads(line)
        if e["kind"] == "call":
            by_ref[e["ref"]] = {"prompt_tokens": e.get("prompt_tokens"), "seconds": e.get("seconds")}
        elif e["kind"] == "reply_parsed" and e["ref"] in by_ref:
            by_ref[e["ref"]]["reply_tokens"] = e.get("reply_tokens")
    out: dict[str, list[dict]] = {}
    for ref, c in by_ref.items():
        out.setdefault(ref.split(" day ")[0], []).append(c)
    return out


def estimate(args) -> None:
    """Every prompt the experiment would send, built with the fake, then priced at the given speeds."""
    fake = ToyFake()
    cfg = {"pressure": args.pressure, "order_delay": 0, "delivery_failure": 0.0, "view": args.view}
    rng = random.Random(args.seed)
    rows, total = [], [0, 0, 0, 0.0]
    for arm in args.arms.split(","):
        calls = prompt_tokens = 0
        for i in range(args.runs):
            s = world.new_state(f"{arm}-{i + 1}", arm, rng.randint(1, 10 ** 6), cfg)
            for t in range(1, args.days + 1):
                prompt = world.proposer_prompt(s, t)
                prompt_tokens += fake.tokens(prompt)
                calls += 1
                world.apply_reply(s, t, parse(fake.chat(prompt)))
                world.observe(s, t + 1)
        cached = fake.tokens(world.static_prefix(s)) * (calls - 1)      # llama-server reuses the fixed prefix
        prompt_tokens -= cached
        reply_tokens = ASSUMED_REPLY[arm] * calls
        seconds = prompt_tokens / args.pp_tps + reply_tokens / args.gen_tps
        rows.append((arm, calls, prompt_tokens, reply_tokens, seconds))
        for k, v in enumerate((calls, prompt_tokens, reply_tokens, seconds)):
            total[k] += v
    print(f"{args.runs} runs x {args.days} days, view {args.view}; speeds: prompt {args.pp_tps:.0f} tok/s, "
          f"generation {args.gen_tps:.1f} tok/s")
    print("arm       calls  prompt tok  reply tok   minutes")
    for arm, c, p, g, sec in rows + [("total", *total)]:
        print(f"{arm:9} {c:5}  {p:10}  {g:9}   {sec / 60:7.1f}")
    print("Reply lengths are assumed (150-200 tokens a call); a real run records the actual counts.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="toyworld.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--arms", default=",".join(world.ARMS), help="bare, belief, goals, monitor, finder")
        sp.add_argument("--runs", type=int, default=3, help="runs per arm")
        sp.add_argument("--days", type=int, default=5)
        sp.add_argument("--view", choices=["compact", "full"], default="compact")
        sp.add_argument("--pressure", action="store_true", help="judge the system on its own report's numbers")
        sp.add_argument("--seed", type=int, default=30)

    r = sub.add_parser("run")
    common(r)
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="llama-server, e.g. http://127.0.0.1:8080")
    src.add_argument("--fake", action="store_true", help="canned replies: tests the plumbing, never the result")
    r.add_argument("--run", required=True, help="run directory (one per model and condition)")
    r.add_argument("--model", default=None, help="needed only if the server offers several models")
    r.add_argument("--order-delay", type=int, default=0, help="days before a new order quantity takes effect")
    r.add_argument("--delivery-failure", type=float, default=0.0, help="chance a delivery falls back to 10 units")
    r.add_argument("--settle", choices=["auto", "ask", "none"], default="auto",
                   help="monitor arm: who settles offers; auto = a stand-in that accepts every offer, labelled as such")
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--window", type=int, default=8192)
    r.add_argument("--timeout", type=float, default=900)
    rs = sub.add_parser("rescore")
    rs.add_argument("--run", required=True)
    e = sub.add_parser("estimate")
    common(e)
    e.add_argument("--pp-tps", type=float, default=200.0, help="prompt-processing speed, tokens a second")
    e.add_argument("--gen-tps", type=float, default=8.0, help="generation speed, tokens a second")
    args = p.parse_args(argv)
    {"run": run, "estimate": estimate, "rescore": rescore}[args.cmd](args)


if __name__ == "__main__":
    main()
