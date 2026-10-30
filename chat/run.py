"""Rung 1's runner (SPEC.md). Shadow mode: talk, then mark, then review.

    python -m chat.run talk   --session sessions/s01 --url http://127.0.0.1:8080
    python -m chat.run mark   --session sessions/s01                  your marks, before you see any codes
    python -m chat.run review --session sessions/s01 --url http://127.0.0.1:8080 --embed-url http://127.0.0.1:8081
    python -m chat.run show   --session sessions/s01                  the session with the reviewer's codes

`--fake` in place of `--url` runs with canned replies: it tests the plumbing, never the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rung2.model import Fake, Server
from rung2.record import Recorder

from . import record, review, talk


class ChatFake(Fake):
    """Canned reviewer codes: each person turn after the first narrows the aim, unless it says "no" (a repair) or
    "ok" (feedback); every reply serves the latest aim."""

    def chat(self, prompt, schema=None, **kw):
        if schema is review.PERSON_SCHEMA:
            new = prompt.rsplit("The person's new message", 1)[1].split('"')[1]
            first = "nothing yet" in prompt
            f = "repair" if new.lower().startswith("no") else "feedback" if new.lower().startswith("ok") else "task"
            return json.dumps({"function": f, "relation": "new" if first else "narrower", "back_to": None,
                               "quote": " ".join(new.split()[:3])})
        if schema is review.MODEL_SCHEMA:
            active = []
            for line in prompt.split("most specific last:\n", 1)[-1].splitlines():
                if not line.startswith("A"):
                    break
                active.append(line.split(" ")[0])
            reply = prompt.rsplit("The assistant's reply", 1)[1].split('"')[1]
            return json.dumps({"addresses": active[-1] if active else "none", "serves": "yes",
                               "quote": " ".join(reply.split()[:3])})
        return super().chat(prompt, schema=schema, **kw)


def connect(args):
    if args.fake:
        return ChatFake()
    return Server(args.url, getattr(args, "embed_url", None), model=args.model,
                  embed_model=getattr(args, "embed_model", None))


def main(argv=None, read=input, write=print) -> None:
    p = argparse.ArgumentParser(prog="chat.run")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("talk", "review"):
        c = sub.add_parser(name)
        c.add_argument("--session", required=True)
        src = c.add_mutually_exclusive_group(required=True)
        src.add_argument("--url")
        src.add_argument("--fake", action="store_true", help="canned replies: tests the plumbing, never the result")
        c.add_argument("--model")
        c.add_argument("--window", type=int, default=8192)
        c.add_argument("--seed", type=int, default=0)
    t = sub.choices["talk"]
    t.add_argument("--temperature", type=float, default=0.7)
    r = sub.choices["review"]
    r.add_argument("--embed-url", help="the embedding model's server")
    r.add_argument("--embed-model")
    r.add_argument("--embed-prefix", default="", help="put before every text embedded, e.g. a task instruction")
    r.add_argument("--name", default="review", help="this review's name: a new reviewer version gets a new name")
    m = sub.add_parser("mark")
    m.add_argument("--session", required=True)
    m.add_argument("--again", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("--session", required=True)
    s.add_argument("--name", default="review")
    s.add_argument("--force", action="store_true", help="show codes before the session is marked")
    args = p.parse_args(argv)
    session = Path(args.session)
    if args.cmd == "talk":
        model = connect(args)
        talk.talk(session, Recorder(model, session / "log.jsonl", args.window),
                  {"window": args.window, "seed": args.seed, "temperature": args.temperature}, read, write)
    elif args.cmd == "mark":
        from . import mark
        write(f"{mark.mark(session, read, write, args.again)} marks written to {session / 'key.jsonl'}")
    elif args.cmd == "review":
        model = connect(args)
        cfg = {"model": model.name, "embed_model": getattr(model, "embed_model", "fake"),
               "embed_prefix": args.embed_prefix, "window": args.window, "seed": args.seed}
        write(review.review(session, model, args.name, cfg))
    else:
        write(show(session, args.name, args.force))


def show(session: Path, name: str, force: bool = False) -> str:
    """The session with the reviewer's codes and the aim stack, for reading after the session is marked."""
    if not force and not any(m.get("kind") == "marked" for m in record.marks(session)):
        raise SystemExit("mark the session first (python -m chat.run mark), so the codes do not shape your marks; "
                         "--force shows them anyway")
    log = record.read_jsonl(session / "reviews" / name / "log.jsonl")
    codes = {e["turn"]: e for e in log if e["kind"] == "code"}
    stacks = {e["after"]: e["stack"] for e in log if e["kind"] == "stack"}
    out = []
    for t in record.turns(session):
        c = codes.get(t["id"], {})
        code = c.get("code", {})
        text = t["text"] if len(t["text"]) <= 300 else t["text"][:300] + " ..."
        out.append(f"{t['id']} {'you' if t['role'] == 'person' else 'model'}> {text}")
        if t["role"] == "person":
            out.append(f"    reviewer: {code.get('function')}, {code.get('relation')}"
                       + (f" to {code['back_to']}" if code.get("back_to") else "")
                       + f"; quote {'ok' if c.get('quote_checked') else 'NOT FOUND'}; sim to aim {c.get('sim_to_aim')}"
                       + f"; stack after: {' > '.join(stacks.get(t['id'], [])) or '(empty)'}")
        else:
            out.append(f"    reviewer: addresses {code.get('addresses')}, serves {code.get('serves')}; quote "
                       f"{'ok' if c.get('quote_checked') else 'NOT FOUND'}; sim to aim {c.get('sim_to_aim')}, "
                       f"to your last turn {c.get('sim_to_last')}")
    return "\n".join(out)


if __name__ == "__main__":
    main()
