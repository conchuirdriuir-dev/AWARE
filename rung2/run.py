"""Rung 2's runner.

    python -m rung2.run build --story DIR --run DIR (--url http://127.0.0.1:8080 [--embed-url URL] | --fake)
    python -m rung2.run ask   --run DIR --questions FILE [--evidence FILE] [--arms 0ABCDE]
    python -m rung2.run score --run DIR --key FILE [--settled FILE]
    python -m rung2.run show  --run DIR [--upto N]               what the store holds, and what the keeper lost
    python -m rung2.run split --text BOOK.txt --out DIR [--chapters N]   a Gutenberg book into ch01.md, ch02.md, ...
    python -m rung2.run check --url URL [--embed-url URL]        each kind of request, alone: which one a server refuses

`build --only keeper` runs just the keeper and the chapter lines: the dry run (SPEC.md, "Before trusting the store").

`build` reads the chapters in order, as if the story were being written: it splits them into passages, runs the
keeper and the chapter lines (arm D), embeds the passages (arm B), and updates the running summary (arm C).
`ask` asks every question in every arm; `score` writes report.md. Everything the model is sent and says is in
the run's log.jsonl. A run directory belongs to one model: `ask` reuses the connection `build` recorded.

Questions (JSONL): {"id": "q01", "type": 1, "after": 18, "question": "..."}; types as in SPEC.md, "Questions".
Evidence for arm E (JSONL, prepared by the owner): {"id": "q01", "quotes": ["a phrase from the passage", ...]}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import keeper, retrieval, store
from .arms import ARMS, Arms
from .model import Fake, Server, check
from .record import Recorder
from .score import read_jsonl, report, score
from .show import show
from .text import chunk, load_chapters, split_book

SUMMARY_WORDS = 1700
SUMMARY = ("Here is the running summary of a novel so far, and its next chapter. Rewrite the summary so it also covers "
           "the new chapter, keeping every detail that may matter later: names, descriptions, who knows what, where "
           "things are, promises and unresolved questions. At most {words} words. Reply with the summary only.\n\n"
           "SUMMARY SO FAR:\n{old}\n\nCHAPTER {n} TEXT:\n{text}")


def pinned(cfg: dict) -> str | None:
    """The model a run was built with, once build has recorded it; until then, the one chosen with --model."""
    recorded = cfg.get("model")
    return recorded if recorded and recorded != "unknown" else cfg.get("model_choice")


def connect(cfg: dict):
    return Fake() if cfg.get("fake") else Server(cfg["url"], cfg.get("embed_url"), model=pinned(cfg),
                                                 embed_model=cfg.get("embed_model"))


def build(args) -> None:
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    cfg = {"fake": args.fake, "url": args.url, "embed_url": args.embed_url, "model_choice": args.model,
           "embed_model": args.embed_model, "window": args.window, "seed": args.seed,
           "story": str(Path(args.story).resolve()), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    model = connect(cfg)
    cfg["model"] = model.name
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    rec = Recorder(model, run / "log.jsonl", args.window)
    rec.event("build", **cfg)
    conn = store.open_store(run / "store.sqlite")
    summaries, summary, previous = {}, "", ""
    for n, text in load_chapters(Path(args.story)):
        t0 = time.time()
        ps = chunk(n, text, rec.tokens)
        store.add_passages(conn, ps)
        for p in ps:
            keeper.keep_passage(conn, rec, p, previous)
            previous = p["text"]
        keeper.keep_chapter_line(conn, rec, n, text)
        keeper.keep_chapter_loops(conn, rec, n, text, ps)
        t_store = time.time() - t0
        print(f"chapter {n}: {len(ps)} passages, keeper {t_store:.0f} s", file=sys.stderr)
        if args.only == "keeper":
            rec.event("chapter_built", chapter=n, passages=len(ps), store_seconds=round(t_store, 2))
            continue
        summary = rec.chat("summary", SUMMARY.format(words=SUMMARY_WORDS, old=summary or "(none yet)", n=n, text=text),
                           ref=f"c{n:02d}", max_tokens=2400).strip()
        summaries[str(n)] = summary
        rec.event("chapter_built", chapter=n, passages=len(ps), store_seconds=round(t_store, 2),
                  summary_seconds=round(time.time() - t0 - t_store, 2))
    if args.only == "keeper":
        return
    (run / "summaries.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    t0 = time.time()
    retrieval.build(store.passages(conn), model.embed).save(run)
    rec.event("index_built", seconds=round(time.time() - t0, 2))


def ask(args) -> None:
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    model = connect(cfg)
    rec = Recorder(model, run / "log.jsonl", cfg["window"])
    conn = store.open_store(run / "store.sqlite")
    evidence = {e["id"]: e["quotes"] for e in read_jsonl(Path(args.evidence))} if args.evidence else {}
    arms = Arms(conn, run, rec.tokens, model.embed, evidence)
    if (run / "answers.jsonl").exists():
        sys.exit(f"{run / 'answers.jsonl'} exists: move it aside to ask again (scores would count answers twice)")
    out = (run / "answers.jsonl").open("w", encoding="utf-8")
    for q in read_jsonl(Path(args.questions)):
        for arm in args.arms:
            if arm == "E" and q["id"] not in evidence:
                continue
            view = arms.view(arm, q)
            prompt = arms.prompt(arm, q, view)
            runs = [(0.7, cfg["seed"] + i) for i in range(3)] if q["type"] == 6 else [(0.0, cfg["seed"])]
            for i, (temperature, seed) in enumerate(runs):
                answer = rec.chat(f"answer:{arm}", prompt, ref=q["id"], max_tokens=700 if q["type"] == 6 else 200,
                                  temperature=temperature, seed=seed).strip()
                row = {"id": q["id"], "arm": arm, "type": q["type"], "after": q["after"], "run": i,
                       "view_tokens": rec.tokens(view), "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            rec.event("view", ref=q["id"], arm=arm, view=view)
        print(f"{q['id']}: asked", file=sys.stderr)


def views_from_log(run: Path) -> dict:
    views = {}
    for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["kind"] == "view":
            views[(e["ref"], e["arm"])] = e["view"]
    return views


def score_cmd(args) -> None:
    run = Path(args.run)
    conn = store.open_store(run / "store.sqlite")
    rows = score(read_jsonl(run / "answers.jsonl"), read_jsonl(Path(args.key)),
                 read_jsonl(Path(args.settled)) if args.settled else [], views_from_log(run), conn)
    (run / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "report.md").write_text(report(rows))
    print(run / "report.md")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="rung2.run")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--story", required=True)
    b.add_argument("--run", required=True)
    b.add_argument("--url")
    b.add_argument("--embed-url")
    b.add_argument("--model", help="which model, when the server offers several")
    b.add_argument("--embed-model", help="which embedding model, when that server offers several")
    b.add_argument("--fake", action="store_true", help="canned replies: tests the plumbing, never the result")
    b.add_argument("--window", type=int, default=8192)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--only", choices=["keeper"], help="keeper: the keeper and chapter lines only (the dry run)")
    sh = sub.add_parser("show")
    sh.add_argument("--run", required=True)
    sh.add_argument("--upto", type=int)
    sp = sub.add_parser("split")
    sp.add_argument("--text", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--chapters", type=int)
    ck = sub.add_parser("check")
    ck.add_argument("--url", required=True)
    ck.add_argument("--embed-url")
    ck.add_argument("--model")
    ck.add_argument("--embed-model")
    a = sub.add_parser("ask")
    a.add_argument("--run", required=True)
    a.add_argument("--questions", required=True)
    a.add_argument("--evidence")
    a.add_argument("--arms", default=ARMS)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--settled")
    args = p.parse_args(argv)
    if args.cmd == "build" and not (args.fake or args.url):
        p.error("build needs --url or --fake")
    {"build": build, "ask": ask, "score": score_cmd, "show": show_cmd, "split": split_cmd,
     "check": check_cmd}[args.cmd](args)


def check_cmd(args) -> None:
    for name, ok, detail in check(args.url, args.embed_url, keeper.SCHEMA, args.model, args.embed_model):
        print(f"{'ok  ' if ok else 'FAIL'} {name}\n     {detail}")


def show_cmd(args) -> None:
    print(show(Path(args.run), args.upto), end="")


def split_cmd(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    chapters = split_book(Path(args.text).read_text(encoding="utf-8", errors="replace"), args.chapters)
    for i, c in enumerate(chapters, 1):
        (out / f"ch{i:02d}.md").write_text(c + "\n", encoding="utf-8")
    words = [len(c.split()) for c in chapters]
    print(f"{len(chapters)} chapters written to {out}: {min(words, default=0)} to {max(words, default=0)} words each")


if __name__ == "__main__":
    main()
