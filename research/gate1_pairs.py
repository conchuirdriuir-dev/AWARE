"""Gate 1's opposites, elicited from Qwen3-8B (PLAN.md): one fixed domain-free prompt, thinking off, greedy. No state is
read here; the owner reads a random 20 pairs before any is.

    python -m research.gate1_pairs --run runs/gate1-pairs-qwen8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.d_jump import MODEL
from substrate.record import Record

PROMPT = "Write the statement that says the opposite of this one. Reply with the statement only."
MAX_NEW = 80
STATEMENTS = "material/negation_statements.jsonl"
PAIRS = "material/negation_pairs.jsonl"
REVIEW = 20


def prompt(statement: str) -> str:
    return f"{PROMPT}\n\n{statement}"


def clean(reply: str) -> str:
    """The reply as the model gave it, less surrounding space and any empty thinking block the template leaves."""
    return reply.replace("<think>", "").replace("</think>", "").strip()


def load_frozen(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(r.get("frozen") for r in rows):
        raise SystemExit("the statements are not frozen; the owner freezes them first")
    return rows


def review_sample(n_pairs: int, k: int = REVIEW, seed: int = 0) -> list[int]:
    return sorted(np.random.default_rng(seed).choice(n_pairs, k, replace=False).tolist())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    a = p.parse_args(argv)
    import torch
    from substrate.states import Reader

    rows = load_frozen(STATEMENTS)
    record = Record(Path(a.run) / "record.jsonl")
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    if not getattr(reader.model, "dispatched", True):
        reader.model.dispatch()
    hf, tok = reader.model._model, reader.tokenizer
    record.append("run", "code", **reader.info, prompt=PROMPT, max_new_tokens=MAX_NEW, thinking=False, sampling="greedy")
    pairs = []
    with torch.no_grad():
        for r in rows:
            for i, statement in enumerate(r["statements"]):
                text = tok.apply_chat_template([{"role": "user", "content": prompt(statement)}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                ids = tok(text, return_tensors="pt")["input_ids"].to(hf.device)
                out = hf.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None)
                new = out[0, ids.shape[1]:]
                reply = tok.decode(new, skip_special_tokens=True)
                pair = {"topic": r["id"], "index": i, "statement": statement, "opposite": clean(reply),
                        "by": "model", "cut_off": len(new) >= MAX_NEW}
                record.append("elicited", pair["by"], raw_reply=reply, **{k: v for k, v in pair.items() if k != "by"})
                pairs.append(pair)
    Path(PAIRS).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pairs), encoding="utf-8")
    chosen = review_sample(len(pairs))
    record.append("review sample", "code", indices=chosen, seed=0)
    print(f"{len(pairs)} pairs; cut off: {sum(x['cut_off'] for x in pairs)}; identical to the statement: "
          f"{sum(x['opposite'] == x['statement'] for x in pairs)}; empty: {sum(not x['opposite'] for x in pairs)}")
    for n, j in enumerate(chosen, 1):
        print(f"{n:>2}. [{pairs[j]['topic']}] {pairs[j]['statement']}\n    -> {pairs[j]['opposite']}")


if __name__ == "__main__":
    main()
