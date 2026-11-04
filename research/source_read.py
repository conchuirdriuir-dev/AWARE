"""A cheap read of the source, on replies the model wrote itself (PLAN.md, direction step 3). Coupled samples with and
without a partner's frame give each reply a key by construction; one forward pass over the reply is the read.

    python -m research.source_read --run runs/source-dry --dry
    python -m research.source_read --run runs/source-qwen8b
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from research import hidden as h
from research.d_jump import MODEL
from substrate.record import Record, read

FRAMES = "material/frames.jsonl"
DRY = "material/frames_dry.jsonl"
STEP2 = "runs/hidden-qwen8b/record.jsonl"
FRAME = "I keep thinking about {frame}."                  # unlabelled, after the goal (amended, PLAN.md)
CANDIDATES = "material/frame_candidates.jsonl"
TARGET = 3
K, TEMPERATURE, MAX_NEW = 24, 0.7, 80     # 24 samples a conversation, amended by the owner after the pilot (PLAN.md)
PILOT_K, BATCH = 8, 8    # the pilot drew 8; samples run in batches of 8, so the run's samples 0-7 are the pilot's draws
PILOT = "runs/source-pilot/record.jsonl"
LAYER, LAYERS = 18, (12, 18, 24)
COMPONENTS, C, FOLDS = 20, 1.0, 5
WORKS, NONE_, ENOUGH, CONVS = 0.7, 0.7, 60, 15


# Material and keys, pure.

def frame_messages(goal: str, frame: str) -> list[dict]:
    return [{"role": "user", "content": goal + " " + FRAME.format(frame=frame)}]


def bare_messages(goal: str) -> list[dict]:
    return [{"role": "user", "content": goal}]


def pick(candidates: list[dict], n_texts: list[str]) -> tuple[int, list[int]]:
    """The candidate whose markers the frame-free replies carry closest to TARGET of them; ties to the earlier drafted."""
    counts = [sum(h.carries(t, c["markers"]) for t in n_texts) for c in candidates]
    return min(range(len(candidates)), key=lambda i: (abs(counts[i] - TARGET), i)), counts


def first_hit(text: str, markers: list[str]) -> int | None:
    """Character position of the first marker hit, or None."""
    low, hits = text.lower(), []
    for m in markers:
        found = re.search(r"(?<![a-z0-9])" + re.escape(m.lower()), low)
        if found:
            hits.append(found.start())
    return min(hits) if hits else None


def hits(text: str, markers: list[str]) -> int:
    low = text.lower()
    return sum(len(re.findall(r"(?<![a-z0-9])" + re.escape(m.lower()), low)) for m in markers)


def key(f_carries: bool, n_carries: bool) -> str | None:
    """Partner-caused if the frame's reply raises it and the coupled reply without the frame does not; own if both do."""
    if not f_carries:
        return None
    return "own" if n_carries else "partner"


def seed(conv_id: str, k: int) -> int:
    return sum(conv_id.encode()) * 100 + k


# The classifier, pure numpy: PCA then L2 logistic regression, both fitted on the training fold only.

def logistic(X: np.ndarray, y: np.ndarray, c: float = C, steps: int = 50) -> np.ndarray:
    """Weights (bias last) minimising the log loss summed over rows plus ||w||^2 / (2c), by Newton's method."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = np.eye(Xb.shape[1]) / c
    reg[-1, -1] = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) + reg @ w
        hess = Xb.T @ (Xb * (p * (1 - p))[:, None]) + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit_predict(Xtr, ytr, Xte, components: int | None) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    if components is not None and components < A.shape[1]:
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        A, B = A @ vt[:components].T, B @ vt[:components].T
    w = logistic(A, ytr.astype(float))
    return np.hstack([B, np.ones((len(B), 1))]) @ w


def folds(groups: list[str], n: int = FOLDS) -> np.ndarray:
    """Fold of each row: conversations in order of first appearance, dealt round-robin."""
    order = list(dict.fromkeys(groups))
    which = {g: i % n for i, g in enumerate(order)}
    return np.array([which[g] for g in groups])


def centred(X: np.ndarray, groups: list[str]) -> np.ndarray:
    """Each row less its conversation's mean, so topic does not fill the components (amended, PLAN.md). Uses no key."""
    X, g = np.asarray(X, float), np.asarray(groups)
    out = X.copy()
    for name in set(groups):
        out[g == name] -= X[g == name].mean(0)
    return out


def out_of_fold(X: np.ndarray, y: np.ndarray, groups: list[str], components: int | None) -> np.ndarray:
    X = centred(X, groups)
    f, out = folds(groups), np.zeros(len(y))
    for k in range(FOLDS):
        te = f == k
        if te.any():
            out[te] = fit_predict(X[~te], y[~te], X[te], components)
    return out


def auc(scores, y) -> float:
    scores, y = np.asarray(scores, float), np.asarray(y, bool)
    return h.auc(scores[~y], scores[y])


def cluster_bootstrap(groups: list[str], stat, n_boot: int = 2000, seed_: int = 0) -> tuple[float, float]:
    """95% interval of stat(index) over conversations resampled with replacement, the scores held."""
    rng = np.random.default_rng(seed_)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    names = list(by)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[names[j]] for j in rng.integers(0, len(names), len(names))])
        v = stat(idx)
        if not np.isnan(v):
            vals.append(v)
    low, high = np.percentile(vals, [2.5, 97.5])
    return float(low), float(high)


def read_verdict(low: float, high: float, diff_low: float) -> str:
    if low >= WORKS and diff_low > 0:
        return "holds"
    if high < NONE_:
        return "no cheap read here"
    return "undecided"


def construction(rows: list[dict], redraw_same: bool) -> dict:
    f = [s for r in rows for s in r["samples"]]
    rate_f = float(np.mean([s["F"]["carries"] for s in f]))
    rate_n = float(np.mean([s["N"]["carries"] for s in f]))
    keyed = [(r["id"], s["key"]) for r in rows for s in r["samples"] if s["key"]]
    count = {k: sum(1 for _, x in keyed if x == k) for k in ("partner", "own")}
    convs = {k: len({c for c, x in keyed if x == k}) for k in ("partner", "own")}
    enough = all(count[k] >= ENOUGH and convs[k] >= CONVS for k in count)
    return {"coupling": {"pass": redraw_same},
            "frame works": {"value": {"F": rate_f, "N": rate_n}, "bar": 0.2, "pass": rate_f - rate_n >= 0.2},
            "enough": {"value": count, "conversations": convs, "bar": [ENOUGH, CONVS], "pass": enough}}


def holds(checks: dict) -> bool:
    return all(c["pass"] for c in checks.values())


def judge(rows: list[dict], states: dict) -> dict:
    """The main case: state read, baselines and the account, on the keyed F replies."""
    items = [(r["id"], i, s) for r in rows for i, s in enumerate(r["samples"]) if s["key"]]
    groups = [g for g, _, _ in items]
    y = np.array([s["key"] == "partner" for _, _, s in items])
    fit_x = np.array([[s["F"]["logprob"]] for _, _, s in items])
    text_x = np.array([[s["F"]["hits"], s["F"]["length"], s["F"]["first_hit_token"]] for _, _, s in items], float)
    scores = {"fit": out_of_fold(fit_x, y, groups, None), "text": out_of_fold(text_x, y, groups, None)}
    for L in LAYERS:
        X = np.array([states[g][str(L)][i] for g, i, _ in items])
        scores[f"layer {L}"] = out_of_fold(X, y, groups, COMPONENTS)
    scores["account"] = np.array([s["F"]["account"]["score"] for _, _, s in items])
    out = {"n": {"partner": int(y.sum()), "own": int((~y).sum())}, "auc": {k: auc(v, y) for k, v in scores.items()}}
    base = max(("fit", "text"), key=lambda k: out["auc"][k])
    main = scores[f"layer {LAYER}"]
    a = lambda s: lambda idx: auc(s[idx], y[idx]) if 0 < y[idx].sum() < len(idx) else float("nan")  # noqa: E731
    low, high = cluster_bootstrap(groups, a(main))
    dlow, dhigh = cluster_bootstrap(groups, lambda idx: a(main)(idx) - a(scores[base])(idx))
    out["read"] = {"layer": LAYER, "baseline": base, "low": low, "high": high, "difference": [dlow, dhigh],
                   "verdict": read_verdict(low, high, dlow)}
    alow, ahigh = cluster_bootstrap(groups, a(scores["account"]))
    out["account"] = {"low": alow, "high": ahigh, "verdict": h.verdict(alow, ahigh)}
    return out


def side_case(rows: list[dict], step2_rows: dict) -> dict:
    """Prefill against the model's own samples in the same goal-only context, by fit."""
    groups, x, y = [], [], []
    for r in rows:
        if r["id"] not in step2_rows:
            continue
        for s in r["samples"]:
            groups.append(r["id"]); x.append(-s["N"]["logprob_goal"]); y.append(False)
        groups.append(r["id"]); x.append(-step2_rows[r["id"]]["P"]["logprob"]); y.append(True)
    x, y = np.array(x), np.array(y)
    low, high = cluster_bootstrap(groups, lambda idx: auc(x[idx], y[idx]) if 0 < y[idx].sum() < len(idx) else float("nan"))
    return {"auc": auc(x, y), "low": low, "high": high, "verdict": "holds" if low >= 0.7 else "does not hold"}


# The model.

class Sampler(h.Hidden):
    def eos(self) -> set[int]:
        e = self.hf.generation_config.eos_token_id
        return set(e if isinstance(e, list) else [e])

    def coupled(self, messages: list[dict], seeds: list[int]) -> list[list[int]]:
        """In batches of BATCH seeds, so a seed's draw does not depend on how many others are drawn with it."""
        return [x for i in range(0, len(seeds), BATCH) for x in self.coupled_batch(messages, seeds[i:i + BATCH])]

    def coupled_batch(self, messages: list[dict], seeds: list[int]) -> list[list[int]]:
        """One sample per seed, temperature-scaled Gumbel-max with the seed's own noise, one draw per step for every
        step, so the same seed under another context shares its noise step by step."""
        import torch

        ctx = self.context(messages)
        ids = torch.tensor([ctx] * len(seeds), device=self.hf.device)
        gens = [torch.Generator().manual_seed(s) for s in seeds]
        stop, out, done, past = self.eos(), [[] for _ in seeds], [False] * len(seeds), None
        with torch.no_grad():
            for _ in range(MAX_NEW):
                o = self.hf(ids, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = o.past_key_values
                logits = o.logits[:, -1].float().cpu() / TEMPERATURE
                noise = torch.stack([-torch.log(-torch.log(torch.rand(logits.shape[-1], generator=g).clamp_min(1e-20)))
                                     for g in gens])
                nxt = (logits + noise).argmax(-1)
                for b, t in enumerate(nxt.tolist()):
                    if not done[b]:
                        if t in stop:
                            done[b] = True
                        else:
                            out[b].append(t)
                if all(done):
                    break
                ids = nxt[:, None].to(self.hf.device)
        return out

    def pass_over(self, messages: list[dict], reply: list[int]) -> tuple[float, dict]:
        """Mean token log-probability of the reply, and its residual mean-pooled over the reply's tokens per layer."""
        import torch

        ctx = self.context(messages)
        with torch.no_grad():
            t = torch.tensor([ctx + reply], device=self.hf.device)
            o = self.hf(t, output_hidden_states=True, use_cache=False)
        n = len(reply)
        lp = o.logits[0, len(ctx) - 1:-1].float().log_softmax(-1).cpu()
        mean_lp = float(lp[torch.arange(n), torch.tensor(reply)].mean())
        pooled = {str(L): o.hidden_states[L + 1][0, len(ctx):].float().mean(0).cpu().numpy() for L in LAYERS}
        return mean_lp, pooled

    def logprob_ids(self, messages: list[dict], reply: list[int]) -> float:
        return self.pass_over(messages, reply)[0]


def read_conversation(bed: Sampler, c: dict, run: Path, record: Record) -> dict:
    goal, m = c["goal"], c["markers"]
    seeds = [seed(c["id"], k) for k in range(K)]
    f_ctx, n_ctx = frame_messages(goal, c["frame"]), bare_messages(goal)
    f_ids, n_ids = bed.coupled(f_ctx, seeds), bed.coupled(n_ctx, seeds)
    samples, states = [], {str(L): [] for L in LAYERS}
    for k in range(K):
        f_text, n_text = bed.text(f_ids[k]), bed.text(n_ids[k])
        record.append("sampled", "model", conversation=c["id"], sample=k, seed=seeds[k], frame=f_text, without=n_text,
                      frame_cut_off=len(f_ids[k]) >= MAX_NEW, without_cut_off=len(n_ids[k]) >= MAX_NEW)
        f_lp, pooled = bed.pass_over(f_ctx, f_ids[k])
        for L in LAYERS:
            states[str(L)].append(pooled[str(L)])
        pos = first_hit(f_text, m)
        s = {"F": {"carries": h.carries(f_text, m), "hits": hits(f_text, m), "length": len(f_ids[k]),
                   "first_hit_token": len(bed.tok(f_text[:pos], add_special_tokens=False)["input_ids"]) if pos is not None
                   else len(f_ids[k]), "logprob": f_lp, "account": bed.account(f_ctx, f_text)},
             "N": {"carries": h.carries(n_text, m), "logprob_goal": bed.logprob_ids(h.goal_messages(goal), n_ids[k])}}
        s["key"] = key(s["F"]["carries"], s["N"]["carries"])
        samples.append(s)
    (run / "states").mkdir(parents=True, exist_ok=True)
    np.savez(run / "states" / f"{c['id']}.npz", **{L: np.array(v) for L, v in states.items()})
    record.append("read", "code", conversation=c["id"], samples=samples)
    return {"id": c["id"], "samples": samples, "n_ids": n_ids}


def load_states(run: Path, ids: list[str]) -> dict:
    out = {}
    for i in ids:
        z = np.load(run / "states" / f"{i}.npz")
        out[i] = {L: z[L] for L in z.files}
    return out


def step2_reads(path: str | Path) -> dict:
    p = Path(path)
    return {r["conversation"]: r for r in read(p) if r["kind"] == "read"} if p.exists() else {}


def pilot(run: Path):
    """Samples only the frame-free replies, which do not depend on the frame, and picks each goal's frame from its
    candidates by them (amended, PLAN.md). Writes the frames for the owner to freeze."""
    from substrate.states import Reader

    goals = {g["id"]: g["goal"][0] for g in h.load(h.GOALS)}
    record = Record(run / "record.jsonl")
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    bed = Sampler(reader)
    record.append("run", "code", **reader.info, pilot=True, samples=PILOT_K, temperature=TEMPERATURE, max_new_tokens=MAX_NEW,
                  target=TARGET)
    done = {r["conversation"]: r for r in read(record.path) if r["kind"] == "picked"}
    chosen = []
    for c in h.load(CANDIDATES):
        if c["id"] not in done:
            start = time.time()
            ids = bed.coupled(bare_messages(goals[c["id"]]), [seed(c["id"], k) for k in range(PILOT_K)])
            texts = [bed.text(x) for x in ids]
            for k, t in enumerate(texts):
                record.append("sampled", "model", conversation=c["id"], sample=k, without=t, pilot=True)
            i, counts = pick(c["candidates"], texts)
            done[c["id"]] = record.append("picked", "code", conversation=c["id"], index=i, counts=counts)
            print(f"{c['id']}: {counts} -> {i} ({time.time() - start:.0f}s)", flush=True)
        best = c["candidates"][done[c["id"]]["index"]]
        chosen.append({"id": c["id"], "by": "model", "frozen": False, "frame": best["frame"], "markers": best["markers"],
                       "picked": {"index": done[c["id"]]["index"], "counts": done[c["id"]]["counts"]}})
    Path(FRAMES).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in chosen), encoding="utf-8")
    print("frames written to", FRAMES)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--dry", action="store_true", help="plumbing and time: the 2 dry conversations, no verdict")
    p.add_argument("--step2", default=STEP2)
    p.add_argument("--pilot", action="store_true", help="the frame-free replies only, to pick each goal's frame")
    a = p.parse_args(argv)
    if a.pilot:
        return pilot(Path(a.run))
    if a.dry:
        convs = h.load(DRY)
    else:
        frames = h.load(FRAMES)
        if not (len(frames) == 40 and all(f.get("frozen") for f in frames)):
            raise SystemExit("the frames are not frozen; the owner freezes them first")
        goals = {g["id"]: g["goal"][0] for g in h.load(h.GOALS)}
        convs = [{**f, "goal": goals[f["id"]]} for f in frames]
    from substrate.states import Reader

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    bed = Sampler(reader)
    record.append("run", "code", **reader.info, dry=a.dry, frame_line=FRAME, samples=K, temperature=TEMPERATURE,
                  max_new_tokens=MAX_NEW, layer=LAYER, read_layers=LAYERS, components=COMPONENTS, c=C, folds=FOLDS,
                  thinking=False)
    done = {r["conversation"]: r for r in read(record.path) if r["kind"] == "read"}
    rows, redraw_same = [], None
    for i, c in enumerate(convs):
        if c["id"] in done:
            rows.append({"id": c["id"], "samples": done[c["id"]]["samples"]})
            continue
        start = time.time()
        row = read_conversation(bed, c, run, record)
        if i == 0:
            again = bed.coupled(bare_messages(c["goal"]), [seed(c["id"], k) for k in range(K)])
            redraw_same = again == row["n_ids"]
            record.append("redrawn", "code", conversation=c["id"], same=redraw_same)
        rows.append({"id": row["id"], "samples": row["samples"]})
        print(f"{c['id']}: {time.time() - start:.0f}s", flush=True)
    if redraw_same is None:
        redraw_same = next(r["same"] for r in read(record.path) if r["kind"] == "redrawn")
    if a.dry:
        for r in rows:
            print(r["id"], [(s["F"]["carries"], s["N"]["carries"], s["key"], round(s["F"]["account"]["score"], 3))
                            for s in r["samples"]])
        print("redrawn the same:", redraw_same)
        return
    pilot_texts = {(r["conversation"], r["sample"]): r["without"] for r in read(PILOT)
                   if r["kind"] == "sampled"} if Path(PILOT).exists() else {}
    if pilot_texts:                                       # reported, not judged: samples 0-7 should be the pilot's draws
        same = [pilot_texts.get((r["conversation"], r["sample"])) == r["without"] for r in read(record.path)
                if r["kind"] == "sampled" and r["sample"] < PILOT_K]
        record.append("pilot redrawn", "code", same=sum(same), of=len(same))
        print(f"pilot's draws redrawn: {sum(same)} of {len(same)}")
    checks = construction(rows, redraw_same)
    record.append("construction", "code", checks=checks, holds=holds(checks))
    print(json.dumps(checks, indent=1))
    side = side_case(rows, step2_reads(a.step2))
    record.append("side case", "code", **side)
    print("side case (prefill):", json.dumps(side))
    if not holds(checks):
        print("the construction does not hold: the main case is not read")
        return
    result = judge(rows, load_states(run, [r["id"] for r in rows]))
    record.append("verdict", "code", **result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
