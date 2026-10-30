#!/usr/bin/env python3
"""
concept_negation_probe.py - Experiment 7 harness (DARA / AWARE).

Question: does a quantized concept space store contrary pairs (brave/cowardly)
as a shared base plus one differing part, with the differing part specific to
each axis? And does status negation ("told" / "not told") stay hard to separate?

Predictions tested (see "Experiment 7 design" in the Negation and Ledger doc):
  P1 Shared genus       contrary pairs share most PQ segments: more than unrelated
                        pairs, close to near-synonyms.
  P2 Determinate diff   the part that differs changes from axis to axis; no
                        universal "not" segment or direction.
  P3 Status blindness   "X" / "not X" sentences do not separate along a consistent
                        direction.
  P4 Compression        coding a contrary pair as base + difference saves bits
                        over coding the two separately. (This follows from P1 by
                        construction; it is reported so the ledger's view is explicit.)

Sources:
  --source server     llama.cpp llama-server with --embeddings (OpenAI-compatible
                      /v1/embeddings). Whole-sentence vectors; final layer only.
                      Use for Qwen3-Embedding-4B (start it with --pooling last) or
                      for the 8B with --pooling mean.
  --source hf         Hugging Face transformers, any layer, mean-pooled over the
                      target phrase's tokens inside its sentence. Use this for
                      mid-to-late layer hidden states of the 8B.
  --source synthetic  planted structure, to check the pipeline itself.

Examples:
  # 1. Pipeline check, no model needed
  python concept_negation_probe.py --source synthetic
  python concept_negation_probe.py --source synthetic --synthetic-universal-not

  # 2. Embedding model through llama.cpp
  llama-server -m Qwen3-Embedding-4B-Q4_K_M.gguf --embeddings --pooling last --port 8081
  python concept_negation_probe.py --source server --url http://localhost:8081 --tag qwen3emb4b

  # 3. Local 8B hidden states at a chosen layer
  python concept_negation_probe.py --source hf --model meta-llama/Llama-3.1-8B --layer 22 --tag llama8b_L22
  (add --load-4bit if memory is tight; needs bitsandbytes)

  # Use your own probe set (e.g. Experiment 6's 144 phrases, Told Hives pairs)
  python concept_negation_probe.py --write-probes probes.json   # dump the default, then edit
  python concept_negation_probe.py --probes probes.json --source server ...

Needs numpy. The hf source also needs torch and transformers.
Thresholds in the verdict section are provisional and set before seeing data.
"""

import argparse
import itertools
import json
import math
import os
import sys
import urllib.request

import numpy as np

# ----------------------------------------------------------------------------
# Default probe set. 12 axes x 3 terms per pole, each in a sentence frame that
# fits both poles, plus 24 status pairs. Not identical to Experiment 6's
# phrases: load those with --probes to keep the two experiments comparable.
# Add Told Hives pairs as extra axes in the same format.
# ----------------------------------------------------------------------------
DEFAULT_PROBES = {
    "instruction": "Represent the meaning of this sentence, focusing on the quality it describes.",
    "axes": [
        {"name": "temperature", "template": "The water in the kettle was {}.",
         "a": ["hot", "scalding", "warm"], "b": ["cold", "icy", "chilly"]},
        {"name": "size", "template": "They found a {} hive in the orchard.",
         "a": ["large", "huge", "big"], "b": ["small", "tiny", "little"]},
        {"name": "wealth", "template": "The family was {} that year.",
         "a": ["rich", "wealthy", "prosperous"], "b": ["poor", "impoverished", "destitute"]},
        {"name": "speed", "template": "The cart moved {} down the lane.",
         "a": ["quickly", "swiftly", "rapidly"], "b": ["slowly", "sluggishly", "leisurely"]},
        {"name": "light", "template": "The room was {} at noon.",
         "a": ["bright", "sunlit", "luminous"], "b": ["dark", "dim", "shadowy"]},
        {"name": "age", "template": "The beekeeper was {}.",
         "a": ["old", "elderly", "aged"], "b": ["young", "youthful", "juvenile"]},
        {"name": "safety", "template": "The road through the marsh was {}.",
         "a": ["safe", "secure", "harmless"], "b": ["dangerous", "perilous", "treacherous"]},
        {"name": "joy", "template": "The house was {} after the wedding.",
         "a": ["joyful", "glad", "cheerful"], "b": ["sorrowful", "grieving", "mournful"]},
        {"name": "honesty", "template": "The steward's account was {}.",
         "a": ["truthful", "honest", "accurate"], "b": ["false", "dishonest", "deceitful"]},
        {"name": "courage", "template": "He was {} when the swarm rose.",
         "a": ["brave", "bold", "fearless"], "b": ["cowardly", "timid", "fearful"]},
        {"name": "trust", "template": "She {} her sister.",
         "a": ["trusted", "confided in", "relied on"], "b": ["betrayed", "deceived", "double-crossed"]},
        {"name": "arrival", "template": "At dusk the stranger {} the house.",
         "a": ["arrived at", "came to", "reached"], "b": ["left", "departed from", "fled"]},
    ],
    "status": [
        ["The bees were told of the death.", "The bees were not told of the death."],
        ["The keeper knew about the marriage.", "The keeper did not know about the marriage."],
        ["The hives stayed through the winter.", "The hives did not stay through the winter."],
        ["The letter reached the house.", "The letter did not reach the house."],
        ["Her mother approved of the match.", "Her mother did not approve of the match."],
        ["The steward signed the will.", "The steward did not sign the will."],
        ["The swarm settled in the oak.", "The swarm did not settle in the oak."],
        ["He kept his promise.", "He did not keep his promise."],
        ["The gate was locked.", "The gate was not locked."],
        ["The child was born at Hollowmere.", "The child was not born at Hollowmere."],
        ["She opened the hive.", "She did not open the hive."],
        ["The village heard the news.", "The village did not hear the news."],
        ["The brace was removed.", "The brace was not removed."],
        ["The candles were lit.", "The candles were not lit."],
        ["He returned before the new moon.", "He did not return before the new moon."],
        ["The honey was sold at market.", "The honey was not sold at market."],
        ["They buried him in the garth.", "They did not bury him in the garth."],
        ["The inheritance passed to her.", "The inheritance did not pass to her."],
        ["The door was painted black.", "The door was not painted black."],
        ["She remembered the old song.", "She did not remember the old song."],
        ["The bees hummed at the window.", "The bees did not hum at the window."],
        ["The priest blessed the hives.", "The priest did not bless the hives."],
        ["The debt was paid.", "The debt was not paid."],
        ["Ambrose wrote to his mother.", "Ambrose did not write to his mother."],
    ],
}


# ----------------------------------------------------------------------------
# Items: one row per vector we need.
# ----------------------------------------------------------------------------
def build_items(probes):
    """Return list of dicts: sentence, span (char start, end) or None, plus labels."""
    items = []
    for ax in probes["axes"]:
        tpl = ax["template"]
        assert tpl.count("{}") == 1, f"template for {ax['name']} needs exactly one {{}}"
        pre = tpl.split("{}")[0]
        for pole in ("a", "b"):
            for term in ax[pole]:
                sent = tpl.format(term)
                items.append({"kind": "term", "axis": ax["name"], "pole": pole, "term": term,
                              "sentence": sent, "span": (len(pre), len(pre) + len(term))})
    for i, (pos, neg) in enumerate(probes["status"]):
        items.append({"kind": "status", "pair": i, "pole": "pos", "sentence": pos, "span": None})
        items.append({"kind": "status", "pair": i, "pole": "neg", "sentence": neg, "span": None})
    return items


def build_pairs(items, rng, n_unrelated=None):
    idx = {}
    for i, it in enumerate(items):
        if it["kind"] == "term":
            idx.setdefault(it["axis"], {"a": [], "b": []})[it["pole"]].append(i)
    contrary, synonym = [], []
    for ax, poles in idx.items():
        for i, j in itertools.product(poles["a"], poles["b"]):
            contrary.append((i, j, ax))
        for pole in ("a", "b"):
            for i, j in itertools.combinations(poles[pole], 2):
                synonym.append((i, j, ax))
    term_ids = [i for i, it in enumerate(items) if it["kind"] == "term"]
    n_unrelated = n_unrelated or len(contrary)
    unrelated, seen = [], set()
    while len(unrelated) < n_unrelated:
        i, j = rng.choice(term_ids, 2, replace=False)
        if items[i]["axis"] == items[j]["axis"] or (i, j) in seen:
            continue
        seen.add((i, j))
        unrelated.append((int(i), int(j), None))
    status = []
    for p in sorted({it["pair"] for it in items if it["kind"] == "status"}):
        pos = next(i for i, it in enumerate(items) if it["kind"] == "status" and it["pair"] == p and it["pole"] == "pos")
        neg = next(i for i, it in enumerate(items) if it["kind"] == "status" and it["pair"] == p and it["pole"] == "neg")
        status.append((pos, neg, p))
    return {"contrary": contrary, "synonym": synonym, "unrelated": unrelated, "status": status}


# ----------------------------------------------------------------------------
# Vector sources
# ----------------------------------------------------------------------------
def embed_server(items, url, model, instruction, batch=16):
    out = []
    endpoint = url.rstrip("/") + "/v1/embeddings"
    texts = [(f"Instruct: {instruction}\nQuery: {it['sentence']}" if instruction else it["sentence"])
             for it in items]
    for k in range(0, len(texts), batch):
        body = json.dumps({"input": texts[k:k + batch], "model": model or "default"}).encode()
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        rows = sorted(data["data"], key=lambda d: d["index"])
        out.extend(row["embedding"] for row in rows)
        print(f"  embedded {min(k + batch, len(texts))}/{len(texts)}", file=sys.stderr)
    return np.asarray(out, dtype=np.float64)


def embed_hf(items, model_name, layer, load_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    kw = {"output_hidden_states": True}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = "auto"
    else:
        kw["torch_dtype"] = torch.float16
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()
    n_layers = model.config.num_hidden_layers
    if layer is None:
        layer = int(round(0.66 * n_layers))
    print(f"  using hidden layer {layer} of {n_layers}", file=sys.stderr)
    out = []
    for n, it in enumerate(items):
        enc = tok(it["sentence"], return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc).hidden_states[layer][0].float().cpu().numpy()
        if it["span"] is None:
            mask = [e > s for s, e in offsets]  # all real tokens (skips BOS with 0,0)
        else:
            s0, e0 = it["span"]
            mask = [(e > s0 and s < e0) for s, e in offsets]
        if not any(mask):
            mask = [True] * len(offsets)
        out.append(hs[np.asarray(mask)].mean(axis=0))
        if (n + 1) % 20 == 0:
            print(f"  encoded {n + 1}/{len(items)}", file=sys.stderr)
    return np.asarray(out, dtype=np.float64), layer


def embed_synthetic(items, rng, universal_not=False, dim=512):
    """Planted structure: each axis has its own genus vector and its own
    difference direction. Status pairs get a shared 'not' direction only if
    universal_not is set (and then contrary pairs share it too)."""
    axes = sorted({it["axis"] for it in items if it["kind"] == "term"})
    genus = {a: rng.normal(size=dim) * 1.0 for a in axes}
    diff = {a: rng.normal(size=dim) * 0.6 for a in axes}
    not_vec = rng.normal(size=dim) * 0.6
    status_base = {}
    X = []
    for it in items:
        if it["kind"] == "term":
            sign = 1.0 if it["pole"] == "a" else -1.0
            d = not_vec if universal_not else diff[it["axis"]]
            v = genus[it["axis"]] + sign * d / 2 + rng.normal(size=dim) * 0.15
        else:
            base = status_base.setdefault(it["pair"], rng.normal(size=dim))
            v = base + rng.normal(size=dim) * 0.15
            if it["pole"] == "neg":
                v = v + (not_vec if universal_not else rng.normal(size=dim) * 0.15)
        X.append(v)
    return np.asarray(X)


# ----------------------------------------------------------------------------
# Product quantization
# ----------------------------------------------------------------------------
def kmeans(X, k, rng, iters=60):
    n = len(X)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):  # k-means++
        d2 = np.min(((X[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centers.append(X[rng.choice(n, p=p)])
    C = np.asarray(centers)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = C.copy()
        for j in range(k):
            pts = X[lab == j]
            if len(pts):
                newC[j] = pts.mean(0)
            else:  # reseed empty cluster at farthest point
                far = ((X - C[lab]) ** 2).sum(-1).argmax()
                newC[j] = X[far]
        if np.allclose(newC, C):
            break
        C = newC
    return ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)


def preprocess(X, pca_dims):
    """Center, PCA-rotate, keep the top components. Returns Z (N, P)."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = min(pca_dims, len(X) - 1, Xc.shape[1])
    return Xc @ Vt[:P].T


def partition(Z, M, mode, rng):
    """Split columns of Z into M segments.
    'pca': principal components dealt round-robin (component k -> segment k mod M),
           so segments carry balanced variance but each stays aligned with
           principal axes. A difference that lives on a few components then
           lands in a few segments.
    'rotated': random orthogonal rotation first, a control that smears any
           direction across all segments. If 'pca' and 'rotated' agree, the
           result does not depend on segment alignment."""
    P = (Z.shape[1] // M) * M
    Z = Z[:, :P]
    if mode == "rotated":
        Q, _ = np.linalg.qr(rng.normal(size=(P, P)))
        Z = Z @ Q
        return [Z[:, s * (P // M):(s + 1) * (P // M)] for s in range(M)]
    return [Z[:, s::M] for s in range(M)]


def pq_codes(segments, K, rng):
    return np.stack([kmeans(seg, K, rng) for seg in segments], axis=1)  # (N, M)


# ----------------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------------
def shared_fraction(codes, pairs):
    return float(np.mean([(codes[i] == codes[j]).mean() for i, j, _ in pairs])) if pairs else float("nan")


def bits_saved(codes, pairs, K):
    """Independent: 2*M*log2K. Base+diff: M*log2K + log2(M+1) + log2 C(M,d) + d*log2K."""
    M = codes.shape[1]
    lk = math.log2(K)
    out = []
    for i, j, _ in pairs:
        d = int((codes[i] != codes[j]).sum())
        joint = M * lk + math.log2(M + 1) + math.log2(math.comb(M, d)) + d * lk
        out.append(2 * M * lk - joint)
    return float(np.mean(out))


def diff_energy_profile(segments, i, j):
    e = np.asarray([((seg[i] - seg[j]) ** 2).sum() for seg in segments])
    return e / e.sum() if e.sum() > 0 else e


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def mean_abs_cos(vecs_a, vecs_b=None):
    if vecs_b is None:
        vals = [abs(float(u @ v)) for u, v in itertools.combinations(vecs_a, 2)]
    else:
        vals = [abs(float(u @ v)) for u in vecs_a for v in vecs_b]
    return float(np.mean(vals)) if vals else float("nan")


def direction_tests(Z, pairs, items):
    """Partition-free tests on difference vectors."""
    items_axis = [it.get("axis") for it in items]
    D = {}
    for cat in ("contrary", "unrelated", "status"):
        D[cat] = [(unit(Z[j] - Z[i]), tag) for i, j, tag in pairs[cat]]
    by_axis = {}
    for v, ax in D["contrary"]:
        by_axis.setdefault(ax, []).append(v)
    within = np.mean([mean_abs_cos(vs) for vs in by_axis.values() if len(vs) > 1])
    axes = list(by_axis)
    across = np.mean([mean_abs_cos(by_axis[a], by_axis[b]) for a, b in itertools.combinations(axes, 2)])
    # Null: unrelated-pair differences compared only when all four terms come
    # from different axes, so shared genus vectors cannot inflate it.
    U = [(unit(Z[j] - Z[i]), {items_axis[i], items_axis[j]}) for i, j, _ in pairs["unrelated"]]
    null_vals = [abs(float(u @ v)) for (u, au), (v, av) in itertools.combinations(U, 2) if not (au & av)]
    null = float(np.mean(null_vals)) if null_vals else float("nan")
    status_within = mean_abs_cos([v for v, _ in D["status"]])
    return {"contrary_within_axis_abs_cos": float(within),
            "contrary_across_axis_abs_cos": float(across),
            "null_unrelated_abs_cos": float(null),
            "status_within_abs_cos": float(status_within)}


def status_sign_test(Z, status_pairs, rng, splits=200):
    """Learn the mean 'not' direction on half the status pairs; on the other
    half, check whether the negated sentence projects further along it.
    0.5 = no consistent direction, 1.0 = cleanly separable."""
    n = len(status_pairs)
    accs = []
    for _ in range(splits):
        perm = rng.permutation(n)
        tr, te = perm[: n // 2], perm[n // 2:]
        d = unit(np.mean([Z[status_pairs[k][1]] - Z[status_pairs[k][0]] for k in tr], axis=0))
        accs.append(np.mean([(Z[status_pairs[k][1]] - Z[status_pairs[k][0]]) @ d > 0 for k in te]))
    return float(np.mean(accs))


def raw_cosines(X, pairs):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return {cat: float(np.mean([Xn[i] @ Xn[j] for i, j, _ in ps])) for cat, ps in pairs.items()}


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def analyse(X, items, pairs, args):
    rng = np.random.default_rng(args.seed)
    Z = preprocess(X, args.pca_dims)
    N = len(X)
    Ks = [k for k in args.K if k <= N // 4]
    if len(Ks) < len(args.K):
        print(f"  note: skipping K > N/4 = {N // 4} (only {N} vectors)", file=sys.stderr)
    res = {"n_vectors": N, "dim_raw": X.shape[1], "dim_pca": Z.shape[1],
           "raw_cosine": raw_cosines(X, pairs),
           "directions": direction_tests(Z, pairs, items),
           "status_sign_test_acc": status_sign_test(Z, pairs["status"], rng),
           "pq": []}
    for mode in ("pca", "rotated"):
        for M in args.M:
            for K in Ks:
                rows = []
                for s in range(args.seeds):
                    r = np.random.default_rng(args.seed + 1000 * s + 7 * M + K)
                    segs = partition(Z, M, mode, r)
                    codes = pq_codes(segs, K, r)
                    row = {cat: shared_fraction(codes, ps) for cat, ps in pairs.items()}
                    row.update({f"bits_{cat}": bits_saved(codes, ps, K) for cat, ps in pairs.items()})
                    # where does each contrary pair's difference concentrate?
                    modal = {}
                    peaks, conc = [], []
                    for i, j, ax in pairs["contrary"]:
                        prof = diff_energy_profile(segs, i, j)
                        peaks.append(int(prof.argmax()))
                        conc.append(float(prof.max()))
                        modal.setdefault(ax, []).append(int(prof.argmax()))
                    counts = np.bincount(peaks, minlength=M)
                    row["global_peak_share"] = float(counts.max() / counts.sum())
                    row["peak_concentration"] = float(np.mean(conc))
                    axis_modes = [int(np.bincount(v, minlength=M).argmax()) for v in modal.values()]
                    row["distinct_axis_modes"] = len(set(axis_modes))
                    rows.append(row)
                agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                agg.update({f"{k}_sd": float(np.std([r[k] for r in rows])) for k in ("contrary", "synonym", "unrelated", "status")})
                agg.update({"mode": mode, "M": M, "K": K})
                res["pq"].append(agg)
    return res


def verdict(res, n_axes):
    """Provisional thresholds, fixed before any real run."""
    pca = [r for r in res["pq"] if r["mode"] == "pca"]
    rot = [r for r in res["pq"] if r["mode"] == "rotated"]
    m = lambda rows, k: float(np.mean([r[k] for r in rows]))
    d = res["directions"]
    out = {}
    con, syn, unr = m(pca, "contrary"), m(pca, "synonym"), m(pca, "unrelated")
    out["P1_shared_genus"] = bool(con > unr + 0.10 and con >= syn - 0.15)
    universal = (d["contrary_across_axis_abs_cos"] - d["null_unrelated_abs_cos"] > 0.10
                 or m(pca, "global_peak_share") > 0.60)
    determinate = d["contrary_within_axis_abs_cos"] - d["null_unrelated_abs_cos"] > 0.10
    out["P2_determinate_difference"] = bool(determinate and not universal)
    out["P2_universal_not_detected"] = bool(universal)
    out["P3_status_blindness"] = bool(res["status_sign_test_acc"] < 0.70)
    out["P4_compression"] = bool(m(pca, "bits_contrary") > m(pca, "bits_unrelated"))
    out["segment_alignment_vs_rotation"] = m(pca, "peak_concentration") - m(rot, "peak_concentration")
    if out["P1_shared_genus"] and out["P2_determinate_difference"] and out["P4_compression"]:
        out["reading"] = "Content negation can be emergent: propose contraries from segment differences; ledger keeps only the status flag."
    elif universal:
        out["reading"] = "A learned negation direction exists across axes: simpler but less dialectical. Revisit the theory."
    elif not out["P1_shared_genus"]:
        out["reading"] = "Codes too coarse or tangled here. Try other layers / more segments before falling back to symbolic negation."
    else:
        out["reading"] = "Mixed result: shared genus without a clear per-axis difference. Inspect per-axis numbers."
    if not out["P3_status_blindness"]:
        out["reading"] += " Status negation separates cleanly, which is unexpected and worth following up."
    return out


def report(res, v, tag):
    print(f"\n=== Experiment 7 probe: {tag} ===")
    print(f"vectors {res['n_vectors']}, raw dim {res['dim_raw']}, PCA dim {res['dim_pca']}")
    rc = res["raw_cosine"]
    print("\nRaw cosine by pair type:  " + "  ".join(f"{k} {rc[k]:.3f}" for k in ("contrary", "synonym", "unrelated", "status")))
    d = res["directions"]
    print("\nDifference directions (|cos|):")
    print(f"  contrary, same axis    {d['contrary_within_axis_abs_cos']:.3f}")
    print(f"  contrary, across axes  {d['contrary_across_axis_abs_cos']:.3f}")
    print(f"  null (unrelated diffs) {d['null_unrelated_abs_cos']:.3f}")
    print(f"  status pairs           {d['status_within_abs_cos']:.3f}")
    print(f"  status sign test acc   {res['status_sign_test_acc']:.3f}   (0.5 = blind, 1.0 = separable)")
    print("\nShared PQ segments (fraction of M) and bits saved by base+diff coding:")
    print(f"  {'mode':8}{'M':>3}{'K':>4}  {'contr':>6}{'syn':>6}{'unrel':>6}{'status':>7}  {'bits_c':>7}{'bits_u':>7}  {'peak':>5}{'conc':>6}{'modes':>6}")
    for r in res["pq"]:
        print(f"  {r['mode']:8}{r['M']:>3}{r['K']:>4}  {r['contrary']:6.2f}{r['synonym']:6.2f}{r['unrelated']:6.2f}{r['status']:7.2f}"
              f"  {r['bits_contrary']:7.1f}{r['bits_unrelated']:7.1f}  {r['global_peak_share']:5.2f}{r['peak_concentration']:6.2f}{r['distinct_axis_modes']:6.1f}")
    print("  peak = share of contrary pairs whose difference peaks in the single most common segment")
    print("  conc = average share of difference energy in the peak segment; modes = distinct peak segments across axes")
    print("\nVerdict (provisional thresholds):")
    for k, val in v.items():
        if k != "reading":
            print(f"  {k:32} {val if isinstance(val, bool) else round(val, 3)}")
    print(f"\n  {v['reading']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["server", "hf", "synthetic"], default="synthetic")
    ap.add_argument("--probes", help="probe set JSON (default: built-in)")
    ap.add_argument("--write-probes", help="write the built-in probe set to this path and exit")
    ap.add_argument("--url", default="http://localhost:8081", help="llama-server base URL")
    ap.add_argument("--model", help="server model name, or HF model id/path for --source hf")
    ap.add_argument("--instruction", default=None, help="override instruction prefix (server); '' for none")
    ap.add_argument("--layer", type=int, default=None, help="hidden layer for --source hf (default ~2/3 depth)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--synthetic-universal-not", action="store_true", help="plant one shared 'not' direction")
    ap.add_argument("--M", type=int, nargs="+", default=[4, 8], help="PQ segment counts")
    ap.add_argument("--K", type=int, nargs="+", default=[4, 8, 16], help="codebook sizes per segment")
    ap.add_argument("--pca-dims", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="runs/exp7")
    args = ap.parse_args()

    if args.write_probes:
        with open(args.write_probes, "w") as f:
            json.dump(DEFAULT_PROBES, f, indent=2)
        print(f"wrote {args.write_probes}")
        return

    probes = DEFAULT_PROBES
    if args.probes:
        with open(args.probes) as f:
            probes = json.load(f)
    rng = np.random.default_rng(args.seed)
    items = build_items(probes)
    pairs = build_pairs(items, rng)
    tag = args.tag or args.source
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, f"vectors_{tag}.npz")

    if args.source == "synthetic":
        X = embed_synthetic(items, rng, universal_not=args.synthetic_universal_not)
        meta = {"universal_not": args.synthetic_universal_not}
    elif os.path.exists(cache):
        print(f"  loading cached vectors from {cache}", file=sys.stderr)
        X = np.load(cache)["X"]
        meta = {"cached": True}
    elif args.source == "server":
        instr = probes.get("instruction", "") if args.instruction is None else args.instruction
        X = embed_server(items, args.url, args.model, instr)
        np.savez(cache, X=X)
        meta = {"url": args.url, "instruction": instr}
    else:
        if not args.model:
            sys.exit("--source hf needs --model")
        X, layer = embed_hf(items, args.model, args.layer, args.load_4bit)
        np.savez(cache, X=X)
        meta = {"model": args.model, "layer": layer}

    res = analyse(X, items, pairs, args)
    v = verdict(res, len(probes["axes"]))
    report(res, v, tag)
    with open(os.path.join(args.out_dir, f"results_{tag}.json"), "w") as f:
        json.dump({"tag": tag, "meta": meta, "args": vars(args), "results": res, "verdict": v}, f, indent=2)
    print(f"results written to {args.out_dir}/results_{tag}.json")


if __name__ == "__main__":
    main()
