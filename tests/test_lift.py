"""Lift v1 (body/lift.py): K from held-out code, its own check, the whitened map, the chunking, shape in the lift."""
import numpy as np
import pytest

from body import lift as L


def _low_rank(n, d=60, r=6, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(r, d)) * np.linspace(4, 1.5, r)[:, None]
    return rng.normal(size=(n, r)) @ B + noise * rng.normal(size=(n, d))


def test_k_is_chosen_by_held_out_code_and_the_check_passes_on_structured_text():
    W = _low_rank(3000, d=200, r=40)                    # more directions than Lift v0's rule may keep (KMAX = 20)
    r = L.fit_layer(W[:2400], W[2400:2700], W[2700:], floor_var=0.01)
    assert 30 <= r["K"] <= 50 and r["k_v0"] <= 20 and r["ok"]
    assert r["held_b_bits_per_chunk"]["K"] < r["held_b_bits_per_chunk"]["k_v0"]


def test_the_check_fails_where_lift_v0_already_holds_the_structure():
    W = _low_rank(1200)                                  # 6 directions: Lift v0's rule finds them too
    r = L.fit_layer(W[:800], W[800:1000], W[1000:], floor_var=0.01)
    assert r["K"] == r["k_v0"] == 6 and not r["ok"]


def test_the_check_fails_where_there_is_no_structure():
    W = np.random.default_rng(1).normal(size=(1200, 60))
    r = L.fit_layer(W[:800], W[800:1000], W[1000:], floor_var=0.01)
    assert not r["ok"]


def test_the_map_whitens_the_training_rows():
    W = _low_rank(2000, seed=2)
    r = L.fit_layer(W[:1600], W[1600:1800], W[1800:], floor_var=0.01)
    lift = {"mu": [r["mu"]], "V": [r["V"]], "scale": [r["scale"]]}
    Z = L.apply(lift, 0, W[:1600])
    assert Z.shape == (1600, r["K"])
    assert np.allclose(Z.mean(0), 0, atol=1e-8) and np.allclose(Z.var(0), 1, atol=0.05)


def test_held_bits_matches_fit_bits():
    from goalspace.formula import Fit
    W = _low_rank(300, seed=3)
    fit = Fit(W[:200], 0.01)
    hb = L.held_bits(fit, W[200:210], 8)
    for k in (0, 3, 8):
        assert np.isclose(hb[k], sum(fit.bits(x, k) for x in W[200:210]))


def test_chunks_follow_drawn_lengths_and_split_by_document(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    for i in range(20):
        (tmp_path / f"doc{i:02d}.txt").write_text(" ".join(f"Sentence {j} of document {i} is here." for j in range(40)))
    parts = L.chunks(tmp_path, tok, seed=5)
    docs = parts["documents"]
    assert not set(docs["train"]) & set(docs["held_a"]) and not set(docs["train"]) & set(docs["held_b"])
    assert len(docs["train"]) == 16 and len(docs["held_a"]) == 2 and len(docs["held_b"]) == 2
    lens = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in parts["train"]]
    assert min(lens) >= 1 and np.median(lens) < 20                     # habitat-like: mostly short
    assert L.chunks(tmp_path, tok, seed=5) == parts                     # deterministic


def test_shape_in_lift_runs_by_source():
    from body.stepm1 import shape_in_lift
    W = _low_rank(1500, seed=4)
    r = L.fit_layer(W[:1000], W[1000:1200], W[1200:], floor_var=0.01)
    lift = {"mu": [r["mu"]], "V": [r["V"]], "scale": [r["scale"]], "floor": [0.05]}
    slots = ["world"] + ["self", "world"] * 40
    turns = [-1] + [t for t in range(40) for _ in (0, 1)]
    S = _low_rank(len(slots), seed=6)[:, None, :]
    out = shape_in_lift(S, turns, slots, lift)
    assert out.shape == (len(slots), 1)
    assert np.isnan(out[: 1 + 2 * 10]).all() and np.isfinite(out[-10:]).all()
