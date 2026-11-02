"""The D jump's analysis on planted causes, without the model."""
import numpy as np

from research import d_jump as dj

LAYERS = [4, 6, 8]
WIDTH = 64


def planted(cause, seed=0, n_sets=10, n_texts=5, n_tokens=12):
    """Per-token states with a small spread at layer 4 and, at layer 8, a jump of the planted kind."""
    rng = np.random.default_rng(seed)
    tokens = []
    for _ in range(n_sets):
        base = rng.normal(size=WIDTH)
        texts = []
        for _ in range(n_texts):
            T = base + rng.normal(scale=0.5, size=(n_tokens, 3, WIDTH))
            if cause == "token":          # one massive token per text, in its own direction
                T[rng.integers(n_tokens), 2] += 400 * rng.normal(size=WIDTH) / np.sqrt(WIDTH)
            elif cause == "coordinate":   # every token, in a few fixed coordinates, varying by text
                T[:, 2, :3] += 100 * rng.normal(size=3)
            elif cause == "scale":        # everything grows
                T[:, 2] *= 15
            texts.append(T.astype(np.float32))
        tokens.append(texts)
    return tokens


def readings(tokens):
    pools = {c: dj.pooled(tokens, c) for c in dj.TOKEN_CONDITIONS}
    J_tokens = {c: dj.jump(dj.D_per_layer(pools[c]), LAYERS) for c in dj.TOKEN_CONDITIONS}
    J_coords = {h: dj.jump(dj.D_without_coordinates(pools["all"], h), LAYERS) for h in ("top", "random")}
    norms = dj.median_norms(tokens)
    scale = float(norms[LAYERS.index(dj.HIGH)] / norms[LAYERS.index(dj.LOW)])
    return J_tokens["all"], dj.read(J_tokens["all"], J_tokens, J_coords, scale)


def test_a_planted_massive_token_is_read_as_tokens_only():
    J, v = readings(planted("token"))
    assert J > 5
    assert v["top 1 token"] and v["top 3 tokens"] and not v["5 coordinates"] and not v["overall scale"]


def test_planted_coordinates_are_read_as_coordinates_only():
    J, v = readings(planted("coordinate"))
    assert J > 5
    assert v["5 coordinates"] and not v["top 1 token"] and not v["top 3 tokens"] and not v["overall scale"]


def test_planted_scale_is_read_as_scale_only():
    J, v = readings(planted("scale"))
    assert J > 5
    assert v["overall scale"] and not v["top 1 token"] and not v["5 coordinates"]


def test_no_cause_reads_as_nothing_held_when_the_control_also_falls():
    v = dj.read(13.0, {"top1": 2.0, "top3": 2.0, "random": 5.0}, {"top": 2.0, "random": 5.0}, 1.0)
    assert not any(v.values())


def test_at_least_one_token_always_stays():
    T = np.ones((1, 8))
    for condition in dj.TOKEN_CONDITIONS:
        assert len(dj.dropped(T, condition, 3)) == 0


def test_the_sink_checks_read_as_written():
    from research.sink_check import read_corpus, read_texts

    assert read_texts([[0], [1]], [False, True])
    assert not read_texts([[1], [1]], [False, True])        # the sink where it should not be
    assert not read_texts([[0, 3]], [False])                # two left out
    assert read_corpus([1] * 99 + [0])[1]
    assert not read_corpus([1] * 98 + [0, 2])[1]


def test_moving_the_origin_along_the_normal_is_what_moves_a_mirror():
    from research.origin_candidates import mirror_shifts

    means = np.array([[0.0, 0.0], [1.0, 0.0]])              # one normal, along x
    assert np.allclose(mirror_shifts(np.array([3.0, 0.0]), means), [6.0])
    assert np.allclose(mirror_shifts(np.array([0.0, 3.0]), means), [0.0])


def test_the_largest_step_is_found():
    D = np.array([5.0, 6.0, 60.0])
    step = dj.largest_step(D, LAYERS)
    assert (step["from"], step["to"]) == (6, 8) and step["share_of_log_J"] > 0.9
