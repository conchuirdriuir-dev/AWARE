import numpy as np
import pytest

from substrate import geometry as g
from substrate.axes import Code
from substrate.record import Record, read


def noise(rng, n, d, D):
    return rng.normal(scale=D / np.sqrt(d), size=(n, d))


def test_reflection_about_centroid_is_its_own_inverse_and_fixes_the_mirror():
    rng = np.random.default_rng(0)
    c, u, X = rng.normal(size=8), rng.normal(size=8), rng.normal(size=(5, 8))
    assert np.allclose(g.reflect(g.reflect(X, c, u), c, u), X)
    on_mirror = c + rng.normal(size=8)
    on_mirror -= ((on_mirror - c) @ g.unit(u)) * g.unit(u)
    assert np.allclose(g.reflect(on_mirror[None], c, u)[0], on_mirror)


def test_spread_recovers_the_planted_noise():
    rng = np.random.default_rng(0)
    sets = [rng.normal(size=256) + noise(rng, 50, 256, 0.3) for _ in range(20)]
    assert abs(g.spread(sets) - 0.3) < 0.03
    assert g.between(sets) > 10 * g.spread(sets)


def test_alignment_is_one_for_the_same_space_and_zero_for_orthogonal_ones():
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(16, 4)))
    Q = q.T
    assert np.isclose(g.alignment(Q[:2], np.array([[0, 1], [1, 0]]) @ Q[:2]), 1.0)
    assert np.isclose(g.alignment(Q[:2], Q[2:]), 0.0)


def test_a_real_direction_is_admitted_and_a_noise_direction_is_retained_as_structural(tmp_path):
    rng = np.random.default_rng(0)
    d, D = 64, 1.0
    c = rng.normal(size=d)
    u = g.unit(rng.normal(size=d))
    held_out = c + 5 * rng.normal(size=(20, 1)) * u + noise(rng, 20, d, D)
    code = Code(c, D, "self")
    real = code.consider(u, held_out, "self", 0)
    junk = code.consider(rng.normal(size=d), held_out, "self", 1)
    assert real.kind == "predictive" and real.gain > 1
    assert junk.kind == "structural" and junk.tags == ["untested"]
    assert len(code.axes) == 2 and len(code.predictive()) == 1      # retained, but out of residual
    code.save(tmp_path / "self")
    again = Code.load(tmp_path / "self")
    assert [a.kind for a in again.axes] == ["predictive", "structural"]
    assert np.allclose(again.predictive(), code.predictive())


def test_chunks_carry_their_provenance_and_sources_are_checked(tmp_path):
    r = Record(tmp_path / "run" / "record.jsonl")
    r.chunk("the task", "person", "external")
    r.chunk("a reply", "model", "self")
    Record(tmp_path / "run" / "record.jsonl").append("states", "code", layers=[4])
    rows = read(tmp_path / "run" / "record.jsonl")
    assert [(e["seq"], e["by"], e.get("provenance")) for e in rows] == [
        (0, "person", "external"), (1, "model", "self"), (2, "code", None)]
    with pytest.raises(ValueError):
        r.append("text", "reviewer", text="a")
    with pytest.raises(ValueError):
        r.chunk("x", "model", "mine")
