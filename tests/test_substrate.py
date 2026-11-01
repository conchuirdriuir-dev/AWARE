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
    r.chunk("the task", "person", "external", cause="spontaneous")
    r.chunk("a reply", "model", "self")
    Record(tmp_path / "run" / "record.jsonl").append("states", "code", layers=[4])
    rows = read(tmp_path / "run" / "record.jsonl")
    assert [(e["seq"], e["by"], e.get("provenance")) for e in rows] == [
        (0, "person", "external"), (1, "model", "self"), (2, "code", None)]
    with pytest.raises(ValueError):
        r.append("text", "reviewer", text="a")
    with pytest.raises(ValueError):
        r.chunk("x", "model", "mine")


def test_an_external_chunk_needs_its_arrival_cause(tmp_path):
    r = Record(tmp_path / "r.jsonl")
    with pytest.raises(ValueError):
        r.chunk("tool output", "code", "external")
    with pytest.raises(ValueError):
        r.chunk("tool output", "code", "external", cause="mine")
    entry = r.chunk("tests passed", "code", "external", cause="elicited")
    assert entry["cause"] == "elicited" and entry["provenance"] == "external"


def test_a_logged_session_becomes_chunks_with_provenance_and_cause(tmp_path):
    import io
    import json
    import tarfile

    from substrate.corpus import load_session

    def event(role, at, uuid, content):
        return {"created_at": at, role: {"uuid": uuid, "internal_anthropic_catchall": {"message": {"content": content}}}}
    page = {"ccr": {"data": [
        event("assistant", "2026-01-01T00:00:02", "a1", [{"type": "thinking", "thinking": "hidden"},
                                                          {"type": "text", "text": "I will look."},
                                                          {"type": "tool_use", "name": "Read", "input": {"path": "x"}}]),
        event("user", "2026-01-01T00:00:01", "u1", "Please check x."),
        event("user", "2026-01-01T00:00:03", "u2", [{"type": "tool_result", "content": [{"type": "text", "text": "x = 1"}]}]),
    ]}}
    raw = ('<envelope untrusted="true"> ' + json.dumps(page) + " </envelope>").encode()
    archive = tmp_path / "S99.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("S99/S99_p01.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    chunks = load_session(archive)
    assert [(c["provenance"], c["cause"]) for c in chunks] == [
        ("external", "spontaneous"), ("self", None), ("self", None), ("external", "elicited")]
    assert chunks[0]["text"] == "Please check x." and chunks[2]["text"].startswith("[tool call Read]")
    assert chunks[3]["text"] == "x = 1" and chunks[1]["ref"] == ["S99", "a1", 1]       # the thinking block is left out


def test_chunks_are_classed_as_exact_near_repeat_or_new_by_text():
    from substrate.echo_floor import classify

    base = "the quick brown fox jumps over the lazy dog near the river bank today"
    texts = [base, base, base + " again", "an entirely different sentence about budgets and gardens and lessons"]
    assert [k for k, _ in classify(texts)] == ["new", "exact repeat", "near repeat", "new"]


def test_tool_calls_and_their_results_share_a_call_id(tmp_path):
    import io
    import json
    import tarfile

    from substrate.corpus import load_session

    def event(role, at, uuid, content):
        return {"created_at": at, role: {"uuid": uuid, "internal_anthropic_catchall": {"message": {"content": content}}}}
    page = {"ccr": {"data": [
        event("assistant", "2026-01-01T00:00:01", "a1", [{"type": "tool_use", "id": "t1", "name": "Read",
                                                          "input": {"path": "x", "offset": 10}}]),
        event("user", "2026-01-01T00:00:02", "u1", [{"type": "tool_result", "tool_use_id": "t1", "content": "x = 1"}]),
    ]}}
    raw = json.dumps(page).encode()
    archive = tmp_path / "S98.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("S98/S98_p01.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    call, result = load_session(archive)
    assert call["tool"] == "Read" and call["call"] == "t1" and json.loads(call["args"]) == {"offset": 10, "path": "x"}
    assert result["call"] == "t1" and result["cause"] == "elicited"


def test_a_corpus_path_without_archives_stops_before_any_model_work(tmp_path):
    from substrate.echo_floor import echo_floor_alone, echo_floor_retries

    for fn in (echo_floor_alone, echo_floor_retries):
        with pytest.raises(SystemExit, match="no session archives"):
            fn(None, [], [0], Record(tmp_path / "r.jsonl"), str(tmp_path))      # reader None: it must never be reached


def test_a_session_checkpoint_is_reused_on_resume_and_a_partial_write_is_ignored(tmp_path):
    from substrate.echo_floor import session_resumable

    record = Record(tmp_path / "run" / "record.jsonl")
    calls = []

    def compute():
        calls.append(1)
        return {"dist": np.arange(6.0).reshape(3, 2)}, [{"ref": ["S1", "u", 0]}]
    first = session_resumable(record, "alone-S1", compute)
    again = session_resumable(record, "alone-S1", compute)
    assert len(calls) == 1 and np.array_equal(first[0]["dist"], again[0]["dist"]) and again[1] == first[1]
    assert [e["kind"] for e in read(record.path)] == ["resumed"]
    (tmp_path / "run" / "checkpoints" / "alone-S2.npz.tmp.npz").write_bytes(b"half")     # a kill mid-write
    session_resumable(record, "alone-S2", compute)
    assert len(calls) == 2
