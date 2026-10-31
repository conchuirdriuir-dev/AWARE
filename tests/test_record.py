import pytest

from record import Record, read


def test_entries_append_in_order_with_their_source(tmp_path):
    r = Record(tmp_path / "run" / "record.jsonl")
    r.append("text", "person", text="a")
    r.append("states", "code", layers=[4])
    again = Record(tmp_path / "run" / "record.jsonl")
    again.append("text", "model", text="b")
    rows = read(tmp_path / "run" / "record.jsonl")
    assert [(e["seq"], e["by"]) for e in rows] == [(0, "person"), (1, "code"), (2, "model")]


def test_an_entry_without_a_known_source_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Record(tmp_path / "r.jsonl").append("text", "reviewer", text="a")
