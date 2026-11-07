"""The `hardware` and `read_end` entries: recorded by the code, and invisible to every reader of episodes."""
import subprocess

from body.machine import REPO, describe, record_machine
from body.stepm1 import episodes
from substrate.record import Record, read


def test_describe_names_the_commit_and_the_software():
    d = describe()
    assert set(d) == {"gpu", "python", "cuda", "packages", "commit", "uncommitted_changes"}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    assert d["commit"] == head
    assert d["packages"]["numpy"]


def test_new_entries_leave_episodes_unchanged(tmp_path):
    r = Record(tmp_path / "record.jsonl")
    r.append("episode", "code", seed=1)
    r.chunk("opening", "code", "external", cause="spontaneous", turn=-1, slot="world")
    record_machine(r, "stepm1 play")                     # a resume can land here, after a cut-off episode
    r.chunk("go east", "code", "self", turn=0, slot="self")
    r.append("end", "code", seed=1, turns=1)
    before = episodes([e for e in read(r.path) if e["kind"] not in ("hardware", "read_end")])
    r.append("read_end", "code", episodes=1, read=1, already_read=0, seconds=0.1, chunks=2)
    entries = read(r.path)
    assert episodes(entries) == before
    hw = [e for e in entries if e["kind"] == "hardware"]
    assert hw and hw[0]["by"] == "code" and hw[0]["command"] == "stepm1 play"
