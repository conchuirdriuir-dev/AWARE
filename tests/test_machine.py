"""The `hardware` and `read_end` entries: recorded by the code, and invisible to every reader of episodes."""
import subprocess

from body.machine import REPO, cpu_quota, describe, record_machine, threads
from body.stepm1 import episodes
from substrate.record import Record, read


def test_describe_names_the_commit_and_the_software():
    d = describe()
    assert {"gpu", "python", "cuda", "packages", "commit", "uncommitted_changes", "thread_settings", "cpus_shown",
            "cpus_allowed", "cpu_quota"} <= set(d)
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


def test_cpu_quota_reads_both_cgroup_versions(tmp_path):
    v2 = tmp_path / "cpu.max"
    v2.write_text("512000 100000\n")
    assert cpu_quota((str(v2),)) == 5.12
    v2.write_text("max 100000\n")
    assert cpu_quota((str(v2),)) is None
    v1 = tmp_path / "cpu.cfs_quota_us"
    v1.write_text("512000\n")
    (tmp_path / "cpu.cfs_period_us").write_text("100000\n")
    assert cpu_quota((str(tmp_path / "missing"), str(v1))) == 5.12
    v1.write_text("-1\n")
    assert cpu_quota((str(v1),)) is None


def test_oversubscription_warns_only_when_threads_are_unset(tmp_path, monkeypatch):
    q = tmp_path / "cpu.max"
    q.write_text("512000 100000\n")
    monkeypatch.setattr("os.cpu_count", lambda: 32)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(32)))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    t = threads((str(q),))
    assert t["cpus_shown"] == 32 and t["cpu_quota"] == 5.12 and "OMP_NUM_THREADS=1" in t["warning"]
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert "warning" not in threads((str(q),))
    monkeypatch.delenv("OMP_NUM_THREADS")
    q.write_text("max 100000\n")
    assert "warning" not in threads((str(q),))          # no quota and all 32 allowed: nothing to warn about
