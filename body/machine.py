"""What a run ran on: the GPU, the software and the code's commit, appended to the run's record by the code itself.

Step -1's 8B run recorded neither its GPU nor its commit (`commit: null`; PLAN.md, Step -1's "Runs"), and the
rented-GPU decision (DECISIONS.md, 2026-11-06) needs every run to record its GPU model. Until now that was a manual
step in the owner's runbook; `play` and `read` now append a `hardware` entry each time they start, so a resume on
another machine is recorded too. The entry keeps the runbook's field names (gpu, python, cuda, packages) and adds the
commit, and the CPU threads: the maths library's thread settings, the CPUs the machine shows, and the CPUs its
quota allows. On a rented instance that showed 32 CPUs under a quota of about 5, shape's fit ran 400 times slower on
the default threads than on one (2026-11-07, PLAN.md); when the threads are unset and the machine shows more than twice
its quota, the entry carries a warning and it is printed. Nothing reads it but people and scoring's reports; anything
that cannot be found is recorded as None.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ("torch", "transformers", "textworld", "nnsight", "nnterp", "numpy")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
QUOTA_FILES = ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=REPO)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _gpu() -> str | None:
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if smi:
        return smi
    try:
        import torch

        if torch.cuda.is_available():
            return ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    except ImportError:
        pass
    return None


def cpu_quota(files=QUOTA_FILES) -> float | None:
    """The CPUs the container's quota allows (cgroup v2 `cpu.max`, or v1's quota over its period), or None if none."""
    for f in files:
        try:
            text = open(f).read().split()
        except OSError:
            continue
        if f.endswith("cpu.max"):                        # "quota period", or "max period" for no quota
            if text and text[0] != "max":
                return int(text[0]) / int(text[1])
            return None
        quota = int(text[0])
        if quota <= 0:                                   # -1: no quota
            return None
        try:
            period = int(open(os.path.join(os.path.dirname(f), "cpu.cfs_period_us")).read().split()[0])
        except (OSError, IndexError, ValueError):
            period = 100_000
        return quota / period
    return None


def threads(quota_files=QUOTA_FILES) -> dict:
    """The thread settings, the CPUs shown and usable, and a warning when the defaults would oversubscribe."""
    shown = os.cpu_count()
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = shown
    quota = cpu_quota(quota_files)
    settings = {v: os.environ.get(v) for v in THREAD_VARS}
    out = {"thread_settings": settings, "cpus_shown": shown, "cpus_allowed": allowed, "cpu_quota": quota}
    usable = min(x for x in (allowed, quota) if x is not None) if (allowed or quota) else None
    if not any(settings.values()) and shown and usable and shown > 2 * usable:
        out["warning"] = (f"the machine shows {shown} CPUs but allows about {usable:g}, and no thread count is set: "
                          f"the maths library may oversubscribe; set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                          f"MKL_NUM_THREADS=1 before running")
    return out


def describe() -> dict:
    """The machine, software and code this process runs on."""
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for p in PACKAGES:
        try:
            packages[p] = version(p)
        except PackageNotFoundError:
            packages[p] = None
    try:
        import torch

        cuda = torch.version.cuda
    except ImportError:
        cuda = None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {"gpu": _gpu(), "python": platform.python_version(), "cuda": cuda, "packages": packages,
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "uncommitted_changes": None if status is None else bool(status), **threads()}


def record_machine(record, command: str) -> dict:
    """Append what this command runs on to the run's record (a `hardware` entry, by the code)."""
    entry = record.append("hardware", "code", command=command, **describe())
    if entry.get("warning"):
        print("warning:", entry["warning"], file=sys.stderr)
    return entry
