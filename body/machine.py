"""What a run ran on: the GPU, the software and the code's commit, appended to the run's record by the code itself.

Step -1's 8B run recorded neither its GPU nor its commit (`commit: null`; PLAN.md, Step -1's "Runs"), and the
rented-GPU decision (DECISIONS.md, 2026-11-06) needs every run to record its GPU model. Until now that was a manual
step in the owner's runbook; `play` and `read` now append a `hardware` entry each time they start, so a resume on
another machine is recorded too. The entry keeps the runbook's field names (gpu, python, cuda, packages) and adds the
commit. Nothing reads it but people and scoring's reports; anything that cannot be found is recorded as None.
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

PACKAGES = ("torch", "transformers", "textworld", "nnsight", "nnterp", "numpy")
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
            "uncommitted_changes": None if status is None else bool(status)}


def record_machine(record, command: str) -> dict:
    """Append what this command runs on to the run's record (a `hardware` entry, by the code)."""
    return record.append("hardware", "code", command=command, **describe())
