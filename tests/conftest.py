"""Shared test setup.

nnsight (under `substrate.states.Reader`) mounts a `save` method on Python's base `object` the first time a trace
saves a value ("pymount"), for the rest of the process. TextWorld's emulator (jericho) then looks up its C function
`save` on its ctypes library and gets nnsight's method instead (`'method' object has no attribute 'argtypes'`), so any
TextWorld test that runs after a reader test in the same process fails. Found 2026-11-07 when
`tests/test_rate_calibration.py` (whose tiny-model test builds a Reader) ran before `tests/test_habitat.py`. After each
test the mount is undone; nnsight mounts it again, lazily, the next time a trace saves.
"""
import sys

import pytest


@pytest.fixture(autouse=True)
def _undo_nnsight_pymount():
    yield
    g = sys.modules.get("nnsight.intervention.tracing.globals")
    if g is not None and getattr(g, "_mounted", False):
        from nnsight._c.py_mount import unmount

        unmount("save")
        g._mounted = False
