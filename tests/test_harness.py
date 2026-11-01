"""The harness, arm A (PLAN.md, step 6, first part), on the tiny model: plumbing, never a result. The step's own test
runs on Qwen3-0.6B (`pytest -m model`)."""
import pytest

torch = pytest.importorskip("torch")

from product.harness import Session, call_tool, tool_calls, verify  # noqa: E402
from substrate.record import read  # noqa: E402

from tests.test_states import tiny  # noqa: E402,F401  (the fixture)


def test_a_scripted_session_replays_exactly_and_its_cache_is_faithful(tiny, tmp_path):  # noqa: F811
    model, tok = tiny
    s = Session(model, tok, tmp_path / "run", max_new_tokens=8, layers=[0, 2, 3])
    s.person("plan a garden")
    s.person("write the lesson")
    rep = verify(tmp_path / "run", model, tok)
    assert rep["problems"] == []
    assert rep["provenance ok"] and rep["replay exact"]
    assert rep["passes"] == 2 and rep["chunks"] == 5                       # system, two person turns, two replies
    entries = read(tmp_path / "run" / "record.jsonl")
    order = [e["kind"] for e in entries]
    assert order.index("pass") > [i for i, e in enumerate(entries) if e.get("role") == "person"][0]
    assert rep["states checked"] > 0 and rep["cache faithful"], rep


def test_tools_stay_in_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    assert call_tool(ws, "write_file", {"path": "notes/a.txt", "content": "hi"}).startswith("wrote 2")
    assert call_tool(ws, "read_file", {"path": "notes/a.txt"}) == "hi"
    assert "outside the workspace" in call_tool(ws, "read_file", {"path": "../secret"})
    assert "outside the workspace" in call_tool(ws, "write_file", {"path": "/etc/x", "content": ""})
    assert call_tool(ws, "rm", {}).startswith("refused")
    assert tool_calls('<tool_call>\n{"name": "read_file", "arguments": {"path": "a"}}\n</tool_call>') == \
        [("read_file", {"path": "a"})]


def test_a_refused_tool_call_is_a_recorded_elicited_result(tiny, tmp_path):  # noqa: F811
    model, tok = tiny
    replies = iter(['<tool_call>{"name": "read_file", "arguments": {"path": "../../etc/passwd"}}</tool_call>', "done"])

    class Canned(Session):
        def _generate(self):
            text = next(replies)
            self._add(text, "model", "self", None, "model")
            return text

    s = Canned(model, tok, tmp_path / "run")
    s.person("read it")
    tool = [e for e in read(tmp_path / "run" / "record.jsonl") if e.get("role") == "tool"]
    assert len(tool) == 1 and "outside the workspace" in tool[0]["text"]
    assert tool[0]["provenance"] == "external" and tool[0]["cause"] == "elicited" and tool[0]["by"] == "code"


@pytest.mark.model
def test_the_harness_on_qwen3_0_6b(tmp_path):
    from product.harness import load

    model, tok = load("Qwen/Qwen3-0.6B")
    s = Session(model, tok, tmp_path / "run", max_new_tokens=200)
    s.person("Please create a file called plan.md with three short steps for planting a small herb garden.")
    s.person("List the files in the workspace.")
    rep = verify(tmp_path / "run", model, tok)
    assert rep["problems"] == [] and rep["provenance ok"] and rep["replay exact"] and rep["cache faithful"], rep
