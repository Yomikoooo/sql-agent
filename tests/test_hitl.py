"""The fast layer of human-in-the-loop tests.

`scripts/check_hitl_server_restart.py` proves the slow half -- that a paused
run survives across processes -- and it needs a subprocess and several seconds.
This file holds the logic instead: what has to pause, what must not, and what
happens after a rejection.

Same reason as `test_degradation.py`: these branches never run in normal use.
They execute only at the moment an approval is actually triggered, which is
exactly when nobody is watching.
"""
import json

import pytest

from sql_agent import agent_graph
from sql_agent import run_store
from sql_agent import server
from sql_agent import tools


def _msg(tool_calls: list[dict] | None = None, content: str = "") -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        return {"choices": [{"message": message, "finish_reason": "tool_calls"}],
                "usage": {}}
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {}}


def _call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


EXPORT_ARGS = {"query": "SELECT COUNT(*) AS n FROM users", "filename": "t.csv"}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the checkpoint database, the run store and the export directory at a temp dir.

    Without the isolation the tests write into the real
    `state/checkpoints.sqlite`, and starting the server afterwards shows a pile
    of runs awaiting approval that the test suite left behind.
    """
    monkeypatch.setattr(agent_graph, "STATE_DIR", tmp_path)
    monkeypatch.setattr(run_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(tools, "EXPORT_DIR", tmp_path / "exports")
    return tmp_path


@pytest.fixture
def fake_model(monkeypatch):
    """Install a fake model and hand back its call count."""
    def install(script):
        calls = {"n": 0}

        def fake(messages, tools_schema, *, model=None, temperature=0.0, **kw):
            calls["n"] += 1
            return script(messages, tools_schema, calls["n"])
        monkeypatch.setattr(agent_graph, "call_with_tools", fake)
        return calls
    return install


def _export_then_answer(messages, tools_schema, n):
    """Ask for the export on the first turn, answer afterwards."""
    if not any(m.get("role") == "tool" for m in messages):
        return _msg([_call("export_csv", EXPORT_ARGS)])
    return _msg(content="finished")


CFG = dict(model=None, temperature=0.0, max_turns=8, enable_export=True)


def test_high_impact_tool_pauses_and_does_not_run(isolated, fake_model):
    """A high-impact action pauses first, and nothing has happened yet at the moment it pauses."""
    fake_model(_export_then_answer)

    with agent_graph.open_checkpointer(isolated / "cp.sqlite") as saver:
        out = agent_graph.start_run("q", thread_id="t1", saver=saver, **CFG)

    assert out["status"] == "awaiting_approval"
    assert out["pending"][0]["tools"][0]["name"] == "export_csv"
    # The point of the test: the file cannot exist while the run is paused.
    # Acting first and asking afterwards is the same as not asking.
    assert not (isolated / "exports" / "t.csv").exists()


def test_approve_executes_and_finishes(isolated, fake_model):
    fake_model(_export_then_answer)

    with agent_graph.open_checkpointer(isolated / "cp.sqlite") as saver:
        agent_graph.start_run("q", thread_id="t2", saver=saver, **CFG)
        out = agent_graph.resume_run(thread_id="t2", saver=saver, approved=True, **CFG)

    assert out["status"] == "done"
    assert out["result"].stop_reason == "done"
    assert (isolated / "exports" / "t.csv").exists()


def test_reject_is_not_a_failure(isolated, fake_model):
    """A rejection is not a failure: the action does not run, but the run finishes normally.

    Turning it into an error would throw away the whole context the moment a
    person declines, losing the work done up to that point.
    """
    fake_model(_export_then_answer)

    with agent_graph.open_checkpointer(isolated / "cp.sqlite") as saver:
        agent_graph.start_run("q", thread_id="t3", saver=saver, **CFG)
        out = agent_graph.resume_run(thread_id="t3", saver=saver, approved=False, **CFG)

    assert out["status"] == "done"
    assert out["result"].stop_reason == "done"          # not "error"
    assert not (isolated / "exports" / "t.csv").exists()
    # what reaches the model is an ordinary tool result saying it was declined
    tool_msgs = [m for m in out["result"].messages if m.get("role") == "tool"]
    assert any("declined" in str(m.get("content", "")).lower() for m in tool_msgs)


def test_interrupt_does_not_double_run_the_readonly_tool(isolated, fake_model, monkeypatch):
    """An interrupt replays the whole node, so a read-only tool in the same batch must not run twice.

    That is why the question is asked before any tool in the batch runs, rather
    than when the high-impact one comes up.  The bug only appears when one
    batch mixes a read-only tool with a high-impact one, and it raises nothing.
    """
    ran = {"n": 0}
    real_run_sql = tools.TOOL_FUNCTIONS["run_sql"]

    def counting_run_sql(query):
        ran["n"] += 1
        return real_run_sql(query)

    monkeypatch.setitem(tools.TOOL_FUNCTIONS, "run_sql", counting_run_sql)

    def both_in_one_batch(messages, tools_schema, n):
        if not any(m.get("role") == "tool" for m in messages):
            return _msg([_call("run_sql", {"query": "SELECT 1"}, "a"),
                         _call("export_csv", EXPORT_ARGS, "b")])
        return _msg(content="finished")

    fake_model(both_in_one_batch)

    with agent_graph.open_checkpointer(isolated / "cp.sqlite") as saver:
        agent_graph.start_run("q", thread_id="t4", saver=saver, **CFG)
        assert ran["n"] == 0          # not approved yet, so no tool has run
        agent_graph.resume_run(thread_id="t4", saver=saver, approved=True, **CFG)

    assert ran["n"] == 1              # asking per tool instead of per batch makes this 2


def test_export_tool_is_hidden_by_default():
    """export_csv stays hidden by default, otherwise the baseline behind every recorded eval number moves."""
    default_names = {t["function"]["name"] for t in tools.TOOLS_SCHEMA}
    all_names = {t["function"]["name"] for t in tools.ALL_TOOLS_SCHEMA}

    assert "export_csv" not in default_names
    assert "export_csv" in all_names
    assert "export_csv" in tools.TOOL_FUNCTIONS     # registered, just not exposed by default


@pytest.mark.parametrize("filename", ["../evil.csv", "/tmp/evil.csv", "evil.txt"])
def test_export_rejects_bad_filenames(isolated, filename):
    """The path check belongs to the action itself.

    The approval step cannot be relied on to catch this: a person sees "export
    to evil.csv" and has no reason to think it points outside the project.
    """
    result = tools.export_csv("SELECT 1", filename)
    assert not result.ok
    assert "bare *.csv" in result.text


def test_http_approve_flow_and_edges(isolated, fake_model):
    """The HTTP layer: approving works, approving twice is 409, an unknown run is 404."""
    from fastapi.testclient import TestClient

    fake_model(_export_then_answer)
    client = TestClient(server.app)

    started = client.post("/runs", json={"question": "export it"}).json()
    assert started["status"] == "awaiting_approval"
    run_id = started["run_id"]

    # after a restart this endpoint is how the pending runs are found again
    pending = client.get("/runs/pending").json()["pending"]
    assert any(p["run_id"] == run_id for p in pending)

    done = client.post(f"/runs/{run_id}/approve").json()
    assert done["status"] == "done"

    # clicking approve twice must not run the whole thing a second time
    assert client.post(f"/runs/{run_id}/approve").status_code == 409
    assert client.post("/runs/deadbeef/approve").status_code == 404
