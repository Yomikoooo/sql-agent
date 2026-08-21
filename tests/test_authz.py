"""Tool authorization: what is allowed and what was approved are two gates.

What this file pins down: "the model cannot see this tool" is not an
authorization boundary.

Background: `export_csv` overwrites a file of the same name and is the only
irreversible action in the project -- the whole human-in-the-loop path exists
for it.  But it also sits in the global `TOOL_FUNCTIONS` registry, and the
executor on the plain `run_agent` path used to consult no allowlist at all, so
any `export_csv` tool call that reached the executor wrote the file.

Where would such a call come from?  A hallucination, a prompt injection (a
payload that landed 5 out of 5 times when measured here), or someone else
reusing `_execute`.  None of the three needs the model to have been shown the
tool's schema.
"""
import json

import pytest

from sql_agent import agent
from sql_agent import agent_graph
from sql_agent import tools

PROBE = "__authz_probe.csv"


@pytest.fixture(autouse=True)
def clean_probe_file():
    path = tools.EXPORT_DIR / PROBE
    path.unlink(missing_ok=True)
    yield path
    path.unlink(missing_ok=True)


def _export_args() -> str:
    return json.dumps({"query": "SELECT 1", "filename": PROBE})


def _calls_export() -> dict:
    """A response asking to call export_csv -- what injection or a hallucination produces."""
    return {"choices": [{"message": {"role": "assistant", "content": "",
                                     "tool_calls": [{"id": "c1", "type": "function",
                                                     "function": {"name": "export_csv",
                                                                  "arguments": _export_args()}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {}}


# -- the gates themselves ------------------------------------------------

def test_execute_refuses_to_run_without_an_allowlist():
    """`allowed` has no default: forgetting it is a TypeError, not a silent pass.

    This guards the fix against being quietly reverted -- if someone later
    gives `allowed` a `= None` default meaning "everything is allowed", this
    test goes red.
    """
    with pytest.raises(TypeError):
        agent._execute("run_sql", '{"query": "SELECT 1"}')      # `allowed` omitted on purpose


def test_a_tool_outside_the_allowlist_is_refused(clean_probe_file):
    r = agent._execute("export_csv", _export_args(),
                       allowed=frozenset({"run_sql", "submit_answer"}))
    assert r.ok is False
    assert not clean_probe_file.exists(), "a refused action still wrote the file"


def test_the_refusal_does_not_leak_hidden_tool_names():
    """The earlier wording handed the whole `TOOL_FUNCTIONS` registry back to the model."""
    r = agent._execute("no_such_tool", "{}", allowed=frozenset({"run_sql"}))
    assert r.ok is False
    assert "export_csv" not in r.text


def test_high_impact_tool_needs_approval_even_when_allowed(clean_probe_file):
    """Being on the allowlist is not permission to run. The two gates are independent."""
    r = agent._execute("export_csv", _export_args(),
                       allowed=frozenset({"export_csv"}))       # allowed, but not approved
    assert r.ok is False
    assert "approval" in r.text.lower()
    assert not clean_probe_file.exists()


def test_approved_high_impact_tool_actually_runs(clean_probe_file):
    """The positive case is required: with only refusal tests, a "refuse everything" implementation passes too."""
    r = agent._execute("export_csv", _export_args(),
                       allowed=frozenset({"export_csv"}),
                       approved=frozenset({"export_csv"}))
    assert r.ok is True
    assert clean_probe_file.exists()


# -- end to end: a forged tool call on the plain agent path ---------------

def test_a_forged_export_call_reaches_the_agent_loop_but_writes_nothing(
        monkeypatch, clean_probe_file):
    """Forge an `export_csv` call on the plain `run_agent` path: the agent
    finishes normally and nothing is written.

    Both halves of the assertion are needed:
      1. the file was not written -- that is the security conclusion
      2. the agent did not crash and the loop continued -- a refusal is an
         ordinary tool result, not an exception.  Raising instead would let a
         single injection tear down the whole context.
    """
    seen = {"n": 0}

    def fake(messages, tools_schema, model=None, temperature=0.0, **kw):
        seen["n"] += 1
        if seen["n"] == 1:
            return _calls_export()
        return {"choices": [{"message": {"role": "assistant", "content": "done"},
                             "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(agent, "call_with_tools", fake)
    r = agent.run_agent("q", max_turns=3)

    assert not clean_probe_file.exists(), "the forged call wrote the file"
    assert r.stop_reason == "done"
    assert r.tool_calls == 1                      # the call happened, it was refused
    refusals = [m for m in r.messages if m.get("role") == "tool"]
    assert len(refusals) == 1 and "not available" in refusals[0]["content"]


# -- graph path: the allowlist follows enable_export ----------------------

def test_the_graph_allowlist_follows_enable_export():
    """The allowlist is derived from the schema shown to the model on this run, not a hand-kept second list."""
    assert "export_csv" not in agent.ALLOWED_TOOLS          # never on the plain path
    names = lambda schema: {t["function"]["name"] for t in schema}
    assert "export_csv" not in names(tools.TOOLS_SCHEMA)
    assert "export_csv" in names(tools.ALL_TOOLS_SCHEMA)
    # build_graph's closure variable is not reachable, so assert instead that
    # the difference between the two schemas is exactly the high-impact tools
    assert names(tools.ALL_TOOLS_SCHEMA) - names(tools.TOOLS_SCHEMA) == tools.HIGH_IMPACT_TOOLS
