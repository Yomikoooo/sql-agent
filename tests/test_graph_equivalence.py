"""Differential test: `agent.run_agent` and `agent_graph.run_agent` must agree
field by field under the same fake model.

Comparing eval scores between the two implementations is too coarse -- token
counts can come out low, the trajectory can be reformatted, submitted_sql can
go missing, and the score still does not move.  What is compared here is every
field of AgentResult plus every message.

The first run of this test caught three bugs, none of which raised anything:
  1. add_messages converted the dicts into LangChain objects, changing the
     shape of tool_calls
  2. the counter fields had no reducer, so they were overwritten instead of
     accumulated and total_input_tokens came out 40 instead of 60
  3. the turn check sat on the model->tools edge, so the tool call of the last
     turn never ran and submitted_sql was lost

A fake model (offline, free, milliseconds) is what makes swapping the
implementation a thing that can have a regression test at all.
"""
import importlib

import httpx
import pytest

from sql_agent import agent
from sql_agent import agent_graph

FIELDS = ("text", "stop_reason", "turns", "tool_calls", "submitted_sql", "last_sql",
          "total_input_tokens", "total_output_tokens", "total_cache_hit_tokens",
          "truncated_turns")

TOOL_CALL = [{"id": "c0", "type": "function", "function": {
    "name": "run_sql", "arguments": '{"query":"SELECT COUNT(*) FROM users"}'}}]
SUBMIT_CALL = [{"id": "c0", "type": "function", "function": {
    "name": "submit_answer", "arguments": '{"query":"SELECT 1","answer":"a"}'}}]

SCRIPTS = {
    # exit 1: no tool wanted on the very first turn
    "done": [
        {"choices": [{"message": {"role": "assistant", "content": "answer"},
                      "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ],
    # one tool call then a wrap-up -- this is the case that catches
    # overwriting instead of accumulating the counters
    "tool_then_done": [
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": TOOL_CALL},
                      "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 5}},
        {"choices": [{"message": {"role": "assistant", "content": "200 users"},
                      "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 40, "completion_tokens": 4}},
    ],
    # exit 2: keeps asking for tools until the turn limit. The tool call of
    # the last turn still has to run -- submit_answer is the one in here
    "max_turns": [
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": SUBMIT_CALL},
                      "finish_reason": "length"}],
         "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
    ] * 10,
    # a weak model: missing id, missing type, arguments None, whole message None
    "sloppy": [
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "get_schema", "arguments": None}}]},
            "finish_reason": "tool_calls"}],
         "usage": {}},
        {"choices": [{"message": None}], "usage": {}},
    ],
}


def _fake(script):
    """Return one scripted response per turn, repeating the last one once exhausted."""
    state = {"i": 0}

    def call(messages, tools_schema, model=None, temperature=0.0):
        i = min(state["i"], len(script) - 1)
        state["i"] += 1
        return script[i]
    return call


def _run_both(make_call, max_turns=3):
    agent.call_with_tools = make_call()
    a = agent.run_agent("q", max_turns=max_turns)

    importlib.reload(agent_graph)          # the closure captured the reference from import time
    agent_graph.call_with_tools = make_call()
    b = agent_graph.run_agent("q", max_turns=max_turns)
    return a, b


def _assert_same(a, b):
    for f in FIELDS:
        assert getattr(a, f) == getattr(b, f), \
            f"{f}: agent={getattr(a, f)!r} agent_graph={getattr(b, f)!r}"
    assert len(a.messages) == len(b.messages), \
        f"message count {len(a.messages)} vs {len(b.messages)}"
    for i, (x, y) in enumerate(zip(a.messages, b.messages)):
        assert x == y, f"messages[{i}]:\n  agent={x!r}\n  graph={y!r}"


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_equivalent(name, monkeypatch):
    script = SCRIPTS[name]
    a, b = _run_both(lambda: _fake(script))
    _assert_same(a, b)


def test_equivalent_on_network_error():
    """Exit 3: when the LLM call raises, both sides land on stop_reason='error' rather than crashing."""
    def boom(*_a, **_k):
        raise httpx.ConnectError("connection refused")

    a, b = _run_both(lambda: boom)
    _assert_same(a, b)
    assert a.stop_reason == "error"
