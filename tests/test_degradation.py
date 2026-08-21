"""Tests for the graceful degradation paths.

Why this file has to exist: degradation code never runs in normal operation --
only at the moment the upstream actually goes down.  Code that does not run is
code nobody notices is broken.  A misspelled `except UpsteamError` sat here
across several rounds without the healthy path ever giving it away.

Everything touched here is module-level state (`agent.call_with_tools`,
`agent.FALLBACK_MODEL`, `reliability.LLM_BREAKER`), so it has to go through
pytest's `monkeypatch` fixture, which restores it after each test.  Assigning
`agent.X = ...` by hand leaks: one test quietly contaminates every test after
it, and the symptom shows up in a different file.
"""
import time

import pytest

from sql_agent import agent
from sql_agent import llm
from sql_agent import reliability
from sql_agent.reliability import CircuitBreaker


def _ok(text: str) -> dict:
    """A normal response that answers without calling a tool."""
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {}}


def _calls_tool(name: str) -> dict:
    """A response asking for a tool, used to push the agent into a second turn.

    The tool name is deliberately one that does not exist: `_execute` returns an
    error result, the loop carries on, and the database is never touched.
    `data/*.db` is gitignored, so a test that depends on it fails on anyone
    else's machine by construction.
    """
    return {"choices": [{"message": {"role": "assistant", "content": "",
                                     "tool_calls": [{"id": "c1", "type": "function",
                                                     "function": {"name": name,
                                                                  "arguments": "{}"}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {}}


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace `agent.call_with_tools` with a fake and record which model each call went to.

    Returns an `install(behaviour)` so each test decides how the primary and
    the fallback model respond.
    """
    calls: list[str | None] = []

    def install(behaviour):
        def fake(messages, tools, *, model=None, temperature=0.0, **kw):
            calls.append(model)
            return behaviour(model, len(calls))
        monkeypatch.setattr(agent, "call_with_tools", fake)
        return calls

    return install


def test_no_breaker_leaves_the_eval_path_untouched(fake_llm):
    """With no breaker passed, the breaker must not be touched at all.

    This guards the baseline behind the recorded eval numbers: `runner.py`
    passes no breaker, so failures during an eval must not accumulate in one.
    Otherwise a single blip from the API blocks the dozens of tasks that follow,
    which silently changes the premise of every earlier conclusion.
    """
    fake_llm(lambda model, n: (_ for _ in ()).throw(llm.APIError(503, "down")))
    spy = CircuitBreaker("spy", fail_max=2, reset_after=30)

    r = agent.run_agent("q", max_turns=1)          # no breaker, on purpose

    assert r.stop_reason == "error"
    assert r.degraded is False
    assert spy.state == "closed"
    assert spy._failures == 0                      # nothing was ever counted


def test_degrades_to_fallback_once_the_breaker_opens(fake_llm, monkeypatch):
    """Once the breaker opens the fallback model answers, and the result is marked degraded."""
    def behaviour(model, n):
        if model == "backup:3b":
            return _ok("answer from backup")
        raise llm.APIError(503, "primary down")

    calls = fake_llm(behaviour)
    monkeypatch.setattr(agent, "FALLBACK_MODEL", "backup:3b")
    b = CircuitBreaker("llm", fail_max=2, reset_after=30)

    for _ in range(2):                             # accumulate until it opens
        r = agent.run_agent("q", breaker=b, max_turns=1)
        assert r.stop_reason == "error"
        assert r.degraded is False                 # failed, not yet degraded
    assert b.state == "open"

    r = agent.run_agent("q", breaker=b, max_turns=1)

    assert r.stop_reason == "done"
    assert r.degraded is True
    assert r.text == "answer from backup"
    # the primary was never contacted -- that is the whole point of the breaker
    assert calls == [None, None, "backup:3b"]


def test_without_a_fallback_it_fails_loudly(fake_llm, monkeypatch):
    """With no fallback configured it fails as it is, rather than pretending nothing happened.

    "No fallback configured" and "degraded successfully" have to stay
    distinguishable.  Merged into one outcome, there is no way to tell "the
    upstream is down" from "I forgot to set the environment variable".
    """
    fake_llm(lambda model, n: (_ for _ in ()).throw(llm.APIError(503, "down")))
    monkeypatch.setattr(agent, "FALLBACK_MODEL", None)
    b = CircuitBreaker("llm", fail_max=1, reset_after=30)

    agent.run_agent("q", breaker=b, max_turns=1)   # opens the breaker
    assert b.state == "open"

    r = agent.run_agent("q", breaker=b, max_turns=1)

    assert r.stop_reason == "error"
    assert r.degraded is False
    assert "unavailable" in r.text.lower()


def test_degraded_flag_survives_a_later_healthy_turn(fake_llm, monkeypatch):
    """`degraded` is set, never cleared.

    Turn one degrades, turn two finds the primary healthy again -- the answer is
    still the product of a degraded run.  Reassigning the flag each turn wipes
    that out, with no error and no symptom.
    """
    def behaviour(model, n):
        if model == "backup:3b":
            # sleep past the cooldown so the next allow() issues a probe,
            # standing in for the upstream coming back
            time.sleep(0.06)
            return _calls_tool("no_such_tool")     # a tool call forces a second turn
        return _ok("primary is back")              # second turn: the primary is healthy

    fake_llm(behaviour)
    monkeypatch.setattr(agent, "FALLBACK_MODEL", "backup:3b")

    # The breaker was opened by an earlier request, not by this one: within a
    # single run_agent the first failure returns an error outright and never
    # reaches the degradation branch.  So the situation is set up in advance.
    b = CircuitBreaker("llm", fail_max=1, reset_after=0.05)
    b.record_failure()
    assert b.state == "open"

    r = agent.run_agent("q", breaker=b, max_turns=3)

    assert r.stop_reason == "done"
    assert r.text == "primary is back"             # the last turn used the primary
    assert r.degraded is True                      # the answer as a whole still counts as degraded
    assert b.state == "closed"                     # the probe succeeded and the breaker recovered


def test_server_falls_back_to_last_sql_and_says_so(fake_llm):
    """Out of budget but holding SQL that ran: the server hands it over and says it is a fallback.

    This can only be done in the server.  Doing it in `agent.py` is not an
    option: the four behaviour tasks in the dev set grade exactly on whether an
    answer was submitted, so filling the field in at the agent layer would be
    forging the thing being measured.
    """
    from fastapi.testclient import TestClient

    from sql_agent import server

    # keeps calling run_sql, never submits, and runs out of budget -- the
    # shape of a real failure
    fake_llm(lambda model, n: {
        "choices": [{"message": {"role": "assistant", "content": "",
                                 "tool_calls": [{"id": "c1", "type": "function",
                                                 "function": {"name": "run_sql",
                                                              "arguments": '{"query": "SELECT 1"}'}}]},
                     "finish_reason": "tool_calls"}],
        "usage": {}})

    body = TestClient(server.app).post(
        "/ask", json={"question": "q", "max_turns": 2}).json()

    assert body["stop_reason"] == "max_turns"       # not rewritten to done: it really did run out of budget
    assert body["sql"] == "SELECT 1"
    assert body["sql_is_fallback"] is True


def test_degraded_reaches_the_http_response(fake_llm, monkeypatch):
    """Degradation has to be visible in the response; invisible is the same as absent.

    `/health` shows the breaker is open while `ok` stays True: an upstream
    outage does not mean this service is broken.  Reporting itself unhealthy
    only invites a restart that fixes nothing and throws away the ability to
    degrade along the way.
    """
    from fastapi.testclient import TestClient

    from sql_agent import server

    def behaviour(model, n):
        if model == "backup:3b":
            return _ok("answer from backup")
        raise llm.APIError(503, "primary down")

    fake_llm(behaviour)
    monkeypatch.setattr(agent, "FALLBACK_MODEL", "backup:3b")
    # Swap out the process-wide breaker.  Left alone, this test leaves it open
    # and contaminates every later test in the same pytest run that uses it.
    monkeypatch.setattr(reliability, "LLM_BREAKER",
                        CircuitBreaker("llm", fail_max=1, reset_after=30))

    client = TestClient(server.app)
    assert client.get("/health").json()["llm_breaker"] == "closed"

    assert client.post("/ask", json={"question": "q", "max_turns": 1}
                       ).json()["degraded"] is False          # failed, but not degraded yet
    body = client.post("/ask", json={"question": "q", "max_turns": 1}).json()

    assert body["degraded"] is True
    assert body["answer"] == "answer from backup"

    assert body["sql_is_fallback"] is False        # no SQL was submitted here, and there is no last_sql

    health = client.get("/health").json()
    assert health["llm_breaker"] == "open"
    assert health["fallback_model"] == "backup:3b"
    assert health["ok"] is True                    # the upstream being down is not this service being unhealthy
