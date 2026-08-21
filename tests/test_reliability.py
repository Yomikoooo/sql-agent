"""State machine tests for CircuitBreaker.

`reset_after` is tiny (0.05s) throughout.  What is under test is the transition
logic, not the ability to wait 30 seconds; a slow test buys no information.

The assertions read `b.state`, not `b._state`, because the external behaviour
is the contract.  Note that `state` is only a snapshot and performs no
transition -- transitions happen inside `allow()`.
"""
import time

from sql_agent import reliability
from sql_agent.reliability import CircuitBreaker


def test_opens_after_consecutive_failures():
    """Consecutive failures reaching fail_max open the breaker and stop letting calls through.

    This is the test that catches a missing assignment to `_opened_at`: the
    state does flip to open, but the cooldown is computed from nothing and
    `allow()` lets the next call straight through anyway.
    """
    b = CircuitBreaker("t", fail_max=3, reset_after=0.05)   # set up the situation

    b.record_failure()                      
    b.record_failure()

    assert b.allow() is True
    assert b.state == "closed"

    b.record_failure()

    assert b.allow() is False
    assert b.state == "open"


def test_cooldown_lets_exactly_one_probe_through():
    """When the cooldown ends exactly one probe goes through, not the full traffic."""
    b = CircuitBreaker("t", fail_max=1, reset_after=0.05)
    b.record_failure()                      # fail_max=1, so one failure opens it

    assert b.allow() is False
    time.sleep(0.06)

    assert b.allow() is True
    assert b.state == "half_open"

    assert b.allow() is False


def test_probe_success_closes_the_breaker():
    """A successful probe closes the breaker again and calls flow from then on."""
    b = CircuitBreaker("t", fail_max=1, reset_after=0.05)
    b.record_failure()
    time.sleep(0.06)
    assert b.allow()                        # this one is the probe; the state is half_open

    b.record_success()
    assert b.state == "closed"

    assert b.allow() == True
    assert b.allow() == True


def test_probe_failure_reopens_and_restarts_cooldown():
    """A failed probe goes straight back to open, and the cooldown starts over.

    Starting over is the point: if record_failure does not refresh _opened_at,
    the previous cooldown still counts, so a failing probe is followed
    immediately by another call and the breaker protects nothing.
    """
    b = CircuitBreaker("t", fail_max=1, reset_after=0.05)
    b.record_failure()
    time.sleep(0.06)
    assert b.allow()                        # enters half_open

    b.record_failure()
    assert b.state == "open"
    assert b.allow() == False
    time.sleep(0.06)
    assert b.allow() == True


# ---------------------------------------------------------------------------
# The four tests above cover the transitions when somebody reports the outcome
# to the breaker correctly.  The ones below cover the path where nobody reports
# anything at all, which is the gap the bug lives in.
#
# Why the four above cannot catch it: they call record_failure and
# record_success by hand.  The real caller is `guarded_call`, which only calls
# record_failure when `is_upstream_failure(e)` is true.  When a probe hits a
# non-upstream exception -- a 400, or anything raised by our own code -- neither
# is called, and half_open waits for an answer that never comes.
# So this test has to go through `guarded_call` rather than drive the breaker
# directly.
# ---------------------------------------------------------------------------
from sql_agent import llm
import pytest
from sql_agent.reliability import BreakerOpen, guarded_call


def _raise(exc):
    """Wrap an exception in the zero-argument callable that guarded_call expects."""
    def fn():
        raise exc
    return fn


def test_a_non_upstream_probe_must_not_wedge_the_breaker():
    """After a probe hits a 4xx the breaker must keep evolving instead of wedging forever.

    This happened for real: a wrong model name sent to a hosted endpoint
    answered 400 on every task, all of them errors.  Had the breaker been in
    half_open at that moment, the service would have stopped issuing requests
    for good -- with a perfectly healthy upstream, and with no error message
    pointing anywhere near the breaker.

    Both halves of the criterion are needed:
      1. a 400 still does not count as an upstream failure -- fixing this must
         not overturn the rule that 4xx is the caller's own fault, or one
         mistyped field would condemn the whole upstream
      2. the probe must still be consumed, so that after another cooldown a
         further probe can be issued
    """
    b = CircuitBreaker("t", fail_max=1, reset_after=0.05)

    # set up the situation: one upstream failure opens the breaker
    with pytest.raises(llm.MalformedResponseError):
        guarded_call(_raise(llm.MalformedResponseError("upstream down")), breaker=b)
    assert b.state == "open"

    # cooldown over, so the next call is the probe -- let it hit a 400
    time.sleep(0.06)
    with pytest.raises(llm.APIError):
        guarded_call(_raise(llm.APIError(400, "bad model name")), breaker=b)

    # 1. the rule still stands
    assert reliability.is_upstream_failure(llm.APIError(400, "x")) is False

    # 2. the probe was consumed: the state is open and cooling down, not stuck in half_open
    assert b.state == "open", (
        f"the probe was not consumed, the breaker sits at {b.state} -- "
        "half_open issues one probe, and this probe never gets an outcome")

    # one more cooldown, and since the upstream is fine it has to recover
    time.sleep(0.06)
    assert guarded_call(lambda: {"ok": 1}, breaker=b) == {"ok": 1}
    assert b.state == "closed"


def test_the_wedge_is_visible_from_the_caller_side():
    """The same bug seen from the caller: the upstream is healthy, yet every call is blocked.

    This is written separately because the test above asserts on internal
    state, while the symptom in production is that the service goes silent.
    Both angles are needed -- with only the state assertion, renaming a state
    would leave the symptom unguarded.
    """
    b = CircuitBreaker("t", fail_max=1, reset_after=0.05)
    with pytest.raises(llm.MalformedResponseError):
        guarded_call(_raise(llm.MalformedResponseError("upstream down")), breaker=b)
    time.sleep(0.06)
    with pytest.raises(llm.APIError):
        guarded_call(_raise(llm.APIError(400, "bad model name")), breaker=b)

    time.sleep(0.06)                       # long past the cooldown
    try:
        guarded_call(lambda: {"ok": 1}, breaker=b)
    except BreakerOpen:
        pytest.fail("the upstream is healthy and the cooldown has passed, yet "
                    "BreakerOpen blocked the call -- the breaker never recovers")
