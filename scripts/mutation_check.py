"""Mutation testing: show that the tests in tests/test_reliability.py bite.

An all-green suite is not evidence that the tests are useful; they may simply
assert nothing that matters.  So the bugs are put back one at a time to see
which test goes red.

reliability.py is left alone -- each bug is faked by subclassing and overriding
the method.

Run it with:
    python scripts/mutation_check.py

How to read the output:
    every bug is caught by at least one test, or nothing guards that bug
    a test that catches something no other test catches is not redundant
"""
import time

from sql_agent.paths import ROOT  # noqa: F401

from sql_agent import llm
from sql_agent.reliability import CircuitBreaker, guarded_call


class ForgotOpenedAt(CircuitBreaker):
    """Bug 1, which really happened: going from closed to open without recording _opened_at."""

    def record_failure(self):
        with self._lock:
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
            else:
                self._failures += 1
                if self._failures >= self.fail_max:
                    self._state = "open"          # the timestamp line is missing


class NoCooldownRestart(CircuitBreaker):
    """Bug 2: a failed probe returns to open, but the cooldown does not restart."""

    def record_failure(self):
        with self._lock:
            if self._state == "half_open":
                self._state = "open"              # _opened_at is never refreshed
            else:
                self._failures += 1
                if self._failures >= self.fail_max:
                    self._state = "open"
                    self._opened_at = time.monotonic()


class NoHalfOpen(CircuitBreaker):
    """Bug 3: only open and closed, so the full traffic returns at once when the cooldown ends (a thundering herd)."""

    def allow(self):
        with self._lock:
            if self._state == "closed":
                return True
            return time.monotonic() - self._opened_at >= self.reset_after


class ForgotToConsumeTheProbe(CircuitBreaker):
    """Bug 4, also real: nobody tells the breaker when a probe hits a non-upstream error.

    half_open issues exactly one probe, and the only ways out of half_open are
    record_success and record_failure.  `guarded_call` calls record_failure only
    for upstream failures, so a 400 calls neither -- the breaker waits forever
    for an answer, and every later call is blocked by BreakerOpen while the
    upstream is perfectly healthy.
    """

    def record_indeterminate(self):
        pass                                  # the behaviour as it was before the fix


# -- five tests, with the assertions copied from tests/test_reliability.py --
# Copied rather than imported: the assertions have to run against a different
# class each time, and the test functions over there name CircuitBreaker
# directly.  The price of the copy is drift, so a change there needs a change
# here.

def t1(C):
    b = C("t", fail_max=3, reset_after=0.05)
    b.record_failure()
    b.record_failure()
    assert b.allow() is True
    assert b.state == "closed"
    b.record_failure()
    assert b.allow() is False
    assert b.state == "open"


def t2(C):
    b = C("t", fail_max=1, reset_after=0.05)
    b.record_failure()
    assert b.allow() is False
    time.sleep(0.06)
    assert b.allow() is True
    assert b.state == "half_open"
    assert b.allow() is False


def t3(C):
    b = C("t", fail_max=1, reset_after=0.05)
    b.record_failure()
    time.sleep(0.06)
    assert b.allow()
    b.record_success()
    assert b.state == "closed"
    assert b.allow() is True
    assert b.allow() is True


def t4(C):
    b = C("t", fail_max=1, reset_after=0.05)
    b.record_failure()
    time.sleep(0.06)
    assert b.allow()
    b.record_failure()
    assert b.state == "open"
    assert b.allow() is False
    time.sleep(0.06)
    assert b.allow() is True


def _raise(exc):
    def fn():
        raise exc
    return fn


def t5(C):
    """Shaped differently from the four above: it has to go through guarded_call.

    Bug 4 is not in the breaker itself, it is in the gap where the caller never
    tells the state machine anything.  t1 to t4 all call record_failure and
    record_success by hand, so they cannot see into that gap.
    """
    b = C("t", fail_max=1, reset_after=0.05)

    try:                                      # one upstream failure opens it
        guarded_call(_raise(llm.MalformedResponseError("down")), breaker=b)
    except llm.MalformedResponseError:
        pass
    assert b.state == "open"

    time.sleep(0.06)                          # cooldown over, so the next call is the probe
    try:                                      # the probe hits a 400, which is not an upstream failure
        guarded_call(_raise(llm.APIError(400, "bad model name")), breaker=b)
    except llm.APIError:
        pass
    assert b.state == "open"                  # the probe must be consumed, not left in half_open

    time.sleep(0.06)                          # the upstream is fine, so it has to recover
    assert guarded_call(lambda: {"ok": 1}, breaker=b) == {"ok": 1}
    assert b.state == "closed"


TESTS = [("t1", t1), ("t2", t2), ("t3", t3), ("t4", t4), ("t5", t5)]
MUTANTS = [
    ("original (all green)", CircuitBreaker),
    ("bug 1 no _opened_at", ForgotOpenedAt),
    ("bug 2 no cooldown restart", NoCooldownRestart),
    ("bug 3 no half_open", NoHalfOpen),
    ("bug 4 probe not consumed", ForgotToConsumeTheProbe),
]


def main() -> int:
    rows = []
    for name, cls in MUTANTS:
        results = []
        for tname, fn in TESTS:
            try:
                fn(cls)
                results.append(f"{tname}:pass")
            except AssertionError:
                results.append(f"{tname}:RED")
        rows.append((name, results))
        print(f"{name:<26} " + "  ".join(results))

    # The expectations are checked here: the original has to pass everything and
    # every mutant has to be caught at least once.  Without this the script
    # could break and still look like it worked.
    ok = all(r.endswith("pass") for r in rows[0][1])
    for name, results in rows[1:]:
        if not any(r.endswith("RED") for r in results):
            print(f"FAIL: no test catches {name} -- nothing guards that bug")
            ok = False
    if not ok:
        print("\nFAILED")
        return 1
    print("\nOK: the original is all green and every bug turns at least one test red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
