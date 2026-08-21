"""Circuit breaker and graceful degradation for the LLM call.

The breaker sits *outside* `llm.call_with_tools`, which already retries with
backoff. Inside the retry loop, one outage would be counted N times.
"""
import threading
import time

from . import llm


class BreakerOpen(Exception):
    """The breaker is open: this call was never sent.

    Kept distinct from "the call failed" on purpose. Merging them hides
    whether the breaker is doing anything at all.
    """


class CircuitBreaker:
    """Three-state breaker.

        closed     --fail_max consecutive failures-->  open
        open       --reset_after seconds elapsed---->  half_open (one probe)
        half_open  --probe succeeds----------------->  closed
        half_open  --probe fails-------------------->  open (cooldown restarts)

    half_open admits exactly one call, so a recovering upstream is not hit by
    the whole backlog at once.

    Every method takes the lock: FastAPI runs sync endpoints in a thread pool,
    so these counters really are mutated concurrently.
    """

    def __init__(self, name: str, *, fail_max: int = 5,
                 reset_after: float = 30.0) -> None:
        # monotonic, never time.time(): a clock adjustment can move time.time()
        # backwards, and the breaker would then wait out its cooldown forever.
        self.name = name
        self.fail_max = fail_max
        self.reset_after = reset_after
        self._failures: int = 0
        self._state: str = "closed"
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Whether to send this call. Mutates state (open -> half_open);
        use `state` for a read-only snapshot."""
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self.reset_after:
                    self._state = "half_open"
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"

    def record_failure(self) -> None:
        """Report an *upstream* failure. 4xx is our own fault and must not
        land here, or one bad field would take the whole upstream down."""
        with self._lock:
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
            else:
                self._failures += 1
                if self._failures >= self.fail_max:
                    self._state = "open"
                    self._opened_at = time.monotonic()

    def record_indeterminate(self) -> None:
        """Report a call that says nothing about the upstream (4xx, or a bug
        of ours). The probe still has to be consumed: half_open is the only
        state that cannot leave on its own, so without this the breaker stays
        there forever and rejects every later call, with a healthy upstream
        and nothing in the logs pointing at the breaker.

        A no-op unless half_open, which keeps the eval path byte-identical.
        """
        with self._lock:
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        """Read-only snapshot for /health and metrics. Deliberately performs
        no transition, so an expired cooldown still reads "open" until
        something calls allow()."""
        with self._lock:
            return self._state


# Process-wide, on purpose: a per-request breaker would never reach fail_max.
LLM_BREAKER = CircuitBreaker("llm", fail_max=5, reset_after=30.0)


def is_upstream_failure(exc: BaseException) -> bool:
    """Whether `exc` means "the upstream is down".

    Reuses llm.py's retry predicate rather than restating it: "worth
    retrying" and "upstream is down" are the same question, and two copies
    would drift silently.

    4xx returns False. Pointing --model at the wrong endpoint once produced
    400s on 76/76 runs; opening the breaker on those would have pointed the
    investigation at the upstream instead of at the flag.
    """
    if isinstance(exc, llm.TRANSIENT_EXC):
        return True
    elif isinstance(exc, llm.APIError) and exc.status_code in llm.TRANSIENT_STATUS:
        return True
    elif isinstance(exc, llm.MalformedResponseError):
        return True
    else:
        return False


def guarded_call(fn, *, breaker: CircuitBreaker = LLM_BREAKER):
    """Run `fn()` behind the breaker and return its result.

    `call_with_tools` retries internally (1s/2s/4s). One whole failed
    `call_with_tools` counts as one failure here, not three.

        data = guarded_call(lambda: llm.call_with_tools(messages, TOOLS_SCHEMA))
    """
    if not breaker.allow():
        raise BreakerOpen
    try:
        result = fn()
    except Exception as e:
        if is_upstream_failure(e):
            breaker.record_failure()
        else:
            breaker.record_indeterminate()
        raise
    else:
        breaker.record_success()
        return result
