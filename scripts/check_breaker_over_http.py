"""Check how retries and the circuit breaker multiply, over a real HTTP path.

Why the unit tests are not enough: tests/test_degradation.py replaces
`call_with_tools` wholesale, so the retry loop inside `llm.py` (backing off
1/2/4 seconds) never takes part.  The failure mode being ruled out is a breaker
and a retry loop nested the wrong way round, where one outage costs N x M
requests.  Proving it does not happen means counting the HTTP requests that
actually reach a server.

How to run it, in two terminals:
    MOCK_FORCE_STATUS=503 python scripts/mock_llm_server.py
    python scripts/check_breaker_over_http.py

Expected, with MAX_RETRIES=3, so 4 HTTP attempts per call:
    1st guarded_call -> 4 hits, one failure recorded on the breaker
    2nd guarded_call -> 8 hits, the breaker opens
    3rd guarded_call -> still 8 hits: BreakerOpen, the request never goes out

The last line is the whole point.  The breaker wraps the retries from the
outside, so once it is open the cost is 0 requests, not 4.

The script takes about 15 seconds, because the backoff really does sleep.
"""
import os
import time

import httpx

from sql_agent.paths import ROOT  # noqa: F401

MOCK = "http://127.0.0.1:8765"
# Set these before importing llm: it reads .env and the environment at import time.
os.environ["DEEPSEEK_API_URL"] = f"{MOCK}/v1/chat/completions"
os.environ["DEEPSEEK_API_KEY"] = "mock"

from sql_agent import llm                       # noqa: E402
from sql_agent import reliability               # noqa: E402


def hits() -> int:
    return httpx.get(f"{MOCK}/__stats", timeout=5).json()["hits"]


def main() -> int:
    try:
        httpx.delete(f"{MOCK}/__stats", timeout=5)
    except httpx.ConnectError:
        print("FAIL: the mock is not running. Start it with: "
              "MOCK_FORCE_STATUS=503 python scripts/mock_llm_server.py")
        return 1

    breaker = reliability.CircuitBreaker("llm", fail_max=2, reset_after=30)
    per_call = llm.MAX_RETRIES + 1          # HTTP attempts one call can cost
    rows = []

    for i in (1, 2, 3):
        t0 = time.monotonic()
        try:
            reliability.guarded_call(
                lambda: llm.call_with_tools([{"role": "user", "content": "q"}], []),
                breaker=breaker)
            outcome = "succeeded, unexpectedly"
        except reliability.BreakerOpen:
            outcome = "BreakerOpen (never sent)"
        except Exception as e:                       # noqa: BLE001
            outcome = type(e).__name__
        rows.append((i, hits(), breaker.state, round(time.monotonic() - t0, 1), outcome))
        print(f"call {i}  hits={rows[-1][1]:<3} breaker={breaker.state:<9} "
              f"took={rows[-1][3]:>4}s  {outcome}")

    print()
    # The expectations are checked here: printing alone would make this a
    # demonstration, not evidence.
    ok = True
    if rows[0][1] != per_call:
        print(f"FAIL: call 1 should cost {per_call} hits "
              f"(1 attempt + {llm.MAX_RETRIES} retries), got {rows[0][1]}")
        ok = False
    if rows[1][1] != per_call * 2:
        print(f"FAIL: the total after call 2 should be {per_call * 2}, got {rows[1][1]}")
        ok = False
    if rows[2][1] != rows[1][1]:
        print(f"FAIL: an open breaker must send nothing, yet the count went "
              f"from {rows[1][1]} to {rows[2][1]} -- the breaker sits inside the retries")
        ok = False
    if rows[2][2] != "open":
        print(f"FAIL: the breaker should be open by call 3, it is {rows[2][2]}")
        ok = False

    print("OK: the breaker wraps the retries -- one outage is recorded once, "
          "and an open breaker sends nothing" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
