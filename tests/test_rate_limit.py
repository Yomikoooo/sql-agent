import time


from sql_agent.rate_limit import TokenBucket


def test_burst_then_throttle():
    bucket = TokenBucket(rate=2, capacity=2)

    t0 = time.monotonic()
    bucket.acquire()
    bucket.acquire()                              # bucket starts full -> both go through at once
    assert time.monotonic() - t0 < 0.05

    bucket.acquire()                              # the third one has to wait ~0.5s
    assert 0.4 < time.monotonic() - t0 < 0.7


def test_returns_waited_time():
    bucket = TokenBucket(rate=10, capacity=1)
    bucket.acquire()
    assert bucket.acquire() >= 0.05               # the second one waits 1/10 s