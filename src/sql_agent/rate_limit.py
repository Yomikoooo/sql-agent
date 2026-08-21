import threading
import time


class TokenBucket:
    """Token bucket: `capacity` bounds a burst, `rate` bounds the long-run
    average. The two knobs are independent."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = rate if capacity is None else capacity
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()


    def acquire(self, n: float = 1.0) -> float:
        """Take n tokens, blocking until they are available. Returns how long it
        waited, so that "is the limiter doing anything" is measurable."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                # Refilled from elapsed time rather than by a timer: no background thread.
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= n:
                    self._tokens -= n
                    return waited
                sleep_for = (n - self._tokens) / self.rate
            # Sleep outside the lock. Holding it here would stop every other thread
            # from even refilling, turning the limiter into a queue.
            time.sleep(sleep_for)
            waited += sleep_for