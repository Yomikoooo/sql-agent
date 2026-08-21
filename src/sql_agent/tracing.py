"""Hand-rolled tracing: one run becomes a tree of timed spans, not one log line.

Written by hand rather than pulled from a library so that the cost of each
part is visible; the four questions that would justify a hosted tracer
(cross-process? shared with others? real-time? high volume?) are all "no" here.

    with trace(task_id="sql_009", model="qwen2.5:3b") as t:
        with span("retrieve", retriever="vector") as s:
            hits = search_vector(q, k=2)
            s.attrs["retrieved_ids"] = [h["id"] for h in hits]
        with span("llm.call", turn=1) as s:
            data = call_with_tools(...)
            s.attrs["prompt_tokens"] = data["usage"]["prompt_tokens"]

    t.to_dict()      # goes into runner's jsonl -- add fields, never change one
    print(t.render())
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass, field

# ContextVar, not a module-level list and not threading.local(): runner drives
# five tasks concurrently, and a shared list silently merges their spans into
# one tree. ContextVar covers threads and async tasks alike.
# None means "not inside a trace" -- span() is then a no-op.
_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "trace_current_span", default=None)


@dataclass
class Span:
    """An event with a duration. The duration is the whole point: intervals
    can be summed and attributed, log lines cannot."""

    name: str
    attrs: dict = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    ms: float = 0.0

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "ms": self.ms}
        if self.attrs:
            d["attrs"] = self.attrs
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @property
    def self_ms(self) -> float:
        """Time spent in this layer alone: total minus the children."""
        return round(self.ms - sum(c.ms for c in self.children), 1)

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        head = f"{pad}{self.name:<22} {self.ms:>8.1f}ms"
        if self.attrs:
            kv = " ".join(f"{k}={v}" for k, v in self.attrs.items())
            head += f"   {{{kv[:100]}}}"
        return "\n".join([head] + [c.render(indent + 1) for c in self.children])


@contextlib.contextmanager
def span(name: str, **attrs):
    """Open a span. A no-op outside a trace, so instrumenting a function does
    not force its callers to start one.

    Closing happens in `finally`: the error path is exactly the one whose
    timing you need most.
    """
    parent = _current.get()
    if parent is None:
        yield Span(name, attrs)
        return

    rec = Span(name, attrs)
    parent.children.append(rec)
    token = _current.set(rec)
    t0 = time.perf_counter()          # perf_counter, not time.time: no clock jumps
    try:
        yield rec                     # the caller may keep adding to rec.attrs
    finally:
        rec.ms = round((time.perf_counter() - t0) * 1000, 1)
        _current.reset(token)


@contextlib.contextmanager
def trace(name: str = "agent.run", **attrs):
    """Open a root span.

    Sets and resets the ContextVar explicitly rather than relying on "new
    thread, new context": thread pools reuse threads.
    """
    root = Span(name, attrs)
    token = _current.set(root)
    t0 = time.perf_counter()
    try:
        yield root
    finally:
        root.ms = round((time.perf_counter() - t0) * 1000, 1)
        _current.reset(token)


def current_trace() -> Span | None:
    """The span we are currently inside (not the root)."""
    return _current.get()


def rollup(root: Span) -> dict[str, float]:
    """Total time per span name, for attribution.

    Uses self_ms, not ms: with ms the parents double-count their children and
    the percentages add up to more than 100.
    """
    out: dict[str, float] = {}

    def walk(s: Span) -> None:
        out[s.name] = round(out.get(s.name, 0.0) + s.self_ms, 1)
        for c in s.children:
            walk(c)

    walk(root)
    return out
