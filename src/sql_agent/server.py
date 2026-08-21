"""The HTTP surface: request/response, SSE streaming, and approvals.

    GET  /health        readiness, for Docker and load balancers
    POST /ask           ask a question, wait, get a structured result
    POST /ask/stream    the same, emitting events as it goes (SSE)
    POST /runs          a run that can stop and ask for approval

Streaming exists so the user can see *that a query is running*, not for a
typewriter effect, which is why tool events are emitted and not just text
deltas: a query takes tens of seconds, and a blank screen for that long reads
as a crash.

`agent.run_agent` blocks -- the underlying HTTP calls are synchronous. Calling
it from an async endpoint would stall the entire event loop and starve every
other request, so /ask/stream hands it to a background thread and drains a
queue.

    uvicorn server:app --reload            # development
    python server.py                       # same, reads HOST and PORT
"""
import asyncio
import json
import logging
import os
import queue
import threading
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import agent
from . import agent_graph
from . import reliability
from . import run_store
from . import tools

logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Agent", version="1.0")

# Streams in flight: run_id -> stop switch.
# Locked, because FastAPI runs sync endpoints in a thread pool, so a cancel and
# the event loop's register/unregister really can overlap -- dict atomicity is
# not enough to make "check, then use" safe.
# Process-local: under uvicorn --workers > 1 a cancel may land on a process
# that has never heard of the run. Multi-process would need shared storage.
_runs: dict[str, threading.Event] = {}
_runs_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    model: str | None = None          # a colon routes to the local server
    max_turns: int = Field(default=8, ge=1, le=20)


def _payload(r: agent.AgentResult) -> dict:
    """Project an AgentResult onto the fields that go out over the wire.

    `messages` is deliberately not exposed: it can run to tens of kilobytes and
    may contain database rows carrying injected instructions. Traces are the
    way to inspect a run.
    """
    return {
        "answer": r.text,
        # Prefer what the agent explicitly submitted, falling back to last_sql
        # only when it never did (having run out of budget, say). The fallback
        # belongs here and in the eval scorer, never in the agent itself: some
        # tasks score exactly whether it submitted, so assigning a value down
        # there would forge the thing being measured.
        "sql": r.submitted_sql or r.last_sql,
        # Like degraded: a fallback answer has to be distinguishable. Unmarked,
        # "the agent submitted this" and "we submitted it for the agent" are
        # the same signal downstream.
        "sql_is_fallback": r.submitted_sql is None and r.last_sql is not None,
        "stop_reason": r.stop_reason,
        "turns": r.turns,
        "tool_calls": r.tool_calls,
        "input_tokens": r.total_input_tokens,
        "output_tokens": r.total_output_tokens,
        "elapsed": round(r.elapsed, 2),
        # Degradation has to be visible in the response. Without this field a
        # week of quietly answering from the fallback model goes unnoticed, and
        # "the model got worse" gets blamed on the prompt.
        "degraded": r.degraded,
    }


@app.get("/health")
def health() -> dict:
    """Readiness.

    Never call the model from here. Health is polled frequently, so every check
    would cost time and money, and one upstream hiccup would mark the whole
    service unhealthy. Check only what this process owns.
    """
    return {
        "ok": True,
        "db": tools.DB_PATH.name,
        "db_exists": tools.DB_PATH.exists(),      # missing -> every query says "no such table"
        "schema_sample_mode": tools.SCHEMA_SAMPLE_MODE,
        # Breaker state is information only and does not feed the ok verdict.
        # Reporting unhealthy because an upstream is down invites Docker or the
        # load balancer to restart or remove this container -- a restart cannot
        # fix someone else's outage and takes the degraded service away too.
        # The question is "can I still work", not "are my dependencies well".
        "llm_breaker": reliability.LLM_BREAKER.state,
        "fallback_model": agent.FALLBACK_MODEL,   # None = fail outright, no degradation
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Ask and wait for the answer.

    Deliberately `def` and not `async def`: FastAPI runs sync endpoints in a
    thread pool, so this blocking call stays off the event loop. Written as
    `async def` it would be wrong.
    """
    # Only the server passes a breaker. The eval runner does not: failures
    # accumulate across calls, so one API hiccup would block dozens of later
    # tasks wholesale and silently move the baseline under every historical
    # number. See agent.run_agent for the full reasoning.
    r = agent.run_agent(req.question, model=req.model, max_turns=req.max_turns,
                        breaker=reliability.LLM_BREAKER)
    return _payload(r)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    """Stream tool events as they happen, then the final state."""

    # Thread-safe queue: the agent puts from a worker thread, the event loop
    # gets from here.
    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    run_id = uuid.uuid4().hex[:12]
    with _runs_lock:
        _runs[run_id] = stop

    def worker() -> None:
        try:
            r = agent.run_agent(
                req.question, model=req.model, max_turns=req.max_turns,
                on_event=events.put,          # one per turn and per tool call
                should_stop=stop.is_set,      # set on disconnect; honoured between turns
                breaker=reliability.LLM_BREAKER,   # the same breaker /ask uses
            )
            events.put({"type": "final", **_payload(r)})
        except Exception as e:                # noqa: BLE001
            # An exception in a worker thread does not propagate to the
            # request context. Uncaught, it vanishes entirely and shows up as
            # "the stream just ended, with no error" -- the worst kind to
            # diagnose.
            logger.exception("agent failed in worker thread")
            events.put({"type": "error", "error": type(e).__name__, "message": str(e)})
        finally:
            events.put(None)                  # sentinel: no more events

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        finished = False
        # run_id goes first, before the agent does anything: a client cannot
        # cancel what it has no handle to.
        yield f"data: {json.dumps({'type': 'run', 'run_id': run_id})}\n\n"
        try:
            while True:
                try:
                    # The blocking get has to be awaited in a thread or it
                    # stalls the event loop again. The 0.5s timeout is what
                    # makes this coroutine cancellable: parked in an untimed
                    # get, it cannot receive a cancellation either.
                    item = await asyncio.to_thread(events.get, True, 0.5)
                except queue.Empty:
                    # An SSE comment line, used as a keep-alive: proxies in
                    # between tend to drop a connection after tens of seconds
                    # with no data, and one turn can take that long.
                    yield ": keepalive\n\n"
                    continue

                if item is None:
                    finished = True
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            # Client disconnect lands here. Verified: about 4 seconds after
            # curl --max-time cuts the connection, the log shows in order
            #    `stream ended early` → `agent: stopped by caller at turn 4`。
            #
            # This path was declared broken three times on false evidence:
            # once because the test command did not exist on the platform and
            # never ran at all, and twice because the expected log lines were
            # absent -- while this module's logger had no handler configured,
            # so those lines could never have appeared whatever happened.
            # "Not in the log" is not "did not happen".
            if not finished:
                stop.set()
                logger.info("stream ended early, asking agent to stop")
            with _runs_lock:
                _runs.pop(run_id, None)      # leaks memory if skipped

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # or nginx buffers the stream into one response
        },
    )


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    """Cancel a run explicitly.

    Both this and the automatic stop are needed, and both are verified:

    - Automatic: the generator gets cancelled and its finally clause runs. It
      costs the user nothing, but it depends on the server noticing the TCP
      disconnect, which varies with the deployment -- reverse proxies, keep
      alive, HTTP/2 all change the behaviour.
    - Explicit: deterministic, and expressible as a rerunnable check. A stop
      button uses this; the automatic path is the safety net.

    Cancellation is not immediate. The flag is only read between turns, since
    cutting a model call mid-flight leaves an incomplete message list, so the
    worst case waits out the turn in progress.
    """
    with _runs_lock:
        ev = _runs.get(run_id)
    if ev is None:
        raise HTTPException(404, f"no active run {run_id!r}")
    ev.set()
    logger.info("run %s cancelled by request", run_id)
    return {"cancelled": run_id}


# ─────────────────────────────────────────────────────────────
# Human in the loop
#
# The runs below are not the same kind of thing as the runs above; they only
# share the /runs prefix. Those live in the in-memory _runs and exist to cancel
# a stream in flight. These live in sqlite and exist to approve a run that has
# stopped for a human. The two id spaces are disjoint, so mixing them up gives
# a 404. Known naming debt, left alone because renaming touches the streaming
# path that is already verified.
# ─────────────────────────────────────────────────────────────


class StartRunRequest(BaseModel):
    question: str = Field(min_length=1)
    model: str | None = None
    max_turns: int = Field(default=8, ge=1, le=20)
    enable_export: bool = True       # the whole point of this endpoint


def _run_payload(run_id: str, outcome: dict) -> dict:
    body = {"run_id": run_id, "status": outcome["status"]}
    if outcome["status"] == "awaiting_approval":
        body["pending"] = outcome["pending"]
    else:
        body.update(_payload(outcome["result"]))
    return body


def _execute_run(run_id: str, question: str, config: dict, *,
                 resume: bool | None = None) -> dict:
    """Start a run, or resume one. Both share the same finishing logic.

    The checkpointer and state database are reopened every time rather than
    held: a resume may well happen in a different process, which is the
    property being demonstrated, so this path has to work with nothing warmed
    up.
    """
    conn = run_store.connect()
    try:
        with agent_graph.open_checkpointer() as saver:
            if resume is None:
                outcome = agent_graph.start_run(
                    question, thread_id=run_id, saver=saver, **config)
            else:
                outcome = agent_graph.resume_run(
                    thread_id=run_id, saver=saver, approved=resume, **config)
        run_store.save(conn, run_id, question, config, outcome["status"],
                       outcome.get("pending"))
        return outcome
    finally:
        conn.close()


@app.post("/runs")
def start_run(req: StartRunRequest) -> dict:
    """Start a run that may stop for approval.

    A high-impact action (`tools.HIGH_IMPACT_TOOLS`) suspends it and returns
    `status="awaiting_approval"` together with the pending actions.
    """
    run_id = uuid.uuid4().hex[:12]
    config = {"model": req.model, "temperature": 0.0,
              "max_turns": req.max_turns, "enable_export": req.enable_export}
    return _run_payload(run_id, _execute_run(run_id, req.question, config))


@app.post("/runs/{run_id}/approve")
def approve_run(run_id: str) -> dict:
    return _decide(run_id, approved=True)


@app.post("/runs/{run_id}/reject")
def reject_run(run_id: str) -> dict:
    """Decline.

    A refusal is not a failure. The action does not run, but the run continues
    to completion: the model receives "the user declined" as a tool result and
    decides what to do next. Made an error instead, one click would throw away
    the whole context.
    """
    return _decide(run_id, approved=False)


def _decide(run_id: str, *, approved: bool) -> dict:
    conn = run_store.connect()
    try:
        record = run_store.load(conn, run_id)
    finally:
        conn.close()

    if record is None:
        raise HTTPException(404, f"unknown run {run_id}")
    if record["status"] != "awaiting_approval":
        # Clicking approve twice must not run it twice
        raise HTTPException(409, f"run {run_id} is {record['status']}, not awaiting approval")

    # The configuration is read back from storage, never reconstructed: see
    # run_store. Resuming with different values does not raise, it just
    # silently makes this a different run.
    outcome = _execute_run(run_id, record["question"], record["config"],
                           resume=approved)
    return _run_payload(run_id, outcome)


@app.get("/runs/pending")
def pending_runs() -> dict:
    """What is waiting on a human. This is how the queue is recovered after a
    restart."""
    conn = run_store.connect()
    try:
        return {"pending": run_store.list_pending(conn)}
    finally:
        conn.close()


@app.get("/runs")
def list_runs() -> dict:
    """Runs currently in flight. For debugging and demos."""
    with _runs_lock:
        return {"active": sorted(_runs)}


def main() -> None:
    import uvicorn

    # Without this line not one logger.info from this module appears: uvicorn
    # configures only its own loggers, and everyone else's default to no
    # handler. That once turned "the lines are not in the log" into "the thing
    # did not happen", when the lines could never have been printed at all.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # 0.0.0.0 rather than 127.0.0.1: bound to loopback inside a container,
    # only the container can connect and everyone outside sees a refusal.
    uvicorn.run(app,
                host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
