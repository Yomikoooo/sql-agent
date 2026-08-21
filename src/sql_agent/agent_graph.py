import json
import logging
import operator
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt

from .agent import AgentResult, SYSTEM_PROMPT, _normalize_tool_calls, _execute
from .context import truncate_tool_result
from .llm import LLMError, call_with_tools
from .tools import ALL_TOOLS_SCHEMA, HIGH_IMPACT_TOOLS, TOOLS_SCHEMA
from . import paths


logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[list, operator.add]
    turns: Annotated[int, operator.add]
    total_in: Annotated[int, operator.add]
    total_out: Annotated[int, operator.add]
    total_cache: Annotated[int, operator.add]
    tool_call_count: Annotated[int, operator.add]
    truncated: Annotated[int, operator.add]
    submitted_sql: str | None
    last_sql: str | None
    stop_reason: str | None
    text: str


def build_graph(*, model: str | None = None,
                temperature: float = 0.0,
                max_turns: int = 8,
                verbose: bool = False,
                enable_export: bool = False,
                checkpointer=None):
    """Build the graph and compile it.

    Split out of `run_agent` because an approval arrives in a *later* call --
    possibly in a different process -- and the same graph has to be rebuilt to
    receive it.

    The configuration is not in the checkpoint. `model`, `max_turns` and
    `enable_export` are closure variables; the checkpoint holds only State. A
    caller resuming a run must pass them again, and passing different ones does
    not raise -- the resumed run silently becomes a different run. run_store
    exists to hand them back unchanged.

    With `enable_export=False` the model is shown a byte-identical tool set to
    the one it saw before approvals existed.
    """
    tools_schema = ALL_TOOLS_SCHEMA if enable_export else TOOLS_SCHEMA
    # Derived from the schema handed to the model on *this* call (see the note
    # on agent.ALLOWED_TOOLS). With enable_export=False, export_csv is neither
    # visible nor executable -- two properties, one source.
    allowed_tools = frozenset(t["function"]["name"] for t in tools_schema)

    def call_model(state: State) -> dict:
        """call llm model, return keys to be updated"""
        try:
            data = call_with_tools(state["messages"],
                                   tools_schema,
                                   model=model,
                                   temperature=temperature)
        except (httpx.HTTPError, LLMError) as e:
            logger.exception("llm call failed at turn %d", state["turns"] + 1)
            return {"turns": 1,
                    "stop_reason": "error", "text": f"LLM call failed: {e}"}
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        message = choice.get("message") or {"role": "assistant", "content": ""}
        _normalize_tool_calls(message)
        
        if verbose:
            print(f"\n--- turn {state['turns'] + 1} ---")
            if message.get("content"):
                print(f"[thought] {message['content'][:200]}")

        truncated = 0
        if choice.get("finish_reason") == "length":
            truncated = 1
            logger.warning("response truncated at turn %d", state["turns"] + 1)

        out = {
            "messages": [message],
            "turns": 1,
            "total_in": usage.get("prompt_tokens", 0),
            "total_out": usage.get("completion_tokens", 0),
            "total_cache": usage.get("prompt_cache_hit_tokens", 0),
            "truncated": truncated,
        }
        if not message.get("tool_calls"):
            out["stop_reason"] = "done"
            out["text"] = message.get("content") or ""
        return out


    def run_tools(state: State) -> dict:
        out, n = [], 0
        submitted_sql = state["submitted_sql"]
        last_sql = state["last_sql"]
        calls = state["messages"][-1]["tool_calls"]

        # Ask before executing *any* tool in the batch, not when the gated
        # tool's turn comes up. LangGraph's `interrupt()` re-runs the whole
        # node on resume, so tools that already ran would run a second time --
        # harmless for a read-only run_sql, a double write for anything else.
        # Nothing raises; it only shows up when one batch mixes read-only and
        # high-impact calls.
        #
        # The criterion is the blast radius of the action, not the model's
        # confidence.
        pending = [c for c in calls if c["function"]["name"] in HIGH_IMPACT_TOOLS]
        # Empty means nothing is approved, and it must never default to "all":
        # the default is only ever consulted when a high-impact call exists,
        # which is precisely when it is the safety boundary.
        approved_tools: frozenset[str] = frozenset()
        if pending:
            decision = interrupt({          # first pass: suspends the graph
                "type": "approval_required",
                "tools": [{"name": c["function"]["name"],
                           "arguments": c["function"]["arguments"]} for c in pending],
                "reason": "These actions write files and overwrite anything of "
                          "the same name. They cannot be undone.",
            })                              # on resume: returns the answer
            if bool((decision or {}).get("approved")):
                approved_tools = frozenset(c["function"]["name"] for c in pending)

        for tool_call in calls:
            name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            n += 1

            if name in HIGH_IMPACT_TOOLS and name not in approved_tools:
                # A refusal is not a failure. The action does not run, but
                # "the user declined" goes back as an ordinary tool result so
                # the model can choose what to do next. Turning it into an
                # error would throw away the whole context on every refusal.
                out.append({"role": "tool", "tool_call_id": tool_call["id"],
                            "content": "The user declined this action. It was NOT "
                                       "performed and nothing was written. Continue "
                                       "without it, or explain what you would need."})
                continue

            result = _execute(name, raw_args, allowed=allowed_tools,
                              approved=approved_tools)

            if name in ("run_sql", "submit_answer") and result.ok:
                try:
                    query =json.loads(raw_args).get("query")
                except json.JSONDecodeError:
                    query = None
                if query:
                    last_sql = query
                    if name == "submit_answer":
                        submitted_sql = query

            if verbose:
                print(f"[tool] {name}({raw_args[:100]})")
                print(f".      -> {result.text[:120]}")

            out.append({"role": "tool", "tool_call_id": tool_call["id"],
                        "content": truncate_tool_result(result.text)}) 

        upd = {"messages": out, "tool_call_count": n,
                               "submitted_sql": submitted_sql, "last_sql": last_sql}
        if state["turns"] >= max_turns:
            logger.warning("hit max_turns (%d)", max_turns)
            upd["stop_reason"] = "max_turns"
            upd["text"] = f"Aborted: reached max_turns limit ({max_turns})."
        return upd

    def after_model(state: State) -> str:
        return END if state["stop_reason"] else "tools"


    def after_tools(state: State) -> str:
        return END if state["stop_reason"] else "model"
    


    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model)
    graph.add_conditional_edges("tools", after_tools)
    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str, prompt_suffix: str = "") -> dict:
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT + prompt_suffix},
                     {"role": "user", "content": question}],
        "turns": 0, "total_in": 0, "total_out": 0, "total_cache": 0,
        "tool_call_count": 0, "truncated": 0,
        "submitted_sql": None, "last_sql": None,
        "stop_reason": None, "text": "",
    }


def to_result(final: dict, elapsed: float) -> AgentResult:
    return AgentResult(
        text=final["text"], stop_reason=final["stop_reason"], turns=final["turns"],
        tool_calls=final["tool_call_count"],
        submitted_sql=final["submitted_sql"], last_sql=final["last_sql"],
        total_input_tokens=final["total_in"],
        total_output_tokens=final["total_out"],
        total_cache_hit_tokens=final["total_cache"],
        elapsed=elapsed,
        messages=final["messages"], truncated_turns=final["truncated"],
    )


def run_agent(question: str,
              *,
              model: str | None = None,
              temperature: float = 0.0,
              max_turns: int = 8,
              verbose: bool = False,
              prompt_suffix: str = "") -> AgentResult:
    """Run to completion with no checkpointer, behaving exactly as it did
    before approvals existed.

    `tests/test_graph_equivalence.py` compares this field by field against
    `agent.run_agent`.
    """
    t0 = time.perf_counter()
    app = build_graph(model=model, temperature=temperature,
                      max_turns=max_turns, verbose=verbose)
    final = app.invoke(initial_state(question, prompt_suffix),
                       {"recursion_limit": max_turns * 2 + 10})
    return to_result(final, time.perf_counter() - t0)


# ─────────────────────────────────────────────────────────────
# Human in the loop: pause -> persist -> confirm -> resume
# ─────────────────────────────────────────────────────────────

STATE_DIR = paths.STATE_DIR


@contextmanager
def open_checkpointer(path: Path | str | None = None):
    """Open the on-disk checkpoint database.

    On-disk rather than InMemorySaver, because the property being demonstrated
    is "trigger a confirmation, restart the server process, confirm, continue
    from where it stopped". An in-memory saver loses exactly the thing under
    test.

    `check_same_thread=False` is required: FastAPI runs sync endpoints in a
    thread pool, so the thread that starts a run is usually not the one that
    resumes it, and sqlite raises without it.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    db = str(path or STATE_DIR / "checkpoints.sqlite")
    conn = sqlite3.connect(db, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


def _pending_approval(out: dict) -> list | None:
    """Return the pending actions if the graph stopped for approval, else None.

    An interrupted `invoke` returns normally rather than raising; the interrupt
    is tucked into the `__interrupt__` key. Miss that key and "waiting for a
    human" reads as "finished", with `stop_reason` still None.
    """
    hits = out.get("__interrupt__")
    if not hits:
        return None
    return [h.value for h in hits]


def start_run(question: str, *, thread_id: str, saver,
              prompt_suffix: str = "", **cfg) -> dict:
    """Start a run that can stop for approval.

    Returns `{"status": "done", "result": AgentResult}` or
    `{"status": "awaiting_approval", "pending": [...]}`.
    """
    app = build_graph(checkpointer=saver, **cfg)
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": cfg.get("max_turns", 8) * 2 + 10}
    t0 = time.perf_counter()
    out = app.invoke(initial_state(question, prompt_suffix), config)
    pending = _pending_approval(out)
    if pending:
        return {"status": "awaiting_approval", "pending": pending}
    return {"status": "done", "result": to_result(out, time.perf_counter() - t0)}


def resume_run(*, thread_id: str, saver, approved: bool, **cfg) -> dict:
    """Continue from the breakpoint once a human has decided.

    `cfg` must match the `start_run` call exactly: the configuration is not in
    the checkpoint (see `build_graph`). A mismatch does not raise, it just
    silently turns this into a different run.
    """
    app = build_graph(checkpointer=saver, **cfg)
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": cfg.get("max_turns", 8) * 2 + 10}
    t0 = time.perf_counter()
    out = app.invoke(Command(resume={"approved": approved}), config)
    pending = _pending_approval(out)
    if pending:                      # one run may ask for approval twice
        return {"status": "awaiting_approval", "pending": pending}
    return {"status": "done", "result": to_result(out, time.perf_counter() - t0)}