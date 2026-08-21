"""The agent loop.

A bounded `for` with three exits, returning an AgentResult. The SQL-specific
parts are the system prompt, the tool set, and recording which query the agent
declared as its answer -- the verifier scores that one.
"""
import json
import logging
import os
import time
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from .context import truncate_tool_result
from .llm import LLMError, call_with_tools
from .reliability import BreakerOpen, CircuitBreaker, guarded_call
from .tools import (HIGH_IMPACT_TOOLS, TOOL_FUNCTIONS, TOOLS_SCHEMA, ToolResult,
                   get_schema)
from .tracing import span

logger = logging.getLogger(__name__)


# Tried and reverted: rewriting "How to work" as a numbered 5-step procedure
# that forced run_sql before submit_answer. Exploration went exactly as
# intended (run_sql 2/31 -> 29/31), but the submission protocol collapsed
# (submit_answer 30/31 -> 5/31), which pushed scoring back onto the unreliable
# last_sql fallback. Adding "This step is required" only recovered it to 9/31.
# The score itself barely moved (17 -> 18 -> 17); measurement quality clearly
# got worse.
#
# That negative result says the *workflow wording* did not help. It does not
# say prompting is exhausted -- few-shot examples, targeted rules, tool
# descriptions and error-message wording were all still untried at that point.
# (An earlier draft concluded "engineering is near its ceiling" from this; that
# was over-generalisation from one experiment.)
#
# Three rules below are kept for domain correctness, not because the eval
# proved them:
#   - GROUP BY id, not name: the database really does contain 59 duplicate
#     names, so grouping by name is simply wrong. A placebo control put its
#     net +2 inside the perturbation noise, so 31 tasks cannot resolve its
#     effect size either way.
#   - "Quantities live on order_items" and the strftime line: a placebo of the
#     same length carrying no information scored 2 higher on the 56 tasks the
#     two arms shared. The statements are true, so they stay; the eval cannot
#     attribute anything to them.
SYSTEM_PROMPT = """You are a data analyst working with a SQLite database.

## Your job
Answer the user's question by querying the database and reporting what you found.

## How to work
- The database schema is given below. Use it. Do not call get_schema.
- Use run_sql to explore, to check your assumptions, and to fix mistakes.
- When you have the one query whose result answers the question, call submit_answer
  with it. That is how you tell us which query is your answer.
- Checking your work after submitting is encouraged. Keep using run_sql for that.
- If the query errors, read the error, fix the query, and try again.
- If a query returns 0 rows, say so explicitly. That may be the correct answer.

## When you should NOT submit an answer
Two cases. In both, reply in text and do not call submit_answer at all.
- The question is ambiguous and different readings give different answers
  (e.g. "top products" — by revenue, by units, or by price?). Ask the user which
  one they mean. Do not pick one silently.
- The database cannot answer the question, because the data simply is not there
  (e.g. profit needs cost, and there is no cost column). Say plainly what is
  missing. Do not submit a placeholder or a near-miss query instead.

If the question IS clear and the data IS there, answer it. Do not ask for
confirmation of things you can look up yourself.

## Rules
- Always get your numbers from the database. Never estimate or recall them.
- order_items.unit_price is the price paid at order time; products.price is the
  current price. Use unit_price for historical revenue, unless the question
  explicitly asks about current list prices.
- When you group rows that belong to a person or a product, GROUP BY their id
  column, not their name. Different rows can share the same name.
- Return exactly the columns the question asks for, and no others. Do not include
  id, helper, or intermediate columns, even if you used them to group rows or to
  break ties in ORDER BY. A column used only in GROUP BY or ORDER BY does not
  belong in SELECT.
- Quantities live on order_items. To count units sold, join orders to order_items
  and sum order_items.quantity.
- This is SQLite. For date parts use strftime: strftime('%Y', o.created_at) for the
  year, strftime('%Y-%m', o.created_at) for the month.
- Report what the data says. Do not add praise, judgement, or business advice.
- Answer in at most 3 sentences. State the number first.
"""
SYSTEM_PROMPT += f"""
## The database
{get_schema().text}
"""


SYSTEM_PROMPT += os.environ.get("SQLAGENT_PROMPT_SUFFIX", "")

# Where degradation falls back to. An environment variable rather than a file
# edit, so competing versions can run in isolated subprocesses without a race.
# Unset means no degradation: the request fails as it would have. "No fallback
# configured" and "degraded successfully" must look different in the logs --
# swallowing that difference is the most expensive mistake available here.
FALLBACK_MODEL = os.environ.get("SQLAGENT_FALLBACK_MODEL") or None

# Which tools this run may execute, derived from the schema the model is shown
# rather than written out a second time. "What the model can see" and "what the
# executor will run" therefore cannot drift: adding a tool to TOOLS_SCHEMA
# allows it, leaving it out denies it.
#
# The previous arrangement -- export_csv is absent from TOOLS_SCHEMA, so the
# model will not call it -- relies on the model, which is not an authorisation
# boundary. A hallucinated call, an injected one, or another caller reusing
# _execute all get past it.
ALLOWED_TOOLS = frozenset(t["function"]["name"] for t in TOOLS_SCHEMA)


class AgentResult(BaseModel):
    text: str
    stop_reason: str                 # "done" | "max_turns" | "error"
    turns: int
    tool_calls: int
    submitted_sql: str | None = None  # what the agent declared as its answer;
                                      # this is what gets scored
    last_sql: str | None = None       # last query that ran successfully. Only a
                                      # fallback when submitted_sql is missing,
                                      # plus diagnostics. Unreliable on its own
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_hit_tokens: int = 0
    elapsed: float = 0.0
    messages: list = []              # full trajectory; failure classification reads it
    degraded: bool = False           # this answer came from the fallback model.
                                     # Set-only: if any turn degraded it stays
                                     # True, even if the primary recovered later
    truncated_turns: int = 0         # turns cut short by max_tokens. A truncated
                                     # turn may hold a partial query, so this is
                                     # measurement contamination and belongs in
                                     # the metrics, not only in a log line

def run_agent(question: str, *, model: str | None = None,
              temperature: float = 0.0,
              max_turns: int = 8, verbose: bool = False,
              prompt_suffix: str = "",
              on_event: Callable[[dict], None] | None = None,
              should_stop: Callable[[], bool] | None = None,
              breaker: CircuitBreaker | None = None) -> AgentResult:
    """Run the agent once.

    `on_event`, `should_stop` and `breaker` are optional hooks. **With all
    three left at None this function behaves byte-for-byte as it did before
    they existed**, which every historical experiment depends on.

    - `on_event(event_dict)`: called at the start of each turn and around every
      tool call. Streaming exists so the user can see *that a query is
      running*, not for a typewriter effect, so tool events have to be
      emitted -- text deltas alone would be useless here.
    - `should_stop()`: asked once per turn. When the user closes the page the
      backend should stop spending tokens. Checked only between turns: a model
      call cannot be interrupted cleanly, and forcing it leaves the message
      list incomplete.
    - `breaker`: enables the circuit breaker and degradation. The None default
      is deliberate, not laziness -- `guarded_call` accumulates failures across
      calls, so enabling it during an eval lets one API hiccup block dozens of
      later tasks wholesale, silently moving the baseline of every historical
      number. Hence runner.py passes nothing and only server.py, where a real
      user is waiting, passes a breaker.
    """
    def emit(kind: str, **fields) -> None:
        if on_event is not None:
            on_event({"type": kind, **fields})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + prompt_suffix},
        {"role": "user", "content": question},
    ]

    total_in = total_out = total_cache = tool_call_count = 0
    truncated = 0
    # Locals, deliberately not module state: runner drives five tasks
    # concurrently, and shared mutable state would overwrite silently.
    submitted_sql: str | None = None
    last_sql: str | None = None
    degraded = False
    t0 = time.perf_counter()

    def _result(text: str, stop_reason: str, turns: int) -> AgentResult:
        return AgentResult(
            text=text, stop_reason=stop_reason, turns=turns,
            tool_calls=tool_call_count,
            submitted_sql=submitted_sql, last_sql=last_sql,
            total_input_tokens=total_in, total_output_tokens=total_out,
            total_cache_hit_tokens=total_cache,
            elapsed=time.perf_counter() - t0,
            messages=messages, truncated_turns=truncated,
            degraded=degraded,
        )

    def _call(m: str | None) -> dict:
        """One model call, factored out so the primary and fallback paths
        differ by exactly one argument."""
        return call_with_tools(messages, TOOLS_SCHEMA, model=m,
                               temperature=temperature)

    for turn in range(max_turns):
        # Exit 3: the caller asked to stop. Turn boundaries only, see docstring.
        if should_stop is not None and should_stop():
            logger.info("stopped by caller at turn %d", turn + 1)
            return _result("Stopped by caller.", "cancelled", turn)

        emit("turn", turn=turn + 1)

        with span("llm.call", turn=turn + 1) as sp:
            try:
                if breaker is None:
                    # default path: identical to before the breaker existed
                    data = _call(model)
                else:
                    data = guarded_call(lambda: _call(model), breaker=breaker)

            except BreakerOpen as e:
                # The primary is already considered down; nothing was sent.
                if FALLBACK_MODEL is None:
                    # Fail as-is. "No fallback configured" and "degraded
                    # successfully" have to stay distinguishable in the logs.
                    logger.error("breaker open, no fallback configured")
                    sp.attrs["error"] = "BreakerOpen"
                    return _result(f"LLM unavailable: {e}", "error", turn + 1)

                logger.warning("breaker open, degrading to %s", FALLBACK_MODEL)
                try:
                    # The fallback is deliberately *not* wrapped in
                    # guarded_call: sharing one breaker would let the primary's
                    # failures condemn the fallback too, killing the
                    # degradation path exactly when it is needed.
                    data = _call(FALLBACK_MODEL)
                except (httpx.HTTPError, LLMError) as e2:
                    logger.exception("fallback model failed at turn %d", turn + 1)
                    sp.attrs["error"] = type(e2).__name__
                    return _result(f"fallback model failed: {e2}", "error", turn + 1)

                # Set-only. Degrading on turn 3 and recovering on turn 4 still
                # makes this answer a degraded one.
                degraded = True
                sp.attrs["degraded"] = True

            except (httpx.HTTPError, LLMError) as e:
                # guarded_call re-raises unchanged, so this still catches it
                logger.exception("llm call failed at turn %d", turn + 1)
                sp.attrs["error"] = type(e).__name__          # the error path needs a span too
                return _result(f"LLM call failed: {e}", "error", turn + 1)

            usage = data.get("usage") or {}
            sp.attrs.update(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0),
                finish_reason=data["choices"][0].get("finish_reason"),   # which turn was cut off
            )

        total_in += usage.get("prompt_tokens", 0)
        total_out += usage.get("completion_tokens", 0)
        total_cache += usage.get("prompt_cache_hit_tokens", 0)

        choice = data["choices"][0]
        message = choice.get("message") or {"role": "assistant", "content": ""}
        _normalize_tool_calls(message)      # fill in fields weak models omit
        messages.append(message)

        if choice.get("finish_reason") == "length":
            truncated += 1
            logger.warning("response truncated at turn %d", turn + 1)

        if verbose:
            print(f"\n--- turn {turn + 1} ---")
            if message.get("content"):
                print(f"[thought] {message['content'][:200]}")

        tool_calls = message.get("tool_calls")
        if not tool_calls:                                    # Exit 1: finished normally
            return _result(message.get("content") or "", "done", turn + 1)

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            tool_call_count += 1

            # Emitted *before* execution: the point is showing that a query is
            # running, which is worthless once it has already finished.
            emit("tool_call", name=name, args=raw_args[:400])

            with span(f"tool.{name}") as sp:
                result = _execute(name, raw_args, allowed=ALLOWED_TOOLS)
                sp.attrs.update(ok=result.ok, raw_len=len(result.text))

            emit("tool_result", name=name, ok=result.ok,
                 preview=result.text[:300])

            if name in ("run_sql", "submit_answer") and result.ok:
                try:
                    query = json.loads(raw_args).get("query")
                except json.JSONDecodeError:
                    query = None
                if query:
                    last_sql = query                    # fallback + diagnostics
                    if name == "submit_answer":
                        submitted_sql = query           # the one that gets scored

            if verbose:
                print(f"[tool] {name}({raw_args[:100]})")
                print(f"       -> {result.text[:120]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": (kept := truncate_tool_result(result.text)),
            })
            sp.attrs["kept_len"] = len(kept)        # record length before and after truncation

    logger.warning("hit max_turns (%d)", max_turns)           # Exit 2: budget exhausted
    return _result(f"Aborted: reached max_turns limit ({max_turns}).",
                   "max_turns", max_turns)


def _normalize_tool_calls(message: dict) -> None:
    """Fill in missing fields in `tool_calls`, in place.

    Hosted APIs return every field; local inference servers and small models
    often do not. A missing `id` crashed 5 of 31 runs outright.

    In place rather than returning a copy, because this message is appended to
    `messages` and sent back to the API, and next turn's tool message has to
    reference it by tool_call_id. Patching a local copy leaves the two sides
    disagreeing and the API returns 400: the id has to be added to the data
    that actually goes back.
    """
    for i, tool_call in enumerate(message.get("tool_calls") or []):
        tool_call.setdefault("type", "function")
        if not tool_call.get("id"):
            tool_call["id"] = f"call_{i}"
        fn = tool_call.setdefault("function", {})
        fn.setdefault("name", "")
        if fn.get("arguments") is None:
            fn["arguments"] = "{}"


def _execute(name: str, raw_args: str, *,
             allowed: frozenset[str],
             approved: frozenset[str] = frozenset()) -> ToolResult:
    """Execute one tool call. Never raises. Returns a ToolResult rather than a
    string so callers know whether it succeeded without guessing from the
    first word of an error message.

    **This is the only gate for tool execution, and it denies by default.**

    `allowed` is the set of tool names this run may execute, and has no
    default on purpose. A default of "None means everything" would let a new
    call site silently restore the old permissive behaviour, which is the
    thing being fixed; forgetting the argument now raises TypeError instead.

    `approved` is the set of tool names a human has approved. Tools in
    HIGH_IMPACT_TOOLS need both. It is a set rather than a bool because one
    batch can contain several high-impact calls, and "the human approved this
    batch" is not the same claim as "the human approved this action".

    A refusal does not list the full registry: dumping TOOL_FUNCTIONS would
    hand the model the names of tools it was not offered.
    """
    if name not in allowed:
        return ToolResult(
            False, f"Error: tool {name!r} is not available in this run. "
                   f"Available: {sorted(allowed)}")

    if name in HIGH_IMPACT_TOOLS and name not in approved:
        # Allowed, but nobody approved this call. Unreachable in the normal
        # human-in-the-loop flow, which asks first. This is the backstop for
        # when that flow is bypassed, forgotten, or _execute is called directly.
        return ToolResult(
            False, f"Error: {name!r} requires human approval and none was granted. "
                   f"Nothing was performed.")

    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        # allowed and the registry disagree: a configuration bug, not the model's
        return ToolResult(False, f"Error: tool {name!r} is allowed but not registered.")

    try:
        args = json.loads(raw_args) if raw_args.strip() else {}
    except json.JSONDecodeError as e:
        return ToolResult(
            False, f"Error: arguments are not valid JSON ({e}). Received: {raw_args}")

    try:
        return func(**args)
    except TypeError as e:
        return ToolResult(False, f"Error: bad arguments for {name} ({e}).")
    except Exception as e:
        logger.exception("tool %s failed", name)
        return ToolResult(False, f"Error: {name} failed ({type(e).__name__}: {e}).")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    for q in [
        "How many users do we have?",
        "Which city has the most users?",
        "Which cities generate the most revenue? Only count completed orders.",
    ]:
        print("=" * 70)
        print(f"Q: {q}")
        r = run_agent(q, verbose=True)
        print(f"\nA: {r.text}")
        print(f"   sql        {r.last_sql}")
        print(f"   {r.stop_reason} | {r.turns} turns | {r.tool_calls} tool calls | "
              f"{r.total_input_tokens:,} in / {r.total_output_tokens:,} out | "
              f"cache {r.total_cache_hit_tokens:,} | {r.elapsed:.1f}s")
