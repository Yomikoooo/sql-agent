"""A local server that pretends to be an OpenAI-compatible API and behaves like a weak model.

Why it exists: a small model (0.5B to 1.5B) takes paths a large hosted model
never does -- answering without calling a tool, tool_call arguments that are not
valid JSON, a missing id, a response that is not even shaped like choices.
Those paths are where the bugs live, and they can be tested before spending any
time installing a model or downloading weights.

It follows the standing rule: get the pipeline working locally against a mock,
and spend real resources only on real runs.

Usage:
    python scripts/mock_llm_server.py              # runs in the foreground, Ctrl+C to stop
    # in another terminal:
    DEEPSEEK_API_URL=http://127.0.0.1:8765/v1/chat/completions \\
    DEEPSEEK_API_KEY=mock python runner.py --model mock-weak --trials 1

It is not a simulator and does not try to answer correctly.  Its only purpose is
to feed malformed responses to the agent and see whether the agent falls over.
"""
import json
import os
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8765

# Force an HTTP failure.  Only takes effect when set, so the default behaviour
# is unchanged for the other scripts that use this mock:
#     MOCK_FORCE_STATUS=503 python scripts/mock_llm_server.py
# It exists for scripts/check_breaker_over_http.py, which counts how retries and
# the breaker multiply -- a faked call_with_tools cannot measure that.
FORCE_STATUS = int(os.environ.get("MOCK_FORCE_STATUS", "0")) or None

# How many POSTs have arrived.  GET /__stats reads it, DELETE resets it.
# The lock is not needed today (HTTPServer is single threaded; ThreadingHTTPServer
# is the concurrent one), it is here so switching to threads later does not
# require remembering to come back: a miscounted total raises nothing, it just
# makes the conclusion wrong.
_hits = 0
_hits_lock = threading.Lock()

# Probability of each failure mode.  They add up to less than 1; the remainder
# is a normal response.
FAILURE_RATES = {
    "no_tool_calls":  0.15,   # invents an answer without calling a tool (the most common weak-model failure)
    "bad_json_args":  0.12,   # arguments is not valid JSON
    "missing_id":     0.08,   # the tool_call has no id field
    "empty_choices":  0.05,   # the response has no choices at all
    "unknown_tool":   0.05,   # calls a tool that does not exist
}

rng = random.Random(42)       # fixed seed, so a failure can be reproduced


def _pick_failure() -> str | None:
    roll = rng.random()
    acc = 0.0
    for mode, rate in FAILURE_RATES.items():
        acc += rate
        if roll < acc:
            return mode
    return None


def _tool_call(name: str, arguments: str, *, with_id: bool = True) -> dict:
    call = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if with_id:
        call["id"] = f"call_{rng.randrange(10**8):08d}"
    return call


def _wrap(message: dict, finish_reason: str = "tool_calls") -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": "mock-weak",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        # prompt_cache_hit_tokens is deliberately absent: it is a provider-specific
        # field that disappears when the provider changes, and the agent has to
        # tolerate it being missing.
        "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940},
    }


def _weak_sql(question: str) -> str:
    """Guess at some SQL.  It is often wrong, which is the point."""
    q = question.lower()
    if "how many" in q or "count" in q:
        return "SELECT COUNT(*) FROM users"
    if "revenue" in q or "spent" in q or "spend" in q:
        return "SELECT SUM(unit_price) FROM order_items"
    if "product" in q:
        return "SELECT name FROM products LIMIT 5"
    return "SELECT * FROM users LIMIT 3"


def build_response(messages: list) -> dict:
    """Decide what to answer from how far the conversation has got."""
    question = next((m["content"] for m in messages if m.get("role") == "user"), "")
    called = {
        tc.get("function", {}).get("name")
        for m in messages if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }

    failure = _pick_failure()

    if failure == "empty_choices":
        # some local servers answer with an entirely different shape when overloaded
        return {"error": {"message": "model is loading", "type": "server_error"}}

    if failure == "no_tool_calls":
        return _wrap({"role": "assistant",
                      "content": "I think the answer is about 100."}, "stop")

    if failure == "unknown_tool":
        return _wrap({"role": "assistant", "content": None,
                      "tool_calls": [_tool_call("query_database", '{"sql": "SELECT 1"}')]})

    if failure == "bad_json_args":
        # weak models often emit unterminated JSON, or natural language inside it
        return _wrap({"role": "assistant", "content": None,
                      "tool_calls": [_tool_call("run_sql", '{"query": "SELECT COUNT(*) FROM users')]})

    with_id = failure != "missing_id"

    if "get_schema" not in called:
        return _wrap({"role": "assistant", "content": None,
                      "tool_calls": [_tool_call("get_schema", "{}", with_id=with_id)]})

    if "submit_answer" not in called:
        sql = _weak_sql(question)
        return _wrap({"role": "assistant", "content": None,
                      "tool_calls": [_tool_call("submit_answer",
                                                json.dumps({"query": sql}),
                                                with_id=with_id)]})

    return _wrap({"role": "assistant", "content": "Done. The answer is above."}, "stop")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        """GET /__stats -> how many POSTs have arrived so far."""
        with _hits_lock:
            self._send(200, json.dumps({"hits": _hits}).encode())

    def do_DELETE(self) -> None:
        """DELETE /__stats -> reset the counter. Call it at the start of a measurement."""
        global _hits
        with _hits_lock:
            _hits = 0
        self._send(200, b'{"hits": 0}')

    def do_POST(self) -> None:
        global _hits
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")
        with _hits_lock:
            _hits += 1

        if FORCE_STATUS is not None:
            # The counter was incremented before this branch: a failure response
            # is still a request that really arrived.  Counting only the
            # successful ones would miss exactly what is being measured.
            self._send(FORCE_STATUS, b'{"error": "forced by MOCK_FORCE_STATUS"}')
            return

        payload = json.dumps(build_response(body.get("messages", []))).encode()
        self._send(200, payload)

    def log_message(self, *args) -> None:
        pass                      # keep the agent's own output readable


if __name__ == "__main__":
    print(f"mock weak model on http://127.0.0.1:{PORT}/v1/chat/completions")
    print(f"failure rates: {FAILURE_RATES}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
