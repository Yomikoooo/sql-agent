import os
import json
import logging
import time
from typing import Iterator

import httpx

from dotenv import load_dotenv

from .tracing import span
from pydantic import BaseModel

from .cache import CACHE


logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass

class ConfigError(LLMError):
    pass

class APIError(LLMError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"API returns {status_code}: {body[:200]}")

class MalformedResponseError(LLMError):
    """HTTP 200, but the body is not the expected shape.

    Not hypothetical: a local inference server returns 200 with
    {"error": {...}} while the model is still loading. Without this class the
    agent does data["choices"] and dies with a bare KeyError, which says
    nothing about what actually happened.
    """
    def __init__(self, body: dict):
        self.body = body
        preview = json.dumps(body, ensure_ascii=False)[:200]
        super().__init__(f"response has no usable 'choices': {preview}")



load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = os.getenv("DEEPSEEK_API_URL")
DEFAULT_MODEL = os.getenv("DEEPSEEK_API_MODEL", "deepseek-chat")

# The endpoint has to follow the model. Passing --model qwen2.5:3b while the
# URL still pointed at the hosted API once produced 76/76 errors: the hosted
# side replied 400 "that is not one of my models". Earlier runs had worked only
# because the URL in .env was edited by hand and edited back afterwards -- an
# undeclared switch that no result file recorded, so months later there is no
# way to tell which endpoint a given run actually hit.
#
# The rule is "does the model name contain a colon": Ollama tags look like
# qwen2.5:3b, hosted names do not. A heuristic, not a spec -- a hosted model
# with a colon in its name would break it.
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/chat/completions")


def resolve_url(model: str | None) -> str:
    """Model name -> which endpoint to call."""
    return OLLAMA_API_URL if ":" in (model or DEFAULT_MODEL) else API_URL


def _check_config() -> None:
    # Name the variable: a bare error means reading the source to find out which
    if not API_KEY:
        raise ConfigError("DEEPSEEK_API_KEY is not set. Check your .env file.")
    if not API_URL:
        raise ConfigError("DEEPSEEK_API_URL is not set. Check your .env file.")

def _headers() -> dict:
    return {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def _normalize(prompt_or_message) -> list:
    if isinstance(prompt_or_message, str):
        return [{"role": "user", "content": prompt_or_message}]
    return prompt_or_message

def _body(messages, model, max_tokens, stream, **extra) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    body.update(extra)
    return body


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResult(BaseModel):
    text: str
    usage: Usage | None = None
    finish_reason: str | None = None

#------------parse sse

DONE = "[DONE]"

def parse_sse_line(line: str) -> str | None:
    if not line or not line.startswith("data: "):
        return None

    payload = line.removeprefix("data: ").strip()
    if payload == DONE:
        return DONE

    chunk = json.loads(payload)
    if not chunk.get("choices"):
        return None

    return chunk["choices"][0]["delta"].get("content")

#------------interface


def chat(prompt_or_message, *, model=DEFAULT_MODEL, max_tokens=1000, timeout=60.0, **extra) -> ChatResult:
    _check_config()
    messages = _normalize(prompt_or_message)
    response = httpx.post(
        resolve_url(model),
        headers=_headers(),
        json=_body(messages, model, max_tokens, stream=False, **extra),
        timeout=timeout,
    )
    if response.status_code != 200:
        logger.error("fail %s", response.status_code)
        raise APIError(response.status_code, response.text)

    data = response.json()
    choice = data["choices"][0]
    result = ChatResult(
        text=choice["message"]["content"],
        usage=Usage(**data["usage"]) if data.get("usage") else None,
        finish_reason=choice.get("finish_reason"),
    )
    if result.usage:
        logger.info("chat finished: input %d / output %d", result.usage.prompt_tokens, result.usage.completion_tokens)
    return result

def chat_stream(prompt_or_message, *, model=DEFAULT_MODEL, max_tokens=1000, timeout=60.0, **extra) -> Iterator[str]:
    _check_config()
    messages = _normalize(prompt_or_message)
    with httpx.stream(
        "POST", 
        resolve_url(model),
        headers=_headers(), 
        json=_body(messages, model, max_tokens, stream=True, **extra)
    ) as response:
        if response.status_code != 200:
            response.read()
            logger.error("response fail: %s", response.status_code)
            raise APIError(response.status_code, response.text)

        for line in response.iter_lines():
            content = parse_sse_line(line)
            if not content:
                continue
            if content == DONE:
                break
            yield content



async def chat_async(client: httpx.AsyncClient, prompt_or_message, *, model=DEFAULT_MODEL, max_tokens=1000, **extra) -> ChatResult:
    _check_config()
    messages = _normalize(prompt_or_message)

    response = await client.post(resolve_url(model), headers=_headers(), json=_body(messages, model, max_tokens, stream=False, **extra))
    if response.status_code != 200:
        raise APIError(response.status_code, response.text)
    data = response.json()
    choice = data["choices"][0]
    return ChatResult(
        text=choice["message"]["content"],
        usage=Usage(**data["usage"]) if data.get("usage") else None,
        finish_reason=choice.get("finish_reason")
    )

if __name__ == "__main__":
    r = chat("explain HTTP with one sentence")
    print(r.text)
    print(r.usage)


# Retried errors: the ones where trying again might work. 4xx is excluded --
# a wrong parameter or a bad key fails identically a hundred times.
TRANSIENT_EXC = (
    httpx.TimeoutException,       # slow local inference, or a model reloading
    httpx.ConnectError,           # connection refused: the restart window
    httpx.RemoteProtocolError,    # Server disconnected without sending a response
    httpx.ReadError,
)
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


def call_with_tools(messages: list, tools: list, *,
                    model: str | None = None, max_tokens: int = 2000,
                    temperature: float = 0.0,
                    timeout: float | None = None,
                    max_retries: int | None = None) -> dict:
    """One tool-enabled call.

    `model=None` falls back to DEFAULT_MODEL from .env. The parameter exists so
    that several models can be compared on the same task set: a baseline has to
    be measured on the model you intend to optimise.

    temperature defaults to 0 because an eval measures capability, not luck.
    The server default of 1.0 pushed the flaky rate to 25.8% here -- even
    "How many users do we have?" failed one run in three. Rejection sampling
    later wants it raised to 0.7-1.0, where diversity is the point. Same code,
    two purposes, two settings.

    Retrying is required, not a nicety. Hosted APIs are stable enough to skip
    it; local inference is not -- a model being unloaded and reloaded fails
    every request for tens of seconds. Without retries that infrastructure
    noise is recorded as "the model got it wrong" and contaminates the metric.
    """
    _check_config()
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    max_retries = MAX_RETRIES if max_retries is None else max_retries

    body = _body(messages, model=model or DEFAULT_MODEL, max_tokens=max_tokens,
                 stream=False, tools=tools, tool_choice="auto",
                 temperature=temperature)

    # exact-match cache
    cached = CACHE.get(body)
    if cached is not None:
        return cached

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            # exponential backoff: 1s, 2s, 4s ... give the server room to recover
            delay = 2 ** (attempt - 1)
            logger.warning("retry %d/%d after %ds (%s)",
                           attempt, max_retries, delay, last_error)
            time.sleep(delay)

        # One child span per HTTP attempt, so the retry count, the time each
        # attempt took, and which layer failed are all visible afterwards.
        with span("http.post", attempt=attempt + 1) as sp:
            try:
                response = httpx.post(url=resolve_url(model), headers=_headers(),
                                      json=body, timeout=timeout)
            except TRANSIENT_EXC as e:
                last_error = e
                sp.attrs["error"] = type(e).__name__
                continue                       # recoverable, retry

            sp.attrs["status"] = response.status_code
            if response.status_code in TRANSIENT_STATUS:
                last_error = APIError(response.status_code, response.text)
                continue                       # rate limit / server error, retry
        break                              # everything else (4xx included) is final
    else:
        # loop ran to completion = retries exhausted without success
        raise last_error if last_error else LLMError("request failed")

    if response.status_code != 200:
        raise APIError(response.status_code, response.text)

    data = response.json()
    # Validate the shape here rather than letting callers index blindly: the
    # response format is this layer's contract. Failing with an LLMError
    # subclass means the agent, which already catches those, records
    # stop_reason="error" in the trace instead of dying on a KeyError.
    if not isinstance(data, dict) or not data.get("choices"):
        raise MalformedResponseError(data if isinstance(data, dict) else {"body": data})

    CACHE.put(body, data)
    return data