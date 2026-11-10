"""A small client for OpenAI-compatible chat completions with tool calling.

Works with any provider that implements the OpenAI chat-completions format: OpenAI, Groq, Gemini's
OpenAI-compatible endpoint, Ollama, vLLM and others. It sends only request fields that all of them
accept (no `name` on messages, no `n`, no logprobs) and retries rate limits and server errors.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio
import httpx2

from .config import LLMSettings

Message = dict[str, Any]
_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """The LLM provider could not produce a reply."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None  # set when the model sent arguments that are not a JSON object


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None

    def as_message(self) -> Message:
        """The assistant message to put back into the conversation (before any tool results)."""
        if not self.tool_calls:
            return {"role": "assistant", "content": self.content}
        return {
            "role": "assistant",
            "content": self.content or None,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in self.tool_calls
            ],
        }


class ChatModel(Protocol):
    async def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Completion: ...


def parse_completion(data: dict[str, Any]) -> Completion:
    """Read the first choice of a chat-completions response."""
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("The LLM provider returned no choices.")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):  # some providers return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

    calls: list[ToolCall] = []
    for i, raw_call in enumerate(message.get("tool_calls") or []):
        function = raw_call.get("function") or {}
        raw_args = function.get("arguments")
        args: dict[str, Any] = {}
        error = None
        if isinstance(raw_args, dict):
            args = raw_args
        elif raw_args not in (None, ""):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                error = f"the arguments were not valid JSON: {str(raw_args)[:200]}"
            else:
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    error = "the arguments must be a JSON object"
        calls.append(ToolCall(id=raw_call.get("id") or f"call_{i}", name=function.get("name") or "", arguments=args, error=error))

    usage = data.get("usage") or {}
    return Completion(
        content=content,
        tool_calls=calls,
        usage=Usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)),
        finish_reason=choice.get("finish_reason"),
    )


def _backoff(attempt: int) -> float:
    return float(min(2**attempt, 20))


def _retry_delay(resp: httpx2.Response, attempt: int) -> float:
    try:
        return min(float(resp.headers["retry-after"]), 60.0)
    except (KeyError, ValueError):
        return _backoff(attempt)


def _error_message(resp: httpx2.Response, settings: LLMSettings) -> str:
    code = resp.status_code
    if code in (401, 403):
        return f"The LLM provider rejected the request (HTTP {code}). Check SCOUT_LLM_API_KEY."
    if code == 404:
        return (
            f"Not found (HTTP 404). Check SCOUT_LLM_BASE_URL ({settings.base_url}) "
            f"and SCOUT_LLM_MODEL ({settings.model})."
        )
    if code == 429:
        return "Rate limited by the LLM provider (HTTP 429), even after retries. Wait a minute or lower --concurrency."
    detail = ""
    try:
        error = resp.json().get("error")
        detail = error.get("message", "") if isinstance(error, dict) else str(error or "")
    except (ValueError, AttributeError):
        detail = resp.text
    return f"The LLM provider returned HTTP {code}: {detail.strip()[:500] or 'no details'}"


class OpenAICompatibleChat:
    """Chat model behind an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: Any = None,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._client = httpx2.AsyncClient(timeout=settings.timeout, headers=headers, transport=transport)
        self._sleep = sleep

    async def __aenter__(self) -> OpenAICompatibleChat:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Completion:
        s = self.settings
        body: dict[str, Any] = {"model": s.model, "messages": messages, "temperature": s.temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        url = f"{s.base_url}/chat/completions"
        for attempt in range(s.max_retries + 1):
            last = attempt == s.max_retries
            try:
                resp = await self._client.post(url, json=body)
            except httpx2.TimeoutException as exc:
                if last:
                    raise LLMError(f"The LLM provider at {s.base_url} timed out after {s.timeout:.0f} s.") from exc
                await self._sleep(_backoff(attempt))
                continue
            except httpx2.HTTPError as exc:
                raise LLMError(f"Could not reach the LLM provider at {s.base_url} ({type(exc).__name__}).") from exc
            if resp.status_code in _RETRY_STATUS and not last:
                await self._sleep(_retry_delay(resp, attempt))
                continue
            if resp.status_code != 200:
                raise LLMError(_error_message(resp, s))
            try:
                data = resp.json()
            except ValueError as exc:
                raise LLMError("The LLM provider did not return JSON.") from exc
            return parse_completion(data)
        raise AssertionError("unreachable")
