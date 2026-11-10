"""Test doubles for the agent tests: a scripted LLM and an offline scout-mcp server."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import httpx2

from scout_agent.llm import Completion, ToolCall, Usage
from scout_mcp.config import Settings
from scout_mcp.server import create_server

PAGE = b"<title>Drip guide</title><main><p>Drip irrigation cuts evaporation by watering the roots directly.</p></main>"


def fake_web(request: httpx2.Request) -> httpx2.Response:
    if request.url.host == "searx.test":
        return httpx2.Response(200, json={"results": [
            {"title": "Drip guide", "url": "https://example.com/drip", "content": "A snippet about drip lines."},
        ]})
    if str(request.url) == "https://example.com/drip":
        return httpx2.Response(200, headers={"content-type": "text/html"}, content=PAGE)
    return httpx2.Response(404)


def offline_server(docs_dir):
    settings = Settings(searxng_url="http://searx.test", docs_dir=docs_dir)
    return create_server(settings, transport=httpx2.MockTransport(fake_web), resolver=lambda host: ["93.184.215.14"])


def call(tool: str, /, *, call_id: str | None = None, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id or f"call_{tool}", name=tool, arguments=arguments)


def reply(content: str = "", *calls: ToolCall) -> Completion:
    return Completion(content=content, tool_calls=list(calls), usage=Usage(10, 5))


class FakeChat:
    """Stands in for an LLM. `respond(messages, tools)` returns the next Completion (or an exception to raise)."""

    def __init__(self, respond: Callable[[list[dict], list[dict] | None], Completion | Exception]) -> None:
        self.respond = respond
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    async def complete(self, messages, tools=None) -> Completion:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        result = self.respond(messages, tools)
        if isinstance(result, Exception):
            raise result
        return result
