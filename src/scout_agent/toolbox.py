"""The agent's side of MCP: connect to scout-mcp and call its tools.

The connection is either a subprocess over stdio, exactly like a desktop MCP client, or an
in-process server object (used by the tests). Tool schemas are turned into the function-calling
format of OpenAI-compatible APIs.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mcp.client.client import Client, StdioServerParameters
from mcp.server import MCPServer


@dataclass
class ToolResult:
    text: str
    is_error: bool
    data: Any = None  # structured output (dict or list), when the tool returned any


def simplify_schema(schema: Any) -> Any:
    """Make a tool's JSON schema acceptable to every provider: drop `title`s and turn `X | None`
    unions into plain optional `X` fields."""
    if isinstance(schema, list):
        return [simplify_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key == "properties" and isinstance(value, dict):  # keys here are field names, not keywords
            out[key] = {name: simplify_schema(sub) for name, sub in value.items()}
        else:
            out[key] = simplify_schema(value)
    variants = out.get("anyOf")
    if isinstance(variants, list):
        non_null = [v for v in variants if v != {"type": "null"}]
        if len(non_null) == 1 and len(non_null) < len(variants):
            del out["anyOf"]
            out = {**non_null[0], **out}
    if "default" in out and out["default"] is None:
        del out["default"]
    return out


class MCPToolBox:
    """Tools of an MCP server, callable by the agent. Use as `async with MCPToolBox(...) as tools:`."""

    def __init__(self, server: MCPServer | StdioServerParameters) -> None:
        self.target = server
        self._client: Client | None = None
        self._tools: dict[str, Any] = {}

    @classmethod
    def stdio(cls, env: Mapping[str, str] | None = None) -> MCPToolBox:
        """Start scout-mcp as a subprocess. The LLM API key is not passed on: the server never needs it."""
        child_env = {k: v for k, v in (os.environ if env is None else env).items() if k != "SCOUT_LLM_API_KEY"}
        child_env.setdefault("SCOUT_LOG_LEVEL", "WARNING")  # keep the agent's terminal output readable
        return cls(StdioServerParameters(command=sys.executable, args=["-m", "scout_mcp"], env=child_env))

    async def __aenter__(self) -> MCPToolBox:
        client = Client(self.target)
        await client.__aenter__()
        try:
            listed = await client.list_tools()
        except BaseException:
            await client.__aexit__(None, None, None)
            raise
        self._client = client
        self._tools = {tool.name: tool for tool in listed.tools}
        return self

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if self._client is None:
            return
        client, self._client = self._client, None
        try:
            await client.__aexit__(exc_type, exc, tb)
        except BaseExceptionGroup as group:
            # The MCP client runs a task group, which wraps an error raised inside `async with` in an
            # ExceptionGroup. Give callers back the error they raised, like any other context manager.
            if exc is not None and group.subgroup(lambda e: not isinstance(e, BaseExceptionGroup) and e is not exc) is None:
                raise exc from None
            raise

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def function_schemas(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = None if names is None else set(names)
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": " ".join((tool.description or "").split()),
                    "parameters": simplify_schema(tool.input_schema),
                },
            }
            for tool in self._tools.values()
            if wanted is None or tool.name in wanted
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._client is None:
            raise RuntimeError("Use 'async with MCPToolBox(...)' before calling tools.")
        if name not in self._tools:
            return ToolResult(f"Unknown tool {name!r}. Available tools: {', '.join(self._tools)}.", True)
        try:
            result = await self._client.call_tool(name, arguments)
        except Exception as exc:  # protocol or transport failure: report it to the model, don't crash the run
            return ToolResult(f"Calling {name} failed: {exc}", True)
        text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "type", None) == "text").strip()
        if result.is_error:
            return ToolResult(text or f"{name} failed.", True)
        data = result.structured_content
        if isinstance(data, dict) and list(data) == ["result"]:  # list outputs are wrapped as {"result": [...]}
            data = data["result"]
        return ToolResult(text, False, data)
