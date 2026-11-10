"""scout-mcp: an MCP server that gives AI agents private web search, safe page fetching and
search over your own notes."""

from .server import create_server, main

__all__ = ["create_server", "main"]
