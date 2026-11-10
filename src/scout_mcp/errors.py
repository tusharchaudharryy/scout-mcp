"""Errors raised by the research tools.

Messages are written for the model that called the tool: each one says what went wrong and,
where possible, what to do next. The server turns them into MCP tool errors.
"""


class ResearchError(Exception):
    """Base class for errors a tool reports back to the caller."""
