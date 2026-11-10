"""A record of everything a run did (LLM calls, tool calls, results), for debugging and evaluation.

Events are kept in memory and, when a path is given, appended to a JSON Lines file as they happen,
so a run that crashes still leaves its trace behind.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .llm import Usage


class Trace:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        self.usage = Usage()
        self._start = time.monotonic()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def log(self, kind: str, **data: Any) -> None:
        event = {"t": round(self.elapsed, 3), "kind": kind, **data}
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]
