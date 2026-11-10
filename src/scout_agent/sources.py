"""The evidence a run has actually read, each piece with a stable label such as [S3].

Only text the agent really read becomes a source: a fetched page, a note, or a passage returned by
notes search. Web-search snippets do not count; the agent must fetch a page before citing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["web", "note"]


@dataclass(frozen=True)
class Source:
    id: str
    kind: SourceKind
    ref: str  # URL for web pages, "name" or "name#chunk" for notes
    title: str
    excerpt: str


class SourceRegistry:
    """Shared by all workers of a run, so the same page gets the same label everywhere."""

    def __init__(self) -> None:
        self._by_id: dict[str, Source] = {}
        self._by_ref: dict[tuple[str, str], Source] = {}

    def add(self, kind: SourceKind, ref: str, title: str, text: str) -> Source:
        key = (kind, ref)
        if key in self._by_ref:
            return self._by_ref[key]
        source = Source(id=f"S{len(self._by_id) + 1}", kind=kind, ref=ref, title=title.strip() or ref, excerpt=" ".join(text.split())[:300])
        self._by_id[source.id] = source
        self._by_ref[key] = source
        return source

    def get(self, source_id: str) -> Source | None:
        return self._by_id.get(source_id)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> list[Source]:
        return list(self._by_id.values())
