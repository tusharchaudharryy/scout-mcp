"""Web search through a self-hosted SearXNG instance (its JSON API)."""

from __future__ import annotations

import re
from typing import Literal

import httpx2
from pydantic import BaseModel

from .errors import ResearchError

TimeRange = Literal["day", "week", "month", "year"]


class SearchError(ResearchError):
    """SearXNG could not answer the query."""


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str
    engines: list[str]
    published: str | None = None


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


async def searxng_search(
    client: httpx2.AsyncClient,
    base_url: str,
    query: str,
    max_results: int,
    *,
    time_range: TimeRange | None = None,
    language: str = "en",
) -> list[SearchHit]:
    """Query SearXNG and return up to `max_results` unique http(s) results."""
    params: dict[str, str | int] = {"q": query, "format": "json", "language": language, "safesearch": 1}
    if time_range:
        params["time_range"] = time_range
    try:
        resp = await client.get(f"{base_url}/search", params=params)
    except httpx2.HTTPError as exc:
        raise SearchError(
            f"SearXNG is not reachable at {base_url} ({type(exc).__name__}). "
            "Start it with `docker compose up -d` in the project folder, or set SEARXNG_URL."
        ) from exc
    if resp.status_code == 403:
        raise SearchError(
            "SearXNG refused the JSON format (HTTP 403). Add `json` under `search.formats` "
            "in searxng/settings.yml and restart SearXNG."
        )
    if resp.status_code != 200:
        raise SearchError(f"SearXNG returned HTTP {resp.status_code} for this query.")
    try:
        data = resp.json()
    except ValueError as exc:
        raise SearchError("SearXNG did not return JSON. Check that SEARXNG_URL points at SearXNG.") from exc

    hits: list[SearchHit] = []
    seen: set[str] = set()
    for item in data.get("results", []):
        url = (item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        engines = item.get("engines") or ([item["engine"]] if item.get("engine") else [])
        hits.append(
            SearchHit(
                title=_clean(item.get("title")) or url,
                url=url,
                snippet=_clean(item.get("content")),
                engines=[str(e) for e in engines],
                published=item.get("publishedDate"),
            )
        )
        if len(hits) >= max_results:
            break
    return hits
