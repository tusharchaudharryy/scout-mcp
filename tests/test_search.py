import httpx2
import pytest

from scout_mcp.search import SearchError, searxng_search

BASE = "http://searx.test"


def client(handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


@pytest.mark.anyio
async def test_parses_dedupes_and_limits_results():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(request.url.params)
        return httpx2.Response(200, json={"results": [
            {"title": " Weevil  guide ", "url": "https://a.example/1", "content": "How to\n detect", "engines": ["duckduckgo", "brave"]},
            {"title": "Duplicate", "url": "https://a.example/1", "content": "same url"},
            {"title": "Not a web page", "url": "magnet:?xt=urn:btih:abc", "content": ""},
            {"title": "", "url": "https://b.example/2", "content": "No title", "engine": "bing", "publishedDate": "2026-01-02"},
            {"title": "Third", "url": "https://c.example/3", "content": "over the limit"},
        ]})

    hits = await searxng_search(client(handler), BASE, "weevil detection", max_results=2, time_range="year")
    assert [(h.title, h.url, h.snippet, h.engines) for h in hits] == [
        ("Weevil guide", "https://a.example/1", "How to detect", ["duckduckgo", "brave"]),
        ("https://b.example/2", "https://b.example/2", "No title", ["bing"]),
    ]
    assert hits[1].published == "2026-01-02"
    assert seen["format"] == "json" and seen["time_range"] == "year" and seen["q"] == "weevil detection"


@pytest.mark.anyio
async def test_json_format_disabled_explains_the_fix():
    with pytest.raises(SearchError, match="search.formats"):
        await searxng_search(client(lambda r: httpx2.Response(403)), BASE, "x", 5)


@pytest.mark.anyio
async def test_unreachable_searxng_says_how_to_start_it():
    def handler(request):
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(SearchError, match="docker compose up -d"):
        await searxng_search(client(handler), BASE, "x", 5)
