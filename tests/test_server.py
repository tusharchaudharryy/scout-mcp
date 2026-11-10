"""End-to-end tests through the MCP protocol, using the SDK's in-memory client."""

import httpx2
import pytest

from mcp.client.client import Client
from scout_mcp.config import Settings
from scout_mcp.server import create_server

pytestmark = pytest.mark.anyio


def fake_web(request: httpx2.Request) -> httpx2.Response:
    if request.url.host == "searx.test":
        return httpx2.Response(200, json={"results": [{"title": "Guide", "url": "https://example.com/guide", "content": "snippet"}]})
    if str(request.url) == "https://example.com/guide":
        return httpx2.Response(200, headers={"content-type": "text/html"}, content=b"<title>Guide</title><p>Full text.</p>")
    return httpx2.Response(404)


@pytest.fixture
def server(docs_dir):
    settings = Settings(searxng_url="http://searx.test", docs_dir=docs_dir)
    return create_server(settings, transport=httpx2.MockTransport(fake_web), resolver=lambda host: ["93.184.215.14"])


async def test_tools_are_listed_and_all_read_only(server):
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"web_search", "fetch_page", "search_docs", "list_docs", "read_doc"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    assert tools["web_search"].input_schema["properties"]["max_results"]["maximum"] == 10


async def test_search_then_read_a_note(server):
    async with Client(server) as client:
        hits = (await client.call_tool("search_docs", {"query": "drip emitters"})).structured_content["result"]
        doc = (await client.call_tool("read_doc", {"name": hits[0]["doc"]})).structured_content
    assert hits[0]["doc"] == "irrigation.md"
    assert doc["text"].startswith("# Drip irrigation")


async def test_web_search_then_fetch(server):
    async with Client(server) as client:
        hits = (await client.call_tool("web_search", {"query": "guide"})).structured_content["result"]
        page = (await client.call_tool("fetch_page", {"url": hits[0]["url"]})).structured_content
    assert page["title"] == "Guide" and page["text"] == "Full text."


async def test_errors_come_back_as_tool_errors_with_advice(server):
    async with Client(server) as client:
        missing = await client.call_tool("read_doc", {"name": "../secret.txt"})
        private = await client.call_tool("fetch_page", {"url": "http://127.0.0.1:8080/"})
        too_many = await client.call_tool("web_search", {"query": "x", "max_results": 50})
    assert missing.is_error and "list_docs" in missing.content[0].text
    assert private.is_error and "non-public address" in private.content[0].text
    assert too_many.is_error


async def test_resource_and_prompt(server):
    async with Client(server) as client:
        templates = (await client.list_resource_templates()).resource_templates
        note = (await client.read_resource("docs://pests.md")).contents[0].text
        prompt = (await client.get_prompt("research_brief", {"topic": "date palm pests"})).messages[0].content.text
    assert [t.uri_template for t in templates] == ["docs://{name}"]
    assert note.startswith("# Red palm weevil")
    assert "date palm pests" in prompt and "search_docs" in prompt
