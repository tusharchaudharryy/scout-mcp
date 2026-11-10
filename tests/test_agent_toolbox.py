import pytest
from fakes import offline_server

from scout_agent.toolbox import MCPToolBox, simplify_schema


def keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            if key != "properties":
                yield from keys(value)
            else:
                for sub in value.values():
                    yield from keys(sub)
    elif isinstance(obj, list):
        for item in obj:
            yield from keys(item)


@pytest.mark.anyio
async def test_tool_schemas_are_simplified_for_every_provider(docs_dir):
    async with MCPToolBox(offline_server(docs_dir)) as tools:
        schemas = tools.function_schemas()
        only_notes = tools.function_schemas(["search_docs", "read_doc"])
    assert {s["function"]["name"] for s in schemas} == {"web_search", "fetch_page", "search_docs", "list_docs", "read_doc"}
    assert [s["function"]["name"] for s in only_notes] == ["search_docs", "read_doc"]
    assert "title" not in set(keys(schemas)) and "anyOf" not in set(keys(schemas))
    web = next(s["function"] for s in schemas if s["function"]["name"] == "web_search")
    assert web["parameters"]["properties"]["time_range"]["type"] == "string"
    assert "\n" not in web["description"]


def test_simplify_schema_keeps_fields_that_are_named_title():
    schema = {
        "type": "object",
        "title": "Args",
        "properties": {
            "title": {"type": "string", "title": "Title"},
            "n": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None, "title": "N", "description": "count"},
        },
    }
    assert simplify_schema(schema) == {
        "type": "object",
        "properties": {"title": {"type": "string"}, "n": {"type": "integer", "description": "count"}},
    }


@pytest.mark.anyio
async def test_calls_return_data_and_errors_come_back_as_results(docs_dir):
    async with MCPToolBox(offline_server(docs_dir)) as tools:
        hits = await tools.call("search_docs", {"query": "drip emitters"})
        missing = await tools.call("read_doc", {"name": "nope.md"})
        no_query = await tools.call("search_docs", {"top_k": 3})
        unknown = await tools.call("delete_everything", {})
    assert not hits.is_error and hits.data[0]["doc"] == "irrigation.md"
    assert missing.is_error and "list_docs" in missing.text
    assert no_query.is_error
    assert unknown.is_error and "Available tools" in unknown.text


@pytest.mark.anyio
async def test_errors_inside_the_block_come_out_unwrapped(docs_dir):
    # the MCP client's task group would otherwise turn this into an ExceptionGroup
    with pytest.raises(KeyError, match="boom"):
        async with MCPToolBox(offline_server(docs_dir)):
            raise KeyError("boom")


def test_stdio_server_gets_the_environment_but_not_the_llm_key():
    box = MCPToolBox.stdio({"SCOUT_LLM_API_KEY": "secret", "SEARXNG_URL": "http://search.local", "PATH": "/bin"})
    env = box.target.env
    assert "SCOUT_LLM_API_KEY" not in env
    assert env["SEARXNG_URL"] == "http://search.local" and env["SCOUT_LOG_LEVEL"] == "WARNING"
    assert box.target.args == ["-m", "scout_mcp"]
