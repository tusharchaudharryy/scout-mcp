import pytest
from fakes import FakeChat, call, offline_server, reply

from scout_agent.config import AgentLimits
from scout_agent.llm import ToolCall
from scout_agent.sources import SourceRegistry
from scout_agent.toolbox import MCPToolBox
from scout_agent.trace import Trace
from scout_agent.worker import FINAL_NUDGE, WORKER_PROMPT, run_worker

pytestmark = pytest.mark.anyio


def steps_so_far(messages) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


async def run(docs_dir, respond, **limits):
    llm, registry, trace = FakeChat(respond), SourceRegistry(), Trace()
    async with MCPToolBox(offline_server(docs_dir)) as tools:
        result = await run_worker(llm, tools, "How does drip irrigation help?", registry=registry,
                                  limits=AgentLimits(**limits), trace=trace)
    return result, llm, registry, trace


def tool_messages(llm):
    return [m["content"] for m in llm.calls[-1][0] if m["role"] == "tool"]


async def test_reads_notes_and_pages_and_labels_what_it_read(docs_dir):
    def respond(messages, tools):
        step = steps_so_far(messages)
        if step == 0:
            return reply("", call("search_docs", query="drip irrigation"))
        if step == 1:
            return reply("", call("web_search", query="drip irrigation evaporation"))
        if step == 2:
            return reply("", call("fetch_page", url="https://example.com/drip", max_chars=999_999))
        return reply("- Drip lines cut evaporation [S1] [S2].")

    result, llm, registry, trace = await run(docs_dir, respond)

    assert (result.findings, result.steps, result.source_ids) == ("- Drip lines cut evaporation [S1] [S2].", 4, ["S1", "S2"])
    assert [(s.id, s.kind, s.ref) for s in registry.all()] == [
        ("S1", "note", "irrigation.md#0"),
        ("S2", "web", "https://example.com/drip"),
    ]
    notes, web, page = tool_messages(llm)
    assert notes.startswith("[S1] note: irrigation.md > Drip irrigation")
    assert "snippets only" in web and "[S" not in web  # search results are not citable
    assert page.startswith("[S2] Drip guide\nURL: https://example.com/drip")
    fetch = next(e for e in trace.of_kind("tool") if e["name"] == "fetch_page")
    assert fetch["arguments"]["max_chars"] == 5_000  # the model asked for more; the agent capped it
    assert llm.calls[0][0][0] == {"role": "system", "content": WORKER_PROMPT}
    assert {t["function"]["name"] for t in llm.calls[0][1]} == {"search_docs", "read_doc", "list_docs", "web_search", "fetch_page"}


async def test_every_tool_call_is_answered_with_a_matching_tool_message(docs_dir):
    def respond(messages, tools):
        if steps_so_far(messages) == 0:  # two calls in one turn
            return reply("", call("search_docs", call_id="a", query="weevil"), call("list_docs", call_id="b"))
        return reply("Done.")

    _, llm, _, _ = await run(docs_dir, respond)
    messages = llm.calls[-1][0]
    assistant = next(m for m in messages if m.get("tool_calls"))
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert [c["id"] for c in assistant["tool_calls"]] == answered == ["a", "b"]


async def test_step_budget_forces_an_answer_without_tools(docs_dir):
    def respond(messages, tools):
        return reply("Findings so far [S1].") if tools is None else reply("", call("search_docs", query="weevil"))

    result, llm, registry, _ = await run(docs_dir, respond, max_steps=2)
    assert (result.findings, result.steps) == ("Findings so far [S1].", 2)
    assert len(llm.calls) == 3 and llm.calls[-1][1] is None
    assert llm.calls[-1][0][-1] == {"role": "user", "content": FINAL_NUDGE}
    assert len(registry) == 1  # the same passage read twice keeps one label


async def test_tool_errors_and_bad_arguments_go_back_to_the_model(docs_dir):
    bad_json = ToolCall("c9", "read_doc", {}, error="the arguments were not valid JSON: {oops")

    def respond(messages, tools):
        step = steps_so_far(messages)
        if step == 0:
            return reply("", call("read_doc", name="../secret.txt"))
        if step == 1:
            return reply("", bad_json)
        if step == 2:
            return reply("", call("format_disk"))
        return reply("Nothing found.")

    result, llm, registry, trace = await run(docs_dir, respond)
    escape, invalid, unknown = tool_messages(llm)
    assert escape.startswith("ERROR:") and "list_docs" in escape
    assert invalid.startswith("ERROR: the arguments were not valid JSON")
    assert unknown.startswith("ERROR: Unknown tool 'format_disk'")
    assert len(registry) == 0 and result.findings == "Nothing found."
    assert [e["error"] for e in trace.of_kind("tool")] == [True, True, True]
