import json
import re

import pytest
from fakes import FakeChat, call, offline_server, reply

from scout_agent.citations import check_citations, claims, labels_in, render_sources
from scout_agent.config import AgentLimits
from scout_agent.llm import LLMError
from scout_agent.orchestrator import WRITER_PROMPT, AgentError, parse_plan, research
from scout_agent.sources import SourceRegistry
from scout_agent.toolbox import MCPToolBox
from scout_agent.trace import Trace
from scout_agent.worker import WORKER_PROMPT

QUESTION = "How do drip irrigation and weevil detection work?"
PLAN = '{"subquestions": ["How does drip irrigation save water?", "How is the red palm weevil detected?"]}'


@pytest.mark.parametrize("text, expected", [
    ('{"subquestions": ["a", "b"]}', ["a", "b"]),
    ('Sure!\n```json\n{"subquestions": ["a", "a", " b ", ""]}\n```', ["a", "b"]),
    ('["x", "y", "z", "w"]', ["x", "y", "z"]),
    ("I cannot help with that.", []),
])
def test_parse_plan(text, expected):
    assert parse_plan(text, limit=3) == expected


def test_labels_and_claims():
    assert labels_in("A [S1]. B [S2, S3]. C [S3][S4]. See [the docs](https://x.example).") == ["S1", "S2", "S3", "S4"]
    text = (
        "# Answer\n\nDrip lines save water at the roots. [S1] They also clog sometimes.\n\n"
        "- Weevils bore into palm trunks [S2]\n\n## Open questions\n\n- How much water is saved exactly?"
    )
    assert claims(text) == ["Drip lines save water at the roots [S1].", "They also clog sometimes.", "Weevils bore into palm trunks [S2]"]


def test_report_and_sources_list():
    registry = SourceRegistry()
    registry.add("web", "https://example.com/a", "https://example.com/a", "text")
    registry.add("note", "irrigation.md#0", "irrigation.md > Drip irrigation", "text")
    report = check_citations("Drip lines save water [S2]. Pages say so too [S1]. Weevils arrived in 1891 [S9].", registry)
    assert (report.cited, report.unknown, report.claims, report.cited_claims) == (["S2", "S1"], ["S9"], 3, 3)
    assert render_sources(report.cited, registry) == (
        "## Sources\n\n- [S2] irrigation.md > Drip irrigation - note: irrigation.md#0\n- [S1] https://example.com/a"
    )


def scripted(writer=None, fail_worker=None):
    """Planner -> two workers (one notes search each) -> writer."""

    def respond(messages, tools):
        system, user = messages[0]["content"], messages[1]["content"]
        if system.startswith("You plan research"):
            return reply(PLAN)
        if system == WORKER_PROMPT:
            drip = "drip" in user
            if fail_worker and fail_worker in user:
                return LLMError("The LLM provider returned HTTP 500: boom")
            read = [m["content"] for m in messages if m["role"] == "tool"]
            if not read:
                return reply("", call("search_docs", query="drip irrigation" if drip else "acoustic sensors chewing"))
            label = re.search(r"\[(S\d+)\]", read[0]).group(1)
            return reply(f"- Drip lines cut evaporation [{label}]." if drip else f"- Chewing larvae can be heard [{label}].")
        assert system == WRITER_PROMPT
        if writer:
            return reply(writer(user))
        drip = re.search(r"Drip lines cut evaporation \[(S\d+)\]", user).group(1)
        weevil = re.search(r"Chewing larvae can be heard \[(S\d+)\]", user).group(1)
        return reply(
            f"Drip lines cut evaporation, and weevil larvae can be heard [{drip}, {weevil}].\n\n"
            f"- Drip lines deliver water to the root zone [{drip}].\n"
            f"- Acoustic sensors pick up larvae chewing inside trunks [{weevil}].\n"
            "- Thermal imaging is also being tested in some places.\n"
            "- The weevil was first recorded in 1891 [S9]."
        )

    return respond


async def run(docs_dir, respond, trace=None, **limits):
    llm = FakeChat(respond)
    async with MCPToolBox(offline_server(docs_dir)) as tools:
        return await research(QUESTION, llm, tools, limits=AgentLimits(**limits), trace=trace), llm


@pytest.mark.anyio
async def test_full_run_plans_researches_writes_and_checks_citations(docs_dir, tmp_path):
    trace = Trace(tmp_path / "run.jsonl")
    result, llm = await run(docs_dir, scripted(), trace=trace)

    assert result.subquestions == json.loads(PLAN)["subquestions"]
    assert all(w.findings and not w.error for w in result.workers)
    assert result.check.unknown == ["S9"]  # an invented citation is caught
    assert {s.ref for s in result.sources} == {"irrigation.md#0", "pests.md#1"}
    assert (result.check.claims, result.check.cited_claims) == (5, 4)  # the thermal bullet has no source
    sources_section = result.answer.split("## Sources")[1]
    assert "note: irrigation.md#0" in sources_section and "note: pests.md#1" in sources_section and "S9" not in sources_section
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (60, 30)  # 6 LLM calls x (10, 5)

    lines = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {e["kind"] for e in lines}
    assert {"question", "plan", "llm", "tool", "worker_done", "write", "answer"} <= kinds
    assert result.to_dict()["check"]["coverage"] == 0.8


@pytest.mark.anyio
async def test_an_unreadable_plan_falls_back_to_the_question_itself(docs_dir):
    def respond(messages, tools):
        if messages[0]["content"].startswith("You plan research"):
            return reply("Let me think about that...")
        return scripted(writer=lambda user: "Drip lines cut evaporation [S1].")(messages, tools)

    result, _ = await run(docs_dir, respond)
    assert result.subquestions == [QUESTION]


@pytest.mark.anyio
async def test_one_failed_worker_does_not_sink_the_run(docs_dir):
    result, _ = await run(docs_dir, scripted(writer=lambda user: "Drip lines cut evaporation [S1].", fail_worker="weevil"))
    failed = [w for w in result.workers if w.error]
    assert len(failed) == 1 and "boom" in failed[0].error
    assert result.body == "Drip lines cut evaporation [S1]."


@pytest.mark.anyio
async def test_the_run_fails_clearly_when_every_worker_fails(docs_dir):
    def respond(messages, tools):
        if messages[0]["content"].startswith("You plan research"):
            return reply(PLAN)
        return LLMError("The LLM provider rejected the request (HTTP 401). Check SCOUT_LLM_API_KEY.")

    with pytest.raises(AgentError, match="SCOUT_LLM_API_KEY"):
        await run(docs_dir, respond)
