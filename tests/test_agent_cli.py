import json

import pytest

from scout_agent import cli
from scout_agent.citations import CitationReport
from scout_agent.llm import LLMError, Usage
from scout_agent.orchestrator import ResearchResult


def fake_result() -> ResearchResult:
    return ResearchResult(
        question="What is MCP?",
        subquestions=["What is MCP?"],
        workers=[],
        body="MCP connects models to tools [S1].",
        answer="MCP connects models to tools [S1].\n\n## Sources\n\n- [S1] https://modelcontextprotocol.io\n",
        sources=[],
        check=CitationReport(cited=["S1"], unknown=["S7"], claims=1, cited_claims=1),
        usage=Usage(100, 20),
        seconds=1.5,
        trace_path="runs/x.jsonl",
    )


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("SCOUT_LLM_MODEL", "test-model")


def test_missing_model_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.delenv("SCOUT_LLM_MODEL", raising=False)
    assert cli.main(["ask", "anything"]) == 2
    assert "SCOUT_LLM_MODEL" in capsys.readouterr().err


def test_invalid_limits_are_rejected(model, capsys):
    assert cli.main(["ask", "q", "--max-steps", "0"]) == 2
    assert "max_steps" in capsys.readouterr().err


def test_prints_the_answer_and_a_summary(model, monkeypatch, capsys, tmp_path):
    seen = {}

    async def fake_ask(question, settings, limits, trace_path):
        seen.update(question=question, model=settings.model, steps=limits.max_steps, path=trace_path)
        return fake_result()

    monkeypatch.setattr(cli, "_ask", fake_ask)
    assert cli.main(["ask", "What is MCP?", "--max-steps", "4", "--runs-dir", str(tmp_path)]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("MCP connects models to tools [S1].") and "## Sources" in out
    assert "citation coverage 100% (1/1 claims)" in err and "match no source: S7" in err and "trace: runs/x.jsonl" in err
    assert (seen["model"], seen["steps"]) == ("test-model", 4)
    assert seen["path"].parent == tmp_path and seen["path"].name.endswith("-what-is-mcp.jsonl")


def test_json_output(model, monkeypatch, capsys):
    async def fake_ask(*args):
        return fake_result()

    monkeypatch.setattr(cli, "_ask", fake_ask)
    assert cli.main(["ask", "What is MCP?", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["check"]["coverage"] == 1.0 and data["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}


def test_provider_errors_exit_with_code_1(model, monkeypatch, capsys):
    async def failing(*args):
        raise LLMError("The LLM provider rejected the request (HTTP 401). Check SCOUT_LLM_API_KEY.")

    monkeypatch.setattr(cli, "_ask", failing)
    assert cli.main(["ask", "q"]) == 1
    err = capsys.readouterr().err
    assert "HTTP 401" in err and "trace:" in err
