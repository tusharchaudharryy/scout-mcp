import json

import httpx2
import pytest

from scout_agent.config import AgentLimits, ConfigError, LLMSettings
from scout_agent.llm import Completion, LLMError, OpenAICompatibleChat, ToolCall, parse_completion

SETTINGS = LLMSettings(model="test-model", base_url="https://llm.test/v1", api_key="sk-secret-123", max_retries=2)
HELLO = [{"role": "user", "content": "hi"}]


def ok(message: dict, usage: dict | None = None) -> httpx2.Response:
    return httpx2.Response(200, json={
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 12, "completion_tokens": 3},
    })


async def no_sleep(seconds: float) -> None:
    pass


@pytest.mark.anyio
async def test_request_shape_and_tool_call_parsing():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(url=str(request.url), auth=request.headers.get("authorization"), body=json.loads(request.content))
        return ok({"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_docs", "arguments": '{"query": "drip"}'}},
        ]})

    tools = [{"type": "function", "function": {"name": "search_docs", "description": "d", "parameters": {"type": "object"}}}]
    async with OpenAICompatibleChat(SETTINGS, transport=httpx2.MockTransport(handler)) as llm:
        out = await llm.complete(HELLO, tools)
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret-123"
    assert seen["body"]["model"] == "test-model" and seen["body"]["tools"] == tools and seen["body"]["tool_choice"] == "auto"
    assert "n" not in seen["body"] and "name" not in seen["body"]["messages"][0]  # fields Groq rejects
    assert out.tool_calls == [ToolCall("c1", "search_docs", {"query": "drip"})]
    assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (12, 3)


@pytest.mark.anyio
async def test_local_models_need_no_key_and_plain_calls_send_no_tools():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(auth=request.headers.get("authorization"), body=json.loads(request.content))
        return ok({"content": "hello"})

    settings = LLMSettings(model="qwen3", base_url="http://localhost:11434/v1")
    async with OpenAICompatibleChat(settings, transport=httpx2.MockTransport(handler)) as llm:
        out = await llm.complete(HELLO)
    assert seen["auth"] is None and "tools" not in seen["body"] and out.content == "hello"


def test_bad_tool_arguments_are_reported_not_raised():
    out = parse_completion({"choices": [{"message": {"tool_calls": [
        {"id": "a", "function": {"name": "fetch_page", "arguments": "{not json"}},
        {"id": "b", "function": {"name": "fetch_page", "arguments": "[1, 2]"}},
        {"function": {"name": "list_docs", "arguments": ""}},
    ]}}]})
    assert "not valid JSON" in out.tool_calls[0].error
    assert "JSON object" in out.tool_calls[1].error
    assert (out.tool_calls[2].id, out.tool_calls[2].arguments, out.tool_calls[2].error) == ("call_2", {}, None)


def test_content_parts_are_joined_and_missing_choices_fail():
    parts = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
    assert parse_completion(parts).content == "ab"
    with pytest.raises(LLMError, match="no choices"):
        parse_completion({"choices": []})


def test_assistant_messages_round_trip_tool_calls():
    with_calls = Completion("", [ToolCall("c1", "read_doc", {"name": "x.md"})]).as_message()
    assert with_calls == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read_doc", "arguments": '{"name": "x.md"}'}},
    ]}
    assert Completion("done").as_message() == {"role": "assistant", "content": "done"}


@pytest.mark.anyio
async def test_rate_limits_and_server_errors_are_retried():
    responses = [httpx2.Response(429, headers={"retry-after": "2"}), httpx2.Response(503), ok({"content": "fine"})]
    delays: list[float] = []

    async def record(seconds: float) -> None:
        delays.append(seconds)

    async with OpenAICompatibleChat(SETTINGS, transport=httpx2.MockTransport(lambda r: responses.pop(0)), sleep=record) as llm:
        out = await llm.complete(HELLO)
    assert out.content == "fine"
    assert delays == [2.0, 2.0]  # Retry-After, then exponential backoff (2**1)


@pytest.mark.anyio
@pytest.mark.parametrize("status, hint", [(401, "SCOUT_LLM_API_KEY"), (404, "SCOUT_LLM_MODEL"), (429, "Rate limited")])
async def test_errors_say_how_to_fix_them_and_never_show_the_key(status, hint):
    async with OpenAICompatibleChat(SETTINGS, transport=httpx2.MockTransport(lambda r: httpx2.Response(status)), sleep=no_sleep) as llm:
        with pytest.raises(LLMError) as err:
            await llm.complete(HELLO)
    assert hint in str(err.value) and "sk-secret-123" not in str(err.value)


@pytest.mark.anyio
async def test_provider_error_details_are_passed_on():
    resp = httpx2.Response(400, json={"error": {"message": "tools[0].parameters: unsupported keyword 'anyOf'"}})
    async with OpenAICompatibleChat(SETTINGS, transport=httpx2.MockTransport(lambda r: resp)) as llm:
        with pytest.raises(LLMError, match="unsupported keyword"):
            await llm.complete(HELLO)


@pytest.mark.anyio
async def test_unreachable_provider():
    def handler(request):
        raise httpx2.ConnectError("refused", request=request)

    async with OpenAICompatibleChat(SETTINGS, transport=httpx2.MockTransport(handler)) as llm:
        with pytest.raises(LLMError, match="Could not reach"):
            await llm.complete(HELLO)


def test_settings_from_env_hide_the_key(monkeypatch):
    monkeypatch.delenv("SCOUT_LLM_MODEL", raising=False)
    with pytest.raises(ConfigError, match="SCOUT_LLM_MODEL"):
        LLMSettings.from_env()
    monkeypatch.setenv("SCOUT_LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("SCOUT_LLM_BASE_URL", "https://api.groq.com/openai/v1/")
    monkeypatch.setenv("SCOUT_LLM_API_KEY", "gsk_abc")
    s = LLMSettings.from_env()
    assert (s.model, s.base_url, s.api_key) == ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", "gsk_abc")
    assert "gsk_abc" not in repr(s)


@pytest.mark.parametrize("bad", [{"max_steps": 0}, {"concurrency": 0}, {"max_subquestions": 0}, {"max_page_chars": 100}])
def test_limits_are_validated(bad):
    with pytest.raises(ConfigError):
        AgentLimits(**bad)
