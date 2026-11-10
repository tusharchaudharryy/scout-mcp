"""The research run: plan sub-questions, research them in parallel, write a cited answer, check it.

    question -> planner (LLM) -> sub-questions
             -> one worker per sub-question (LLM + tools, run concurrently)
             -> writer (LLM) -> answer with [S#] labels
             -> citation check (code) -> answer + sources list built from what was actually read
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import anyio

from .citations import CitationReport, check_citations, render_sources
from .config import AgentLimits
from .llm import ChatModel, LLMError, Usage
from .sources import Source, SourceRegistry
from .toolbox import MCPToolBox
from .trace import Trace
from .worker import WorkerResult, run_worker

PLANNER_PROMPT = """\
You plan research. Split the user's question into 1 to {n} focused sub-questions that together
answer it; use a single sub-question when the question is simple. Reply with JSON only, in this
form: {{"subquestions": ["...", "..."]}}"""

WRITER_PROMPT = """\
You write the final answer to a research question from the findings of several researchers.

Rules:
- Use only the findings below. Do not add facts of your own.
- End every factual sentence or bullet with the label(s) of its sources, e.g. [S2] or [S1, S4],
  exactly as they appear in the findings. Never invent labels, URLs or titles.
- Start with a direct answer of two or three sentences, then key points as bullets.
- If the findings leave gaps or disagree, finish with a section titled "Open questions".
- Do not add a list of sources; it is added automatically."""


class AgentError(Exception):
    """The research run could not produce an answer."""


@dataclass
class ResearchResult:
    question: str
    subquestions: list[str]
    workers: list[WorkerResult]
    body: str  # the writer's text
    answer: str  # body + sources list
    sources: list[Source]  # the sources the answer cites
    check: CitationReport
    usage: Usage
    seconds: float
    trace_path: str | None = None
    all_sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["check"]["coverage"] = round(self.check.coverage, 3)
        return data


def parse_plan(text: str, limit: int) -> list[str]:
    """Read the planner's sub-questions from JSON, tolerating code fences and surrounding prose."""
    candidates = [text.strip()]
    if match := re.search(r"\{.*\}|\[.*\]", text, re.DOTALL):
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = data.get("subquestions") if isinstance(data, dict) else data
        if isinstance(items, list):
            seen: list[str] = []
            for item in items:
                if isinstance(item, str) and item.strip() and item.strip() not in seen:
                    seen.append(item.strip())
            return seen[:limit]
    return []


async def plan(llm: ChatModel, question: str, limits: AgentLimits, trace: Trace) -> list[str]:
    reply = await llm.complete([
        {"role": "system", "content": PLANNER_PROMPT.format(n=limits.max_subquestions)},
        {"role": "user", "content": question},
    ])
    trace.add_usage(reply.usage)
    subquestions = parse_plan(reply.content, limits.max_subquestions) or [question]
    trace.log("plan", subquestions=subquestions, raw=reply.content[:2000],
              prompt_tokens=reply.usage.prompt_tokens, completion_tokens=reply.usage.completion_tokens)
    return subquestions


async def write(llm: ChatModel, question: str, findings: list[WorkerResult], registry: SourceRegistry, trace: Trace) -> str:
    blocks = [f"### Sub-question {i}: {w.question}\n{w.findings}" for i, w in enumerate(findings, 1)]
    catalogue = "\n".join(f"[{s.id}] {s.title} ({s.ref})" for s in registry.all()) or "(none)"
    reply = await llm.complete([
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nFindings:\n\n" + "\n\n".join(blocks)
                                    + f"\n\nSources read during the research:\n{catalogue}"},
    ])
    trace.add_usage(reply.usage)
    trace.log("write", prompt_tokens=reply.usage.prompt_tokens, completion_tokens=reply.usage.completion_tokens)
    return reply.content.strip()


async def research(
    question: str,
    llm: ChatModel,
    toolbox: MCPToolBox,
    *,
    limits: AgentLimits | None = None,
    trace: Trace | None = None,
) -> ResearchResult:
    """Answer a research question with cited sources."""
    limits = limits or AgentLimits()
    trace = trace or Trace()
    registry = SourceRegistry()
    trace.log("question", question=question, limits=asdict(limits))

    subquestions = await plan(llm, question, limits, trace)
    results: list[WorkerResult] = [WorkerResult(q, "") for q in subquestions]
    limiter = anyio.CapacityLimiter(limits.concurrency)

    async def work(i: int, sub: str) -> None:
        async with limiter:
            try:
                results[i] = await run_worker(llm, toolbox, sub, registry=registry, limits=limits, trace=trace, worker=i)
            except LLMError as exc:  # one failed worker should not sink the whole run
                results[i] = WorkerResult(sub, "", error=str(exc))
            trace.log("worker_done", worker=i, question=sub, steps=results[i].steps,
                      sources=results[i].source_ids, error=results[i].error, findings=results[i].findings[:4000])

    async with anyio.create_task_group() as tg:
        for i, sub in enumerate(subquestions):
            tg.start_soon(work, i, sub)

    usable = [r for r in results if r.findings and not r.error]
    if not usable:
        errors = "; ".join(sorted({r.error for r in results if r.error})) or "the workers returned no findings"
        raise AgentError(f"No research worker produced findings: {errors}")

    body = await write(llm, question, usable, registry, trace)
    check = check_citations(body, registry)
    answer = f"{body}\n\n{render_sources(check.cited, registry)}\n"
    trace.log("answer", answer=answer, cited=check.cited, unknown=check.unknown, claims=check.claims,
              cited_claims=check.cited_claims, coverage=round(check.coverage, 3),
              prompt_tokens=trace.usage.prompt_tokens, completion_tokens=trace.usage.completion_tokens)
    return ResearchResult(
        question=question,
        subquestions=subquestions,
        workers=results,
        body=body,
        answer=answer,
        sources=[s for s in (registry.get(sid) for sid in check.cited) if s is not None],
        check=check,
        usage=trace.usage,
        seconds=round(trace.elapsed, 2),
        trace_path=str(trace.path) if trace.path else None,
        all_sources=registry.all(),
    )
