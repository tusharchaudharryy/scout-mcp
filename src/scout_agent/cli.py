"""Command line: `scout ask "question"` prints a cited answer and writes a trace of the run."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import anyio

from .config import AgentLimits, ConfigError, LLMSettings
from .llm import LLMError, OpenAICompatibleChat
from .orchestrator import AgentError, ResearchResult, research
from .toolbox import MCPToolBox
from .trace import Trace


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-") or "question"


def trace_path_for(question: str, runs_dir: str) -> Path:
    return Path(runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(question)}.jsonl"


async def _ask(question: str, settings: LLMSettings, limits: AgentLimits, trace_path: Path) -> ResearchResult:
    async with OpenAICompatibleChat(settings) as llm, MCPToolBox.stdio() as tools:
        return await research(question, llm, tools, limits=limits, trace=Trace(trace_path))


def _summary(result: ResearchResult) -> str:
    c = result.check
    parts = [
        f"{len(result.subquestions)} sub-questions",
        f"{len(result.all_sources)} sources read, {len(c.cited)} cited",
        f"citation coverage {c.coverage:.0%} ({c.cited_claims}/{c.claims} claims)",
        f"tokens {result.usage.prompt_tokens} in / {result.usage.completion_tokens} out",
        f"{result.seconds:.1f} s",
    ]
    lines = ["-- " + " | ".join(parts)]
    if c.unknown:
        lines.append(f"-- warning: the answer cites labels that match no source: {', '.join(c.unknown)}")
    failed = [w for w in result.workers if w.error]
    if failed:
        lines.append(f"-- warning: {len(failed)} worker(s) failed: {failed[0].error}")
    lines.append(f"-- trace: {result.trace_path}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scout", description="Research questions with cited sources, using the scout-mcp tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="research a question and print a cited answer")
    ask.add_argument("question", help="the research question, in quotes")
    d = AgentLimits()
    ask.add_argument("--max-subquestions", type=int, default=d.max_subquestions, help=f"(default {d.max_subquestions})")
    ask.add_argument("--max-steps", type=int, default=d.max_steps, help=f"tool rounds per worker (default {d.max_steps})")
    ask.add_argument("--max-page-chars", type=int, default=d.max_page_chars, help=f"(default {d.max_page_chars})")
    ask.add_argument("--concurrency", type=int, default=d.concurrency, help=f"workers at once (default {d.concurrency})")
    ask.add_argument("--runs-dir", default="runs", help="folder for run traces (default ./runs)")
    ask.add_argument("--json", action="store_true", help="print the whole result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on characters the console cannot show
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        settings = LLMSettings.from_env()
        limits = AgentLimits(
            max_subquestions=args.max_subquestions,
            max_steps=args.max_steps,
            max_page_chars=args.max_page_chars,
            concurrency=args.concurrency,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    trace_path = trace_path_for(args.question, args.runs_dir)
    try:
        result = anyio.run(_ask, args.question, settings, limits, trace_path)
    except (LLMError, AgentError) as exc:
        print(f"error: {exc}\n-- trace: {trace_path}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) if args.json else result.answer)
    print(_summary(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
