"""Agent settings: the LLM connection (from environment variables) and per-run limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigError(Exception):
    """The agent is not configured well enough to run."""


@dataclass(frozen=True)
class LLMSettings:
    """Connection to any OpenAI-compatible chat-completions API (OpenAI, Groq, Gemini, Ollama, vLLM...)."""

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default="", repr=False)  # never shown in logs or traces
    timeout: float = 60.0
    max_retries: int = 3
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> LLMSettings:
        model = os.environ.get("SCOUT_LLM_MODEL", "").strip()
        if not model:
            raise ConfigError(
                "Set SCOUT_LLM_MODEL, plus SCOUT_LLM_BASE_URL and SCOUT_LLM_API_KEY for your provider. "
                "The README lists settings for OpenAI, Groq, Gemini and Ollama."
            )
        d = cls(model=model)
        return cls(
            model=model,
            base_url=os.environ.get("SCOUT_LLM_BASE_URL", d.base_url).strip().rstrip("/"),
            api_key=os.environ.get("SCOUT_LLM_API_KEY", "").strip(),
            timeout=float(os.environ.get("SCOUT_LLM_TIMEOUT", d.timeout)),
            max_retries=int(os.environ.get("SCOUT_LLM_MAX_RETRIES", d.max_retries)),
            temperature=float(os.environ.get("SCOUT_LLM_TEMPERATURE", d.temperature)),
        )


@dataclass(frozen=True)
class AgentLimits:
    """Budgets for one research run."""

    max_subquestions: int = 3  # how many sub-questions the planner may create
    max_steps: int = 6  # tool-calling rounds per worker before it must answer
    max_page_chars: int = 5_000  # cap on page or note text handed to the model per tool call
    concurrency: int = 2  # workers running at the same time (keep low on free API tiers)

    def __post_init__(self) -> None:
        for name in ("max_subquestions", "max_steps", "concurrency"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1.")
        if self.max_page_chars < 500:
            raise ConfigError("max_page_chars must be at least 500.")
