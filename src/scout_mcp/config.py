"""Server settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    searxng_url: str = "http://localhost:8080"
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    http_timeout: float = 15.0
    fetch_max_bytes: int = 2_000_000
    allow_private_urls: bool = False
    user_agent: str = "scout-mcp/0.1"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            searxng_url=os.environ.get("SEARXNG_URL", d.searxng_url).rstrip("/"),
            docs_dir=Path(os.environ.get("SCOUT_DOCS_DIR", str(d.docs_dir))).expanduser().resolve(),
            http_timeout=float(os.environ.get("SCOUT_HTTP_TIMEOUT", d.http_timeout)),
            fetch_max_bytes=int(os.environ.get("SCOUT_FETCH_MAX_BYTES", d.fetch_max_bytes)),
            allow_private_urls=_env_bool("SCOUT_ALLOW_PRIVATE_URLS", d.allow_private_urls),
            user_agent=os.environ.get("SCOUT_USER_AGENT", d.user_agent),
        )
