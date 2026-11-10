from pathlib import Path

from scout_mcp.config import Settings


def test_default_user_agent_identifies_the_project():
    # Wikipedia (and others) return 403 to anonymous bots; the UA must say where the bot comes from.
    assert "https://github.com/tusharchaudharryy/scout-mcp" in Settings().user_agent


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARXNG_URL", "http://search.local:9000/")
    monkeypatch.setenv("SCOUT_DOCS_DIR", str(tmp_path))
    monkeypatch.setenv("SCOUT_ALLOW_PRIVATE_URLS", "yes")
    monkeypatch.setenv("SCOUT_FETCH_MAX_BYTES", "1234")
    s = Settings.from_env()
    assert s.searxng_url == "http://search.local:9000"
    assert s.docs_dir == Path(tmp_path).resolve()
    assert s.allow_private_urls is True and s.fetch_max_bytes == 1234


def test_log_level_is_read_and_validated(monkeypatch):
    monkeypatch.setenv("SCOUT_LOG_LEVEL", "warning")
    assert Settings.from_env().log_level == "WARNING"
    monkeypatch.setenv("SCOUT_LOG_LEVEL", "loud")
    assert Settings.from_env().log_level == "INFO"
