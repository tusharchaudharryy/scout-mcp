"""Fetch a web page and reduce it to readable text.

The fetcher is reachable by a model, and a model can be steered by text on the pages it reads
(prompt injection). So by default it only fetches public addresses: every URL, including each
redirect hop, is resolved and rejected if any address is loopback, private, link-local or reserved.
This stops the classic "fetch http://169.254.169.254/ or http://localhost:6379/" attacks. It does not
stop DNS rebinding (the HTTP client resolves the name again); run the server on a machine where
that risk is acceptable, or put it behind an egress proxy.
"""

from __future__ import annotations

import codecs
import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit

import httpx2
from pydantic import BaseModel

from .errors import ResearchError
from .html_text import html_to_text, tidy

Resolver = Callable[[str], list[str]]

_TEXTUAL_TYPES = {"application/xhtml+xml", "application/json", "application/xml"}
_HTML_TYPES = {"text/html", "application/xhtml+xml"}


class FetchError(ResearchError):
    """The page could not be fetched or is not allowed."""


class FetchedPage(BaseModel):
    url: str
    final_url: str
    title: str
    content_type: str
    text: str
    truncated: bool
    total_chars: int


def resolve_host(host: str) -> list[str]:
    """Return every address the host name resolves to."""
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def _ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return ip.ipv4_mapped
    return ip


def check_url(url: str, *, allow_private: bool, resolver: Resolver = resolve_host) -> None:
    """Raise FetchError unless `url` is an http(s) URL that points at public addresses only."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("Only http:// and https:// URLs can be fetched.")
    if not parts.hostname:
        raise FetchError("The URL has no host name.")
    if parts.username or parts.password:
        raise FetchError("URLs that contain credentials are not allowed.")
    if allow_private:
        return
    host = parts.hostname
    literal = _ip(host)
    if literal is not None:
        addresses = [literal]
    else:
        try:
            addresses = [ip for ip in (_ip(a) for a in resolver(host)) if ip is not None]
        except OSError as exc:
            raise FetchError(f"Could not resolve the host {host!r}.") from exc
        if not addresses:
            raise FetchError(f"Could not resolve the host {host!r}.")
    for ip in addresses:
        if not ip.is_global:
            raise FetchError(
                f"Refusing to fetch {host}: it points at a non-public address ({ip}). "
                "Set SCOUT_ALLOW_PRIVATE_URLS=true to allow local and private addresses."
            )


async def _read_capped(resp: httpx2.Response, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    async for chunk in resp.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= max_bytes:
            return b"".join(chunks)[:max_bytes], True
    return b"".join(chunks), False


def _codec(name: str | None) -> str:
    try:
        return codecs.lookup(name or "utf-8").name
    except LookupError:
        return "utf-8"


async def fetch_page(
    client: httpx2.AsyncClient,
    url: str,
    *,
    max_chars: int,
    max_bytes: int,
    allow_private: bool,
    resolver: Resolver = resolve_host,
    max_redirects: int = 5,
) -> FetchedPage:
    """Fetch `url` (following redirects, each one re-checked) and return its readable text."""
    current = url
    for _ in range(max_redirects + 1):
        check_url(current, allow_private=allow_private, resolver=resolver)
        try:
            async with client.stream("GET", current, follow_redirects=False) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError(f"{current} redirected without a Location header.")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    raise FetchError(f"{current} returned HTTP {resp.status_code}.")
                content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type and not (content_type.startswith("text/") or content_type in _TEXTUAL_TYPES):
                    raise FetchError(
                        f"Unsupported content type {content_type!r} at {current}; "
                        "only HTML and plain-text pages are supported."
                    )
                body, cut = await _read_capped(resp, max_bytes)
                encoding = _codec(resp.charset_encoding)
        except httpx2.TimeoutException as exc:
            raise FetchError(f"Timed out fetching {current}.") from exc
        except httpx2.HTTPError as exc:
            raise FetchError(f"Could not fetch {current} ({type(exc).__name__}).") from exc

        raw = body.decode(encoding, errors="replace")
        if content_type in _HTML_TYPES or (not content_type and "<html" in raw[:2000].lower()):
            title, text = html_to_text(raw)
        else:
            title, text = "", tidy(raw)
        return FetchedPage(
            url=url,
            final_url=current,
            title=title,
            content_type=content_type or "unknown",
            text=text[:max_chars],
            truncated=cut or len(text) > max_chars,
            total_chars=len(text),
        )
    raise FetchError(f"Too many redirects (more than {max_redirects}) starting from {url}.")
