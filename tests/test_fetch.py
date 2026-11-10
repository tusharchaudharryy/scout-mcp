import httpx2
import pytest

from scout_mcp.fetch import FetchError, check_url, fetch_page, html_to_text

PUBLIC_IP = "93.184.215.14"


def public_resolver(host: str) -> list[str]:
    return {"localhost": ["127.0.0.1"], "intranet.example": ["10.1.2.3"]}.get(host, [PUBLIC_IP])


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost:6379/",
        "http://intranet.example/wiki",
        "http://10.0.0.5/",
        "http://192.168.1.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
    ],
)
def test_non_public_addresses_are_blocked(url):
    with pytest.raises(FetchError, match="non-public address"):
        check_url(url, allow_private=False, resolver=public_resolver)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", "https://user:pw@example.com/"])
def test_other_schemes_and_credentials_are_blocked(url):
    with pytest.raises(FetchError):
        check_url(url, allow_private=False, resolver=public_resolver)


def test_public_url_passes_and_private_can_be_allowed():
    check_url("https://example.com/page", allow_private=False, resolver=public_resolver)
    check_url("http://localhost:8080/", allow_private=True, resolver=public_resolver)


def test_html_to_text_keeps_structure_and_drops_boilerplate():
    title, text = html_to_text(
        "<html><head><title>Weevil &amp; palms</title><style>p{}</style></head><body>"
        "<nav>Home | About</nav><h1>Detection</h1><p>Acoustic <b>sensors</b> work.</p>"
        "<ul><li>VOC</li><li>Thermal</li></ul><script>alert('x')</script><footer>(c) site</footer></body></html>"
    )
    assert title == "Weevil & palms"
    assert text == "# Detection\n\nAcoustic sensors work.\n\n- VOC\n- Thermal"


def client_for(routes: dict[str, httpx2.Response]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return routes[str(request.url)]

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def fetch(client, url, **kw):
    opts = dict(max_chars=10_000, max_bytes=100_000, allow_private=False, resolver=public_resolver)
    return await fetch_page(client, url, **{**opts, **kw})


@pytest.mark.anyio
async def test_fetch_follows_public_redirects_and_extracts_text():
    client = client_for({
        "https://example.com/old": httpx2.Response(301, headers={"location": "/new"}),
        "https://example.com/new": httpx2.Response(
            200, headers={"content-type": "text/html; charset=utf-8"},
            content="<title>New</title><p>Palm café notes</p>".encode(),
        ),
    })
    page = await fetch(client, "https://example.com/old")
    assert (page.final_url, page.title, page.text, page.truncated) == ("https://example.com/new", "New", "Palm café notes", False)


@pytest.mark.anyio
async def test_redirect_to_a_private_address_is_blocked():
    client = client_for({"https://example.com/": httpx2.Response(302, headers={"location": "http://127.0.0.1:8080/admin"})})
    with pytest.raises(FetchError, match="non-public address"):
        await fetch(client, "https://example.com/")


@pytest.mark.anyio
async def test_long_pages_are_truncated():
    client = client_for({"https://example.com/big": httpx2.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 5000)})
    page = await fetch(client, "https://example.com/big", max_chars=1000)
    assert page.truncated and len(page.text) == 1000 and page.total_chars == 5000
    capped = await fetch(client, "https://example.com/big", max_bytes=2000)
    assert capped.truncated and capped.total_chars == 2000


@pytest.mark.anyio
async def test_binary_content_and_http_errors_are_reported():
    client = client_for({
        "https://example.com/a.pdf": httpx2.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"),
        "https://example.com/gone": httpx2.Response(404),
    })
    with pytest.raises(FetchError, match="Unsupported content type"):
        await fetch(client, "https://example.com/a.pdf")
    with pytest.raises(FetchError, match="HTTP 404"):
        await fetch(client, "https://example.com/gone")


@pytest.mark.anyio
async def test_redirect_loops_stop():
    client = client_for({"https://example.com/loop": httpx2.Response(302, headers={"location": "/loop"})})
    with pytest.raises(FetchError, match="Too many redirects"):
        await fetch(client, "https://example.com/loop")
