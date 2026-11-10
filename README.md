# scout-mcp

An [MCP](https://modelcontextprotocol.io) server that gives AI agents three research abilities:

- **Private web search** through a self-hosted [SearXNG](https://docs.searxng.org) instance, with no
  search API key and no tracking.
- **Safe page fetching** that turns a web page into clean, readable text and refuses private or
  internal addresses.
- **Search over your own notes**: BM25 ranking over paragraph-bound Markdown chunks, so the model sees
  whole paragraphs with their section heading.

It works with any MCP client, such as the MCP Inspector, VS Code or Cursor. It is the first part of a
larger project: a research agent built on these tools, plus an evaluation suite for it (see
[Roadmap](#roadmap)).

## Tools, resource and prompt

| Name | Kind | What it does |
|---|---|---|
| `web_search(query, max_results=5, time_range=None)` | tool | Searches the web through SearXNG and returns title, URL, snippet and engines for each result |
| `fetch_page(url, max_chars=8000)` | tool | Downloads a page and returns its readable text, with title, final URL and truncation info |
| `search_docs(query, top_k=5)` | tool | Searches your notes with BM25 and returns passages with document name, section and score |
| `list_docs()` | tool | Lists your notes with size and passage count |
| `read_doc(name, max_chars=20000)` | tool | Returns the full text of one note |
| `docs://{name}` | resource | One note, for clients that let you attach resources |
| `research_brief(topic)` | prompt | A template that walks the model through search, reading and a cited brief |

All tools are read-only, take bounded inputs and return structured output (JSON with a schema).
Errors come back as tool errors that tell the model what to do next. For example, if SearXNG is down,
the error says how to start it.

## Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/). You need Docker only for web search;
the other tools work without it.

```bash
git clone https://github.com/tusharchaudharryy/scout-mcp.git
cd scout-mcp
uv sync              # create .venv and install dependencies
uv run pytest        # run the test suite (offline, a few seconds)
```

Put your notes (`.md`, `.markdown` or `.txt`) in `data/docs/`, or point `SCOUT_DOCS_DIR` at another
folder. The index rebuilds itself when files change.

### Start SearXNG (for `web_search`)

Install [Docker](https://docs.docker.com/get-started/get-docker/), set a random secret and start the
container:

```bash
export SEARXNG_SECRET="$(openssl rand -hex 32)"     # PowerShell: $env:SEARXNG_SECRET = "<long random string>"
docker compose up -d
# check it works: open http://localhost:8080/search?q=test&format=json
```

SearXNG listens on `127.0.0.1` only. `searxng/settings.yml` turns on the JSON output the server needs.

## Connect a client

**MCP Inspector**, to try the tools in a browser:

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/scout-mcp run scout-mcp
```

**Desktop and editor clients**: most MCP clients take a JSON config like this. Use the absolute path
to your clone; on Windows, write it with double backslashes (`C:\\path\\to\\scout-mcp`). If `uv` is
not found, use its full path (`which uv`, or `where uv` on Windows).

```json
{
  "mcpServers": {
    "scout": {
      "command": "uv",
      "args": ["--directory", "/path/to/scout-mcp", "run", "scout-mcp"],
      "env": { "SEARXNG_URL": "http://localhost:8080" }
    }
  }
}
```

The server speaks MCP over stdio, so `uv run scout-mcp` (or `python -m scout_mcp`) is all a client
needs to start it.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SEARXNG_URL` | `http://localhost:8080` | SearXNG base URL |
| `SCOUT_DOCS_DIR` | `data/docs` in this project | Folder with your notes |
| `SCOUT_HTTP_TIMEOUT` | `15` | Seconds before a search or fetch gives up |
| `SCOUT_FETCH_MAX_BYTES` | `2000000` | Maximum bytes downloaded per page |
| `SCOUT_ALLOW_PRIVATE_URLS` | `false` | Allow `fetch_page` to reach localhost and private networks |
| `SCOUT_USER_AGENT` | `scout-mcp/0.1 (+https://github.com/tusharchaudharryy/scout-mcp)` | User-Agent sent with fetches and searches. Keep a URL or contact in it: sites such as Wikipedia block anonymous bots |

## Design notes

- **Tool design.** Each tool has a short description written for the model, bounded parameters (for
  example `max_results` is 1-10), structured output and errors that say how to recover. Agents make
  fewer mistakes with tools that are hard to misuse.
- **Fetching safely.** A model can be steered by text on pages it reads (prompt injection), so
  `fetch_page` only allows `http`/`https`. It resolves every host, including each redirect hop, and
  refuses loopback, private, link-local and reserved addresses, such as the cloud metadata address
  169.254.169.254. It caps download size and rejects non-text content. Limitation: it does not
  defend against DNS rebinding.
- **Readable text.** The page is parsed into an element tree. If it marks a main area (`<main>`,
  `role="main"` or a single `<article>`), only that area is used, so classes on the layout wrappers
  around it cannot hide the content. Menus, sidebars, dropdowns, footers, hidden and screen-reader-only
  text, scripts and forms are dropped. A class word only counts at the start or end of a class name
  (`layout-sidebar` yes, `feature-main-menu-disabled` no). If the strict rules would remove almost
  everything, gentler ones are used instead. Checked against Wikipedia, docs.python.org, GitHub,
  hamel.dev and fao.org.
- **Chunking.** Notes are split at blank lines and merged into chunks of up to about 1,200 characters.
  A paragraph is never split, and each chunk keeps its Markdown heading, which is also indexed.
- **Ranking.** BM25 (k1 = 1.5, b = 0.75) with a small stopword list, implemented in `docs.py` with
  no extra dependencies.
- **stdio.** The client starts the server as a subprocess and talks JSON-RPC over stdin/stdout, so
  the server never prints to stdout. Logs go to stderr.

## Project layout

```
src/scout_mcp/
  server.py   MCP server: tools, resource, prompt, entry point
  search.py   SearXNG client
  fetch.py    URL safety checks and fetching
  html_text.py  HTML to readable text: main-content detection, menu and sidebar removal
  docs.py     chunking, BM25 index, note store
  config.py   settings from environment variables
tests/        unit tests + end-to-end tests through the MCP protocol (in-memory client)
searxng/      SearXNG settings used by docker-compose.yml
data/docs/    your notes
```

## Roadmap

1. **Research agent.** An orchestrator that plans sub-questions, sends worker calls through these
   tools and merges the findings with citations.
2. **Evaluation suite.** A fixed set of test questions, automatic checks (every claim cited, no
   invented URLs), trace review, and an LLM judge measured against human labels, following Hamel
   Husain's [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/).
3. **Learning from feedback.** Store the user's corrections and feed them back into later runs.
4. **Better retrieval.** Hybrid search (BM25 + embeddings with rank fusion), and PDF support.

## Credits

Built with the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), starting
from the [Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server) guide.
