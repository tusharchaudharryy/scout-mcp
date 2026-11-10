# scout-mcp

[![CI](https://github.com/tusharchaudharryy/scout-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/tusharchaudharryy/scout-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Two parts in one repository:

**`scout-mcp`**, an [MCP](https://modelcontextprotocol.io) server that gives AI agents three research
abilities:

- **Private web search** through a self-hosted [SearXNG](https://docs.searxng.org) instance, with no
  search API key and no tracking.
- **Safe page fetching** that turns a web page into clean, readable text and refuses private or
  internal addresses.
- **Search over your own notes**: BM25 ranking over paragraph-bound Markdown chunks, so the model sees
  whole paragraphs with their section heading.

**`scout`**, a research agent that uses those tools to answer questions with citations that are checked
against what it actually read (see [Research agent](#research-agent)).

The server works with any MCP client, such as the MCP Inspector, VS Code or Cursor.

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

## Research agent

`scout ask` answers a question with cited sources. It starts scout-mcp as a subprocess and talks to it
over MCP, like any other client.

```
question -> planner (LLM) -> sub-questions
         -> workers, one per sub-question, in parallel: LLM + scout-mcp tools in a loop
         -> writer (LLM) -> answer with [S1]-style labels
         -> citation check (code) -> answer + sources list
```

- **Only what was read can be cited.** Every page the agent fetches and every note passage it reads
  gets a label such as `[S3]`. Search snippets get no label: the agent has to fetch a page before
  citing it.
- **Citations are checked by code, not by the model.** Labels that match nothing are flagged, the share
  of claims (sentences and bullets) that carry a label is measured, and the sources list is built from
  what was read, so the model cannot invent a URL.
- **Every run leaves a trace.** Each LLM call, tool call, result size and token count is appended to a
  JSON Lines file in `runs/`, for debugging and for the evaluation suite.
- **Budgets.** Sub-questions (default 3), tool rounds per worker (6), page text per tool call (5,000
  characters) and parallel workers (2) are capped. A worker that runs out of steps must answer from
  what it has read. Rate limits and server errors are retried, honouring `Retry-After`.
- **Failures stay contained.** A worker whose LLM calls fail is reported and the others carry on; bad
  tool arguments and tool errors go back to the model so it can correct itself.

### Choose an LLM

Any provider with an OpenAI-compatible chat-completions API and tool calling works. Set three variables:

| Provider | `SCOUT_LLM_BASE_URL` | `SCOUT_LLM_MODEL` (example) | `SCOUT_LLM_API_KEY` |
|---|---|---|---|
| Groq (free tier) | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` | from console.groq.com |
| Gemini (free tier) | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-3.8-flash` | from aistudio.google.com |
| OpenAI | `https://api.openai.com/v1` (the default) | a current chat model with tool calling | from platform.openai.com |
| Ollama (local) | `http://localhost:11434/v1` | a local model that supports tools | not needed |

Model names change often: pick a current one from the provider's model list that supports tool calling.

### Ask a question

```powershell
$env:SCOUT_LLM_BASE_URL = "https://api.groq.com/openai/v1"   # bash: export SCOUT_LLM_BASE_URL=...
$env:SCOUT_LLM_MODEL = "openai/gpt-oss-120b"
$env:SCOUT_LLM_API_KEY = "<your key>"
uv run scout ask "How do acoustic sensors detect red palm weevil?"
```

The answer, with its sources list, goes to stdout. A summary goes to stderr; this one is from a test
run with a scripted model:

```
-- 2 sub-questions | 2 sources read, 2 cited | citation coverage 100% (3/3 claims) | tokens 600 in / 120 out | 0.2 s
-- trace: runs/20261111-000650-what-is-mcp-and-how-does-stdio-work.jsonl
```

Options: `--max-subquestions`, `--max-steps`, `--max-page-chars`, `--concurrency`, `--runs-dir` and
`--json` (the whole result as JSON). Without SearXNG the agent still works from your notes; web search
then returns an error that the agent sees and works around. The API key is never passed to the MCP
server subprocess and never written to traces.

## Configuration

Server:

| Variable | Default | Meaning |
|---|---|---|
| `SEARXNG_URL` | `http://localhost:8080` | SearXNG base URL |
| `SCOUT_DOCS_DIR` | `data/docs` in this project | Folder with your notes |
| `SCOUT_HTTP_TIMEOUT` | `15` | Seconds before a search or fetch gives up |
| `SCOUT_FETCH_MAX_BYTES` | `2000000` | Maximum bytes downloaded per page |
| `SCOUT_ALLOW_PRIVATE_URLS` | `false` | Allow `fetch_page` to reach localhost and private networks |
| `SCOUT_USER_AGENT` | `scout-mcp/0.1 (+https://github.com/tusharchaudharryy/scout-mcp)` | User-Agent sent with fetches and searches. Keep a URL or contact in it: sites such as Wikipedia block anonymous bots |
| `SCOUT_LOG_LEVEL` | `INFO` (`WARNING` when started by the agent) | Server log level; logs go to stderr |

Agent (besides the three `SCOUT_LLM_*` variables above):

| Variable | Default | Meaning |
|---|---|---|
| `SCOUT_LLM_TIMEOUT` | `60` | Seconds before an LLM call gives up |
| `SCOUT_LLM_MAX_RETRIES` | `3` | Retries for rate limits (HTTP 429), server errors and timeouts |
| `SCOUT_LLM_TEMPERATURE` | `0.2` | Sampling temperature |

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
src/scout_agent/
  cli.py          `scout ask` command
  orchestrator.py planner, parallel workers, writer
  worker.py       tool loop for one sub-question
  toolbox.py      MCP client: connects to scout-mcp, converts tool schemas
  llm.py          OpenAI-compatible chat client with tool calling and retries
  sources.py      labels for everything a run has read
  citations.py    citation check and sources list
  trace.py        JSON Lines trace of a run
tests/        unit tests, end-to-end tests through the MCP protocol, agent tests with a scripted LLM
searxng/      SearXNG settings used by docker-compose.yml
data/docs/    your notes
```

## Roadmap

1. **Evaluation suite.** A fixed set of test questions run through `scout ask`, scored from the run
   traces: citation coverage, invented labels, cost and latency, plus an LLM judge measured against
   human labels, following Hamel Husain's [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/).
2. **Learning from feedback.** Store the user's corrections and feed them back into later runs.
3. **Better retrieval.** Hybrid search (BM25 + embeddings with rank fusion), and PDF support.

## Credits

Built with the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), starting
from the [Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server) guide.

## License

[MIT](LICENSE) (c) 2026 Tushar Chaudhary
