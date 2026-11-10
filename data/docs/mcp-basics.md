# MCP basics

Example note so `search_docs` works out of the box. Replace the files in this folder with your own
notes (.md, .markdown or .txt), or point `SCOUT_DOCS_DIR` at another folder.

## Server capabilities

An MCP server can offer three kinds of capabilities. Tools are functions the model can call, usually
with the user's approval. Resources are file-like data a client can read, such as API responses or file
contents. Prompts are reusable templates that help the user start a task.

## Transports

This server uses the stdio transport: the client starts the server as a subprocess and they exchange
JSON-RPC messages over stdin and stdout. Because stdout carries the protocol, the server must never
print to it; logs go to stderr instead.
