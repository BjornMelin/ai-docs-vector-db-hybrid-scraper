---
title: Use AI Docs from the CLI and MCP
audience: users
status: active
owner: product-education
last_reviewed: 2026-07-12
meta:
  contentType: How-to
  category: Getting started
---

# Use AI Docs from the CLI and MCP

Use the `ai-docs` command for terminal workflows. Invoke Model Context Protocol (MCP) tools through a connected client such as Claude Desktop or Claude Code.

## Choose the correct interface

The two command surfaces serve different callers:

| Interface | Use it for | Invocation |
| --- | --- | --- |
| `ai-docs` CLI | Configuration and collection administration | Run `uv run ai-docs …` in a terminal |
| MCP tools | Agent-driven crawling, ingestion, retrieval, and health checks | Ask the connected MCP client to invoke a named tool |

The dependency-provided `mcp` shell command manages MCP development servers. It doesn't expose this project's document or search tools.

## Ingest URLs from the terminal

Put one URL per line in `urls.txt`, then run the bulk ingestion module:

```bash
uv run python -m src.crawl4ai_bulk_embedder \
  --file urls.txt \
  --collection documents
```

The `ai-docs batch index-documents` command is reserved and does not persist
documents yet. Use the MCP `add_document` and `add_documents_batch` tools below
when an agent should ingest URLs.

List collections from the terminal:

```bash
uv run ai-docs database list --format table
```

The `ai-docs database search` command is also reserved. Search through the API
or an MCP client until its implementation lands.

## Search through an MCP client

Invoke the `search_documents` tool with a typed request:

```json
{
  "request": {
    "query": "authentication configuration",
    "collection": "documents",
    "limit": 5,
    "include_metadata": true
  }
}
```

Use the `filtered_search` tool with the same request shape when you need structured payload filters:

```json
{
  "request": {
    "query": "production deployment",
    "collection": "documents",
    "limit": 10,
    "filters": {
      "category": "operations"
    }
  }
}
```

## Ingest a URL through an MCP client

Invoke `add_document` to crawl, chunk, embed, and persist one URL:

```json
{
  "request": {
    "url": "https://docs.example.com/getting-started",
    "collection": "documents",
    "chunk_size": 1600,
    "chunk_overlap": 200
  }
}
```

Invoke `add_documents_batch` for multiple URLs:

```json
{
  "request": {
    "urls": [
      "https://docs.example.com/getting-started",
      "https://docs.example.com/deployment"
    ],
    "collection": "documents",
    "max_concurrent": 2
  }
}
```

## Crawl without indexing

Invoke `enhanced_5_tier_crawl` when you need normalized page content without vector persistence:

```json
{
  "url": "https://docs.example.com/changelog",
  "timeout_ms": 30000
}
```

Omit `tier` to let the router select a provider. Set it only when you have evidence that a specific provider is required.

## List collections through an MCP client

Invoke `list_collections` with an empty argument object:

```json
{}
```

The response includes collection names, point counts, and vector metadata when Qdrant provides them.

## Change the retrieval mode

Retrieval mode is a startup setting, not a per-command shell flag. Set it in `.env`, then restart the API or MCP server:

```dotenv
AI_DOCS_EMBEDDING__RETRIEVAL_MODE=hybrid
```

Use `dense` for the default dense-vector path, `sparse` for lexical signals, or `hybrid` for both configured vector modalities.

## Inspect the live surface

Use CLI help for terminal commands:

```bash
uv run ai-docs --help
uv run ai-docs database --help
uv run ai-docs batch --help
```

Your MCP client displays the registered tool schemas after it connects. Use [Set up AI Docs locally](./quick-start.md) to configure the transport and [Troubleshooting](./troubleshooting.md) when a provider or service fails.
