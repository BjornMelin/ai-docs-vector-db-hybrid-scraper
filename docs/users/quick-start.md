---
title: Set up AI Docs locally
audience: users
status: active
owner: product-education
last_reviewed: 2026-07-11
meta:
  contentType: Tutorial
  category: Getting started
---

# Set up AI Docs locally

Install the locked Python environment, start Qdrant, and run the FastAPI application with the default FastEmbed provider.

## Prerequisites

Install these tools before you clone the repository:

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Git
- Docker Engine or Docker Desktop with Compose

You don't need a provider credential for the default FastEmbed and Crawl4AI configuration. Add a credential only when you select its provider.

## Install the project

Clone the repository and create its locked environment:

```bash
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
uv python install 3.11
uv sync --dev --frozen
cp .env.example .env
```

Install Crawl4AI's browser assets when you plan to crawl web pages:

```bash
uv run crawl4ai-setup
```

## Configure the application

`Settings` reads `.env` and variables with the `AI_DOCS_` prefix. Nested fields use a double underscore.

The copied template starts with these local defaults:

```dotenv
AI_DOCS_ENVIRONMENT=development
AI_DOCS_EMBEDDING_PROVIDER=fastembed
AI_DOCS_CRAWL_PROVIDER=crawl4ai
AI_DOCS_QDRANT__URL=http://localhost:6333
```

Select OpenAI embeddings only when you have configured its application key:

```dotenv
AI_DOCS_EMBEDDING_PROVIDER=openai
AI_DOCS_OPENAI__API_KEY=your_openai_api_key_here
```

Select Firecrawl only when you have configured its browser-provider key:

```dotenv
AI_DOCS_CRAWL_PROVIDER=firecrawl
AI_DOCS_BROWSER__FIRECRAWL__API_KEY=your_firecrawl_api_key_here
```

## Start Qdrant

Start the vector database without building the application image:

```bash
docker compose --profile simple up -d qdrant
docker compose ps qdrant
```

Qdrant stores its data in the `qdrant_data` volume.

## Run the API

Start FastAPI from the locked environment:

```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Verify the public health endpoint from another terminal:

```bash
curl --fail http://localhost:8000/health
```

Open `http://localhost:8000/docs` to inspect the generated OpenAPI interface.

## Run the MCP server

The checked-in Claude Desktop configuration launches the Model Context Protocol (MCP) server over stdio. Copy `config/claude-mcp-config.example.json` into your Claude settings, replace its `cwd` value, and let Claude start the process.

To run the HTTP transport beside FastAPI, use port 8001:

```bash
FASTMCP_TRANSPORT=streamable-http FASTMCP_PORT=8001 \
  uv run python src/unified_mcp_server.py
```

The MCP endpoint is `http://127.0.0.1:8001/mcp`. Stop the process with `Ctrl+C`.

Set `FASTMCP_TRANSPORT`, `FASTMCP_HOST`, and `FASTMCP_PORT` in the process environment when you need another transport or bind address.

## Validate the installation

Run the focused unit profile before changing code:

```bash
uv run python scripts/dev.py test --profile quick
```

Print a non-sensitive resolved value when configuration doesn't load as expected:

```bash
uv run python -c 'from src.config.loader import Settings; print(Settings().environment.value)'
```

Read [Configure AI Docs](../operators/configuration.md) for the supported environment surface. Use [Troubleshooting](./troubleshooting.md) when a service or provider doesn't start.
