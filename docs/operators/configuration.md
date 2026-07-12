---
title: Configure AI Docs
audience: operators
status: active
owner: platform-engineering
last_reviewed: 2026-07-11
meta:
  contentType: Reference
  category: Operations
---

# Configure AI Docs

Use the Pydantic `Settings` model for application configuration. `.env.example` is the executable environment template, and `src/config/loader.py` plus `src/config/models.py` define the complete schema.

## Environment variable format

Application variables follow two forms:

- Top-level fields: `AI_DOCS_<FIELD>`
- Nested fields: `AI_DOCS_<SECTION>__<FIELD>`

For example, `AI_DOCS_QDRANT__URL` maps to `Settings.qdrant.url`. Don't insert a double underscore after `AI_DOCS`.

## Core runtime settings

These settings select the environment and providers:

| Variable | Model default | Purpose |
| --- | --- | --- |
| `AI_DOCS_MODE` | `production` | Deployment mode label |
| `AI_DOCS_ENVIRONMENT` | `development` | Runtime environment |
| `AI_DOCS_LOG_LEVEL` | `INFO` | Application log level |
| `AI_DOCS_EMBEDDING_PROVIDER` | `fastembed` | Embedding provider |
| `AI_DOCS_CRAWL_PROVIDER` | `crawl4ai` | Crawl provider |

The local `.env.example` overrides `AI_DOCS_MODE` to `simple`.

## Provider credentials

Configure credentials only for providers you enable:

| Variable | Required when |
| --- | --- |
| `AI_DOCS_OPENAI__API_KEY` | `AI_DOCS_EMBEDDING_PROVIDER=openai` |
| `AI_DOCS_BROWSER__FIRECRAWL__API_KEY` | `AI_DOCS_CRAWL_PROVIDER=firecrawl` |
| `AI_DOCS_QDRANT__API_KEY` | The Qdrant deployment requires authentication |

Keep credentials in a local `.env` file or your deployment platform's secret store. Don't commit populated credentials.

## Storage and cache settings

Configure Qdrant and Dragonfly through their nested models:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_DOCS_QDRANT__URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `AI_DOCS_QDRANT__COLLECTION_NAME` | `documents` | Primary collection |
| `AI_DOCS_QDRANT__USE_GRPC` | `false` | Enable the Qdrant gRPC client |
| `AI_DOCS_CACHE__ENABLE_DRAGONFLY_CACHE` | `false` in `.env.example` | Enable distributed caching |
| `AI_DOCS_CACHE__DRAGONFLY_URL` | `redis://localhost:6379` | Dragonfly endpoint |
| `AI_DOCS_FASTEMBED__CACHE_DIR` | `./cache/fastembed` | Persist downloaded embedding models |

The enterprise Compose profile exposes Dragonfly at `redis://dragonfly:6379`. Set `AI_DOCS_CACHE__ENABLE_DRAGONFLY_CACHE=true` before starting that profile when the application should use it.

## Retrieval settings

Tune retrieval through the canonical nested sections:

| Variable | Default in `.env.example` |
| --- | --- |
| `AI_DOCS_EMBEDDING__RETRIEVAL_MODE` | `dense` |
| `AI_DOCS_CHUNKING__STRATEGY` | `enhanced` |
| `AI_DOCS_CHUNKING__CHUNK_SIZE` | `1600` |
| `AI_DOCS_CHUNKING__CHUNK_OVERLAP` | `200` |
| `AI_DOCS_HYDE__ENABLE_HYDE` | `false` |
| `AI_DOCS_RAG__ENABLE_RAG` | `false` |
| `AI_DOCS_RERANKING__ENABLED` | `false` |

## Validate configuration

Load the current environment without printing credentials:

```bash
uv run python -c 'from src.config.loader import Settings; print(Settings().environment.value)'
```

Run the repository validation harness after changing configuration assets:

```bash
uv run python scripts/dev.py validate --check-docs --strict
```

Run the focused contract tests after changing `Settings` or `.env.example`:

```bash
uv run pytest -q tests/unit/config/test_settings_defaults.py
```
