# Unified Configuration Guide

The application reads configuration through `src/config/loader.Config`, a
Pydantic `BaseSettings` class that maps environment variables into typed
sections. Settings are grouped under the `AI_DOCS__` prefix and support nested
keys via double underscores (e.g. `AI_DOCS__QDRANT__URL`).

## Loading Behaviour

- `.env` at the project root is loaded automatically when `Config()` is
  instantiated.
- `env_nested_delimiter="__"` allows nested structures such as
  `AI_DOCS__CACHE__REDIS_URL`.
- `validate_assignment=True` ensures runtime overrides stay type-safe.
- Default values target the **simple** profile (local development) but can be
  overridden for enterprise deployments.

```python
from src.config.loader import Config

config = Config()
print(config.mode)           # ApplicationMode.SIMPLE by default
print(config.qdrant.url)     # http://localhost:6333 in simple mode
print(config.cache.redis_url)
```

## Core Sections

| Section                        | Model                   | Purpose                                                   |
| ------------------------------ | ----------------------- | --------------------------------------------------------- |
| `cache`                        | `CacheConfig`           | Cache backends, eviction policies, TTLs.                  |
| `database`                     | `DatabaseConfig`        | Relational database connection settings.                  |
| `qdrant`                       | `QdrantConfig`          | Vector store URL, API key, collection defaults.           |
| `agentic`                      | `AgenticConfig`         | LangGraph runner limits (parallel tools, timeouts).       |
| `query_processing`             | `QueryProcessingConfig` | Retrieval pipeline knobs (hybrid ratios, rerank budgets). |
| `playwright` / `browser_use`   |                         | Automation stacks for the five-tier browser manager.      |
| `monitoring` / `observability` |                         | Prometheus, OpenTelemetry, log exporters.                 |
| `security`                     | `SecurityConfig`        | Rate limiting, CSP, feature flags for sensitive modes.    |

Refer to `src/config/models.py` for the complete schema and validation rules.

## Profiles and Modes

`Config.mode` stores the active `ApplicationMode` (`simple` or `enterprise`). The
FastAPI dependency layer reads this value to select the correct routers and
service installers (see `docs/developers/app-profiles.md`). Differences between
modes include:

- **Simple**: local Qdrant + filesystem cache, no Redis/Postgres requirement.
- **Enterprise**: expects Redis/Dragonfly, Postgres, Prometheus, and additional
  routers; see `docker-compose.yml` for profile-specific services.

Switch modes via environment:

```bash
export AI_DOCS__MODE=enterprise
uv run python -m src.cli.unified dev --mode enterprise
```

## Common Overrides

```bash
# Point to managed Qdrant
export AI_DOCS__QDRANT__URL=https://qdrant.internal:6333
export AI_DOCS__QDRANT__API_KEY=***

# Enable Firecrawl provider
export AI_DOCS__CRAWL_PROVIDER=firecrawl
export AI_DOCS__FIRECRAWL__API_KEY=***

# Tighten agentic runtime budgets
export AI_DOCS__AGENTIC__MAX_PARALLEL_TOOLS=2
export AI_DOCS__AGENTIC__RUN_TIMEOUT_SECONDS=45
```

Nested sections share a consistent naming scheme. For example, cache tuning for
Dragonfly lives under `AI_DOCS__CACHE__DRAGONFLY__*`, while the HTTP retry
policy for MCP clients sits under `AI_DOCS__MCP_CLIENT__RETRY__*`.

## Secrets and Sensitive Values

- Prefer injecting secrets via environment variables supplied by your orchestrator.
- Local development uses `.env`; exclude it from commits.
- The config loader mirrors top-level API keys into nested sections (e.g.
  setting `AI_DOCS__OPENAI_API_KEY` also populates `config.openai.api_key`).
- For enterprise deployments integrate with your secret manager and export
  environment variables at container start.

## Hot Reloading

`src/config/reloader.py` provides a simple watcher that reloads configuration on
file changes. It is used in local development but should be disabled in
production environments to avoid surprise restarts.

---

Keep this document in sync with the Pydantic models and update it whenever new
sections or prefixes are introduced.
