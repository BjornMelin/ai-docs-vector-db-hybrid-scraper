# Set up and configure AI Docs

This guide consolidates environment preparation, profile selection, and
configuration management for the AI Docs platform.

## 1. Prerequisites

| Tool                 | Windows                                                               | macOS (Homebrew)                                         | Ubuntu/Debian                                               |
| -------------------- | --------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| Python 3.11          | `choco install python311`                                              | `brew install python@3.11`                               | `sudo apt install python3.11 python3.11-venv python3.11-dev` |
| uv (package manager) | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"`        | `curl -LsSf https://astral.sh/uv/install.sh \| sh`      | `curl -LsSf https://astral.sh/uv/install.sh \| sh`         |
| Docker + Compose     | Docker Desktop installer                                              | Docker Desktop installer                                 | `sudo apt install docker.io docker-compose-plugin`          |
| Git                  | Git installer                                                         | `brew install git`                                       | `sudo apt install git`                                      |

## 2. Repository Setup

```bash
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
uv sync --dev --frozen
cp .env.example .env
```

`uv sync` respects the lockfile and creates the virtual environment. Supply API
keys in `.env` before starting the stack.

BGE reranking is optional because its PyTorch stack substantially increases the
installation and container size. Enable it explicitly when needed:

```bash
uv sync --dev --frozen --extra reranking
```

Then set `AI_DOCS_RERANKING__ENABLED=true`. The development group already
includes `browser-use` for adapter contract tests. A non-development install
needs `uv sync --frozen --no-dev --extra agentic-browser` only when enabling the
`browser_use` provider.

## 3. Application Configuration

The API server runs with one unified configuration. Every deployment exposes the same FastAPI surface, simplifying integration testing and automation scripts.

Feature flags and nested configuration models are resolved during startup via the dependency-injector container. Health status for each registered service remains available from `/health`, and `/features` exposes the resolved flag values for observability dashboards.

## 4. Configuration Loader

`src/config/loader.Settings` is a Pydantic `BaseSettings` class that reads
configuration from environment variables. Key behaviours:

- Nested keys use double underscores (e.g. `AI_DOCS_QDRANT__URL`).
- `.env` is loaded automatically for local development.
- Runtime settings are immutable by convention; restart or refresh the settings cache after an environment change.
- Defaults favour local development; production deployments override cache, database, monitoring, and observability sections as needed.

### Core Sections

| Section                        | Model                   | Notes                                                |
| ------------------------------ | ----------------------- | ---------------------------------------------------- |
| `cache`                        | `CacheConfig`           | Controls local + distributed caches, TTLs, eviction. |
| `database`                     | `DatabaseConfig`        | Postgres connection settings; optional for lightweight deployments. |
| `qdrant`                       | `QdrantConfig`          | Vector store URL, API key, collection defaults.      |
| `agentic`                      | `AgenticConfig`         | LangGraph runner budgets (parallelism, timeouts).    |
| `query_processing`             | `QueryProcessingConfig` | Retrieval knobs (hybrid ratios, rerank budgets).     |
| `browser`                       | `BrowserAutomationConfig` | Browser automation provider settings.                |
| `monitoring` / `observability` |                         | Prometheus + OpenTelemetry exporters.                |
| `security`                     | `SecurityConfig`        | Rate limiting, CSP, feature flags.                   |

Refer to `src/config/models.py` for full schema definitions.

### Common Overrides

```bash
# Point to managed Qdrant
export AI_DOCS_QDRANT__URL=https://qdrant.internal:6333
export AI_DOCS_QDRANT__API_KEY=your_qdrant_api_key_here

# Enable Firecrawl provider
export AI_DOCS_CRAWL_PROVIDER=firecrawl
export AI_DOCS_BROWSER__FIRECRAWL__API_KEY=your_firecrawl_api_key_here

# Tighten agentic runtime budgets
export AI_DOCS_AGENTIC__MAX_PARALLEL_TOOLS=2
export AI_DOCS_AGENTIC__RUN_TIMEOUT_SECONDS=45
```

### Loading configuration files

The CLI validates JSON and YAML files through the shared
`load_settings_from_file` helper. Generate and activate a packaged profile:

```bash
uv run ai-docs setup
```

Export resolved settings when you need a YAML file, then validate that file:

```bash
uv run ai-docs config export --format yaml --output /tmp/ai-docs.yaml
uv run ai-docs config load /tmp/ai-docs.yaml --validate-only
```

Use `uv run ai-docs config export --format json` to snapshot the current
settings as JSON.

### Secrets

Keep API keys out of the repository and inject them through environment variables or your orchestrator's secret manager. `AI_DOCS_OPENAI__API_KEY` maps directly to `Settings.openai.api_key`.

### Refreshing Settings

Hot reloading has been removed. When configuration changes are required, refresh settings through the `/api/v1/config/refresh` API or restart the process to ensure a clean environment.

## 5. Running Services

```bash
# Launch core services
docker compose --profile simple up -d

# Check status
docker compose ps

# Tail logs
docker compose logs -f app
```

## 6. Validation

```bash
uv run pytest -q                    # unit tests
uv run ruff check .                 # lint
uv run ruff format --check .        # formatting
```

Re-run the test and lint suites after modifying configuration models or adding
new environment variables.

## 7. Service Access Patterns

- **Client access**: FastAPI, MCP, and CLI layers resolve services via dependency helpers in `src/services/service_resolver.py` and `src/services/fastapi/dependencies.py`, backed by the global `ApplicationContainer`.
- Avoid constructing bespoke managers; call `initialize_container()` once at startup and retrieve providers through `Provide[...]` or `get_container()`.
- **Observability**: Configure traces through `src/services/observability/`. The FastAPI middleware manager calls `setup_prometheus` when metrics are enabled.
- **Health checks**: `HealthCheckManager` in `src/services/observability/health_manager.py` owns service probes used by the public health endpoint.
