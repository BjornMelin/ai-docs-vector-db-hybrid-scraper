# Setup and Configuration

This guide consolidates environment preparation, profile selection, and
configuration management for the AI Docs platform.

## 1. Prerequisites

| Tool                 | Windows                                              | macOS (Homebrew)                     | Ubuntu/Debian                                                |
| -------------------- | ---------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ | --- | ------------- |
| Python 3.11/3.12     | `choco install python --version=3.12.3`              | `brew install python@3.12`           | `sudo apt install python3.12 python3.12-venv python3.12-dev` |
| uv (package manager) | `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`                                | `curl -LsSf https://astral.sh/uv/install.sh                  | sh` | same as macOS |
| Docker + Compose     | Docker Desktop installer                             | `brew install docker docker-compose` | `sudo apt install docker.io docker-compose`                  |
| Git                  | Git installer                                        | `brew install git`                   | `sudo apt install git`                                       |

## 2. Repository Setup

```bash
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
uv sync --all-extras
cp .env.example .env
```

`uv sync` respects the lockfile and creates the virtual environment. Supply API
keys in `.env` before starting the stack.

## 3. Application Profiles

The API server exposes two profiles controlled by `AI_DOCS__MODE`:

| Profile      | Description                                                                                   | Activation                        |
| ------------ | --------------------------------------------------------------------------------------------- | --------------------------------- |
| `simple`     | Solo-developer defaults: minimal external services, same `/api/v1` surface as enterprise.     | default (`AI_DOCS__MODE=simple`)  |
| `enterprise` | Full surface area: Redis/Dragonfly, Postgres, Prometheus, extended orchestration & telemetry. | `export AI_DOCS__MODE=enterprise` |

Profiles are resolved by `AppProfile` (`src/api/app_profiles.py`). Router and
service installers consult the profile to decide which modules to mount. Health
status for each service is exposed via `/health`.

### Switching Profiles

```bash
# Simple smoke test
uv run python scripts/dev.py test --profile quick

# Enterprise validation
AI_DOCS__MODE=enterprise uv run python scripts/dev.py test --profile quick
```

## 4. Configuration Loader

`src/config/loader.Settings` is a Pydantic `BaseSettings` class that reads
configuration from environment variables. Key behaviours:

- Nested keys use double underscores (e.g. `AI_DOCS__QDRANT__URL`).
- `.env` is loaded automatically for local development.
- `validate_assignment=True` keeps runtime overrides type-safe.
- Defaults favour the developer (simple) profile; enterprise deployments override relevant
  sections (cache, database, monitoring, etc.).

### Core Sections

| Section                        | Model                   | Notes                                                |
| ------------------------------ | ----------------------- | ---------------------------------------------------- |
| `cache`                        | `CacheConfig`           | Controls local + distributed caches, TTLs, eviction. |
| `database`                     | `DatabaseConfig`        | Postgres connection settings; optional for the simple profile. |
| `qdrant`                       | `QdrantConfig`          | Vector store URL, API key, collection defaults.      |
| `agentic`                      | `AgenticConfig`         | LangGraph runner budgets (parallelism, timeouts).    |
| `query_processing`             | `QueryProcessingConfig` | Retrieval knobs (hybrid ratios, rerank budgets).     |
| `playwright` / `browser_use`   |                         | Browser automation tier settings.                    |
| `monitoring` / `observability` |                         | Prometheus + OpenTelemetry exporters.                |
| `security`                     | `SecurityConfig`        | Rate limiting, CSP, feature flags.                   |

Refer to `src/config/models.py` for full schema definitions.

### Common Overrides

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

### Loading configuration files

The CLI now understands JSON _and_ YAML configuration files via the shared
`load_settings_from_file` helper. Example:

```bash
# Validate without mutating ~/.ai-docs config
uv run python -m src.cli.main config load config/production.json --validate-only

# Load overrides into the current session context
uv run python -m src.cli.main config load config/staging.yaml
```

Use `uv run python -m src.cli.main config export --format json` to snapshot the
current in-memory settings or `--format yaml` when PyYAML is available.

### Secrets

Keep API keys out of the repository and inject them via environment variables or
your orchestrator's secret manager. Setting top-level keys such as
`AI_DOCS__OPENAI__API_KEY` automatically mirrors values into nested config
sections.

### Refreshing Settings

Hot reloading has been removed. When configuration changes are required, refresh settings through the `/config/refresh` API or restart the process to ensure a clean environment.

## 5. Running Services

```bash
# Launch developer defaults (simple profile)
docker compose --profile simple up -d

# Enterprise stack
docker compose --profile enterprise up -d

# Check status
docker compose ps

# Tail logs
docker compose logs -f app
```

## 6. Validation

```bash
uv run pytest -q                    # unit tests
uv run ruff check .                 # lint
uv run ruff format . --check        # formatting
```

Re-run the test and lint suites after modifying configuration models or adding
new environment variables.

## 7. Service Access Patterns

- **Client access**: FastAPI, MCP, and CLI layers resolve services via dependency helpers in `src/services/dependencies.py`, backed by the global `ApplicationContainer`.
- Avoid constructing bespoke managers; call `initialize_container()` once at startup and retrieve providers through `Provide[...]` or `get_container()`.
- **Observability**: Metrics, traces, and health checks are configured through `src/services/observability/` and `src/services/monitoring/`. Use `ObservabilityConfig` to enable OpenTelemetry exporters and rely on `setup_prometheus` for registry bootstrap.
- **Health checks**: The centralized `HealthCheckManager` (`src/services/monitoring/health.py`) tracks service probes and feeds `/health` endpoints. When adding new services, register probes via the manager instead of ad-hoc endpoints.
