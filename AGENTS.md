# Repository Guidelines

## Project Structure & Module Organization

Runtime code lives in `src/` (FastAPI `api`, LangGraph `services`, CLI `cli`, MCP tooling `mcp_services`/`mcp_tools`). Configs and deploy assets: `config/`, `docker-compose.yml`, `k8s/`; automation lives in `scripts/`; docs in `docs/`; tests span unit, integration, performance,
property, data_quality, scripts suites. `src/services` highlights: agents (LangGraph flows), analytics (search dashboards), browser (tiered scraping), cache (Dragonfly caches), circuit_breaker (resilience), content_intelligence (classifiers/metadata), core (project storage & Qdrant aliases),
crawling (Firecrawl/Crawl4AI), embeddings (FastEmbed/OpenAI), fastapi (dependency wiring), functional (DI facades), hyde (HyDE augmentation), managers (platform managers), middleware (rate limiting), monitoring (metrics/health/telemetry), observability (tracing configs),
processing (batch pipelines), query_processing (intent models + orchestrator), rag (pipeline/generator/retriever), utilities (HNSW tuning/rate limits), vector_db (Qdrant service layer).

## Build, Test, and Development Commands

Install deps via `uv sync --dev`. Drive daily flows with `uv run python -m src.cli.unified …` (`dev --mode simple`, `docs`, `services --action start --stack vector|monitoring`). Use `python scripts/dev.py` for profiles: `test --profile quick`, `eval --dataset tests/data/rag/golden_set.jsonl`,
`benchmark --suite performance`, `validate --check-docs --check-services`, `quality --fix-lint`. Targeted gates: `uv run ruff format/check <paths>`, `uv run pyright <paths>`, `uv run pylint <modules>` (score ≥9.5). Start local services
with `docker compose --profile simple up -d`; `make quality-unit` yields a lint/type/test gate.

## Coding Style & Naming Conventions

Use 4-space indentation, full type hints, and Google-style docstrings on public APIs. Keep modules lowercase_with_underscores, classes PascalCase, and functions/tests snake_case. Favor maintained components (LangGraph nodes, Prometheus helpers, FastEmbed) over bespoke utilities,
and run `uv run ruff format .`, `uv run ruff check . --fix`, `uv run pylint --fail-under=9.5 src tests`, and `uv run pyright src tests` before pushing.

## Testing Guidelines

Pytest modules follow `test_*.py`; fixtures and corpora live in `tests/fixtures/` and `tests/data_quality/`. Run `python scripts/dev.py test --profile unit` (or other profiles) or `uv run pytest -q tests/integration/rag/test_pipeline.py` when iterating.
Tag flows with `pytest.ini` markers such as `@pytest.mark.integration`, `@pytest.mark.performance`, `@pytest.mark.rag`, and keep runs deterministic via stubs and cached embeddings.

## Commit & Pull Request Guidelines

Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`), explain motivation, link issues, list validation commands, and confirm `python scripts/dev.py quality`, relevant pytest profiles, and required Docker checks pass before review—update `CHANGELOG.md` or docs for behaviour shifts.

## Security & Configuration Tips

Store secrets in `.env`, `.env.local`, or a vault—never commit `OPENAI_API_KEY` or `AI_DOCS__FIRECRAWL__API_KEY`. Align FastAPI and MCP via `AI_DOCS__MODE`, `AI_DOCS__QDRANT__URL`, and `FASTMCP_TRANSPORT/FASTMCP_HOST/FASTMCP_PORT`.
Prefer precise exceptions, justify any `# pylint: disable` or `# pyright: ignore`, trim `cache/`, `logs/`, `tmp/`, and run `python scripts/dev.py validate --strict` regularly.

## Error Handling & Observability

Prefer precise exception classes (`ValueError`, `HTTPStatusError`, etc.) over
generic `Exception`. When suppression is unavoidable, justify it with targeted
`# pylint: disable=` or `# pyright: ignore` comments. Inject structured logging
and tracing where it improves operability—import the helpers from
`src/services/logging_config.py` and emit actionable messages (context, impact,
next steps). Use existing OpenTelemetry utilities under `src/services/monitoring`
to attach spans/metrics when expanding service boundaries.

## DISALLOWED GIT COMMANDS

- You must NEVER run the any `git restore` or `git checkout --` commands unless explicitly asked to, never restore or revert changes unles you have been explicitly asked to by the user.
