# Repository Guidelines

## Project Structure & Module Organization

Source code lives in `src/`, organized by service area (API routers under `src/api/`,
vector logic in `src/services/vector_db/`, crawling stack in `src/services/browser/`).
Shared configuration resides in `src/config/`. Tests mirror this layout in `tests/`,
with unit, integration, and specialized suites (e.g., `tests/performance/`,
`tests/chaos/`). CLI utilities and helper scripts are collected in `scripts/`, while
documentation assets live in `docs/`.

## General Instructions

- Strict typing always
- Lazy logging always
- Aggressively clean up and deduplicate code and modules while maintaining full functionality wherever possible while working

## Build, Test, and Development Commands

- `uv sync --dev` — install Python dependencies using the repository’s locked set.
- `uv run python -m src.api.main` — launch the FastAPI service in the current mode.
- `uv run python src/unified_mcp_server.py` — start the MCP server with tool
  registration.
- `python scripts/dev.py test --profile quick` — run the fast unit/integration
  subset.
- `python scripts/dev.py quality` — execute format, lint, type-check, and full
  test gate (Ruff format/check, Pylint, Pyright).
- `uv run pylint src tests` — run static analysis; keep the global score ≥9.5.

## Quality Gate Checklist

Heavy GPU dependencies make repository-wide gates slow. For each change:

1. Identify touched Python modules using `git status --short`.
2. Run targeted static checks:
   - `uv run ruff format <paths>`
   - `uv run ruff check <paths>`
   - `uv run pylint <module_dirs>` (scores ≥9.5)
   - `uv run pyright <paths>`
3. Execute focused pytest suites for affected components (e.g.,
   `python scripts/dev.py test --profile quick` plus
   `uv run pytest tests/services/vector_db/`).
4. Update docs/CHANGELOG when behaviour changes. Run full gates for release
   branches or broad refactors.

## Coding Style & Naming Conventions

Python code follows Ruff formatting (`ruff format .`) and linting (`ruff check . --fix`)
plus Pyright static analysis via the quality gate. Maintain descriptive, typed
functions; use Google-style docstrings (module, class, function) with succinct
summaries and parameter/return sections. Keep comments technical and avoid
marketing language. Modules should expose `snake_case.py`, classes use
`PascalCase`, and constants remain uppercase. Blank line after the docstrings in functions

## Testing Guidelines

Pytest drives all suites. New tests belong beside the feature under `tests/` and
should be named `test_<feature>.py`. Maintain the minimum 80% coverage enforced by
the CI profile (`python scripts/dev.py test --profile ci`). Use markers defined in
`pytest.ini` (`@pytest.mark.integration`,
`@pytest.mark.performance`, etc.) so suites can be targeted via
`scripts/dev.py test --profile <profile>`.

## Commit & Pull Request Guidelines

Use Conventional Commits (e.g., `docs: refine readme for oss discoverability`). Each
pull request should:

1. Describe the change and motivation, linking issues when applicable.
2. Note test commands executed and attach logs or screenshots for UI/diagram updates.
3. Request review from the appropriate code owners and ensure CI passes before
   merge.

## Security & Configuration Tips

Load sensitive settings (such as `OPENAI_API_KEY`, `AI_DOCS__FIRECRAWL__API_KEY`,
`AI_DOCS__MODE`) through `.env` files or your secrets manager; do not commit
credentials. For MCP HTTP transport, configure `FASTMCP_TRANSPORT`,
`FASTMCP_HOST`, and `FASTMCP_PORT` consistently between the environment and
Claude configuration templates.

## Error Handling & Observability

Prefer precise exception classes (`ValueError`, `HTTPStatusError`, etc.) over
generic `Exception`. When suppression is unavoidable, justify it with targeted
`# pylint: disable=` or `# pyright: ignore` comments. Inject structured logging
and tracing where it improves operability—import the helpers from
`src/services/logging_config.py` and emit actionable messages (context, impact,
next steps). Use existing OpenTelemetry utilities under `src/services/monitoring`
to attach spans/metrics when expanding service boundaries.
