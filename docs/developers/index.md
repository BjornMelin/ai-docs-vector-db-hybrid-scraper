---
title: Developer Documentation
audience: developers
status: active
owner: platform-engineering
last_reviewed: 2025-03-13
---

## Developer Hub

This page is the single stop for engineers who build, extend, or operate the AI Docs Vector DB
platform. It keeps the essential setup steps close at hand and points to the deeper references when
you need the full detail.

## 1. Quick Setup

```bash
# Clone and enter the repository
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper

# Install dependencies (project + dev extras)
uv sync --dev

# Launch the local services stack
python scripts/dev.py services start

# Validate the environment before coding
uv run pytest -q
```

| Tool            | Notes                                                        |
| --------------- | ------------------------------------------------------------ |
| Python          | 3.11 or 3.12 (pin with `uv python pin 3.12` if unsure)       |
| Package manager | [uv](https://github.com/astral-sh/uv) for virtualenv + deps  |
| Docker          | Desktop/Engine 24+ for Qdrant, Dragonfly, and API containers |
| Optional        | `docker compose`, `just`, VS Code devcontainer configuration |

## 2. Essential References

Keep these guides bookmarked—together they answer the majority of day-to-day questions:

- **[Architecture & Orchestration](./architecture-and-orchestration.md)** – System design, LangGraph stack, MCP guidance.
- **[Configuration](./configuration.md)** – Environment variables, profile overrides, tuning knobs.
- **[Architecture & Orchestration](./architecture-and-orchestration.md)** – Drop-in patterns for optional cache/search services.
- **[Cache & Performance](./cache-and-performance.md)** – Persistent cache, GPU notes, performance checklist.
- **[Agentic Orchestration](./agentic-orchestration.md)** – LangGraph runner, tool discovery, and testing guidance.
- **[FastMCP Integration](./mcp-integration.md)** – Server structure, middleware, and migration roadmap.
- **[API & Contracts](./api-and-contracts.md)** – REST + MCP endpoints, request/response schemas.
- **[Query Response Contract](./queries/response-contract.md)** – Canonical payloads returned by the pipeline.
- **[Platform Operations](./platform-operations.md)** – Pipeline layout, required checks, reusable jobs.
- **[Contributing & Testing](./contributing-and-testing.md)** – Coding standards, quality gates, review workflow.

Supporting docs when you need more detail:

- **[Developer Setup Guide](./getting-started.md)** – OS-specific prerequisites and local tooling.
- **[GPU Acceleration Guide](./gpu-acceleration.md)** – Optional CUDA/MPS configuration and helper APIs.
- **[Setup & Configuration](./setup-and-configuration.md)** – Environment prerequisites, profiles, and configuration loader.
- **[Deployment Guide](./deployment.md)** – Production rollout patterns and rollback steps.
- **[Document Metadata](./document-metadata.md)** – Indexing schema and required fields.

## 3. Common Tasks

| Task               | Checklist                                                                  |
| ------------------ | -------------------------------------------------------------------------- | ---- | --------------------------- |
| Start a feature    | Sync `main`, run `uv run pytest -q`, branch, follow lint/type gates        |
| Add a dependency   | Edit `pyproject.toml`, run `uv sync`, capture notes in configuration docs  |
| Touch the API      | Update Pydantic models + OpenAPI schema, extend API tests, document change |
| Tune performance   | Adjust config knobs, capture metrics via `/metrics`, note changes for ops  |
| Operate locally    | `python scripts/dev.py services start                                      | stop | logs`, use `docker compose` |
| Ship to production | Follow CI/CD checklist, update release notes, notify operators             |

## 4. Development Patterns

### FastAPI & Dependency Injection

- Register services with `ModeAwareServiceFactory`; prefer functional providers over classes.
- Keep route handlers thin—domain logic belongs in `src/services/**`.
- Use Pydantic models for validation and log structured errors.

### Testing & Quality Gates

- Unit tests live under `tests/unit/**`; mark integration suites with `@pytest.mark.integration`.
- Required gates: `ruff format`, `ruff check`, `pylint`, `pyright`, `uv run pytest`.
- Snapshot data belongs in `tests/fixtures/**`; avoid ad-hoc test assets.

### Observability

- Expose metrics through Prometheus (`/metrics`) and structured logs.
- Emit OpenTelemetry spans when introducing new services and update operator dashboards accordingly.

## 5. Optional Services

Optional caches or search engines should be wired via the patterns documented in [Architecture & Orchestration](./architecture-and-orchestration.md).
## 6. Where to Go Next

| If you need…         | Read…                                 |
| -------------------- | ------------------------------------- |
| Environment details  | [Configuration](./configuration.md)   |
| System overview      | [Architecture](./architecture.md)     |
| API contracts        | [API & Contracts](./api-and-contracts.md)   |
| Automation basics    | [Platform Operations](./platform-operations.md)         |
| Contribution process | [Contributing & Testing](./contributing-and-testing.md)     |
| Deployment guidance  | [Deployment](./deployment.md)         |
| Operator procedures  | [Operator Hub](../operators/index.md) |

See something missing? Open an issue or submit a PR—keeping this hub concise and accurate helps
every engineer move faster.
