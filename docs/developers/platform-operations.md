# Platform Operations

This guide combines deployment procedures, CI/CD workflows, GitHub composite
actions, and compatibility tracking for the retrieval stack.

## 1. Deployment

### Docker Image

```bash
DOCKER_BUILDKIT=1 docker build -t ai-docs-app:latest .
```

The multi-stage Dockerfile installs dependencies with uv and copies the runtime
virtual environment into a slim Python 3.12 base image.

### Compose Orchestration

The stack deploys with a single command and optional feature services may be
enabled via environment variables. Start the defaults with:

```bash
docker compose up -d
```

Services such as Redis, Postgres, Prometheus, Grafana, and Alertmanager can be
enabled by exporting their respective `AI_DOCS__` configuration values before
launching `docker compose`.

### Environment Variables

Inject overrides via the `AI_DOCS__` prefix:

```bash
export AI_DOCS__QDRANT__URL=https://qdrant.internal:6333
export AI_DOCS__CACHE__REDIS_URL=redis://redis:6379
export AI_DOCS__OPENAI__API_KEY=sk-***
```

Use `.env` for local development and a secret manager in production.

### Health & Persistence

- Qdrant exposes `/health`; compose uses it to gate app startup.
- FastAPI exposes `/health` and `/metrics`.
- Named volumes store Qdrant, Redis/Dragonfly, Postgres, and Prometheus data.
  Snapshot volumes regularly:

```bash
docker run --rm -v qdrant_data:/data -v "$PWD/backups":/backup alpine \
  tar czf /backup/qdrant-$(date +%Y%m%d).tar.gz -C /data .
```

### Rolling Updates

1. Build and push the updated image.
2. `docker compose pull && docker compose up -d --no-deps app`
3. Monitor health endpoints.
4. Re-run integration smoke tests (`uv run pytest tests/integration/rag/test_pipeline.py -q`).

## 2. CI/CD Workflows

Six GitHub Actions workflows manage validation, documentation, releases, and
labelling:

| Workflow | Purpose | Trigger |
| --- | --- | --- |
| `ci.yml` | Change-aware linting, tests, builds, security scans | Push/PR to `main`/`develop`, manual |
| `config-deployment.yml` | Config/template validation | Config file changes, manual |
| `docs.yml` | MkDocs build + doc checks | Doc changes, manual |
| `release.yml` | Tagged releases, GHCR image, PyPI publish | Tags `v*.*.*`, manual |
| `test-composite-actions.yml` | Regression for local composite actions | PRs touching `.github/actions/**` |
| `labeler.yml` | PR/issue labelling and reviewer assignment | PR/issue events |

`ci.yml` uses composite actions to set up environments, runs Ruff/Pylint/Pyright,
executes `python scripts/dev.py test --profile ci`, builds distributions, and
runs `pip-audit`/Bandit.

## 3. Composite Actions

Two reusable actions live under `.github/actions/`:

- `setup-environment`: Installs Python + uv, restores dependency cache, exposes a
  `cache-hit` output. Accepts `python-version`, `cache-suffix`, and `install-dev` inputs.
- `validate-config`: Runs `scripts/ci/validate_config.py` with configurable
  `config-root`, `templates-dir`, and `environment` inputs.

`test-composite-actions.yml` exercises both actions to catch regressions.

## 4. Compatibility Matrix

Keep the retrieval stack versions aligned with the matrix below (from
`compatibility-matrix.md`):

| Component | Package | Range | Locked | Notes |
| --- | --- | --- | --- | --- |
| Embedding | `fastembed` | `>=0.7.3,<0.7.4` | `0.7.3` | Default CPU-friendly embeddings. |
| Vector store | `qdrant-client[fastembed]` | `>=1.15.1,<1.15.2` | `1.15.1` | REST/gRPC client with FastEmbed integration. |
| Orchestration | `langchain` | `>=0.3.12,<0.4.0` | `0.3.27` | Use internal wrappers to minimise churn. |
| LangChain core | `langchain-core` | `>=0.3.76` | `0.3.76` | Implicit dependency. |
| LangChain community | `langchain-community` | `>=0.3.12,<0.4.0` | `0.3.30` | Houses vector store integrations. |
| LangChain OpenAI | `langchain-openai` | `>=0.3.33,<1.0.0` | `0.3.33` | Required for evaluator flows. |

Update the matrix when Renovate bumps dependencies; include regression metrics,
contract tests, and ADR updates with each change.

## 5. Local Validation

Before opening a PR, run the same checks that CI enforces:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pylint --fail-under=9.5 src scripts
uv run pyright
uv run python scripts/dev.py test --profile ci
uv build
```

Documentation contributors:

```bash
uv sync --frozen --group docs
uv run python scripts/dev.py validate --check-docs --strict
uv run mkdocs build --strict
```

Keep this guide aligned with `.github/workflows/` and `docker-compose.yml`.
