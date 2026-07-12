# Operate the platform

This guide combines deployment procedures, CI/CD workflows, GitHub composite
actions, and compatibility tracking for the retrieval stack.

## 1. Deploy the application

### Build the Docker image

```bash
DOCKER_BUILDKIT=1 docker build -t ai-docs-app:latest .
```

The multi-stage Dockerfile installs dependencies with uv and copies the runtime
environment into the pinned Playwright Python 3.11 Noble image.

### Run Docker Compose

The stack deploys with a single command and optional feature services may be
enabled via environment variables. Start the defaults with:

```bash
docker compose --profile simple up -d
```

Use `docker compose --profile enterprise up -d` to include Dragonfly, PostgreSQL, Prometheus, and Grafana.

### Environment Variables

Inject overrides via the `AI_DOCS_` prefix:

```bash
export AI_DOCS_QDRANT__URL=https://qdrant.internal:6333
export AI_DOCS_CACHE__DRAGONFLY_URL=redis://dragonfly:6379
export AI_DOCS_OPENAI__API_KEY=your_openai_api_key_here
```

Use `.env` for local development and a secret manager in production.

### Health & Persistence

- Compose checks Qdrant's TCP port before it starts the application.
- FastAPI exposes `/health` and `/metrics`.
- Named volumes store Qdrant, Dragonfly, Postgres, and Prometheus data. Use
  Qdrant's collection snapshot API instead of copying a live volume:

```bash
mkdir -p backups
snapshot_name="$(
  curl --fail --silent --show-error --request POST \
    'http://localhost:6333/collections/documents/snapshots?wait=true' |
    uv run python -c 'import json, sys; print(json.load(sys.stdin)["result"]["name"])'
)"
curl --fail --output "backups/${snapshot_name}" \
  "http://localhost:6333/collections/documents/snapshots/${snapshot_name}"
```

Replace `documents` when you use a different collection name. Test snapshot
recovery before deleting the source volume.

### Rolling Updates

1. Build and push the updated image.
2. `docker compose --profile simple pull && docker compose --profile simple up -d --no-deps app`
3. Monitor health endpoints.
4. Re-run the evaluation harness (`uv run python scripts/dev.py benchmark --suite performance`).

## 2. Run CI and release workflows

GitHub Actions workflows manage validation, documentation, releases, and
labelling:

| Workflow | Purpose | Trigger |
| --- | --- | --- |
| `ci.yml` | Change-aware linting, tests, builds, security scans | Push/PR to `main`/`develop`, manual |
| `config-deployment.yml` | Config/template validation | Config file changes, manual |
| `docs.yml` | MkDocs build + doc checks | Doc changes, manual |
| `embeddings.yml` | Focused embedding regression tests | Embedding or dependency changes |
| `labeler.yml` | Pull request and issue labelling | Pull request and issue events |
| `regression-opt-in.yml` | Full regression and benchmark profiles | Manual |
| `release.yml` | GitHub release and optional PyPI publish | Tags `v*.*.*`, manual |
| `test-composite-actions.yml` | Regression for local composite actions | PRs touching `.github/actions/**` |
| `validation.yml` | CPU validation and opt-in self-hosted GPU validation | Push/PR, manual |

Tagged releases create GitHub release artifacts after the ordinary release
gates pass. PyPI publication is skipped unless the repository or organization
variable `PYPI_PUBLISH_ENABLED` is set to `true`; enable it only after the
protected `pypi` environment and `PYPI_API_TOKEN` are configured.

`ci.yml` uses composite actions to set up environments, runs
actionlint/Ruff/Pylint/Pyright, executes
`uv run python scripts/dev.py test --profile ci`, builds distributions, and
runs `pip-audit`/Bandit.

## 3. Reuse composite actions

Two reusable actions live under `.github/actions/`:

- `setup-environment`: Installs Python + uv, restores dependency cache, exposes a
  `cache-hit` output. Accepts `python-version`, `cache-suffix`, and `install-dev` inputs.
- `validate-config`: Runs `scripts/ci/validate_config.py` with configurable
  `config-root`, `templates-dir`, and `environment` inputs.

`test-composite-actions.yml` exercises both actions to catch regressions.

## 4. Resolve dependency versions

`pyproject.toml` is the only source for supported dependency ranges and Python
compatibility. `uv.lock` is the resolved installation authority. Do not copy
version matrices into operations documentation; verify changes through the
lockfile, regression profiles, and relevant architecture decision records.

## 5. Validate changes locally

Before opening a PR, run the same checks that CI enforces:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pylint --fail-under=9.5 src scripts
uv run pyright
actionlint
uv run python scripts/dev.py test --profile ci
uv build
```

Documentation contributors:

```bash
uv sync --frozen --extra docs
uv run python scripts/dev.py validate --check-docs --strict
uv run mkdocs build --strict
```

Keep this guide aligned with `.github/workflows/` and `docker-compose.yml`.
