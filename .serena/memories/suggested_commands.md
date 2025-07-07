# Essential Commands for AI Docs Vector DB Hybrid Scraper

## Core Development Commands

**Note:** Prefix all Python and test commands with `uv` since we're in a uv-managed environment.

### Testing & Quality
- **Test (with coverage)**: `uv run pytest --cov=src`
- **Fast tests**: `uv run pytest tests/benchmarks/ --benchmark-only`
- **Lint & Format**: `ruff check . --fix && ruff format .`
- **Type check**: `mypy src/ --config-file pyproject.toml`

### Services & Development
- **Start services**: `./scripts/start-services.sh`
- **Health check**: `curl localhost:6333/health`
- **Dev server**: `uv run python -m src.api.main`
- **Dev server (enterprise)**: `DEPLOYMENT_TIER=production uv run python -m src.api.main`

### CLI Tools
- **Unified CLI**: `uv run python -m src.cli.unified`
- **Main CLI**: `uv run python -m src.cli.main`

### Package Management
- **Install dependencies**: `uv sync --dev`
- **Add dependency**: `uv add <package>`
- **Add dev dependency**: `uv add --dev <package>`

### Git & System
- **Git status**: `git status`
- **Git log**: `git log --oneline -10`
- **List files**: `ls -la`
- **Find files**: `find . -name "*.py" -type f`
- **Search in files**: `rg "pattern" --type py`