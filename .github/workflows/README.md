# GitHub Actions Workflows

Simple CI/CD workflows for the AI Docs Vector DB Hybrid Scraper project.

## Workflows

### main.yml - Continuous Integration
- **Trigger**: Push to main/develop branches, pull requests
- **Actions**: 
  - Lint with ruff
  - Validate configurations
  - Run tests with coverage check (80% minimum)
- **Duration**: ~2-3 minutes

### deploy.yml - Manual Deployment
- **Trigger**: Manual workflow dispatch
- **Inputs**: Target environment (development/staging/production)
- **Actions**:
  - Validate configuration
  - Build Docker image (non-dev environments)
  - Deploy configuration
  - Run smoke tests
- **Duration**: ~5 minutes

### pr.yml - Pull Request Checks
- **Trigger**: Pull request events
- **Actions**:
  - Quick validation checks
  - Check for merge conflicts
  - Python syntax validation
  - PR size analysis comment
- **Duration**: ~1 minute

### claude.yml - Claude Code Integration
- **Trigger**: Comments with @claude mention
- **Actions**: Runs Claude Code for automated assistance

## Configuration

All workflows use:
- Python 3.12
- uv for dependency management
- Standard GitHub Actions (no custom composite actions)

## Environment Variables

Set in repository secrets:
- `ANTHROPIC_API_KEY` - For Claude Code integration

## Local Testing

To test workflows locally:
```bash
# Lint and format
uv run ruff check . --fix && uv run ruff format .

# Validate configs
uv run python scripts/validate_config.py

# Run tests
uv run pytest tests/ --cov=src --cov-report=term-missing -v
```