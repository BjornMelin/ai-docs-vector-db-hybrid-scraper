---
title: Manage AI Docs configuration
audience: users
status: active
owner: product-education
last_reviewed: 2026-07-12
meta:
  contentType: How-to
  category: Getting started
---

# Manage AI Docs configuration

Create a profile with the setup wizard, inspect the resolved settings, and validate or export JSON and YAML configuration files with the `ai-docs` CLI.

## Configure environment variables

Copy the executable environment template before you run the application locally:

```bash
cp .env.example .env
```

Application variables use the `AI_DOCS_` prefix. Add a double underscore only between a nested section and field:

```dotenv
AI_DOCS_ENVIRONMENT=development
AI_DOCS_QDRANT__URL=http://localhost:6333
AI_DOCS_EMBEDDING__RETRIEVAL_MODE=dense
```

Configure credentials only when you enable their provider:

```dotenv
AI_DOCS_EMBEDDING_PROVIDER=openai
AI_DOCS_OPENAI__API_KEY=sk-your_openai_api_key_here
```

Restart the application after changing `.env` so every service receives the same immutable settings instance.

## Create a profile

Run the interactive setup wizard with a preselected profile:

```bash
uv run ai-docs setup --profile development
```

Supported profiles are `personal`, `development`, `production`, `testing`, `local-only`, and `minimal`. The wizard writes the selected profile under `config/profiles/` and can activate it as `config.json`.

Inspect the live setup options before automating the wizard:

```bash
uv run ai-docs setup --help
```

## Inspect resolved settings

Show settings loaded from `.env` and process environment variables:

```bash
uv run ai-docs config show --format table
uv run ai-docs config show --format json
uv run ai-docs config show --format yaml
```

Pass a JSON or YAML file through the global `--config` option when another command should use it:

```bash
uv run ai-docs \
  --config config/profiles/development.json \
  config show --format table
```

The display masks no fields by itself. Don't attach configuration output to an issue until you have removed credentials.

## Validate configuration

Validate a file without promoting it to the current CLI context:

```bash
uv run ai-docs config load \
  config/profiles/development.json \
  --validate-only
```

Validate the resolved settings for a file:

```bash
uv run ai-docs \
  --config config/profiles/development.json \
  config validate
```

Use the repository harness after changing `.env.example`, templates, or the Pydantic schema:

```bash
uv run python scripts/dev.py validate --check-docs --strict
uv run pytest -q tests/unit/config
```

## Export configuration

Export the resolved settings as JSON or YAML:

```bash
mkdir -p exports
uv run ai-docs \
  --config config/profiles/development.json \
  config export --format json --output exports/development.json
uv run ai-docs \
  --config config/profiles/development.json \
  config export --format yaml --output exports/development.yaml
```

Exported files can contain credentials. Keep populated files outside version control.

## Use configuration with Docker Compose

The application service loads `.env` when the file exists. Compose overrides only container-network endpoints and the persisted FastEmbed cache path.

```bash
docker compose --profile simple config --quiet
docker compose --profile simple up -d
docker compose --profile simple ps
```

Use the enterprise profile when you also need Dragonfly, PostgreSQL, Prometheus, and Grafana:

```bash
docker compose --profile enterprise up -d
```

Set `AI_DOCS_CACHE__ENABLE_DRAGONFLY_CACHE=true` in `.env` when the application should use the Dragonfly service.

## Diagnose configuration errors

Read the live command surface when a documented option no longer matches the installed version:

```bash
uv run ai-docs --help
uv run ai-docs config --help
```

Print one non-sensitive resolved value when `.env` doesn't load as expected:

```bash
uv run python -c 'from src.config.loader import Settings; print(Settings().environment.value)'
```

Use [Configure AI Docs](../operators/configuration.md) for the canonical environment-variable reference and [Troubleshooting](./troubleshooting.md) for runtime failures.
