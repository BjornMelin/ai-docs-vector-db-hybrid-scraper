---
title: Configuration Guide
audience: developers
status: active
owner: platform-engineering
last_reviewed: 2025-03-13
---

# Configuration Guide

Reference for configuring services and environments when developing or deploying the AI Docs Vector
DB platform.

## Configuration Layout

```text
config/
├── settings.py              # Pydantic settings entry point
├── environments/
│   ├── simple.toml          # Local/dev defaults
│   └── enterprise.toml      # Production defaults
└── secrets.toml.example     # Example secrets structure
```

- **settings.py** stitches environment variables, config files, and defaults into a single Pydantic
  model.
- The `AI_DOCS_ENV` variable selects the active configuration profile.
- Secrets are expected from environment variables (or a secrets manager in production) rather than
  hard-coded files.

## Key Settings

| Setting | Description |
| --- | --- |
| `API_HOST`, `API_PORT` | FastAPI application binding |
| `QDRANT_URL`, `QDRANT_API_KEY` | Vector database connection parameters |
| `EMBEDDING_PROVIDER` | Dense embedding provider (`openai`, `fastembed`, etc.) |
| `SPARSE_PROVIDER` | Sparse embedding provider (e.g., `bge-small-en`) |
| `CACHE_URL` | Dragonfly cache endpoint |
| `BROWSER_AUTOMATION_MODE` | Preferred automation tier override |
| `TELEMETRY_ENABLED` | Enables metrics/log forwarding |

Refer to `src/config/settings.py` for the full schema and default values.

## Environment Profiles

- **simple** – Minimal stack for local development. Uses in-process services and relaxed timeouts.
- **enterprise** – Production-ready configuration enabling observability, high concurrency limits, and
  expanded automation tiers.
- **test** – Fixtures for integration/unit tests with deterministic behaviour.

Switch profiles by exporting `AI_DOCS_ENV` or by invoking `UnifiedConfig.with_env("enterprise")` from
code.

## Secrets Management

- Store credentials in environment variables or a secrets manager (e.g., Vault, AWS Secrets Manager).
- Avoid committing `.toml` files containing secrets; template any required structure in
  `secrets.toml.example`.
- Rotate sensitive keys regularly and ensure the configuration loader can refresh secrets without
  restarts if needed.

## Configuration Overrides

Use environment variables to override individual settings without editing configuration files:

```bash
export AI_DOCS_ENV=enterprise
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python -m src.app
```

For temporary overrides in code, call `UnifiedConfig().copy(update={"embedding_provider": "openai"})`
within scoped contexts.

## Related Material

- [System Architecture](./architecture.md)
- [Operations Configuration](../operators/configuration.md)
- [Security Checklist](../security/essential-security-checklist.md)
