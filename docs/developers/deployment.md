# Production Deployment Guide

Use Docker Compose profiles to run the platform in either **simple** (local) or
**enterprise** (full stack) mode. The supplied Dockerfile builds a slim Uvicorn
image using UV-managed dependencies.

## Build the Runtime Image

```bash
# Build the multi-stage image defined in Dockerfile
DOCKER_BUILDKIT=1 docker build -t ai-docs-app:latest .
```

The build stage installs dependencies with UV and copies the runtime virtual
environment into a slim Python 3.12 container. No manual pip steps are required.

## Compose Profiles

| Profile | Services | Command |
| ------- | -------- | ------- |
| `simple` | `app`, `qdrant` | `docker compose --profile simple up -d` |
| `enterprise` | `app`, `qdrant`, `redis`, `postgres`, `prometheus`, `grafana`, `alertmanager` | `docker compose --profile enterprise up -d` |

Profiles reuse the same `app` image but toggle supporting services and resource
limits. Switch profiles by setting `AI_DOCS__MODE` in the application
container or by exporting the variable locally before running CLI commands.

## Environment Variables

Key settings are injected via the `AI_DOCS__` prefix. Example overrides:

```bash
export AI_DOCS__MODE=enterprise
export AI_DOCS__QDRANT__URL=http://qdrant:6333
export AI_DOCS__CACHE__REDIS_URL=redis://redis:6379
export AI_DOCS__OPENAI__API_KEY=sk-***
```

Enterprise deployments typically mount the `.env` file as a secret or supply the
variables through your orchestrator. See `docs/developers/configuration.md` for
section-specific keys.

## Health Checks and Monitoring

- Qdrant exposes `/health`. Compose defines a health check to gate the FastAPI
  startup.
- The FastAPI container provides `/health` and `/metrics` (Prometheus exporter).
- Prometheus and Grafana profiles ship with default configuration under
  `config/prometheus/` and `config/grafana/`.

## Persistence

Named volumes are declared for Qdrant, Redis/Dragonfly, Postgres, and Prometheus.
Ensure you snapshot these volumes for disaster recovery. Example backup command:

```bash
docker run --rm -v qdrant_data:/data -v "$PWD/backups":/backup alpine \
  tar czf /backup/qdrant-$(date +%Y%m%d).tar.gz -C /data .
```

## Rolling Updates

1. Build and push the updated image.
2. `docker compose pull && docker compose up -d --no-deps app`
3. Watch health checks (`docker compose ps`, `/health`).
4. Re-run smoke tests (`uv run pytest tests/integration/rag/test_pipeline.py -q`).

For orchestrators such as Kubernetes, reuse the same image and translate the
compose services into Deployments/StatefulSets. Mount `.env` as a secret and
recreate the readiness checks defined in the compose file.

---

Keep this guide in sync with `docker-compose.yml` and update it whenever service
profiles or health checks change.
