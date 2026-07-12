---
title: Operate AI Docs
audience: operators
status: active
owner: operations-engineering
last_reviewed: 2026-07-11
meta:
  contentType: How-to
  category: Operations
---

# Operate AI Docs

Use Docker Compose for service lifecycle commands and the public FastAPI endpoint for application health. Run commands from the repository root so Compose uses the checked-in manifest and `.env` file.

## Check service health

Inspect the enterprise profile and application endpoint:

```bash
docker compose --profile enterprise ps
curl --fail http://localhost:8000/health
```

Inspect container resource usage when a service slows down:

```bash
docker stats --no-stream
```

## Read service logs

Read recent application and storage logs before restarting a container:

```bash
docker compose logs --tail=100 app qdrant dragonfly
```

Follow one service during an incident:

```bash
docker compose logs --follow --tail=100 app
```

## Restart services

Restart one service when its logs identify a local failure:

```bash
docker compose restart app
```

Restart the full enterprise profile without deleting volumes:

```bash
docker compose --profile enterprise down
docker compose --profile enterprise up -d
```

## Apply an image update

Update the pinned image tag in `docker-compose.yml`, then pull upstream service images, rebuild the application, and wait for health checks:

```bash
docker compose --profile enterprise pull
docker compose --profile enterprise build app
docker compose --profile enterprise up -d
docker compose --profile enterprise ps
```

Upgrade persistent Qdrant data one minor version at a time. Snapshot the volume before each step and confirm `/readyz` before continuing to the next version.

Run the release test profile before you deploy an application commit:

```bash
uv run python scripts/dev.py test --profile ci
```

## Preserve local data

Compose stores persistent state in named volumes. List the resolved volume names before a host migration:

```bash
docker compose --profile enterprise config --volumes
docker volume ls
```

Stop writers before you snapshot or copy a volume. Follow the storage provider's recovery procedure and test the restored data before you remove the original volume.

Don't run `docker compose down --volumes` during a routine restart. That option deletes local Qdrant, Dragonfly, PostgreSQL, Prometheus, and Grafana state.

## Collect failure evidence

Capture these outputs when a deployment remains unhealthy:

```bash
docker compose --profile enterprise ps
docker compose --profile enterprise config
docker compose logs --tail=200 app qdrant dragonfly postgres
docker version
docker compose version
```

Redact credentials before attaching output to an issue.
