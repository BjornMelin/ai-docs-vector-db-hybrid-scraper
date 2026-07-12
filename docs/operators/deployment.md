---
title: Deploy AI Docs with Docker Compose
audience: operators
status: active
owner: platform-engineering
last_reviewed: 2026-07-11
meta:
  contentType: How-to
  category: Operations
---

# Deploy AI Docs with Docker Compose

Build and run AI Docs with one of the two profiles defined in `docker-compose.yml`. The `simple` profile runs Qdrant and the API; the `enterprise` profile also runs Dragonfly, PostgreSQL, Prometheus, and Grafana.

## Prepare the host

Install Docker Engine with the Compose plugin, then clone the repository:

```bash
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
cp .env.example .env
```

Edit `.env` before you start the stack. [Configure AI Docs](./configuration.md) lists the supported variables.

## Run the default profile

Start Qdrant and the API with the default local settings:

```bash
docker compose --profile simple up -d
docker compose --profile simple ps
```

Verify the public API health endpoint:

```bash
curl --fail http://localhost:8000/health
```

## Run the enterprise profile

Enable Dragonfly in `.env` when the application should use the cache:

```dotenv
AI_DOCS_CACHE__ENABLE_DRAGONFLY_CACHE=true
AI_DOCS_CACHE__DRAGONFLY_URL=redis://dragonfly:6379
```

Start every enterprise service:

```bash
docker compose --profile enterprise up -d
docker compose --profile enterprise ps
```

The enterprise profile exposes these local ports:

| Service | Port |
| --- | --- |
| FastAPI | `8000` |
| Qdrant HTTP | `6333` |
| Qdrant gRPC | `6334` |
| Dragonfly | `6379` |
| PostgreSQL | `5432` |
| Prometheus | `9090` |
| Grafana | `3000` |

## Inspect a failed deployment

Read service state and recent logs before restarting containers:

```bash
docker compose --profile enterprise ps
docker compose logs --tail=100 app qdrant dragonfly
```

Validate the rendered manifest when environment interpolation fails:

```bash
docker compose --profile enterprise config --quiet
```

## Stop the deployment

Stop containers without deleting persistent volumes:

```bash
docker compose --profile enterprise down
```

Add `--volumes` only when you intend to delete local Qdrant, Dragonfly, PostgreSQL, Prometheus, and Grafana data.
