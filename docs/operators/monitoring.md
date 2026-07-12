---
title: Monitor AI Docs
audience: operators
status: active
owner: operations-engineering
last_reviewed: 2026-07-11
meta:
  contentType: How-to
  category: Operations
---

# Monitor AI Docs

Use FastAPI health and metrics endpoints for the application. The enterprise Compose profile also runs Prometheus and Grafana from the checked-in configuration.

## Enable application metrics

Set the monitoring fields in `.env` before you start the application:

```dotenv
AI_DOCS_MONITORING__ENABLED=true
AI_DOCS_MONITORING__ENABLE_METRICS=true
AI_DOCS_MONITORING__METRICS_PATH=/metrics
```

Start the enterprise profile:

```bash
docker compose --profile enterprise up -d
```

## Check application health

Verify FastAPI health from the deployment host:

```bash
curl --fail http://localhost:8000/health
```

Query Prometheus-formatted application metrics when metrics are enabled:

```bash
curl --fail http://localhost:8000/metrics
```

## Check Prometheus and Grafana

Verify their containers and local ports:

```bash
docker compose --profile enterprise ps prometheus grafana
curl --fail http://localhost:9090/-/ready
curl --fail http://localhost:3000/api/health
```

Open Prometheus at `http://localhost:9090` and Grafana at `http://localhost:3000` on a local deployment.

## Inspect collection failures

Read application and Prometheus logs when targets are missing:

```bash
docker compose logs --tail=100 app prometheus
```

Validate the mounted Prometheus configuration before restarting it:

```bash
docker compose --profile enterprise config --quiet
docker compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
```

## Configure OpenTelemetry

Enable OpenTelemetry through the canonical settings namespace:

```dotenv
AI_DOCS_OBSERVABILITY__ENABLED=true
AI_DOCS_OBSERVABILITY__OTLP_ENDPOINT=http://your_collector_here:4317
AI_DOCS_OBSERVABILITY__OTLP_INSECURE=true
```

Restart the application after changing observability settings. The runtime no longer reads a second set of flat `AI_DOCS_OBSERVABILITY_*` aliases.

## Build an alert

Alert on the application signals that match your service-level objectives. At minimum, monitor these conditions:

- The `/health` request fails
- Prometheus can't scrape the application target
- Container restart counts increase
- Qdrant, Dragonfly, or PostgreSQL becomes unavailable

Test every alert against a disposable deployment before you enable notifications for production.
