# Monitoring Guide

## Essential Monitoring

### Health Checks

```bash
# Service health
curl -s http://localhost:8000/health | jq '.'
curl -s http://localhost:6333/health | jq '.'
redis-cli ping

# System resources
docker stats --no-stream
free -h
df -h
```

### Key Metrics

```bash
# Application metrics
curl -s http://localhost:8000/metrics | grep -E "(requests_total|request_duration|errors_total)"

# Browser anti-bot metrics
curl -s http://localhost:8000/metrics | grep -E "browser_(requests_total|challenges_total|response_time_seconds)"

# Vector database metrics
curl -s http://localhost:6333/metrics | grep -E "(qdrant_collections_total|qdrant_points_total|search_duration)"

# Cache metrics
redis-cli info stats | grep -E "(hit_rate|miss_rate|ops_per_sec)"
```

## Prometheus Setup

### Basic Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "ai-docs-api"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"

  - job_name: "qdrant"
    static_configs:
      - targets: ["localhost:6333"]
    metrics_path: "/metrics"

  - job_name: "node-exporter"
    static_configs:
      - targets: ["localhost:9100"]
```

### Start Monitoring Stack

```bash
# Start Prometheus and Grafana
docker-compose -f monitoring/docker-compose.yml up -d

# Verify services
curl http://localhost:9090/targets
curl http://localhost:3000 # Grafana (admin/admin)
```

## Alerting Rules

### Critical Alerts

```yaml
# alerts.yml
groups:
  - name: ai-docs-critical
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"

      - alert: VectorDBErrors
        expr: increase(qdrant_errors_total[5m]) > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in vector database"

      - alert: BrowserChallengesSpike
        expr: increase(ml_app_browser_challenges_total{outcome="detected"}[15m]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Detected bot challenges exceeding normal baseline"
          description: "Investigate proxy/captcha configuration for tier {{ $labels.tier }} runtime {{ $labels.runtime }}"
```

### Configure Alertmanager

```yaml
# alertmanager.yml
global:
  smtp_smarthost: "localhost:587"
  smtp_from: "alerts@ai-docs.com"

route:
  group_by: ["alertname"]
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: "web.hook"

receivers:
  - name: "web.hook"
    webhook_configs:
      - url: "http://localhost:5001/"
```

## Log Monitoring

### Centralized Logging

```bash
# Configure log forwarding
# docker-compose.yml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

# Log aggregation with Loki
docker run -d --name=loki \
  -p 3100:3100 \
  grafana/loki:latest
```

### Log Analysis

```bash
# Search error logs
docker logs api 2>&1 | grep -i error | tail -50

# Monitor real-time logs
docker logs -f api | grep -E "(ERROR|WARNING|CRITICAL)"

# Log volume analysis
docker logs api --since 1h | wc -l
```

## Performance Monitoring

### Response Time Tracking

```bash
# API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/search

# Vector search latency
curl -s http://localhost:6333/metrics | grep search_duration_seconds

# Cache performance
redis-cli info commandstats | grep -E "(get|set)"
```

### Resource Utilization

```bash
# Container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# System load
uptime
iostat -x 1 5
```

## Custom Dashboards

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "AI Docs Monitor",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Vector Search Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(qdrant_search_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## Automated Monitoring

### Health Check Script

```bash
#!/bin/bash
# health-monitor.sh

LOG_FILE="/var/log/ai-docs/health-check.log"
ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

check_service() {
  local service=$1
  local url=$2

  if ! curl -sf "$url" > /dev/null; then
    echo "$(date): $service is DOWN" >> "$LOG_FILE"
    curl -X POST "$ALERT_WEBHOOK" \
      -H 'Content-type: application/json' \
      --data "{\"text\":\"ALERT: $service is down\"}"
    return 1
  fi
  return 0
}

# Check all services
check_service "API" "http://localhost:8000/health"
check_service "Qdrant" "http://localhost:6333/health"
check_service "Redis" "redis://localhost:6379"

echo "$(date): Health check completed" >> "$LOG_FILE"
```

### Scheduled Monitoring

```bash
# Add to crontab
crontab -e

# Check every 5 minutes
*/5 * * * * /opt/ai-docs/scripts/health-monitor.sh

# Generate daily report
0 8 * * * /opt/ai-docs/scripts/daily-report.sh
```

## Troubleshooting Monitoring

### Common Issues

```bash
# Prometheus not scraping
curl http://localhost:9090/api/v1/targets

# Missing metrics
curl http://localhost:8000/metrics | grep http_requests

# Grafana connection issues
docker logs grafana
curl http://localhost:3000/api/health
```

### Reset Monitoring

```bash
# Restart monitoring stack
docker-compose -f monitoring/docker-compose.yml down
docker-compose -f monitoring/docker-compose.yml up -d

# Clear Prometheus data
docker volume rm monitoring_prometheus_data

# Reset Grafana
docker volume rm monitoring_grafana_data
```
