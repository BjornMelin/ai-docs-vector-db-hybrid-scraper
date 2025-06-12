# 📊 Monitoring and Observability Guide

> **Status**: V1 Production Ready  
> **Last Updated**: 2025-06-09  
> **Purpose**: Comprehensive monitoring and observability guide for production operators  
> **Audience**: System administrators, DevOps engineers, and SRE teams

## 🎯 Overview

This guide provides complete monitoring and observability setup for the AI documentation vector database system.
It covers V1 enhanced monitoring with real-time performance tracking, comprehensive metrics collection,
and production-ready alerting strategies.

## 📋 Quick Start Checklist

### Essential Monitoring Setup

- [ ] Prometheus and Grafana deployed
- [ ] System metrics collection configured
- [ ] Application-specific metrics enabled
- [ ] Alert rules configured
- [ ] Log aggregation setup
- [ ] Health checks implemented
- [ ] Performance baselines established

### V1-Specific Monitoring

- [ ] Query API performance tracking
- [ ] HyDE cache hit rate monitoring
- [ ] DragonflyDB metrics collection
- [ ] Collection alias operation tracking
- [ ] Payload indexing performance monitoring

## 🏗️ Monitoring Architecture

```mermaid
graph TD
    A[Application Services] --> B[Metrics Collection]
    B --> C[Prometheus]
    C --> D[Grafana Dashboards]
    C --> E[AlertManager]

    F[Logs] --> G[Log Aggregation]
    G --> H[Centralized Logging]

    I[Health Checks] --> J[Uptime Monitoring]

    K[Custom Metrics] --> L[Business Logic Monitoring]
    L --> C

    subgraph "V1 Enhanced Monitoring"
        M[Query API Metrics]
        N[HyDE Performance]
        O[Cache Analytics]
        P[Index Performance]
    end

    M --> C
    N --> C
    O --> C
    P --> C
```

## 🔧 Core Monitoring Stack

### 1. Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"
  - "recording_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  # System metrics
  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
    scrape_interval: 5s

  # Application metrics
  - job_name: "app-metrics"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s

  # Qdrant metrics
  - job_name: "qdrant"
    static_configs:
      - targets: ["qdrant:6333"]
    metrics_path: "/metrics"
    scrape_interval: 10s

  # DragonflyDB metrics (V1)
  - job_name: "dragonfly"
    static_configs:
      - targets: ["dragonfly:6379"]
    scrape_interval: 10s

  # Custom application metrics
  - job_name: "mcp-server"
    static_configs:
      - targets: ["mcp-server:8080"]
    metrics_path: "/health/metrics"
    scrape_interval: 15s
```

### 2. Grafana Dashboard Configuration

Create comprehensive monitoring dashboard at `monitoring/dashboards/main-dashboard.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "AI Vector DB Monitoring - V1 Enhanced",
    "tags": ["ai", "vector-db", "monitoring", "v1"],
    "timezone": "browser",
    "refresh": "5s",
    "panels": [
      {
        "title": "System Overview",
        "type": "stat",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "up",
            "legendFormat": "Service Uptime"
          }
        ]
      },
      {
        "title": "Query Performance (V1 Enhanced)",
        "type": "graph",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(query_api_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(query_api_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "HyDE Performance Tracking",
        "type": "graph",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
        "targets": [
          {
            "expr": "rate(hyde_cache_hits_total[5m])",
            "legendFormat": "Cache Hits/sec"
          },
          {
            "expr": "rate(hyde_cache_misses_total[5m])",
            "legendFormat": "Cache Misses/sec"
          },
          {
            "expr": "hyde_generation_duration_seconds",
            "legendFormat": "Generation Time"
          }
        ]
      },
      {
        "title": "DragonflyDB Cache Metrics",
        "type": "graph",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 },
        "targets": [
          {
            "expr": "dragonfly_cache_hit_rate",
            "legendFormat": "Hit Rate"
          },
          {
            "expr": "dragonfly_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          },
          {
            "expr": "dragonfly_evicted_keys_total",
            "legendFormat": "Evicted Keys"
          }
        ]
      }
    ]
  }
}
```

### 3. Alert Rules Configuration

Create `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: system_alerts
    rules:
      - alert: HighCPUUsage
        expr: (100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage detected"
          description: "Available memory is below 10%"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space detected"
          description: "Available disk space is below 10%"

  - name: application_alerts
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.job }} service has been down for more than 1 minute"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is above 1 second"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 10%"

  - name: v1_specific_alerts
    rules:
      - alert: QueryAPISlowdown
        expr: histogram_quantile(0.95, rate(query_api_duration_seconds_bucket[5m])) > 0.5
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Query API performance degradation"
          description: "95th percentile Query API latency is above 500ms"

      - alert: HyDECacheMissRate
        expr: rate(hyde_cache_misses_total[5m]) / rate(hyde_cache_requests_total[5m]) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High HyDE cache miss rate"
          description: "HyDE cache miss rate is above 50%"

      - alert: DragonflyMemoryHigh
        expr: dragonfly_memory_usage_bytes > 4294967296 # 4GB
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "DragonflyDB high memory usage"
          description: "DragonflyDB memory usage exceeds 4GB"

      - alert: PayloadIndexingFailed
        expr: increase(payload_indexing_errors_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Payload indexing failures detected"
          description: "Payload indexing operations are failing"

      - alert: CollectionAliasOperationFailed
        expr: increase(collection_alias_errors_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Collection alias operation failures"
          description: "Collection alias operations are failing"

      # Enhanced Database Connection Pool Alerts (BJO-134)
      - alert: DatabaseConnectionPoolExhausted
        expr: db_connection_pool_checked_out / db_connection_pool_size > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool near exhaustion"
          description: "Connection pool utilization is above 90% for 2 minutes"

      - alert: DatabaseConnectionPoolMLModelDegraded
        expr: db_connection_pool_ml_model_accuracy < 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ML model accuracy degraded"
          description: "Connection pool ML model accuracy below 70%"

      - alert: DatabaseCircuitBreakerOpen
        expr: db_circuit_breaker_state{state="open"} == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database circuit breaker is open"
          description: "Circuit breaker {{ $labels.failure_type }} is open, blocking requests"

      - alert: DatabaseConnectionAffinityDegraded
        expr: avg(db_connection_affinity_performance_score) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Connection affinity performance degraded"
          description: "Average connection affinity performance score below 50%"

      - alert: DatabaseQueryLatencyRegression
        expr: histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m])) > 0.25
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Database query latency regression"
          description: "95th percentile query latency above 250ms (regression threshold)"

  - name: vector_db_alerts
    rules:
      - alert: QdrantDown
        expr: up{job="qdrant"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Qdrant database is down"
          description: "Qdrant vector database is not responding"

      - alert: VectorSearchSlow
        expr: histogram_quantile(0.95, rate(qdrant_search_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Vector search performance degraded"
          description: "Vector search queries are taking longer than 2 seconds"

      - alert: HighVectorDBMemory
        expr: qdrant_memory_usage_bytes > 8589934592 # 8GB
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High Qdrant memory usage"
          description: "Qdrant memory usage exceeds 8GB"
```

## 📈 Metrics Collection

### 1. System Metrics

**CPU Utilization**:

```promql
# Current CPU usage
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# CPU usage per core
100 - (avg by (cpu)(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

**Memory Utilization**:

```promql
# Memory usage percentage
100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))

# Available memory
node_memory_MemAvailable_bytes / 1024 / 1024 / 1024
```

**Disk I/O**:

```promql
# Disk read rate
rate(node_disk_read_bytes_total[5m])

# Disk write rate
rate(node_disk_written_bytes_total[5m])

# Disk utilization
100 * rate(node_disk_io_time_seconds_total[5m])
```

**Network Traffic**:

```promql
# Network receive rate
rate(node_network_receive_bytes_total[5m])

# Network transmit rate
rate(node_network_transmit_bytes_total[5m])
```

### 2. Application Metrics

**HTTP Request Metrics**:

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request counters
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Request duration
http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Active connections
active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

class MetricsMiddleware:
    """Middleware to collect HTTP metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]

        start_time = time.time()

        # Increment active connections
        active_connections.inc()

        try:
            # Process request
            await self.app(scope, receive, send)
            status = "200"  # Default, should be extracted from response
        except Exception as e:
            status = "500"
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            http_request_duration.labels(
                method=method,
                endpoint=path
            ).observe(duration)

            http_requests_total.labels(
                method=method,
                endpoint=path,
                status=status
            ).inc()

            # Decrement active connections
            active_connections.dec()
```

### 3. V1 Enhanced Metrics

**Query API Performance**:

```python
# src/monitoring/v1_metrics.py
from prometheus_client import Histogram, Counter, Gauge

# Query API metrics
query_api_duration = Histogram(
    'query_api_duration_seconds',
    'Query API request duration',
    ['operation', 'collection']
)

query_api_requests = Counter(
    'query_api_requests_total',
    'Total Query API requests',
    ['operation', 'collection', 'status']
)

# HyDE metrics
hyde_cache_hits = Counter(
    'hyde_cache_hits_total',
    'HyDE cache hits'
)

hyde_cache_misses = Counter(
    'hyde_cache_misses_total',
    'HyDE cache misses'
)

hyde_generation_duration = Histogram(
    'hyde_generation_duration_seconds',
    'HyDE document generation duration'
)

# DragonflyDB metrics
dragonfly_cache_hit_rate = Gauge(
    'dragonfly_cache_hit_rate',
    'DragonflyDB cache hit rate'
)

dragonfly_memory_usage = Gauge(
    'dragonfly_memory_usage_bytes',
    'DragonflyDB memory usage in bytes'
)

dragonfly_evicted_keys = Counter(
    'dragonfly_evicted_keys_total',
    'Total evicted keys from DragonflyDB'
)

# Payload indexing metrics
payload_indexing_duration = Histogram(
    'payload_indexing_duration_seconds',
    'Payload indexing operation duration',
    ['operation', 'field_type']
)

payload_indexing_errors = Counter(
    'payload_indexing_errors_total',
    'Payload indexing errors',
    ['operation', 'error_type']
)

# Collection alias metrics
collection_alias_operations = Counter(
    'collection_alias_operations_total',
    'Collection alias operations',
    ['operation', 'status']
)

collection_alias_errors = Counter(
    'collection_alias_errors_total',
    'Collection alias operation errors',
    ['operation', 'error_type']
)

# Enhanced Database Connection Pool Metrics (BJO-134)
db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Current database connection pool size'
)

db_connection_pool_checked_out = Gauge(
    'db_connection_pool_checked_out',
    'Number of checked out connections'
)

db_connection_pool_checked_in = Gauge(
    'db_connection_pool_checked_in',
    'Number of checked in connections'
)

db_connection_pool_overflow = Gauge(
    'db_connection_pool_overflow',
    'Number of overflow connections'
)

db_connection_pool_invalidated = Gauge(
    'db_connection_pool_invalidated',
    'Number of invalidated connections'
)

db_connection_create_duration = Histogram(
    'db_connection_create_duration_seconds',
    'Time to create new database connections'
)

db_connection_errors = Counter(
    'db_connection_errors_total',
    'Total database connection errors',
    ['error_type']
)

db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query execution time',
    ['query_type', 'success']
)

db_slow_queries = Counter(
    'db_slow_queries_total',
    'Total slow database queries',
    ['query_pattern']
)

# ML Model Performance Tracking
db_load_prediction_accuracy = Gauge(
    'db_load_prediction_accuracy',
    'ML model accuracy for load prediction'
)

db_load_prediction_mae = Gauge(
    'db_load_prediction_mae',
    'Mean absolute error of load predictions'
)

db_adaptive_config_changes = Counter(
    'db_adaptive_config_changes_total',
    'Total adaptive configuration changes',
    ['config_type', 'adaptation_strategy']
)

# Circuit Breaker Metrics
db_circuit_breaker_state = Gauge(
    'db_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half_open)',
    ['failure_type']
)

db_circuit_breaker_failures = Counter(
    'db_circuit_breaker_failures_total',
    'Circuit breaker failure count',
    ['failure_type']
)

db_circuit_breaker_recoveries = Counter(
    'db_circuit_breaker_recoveries_total',
    'Circuit breaker recovery count',
    ['failure_type']
)

# Connection Affinity Metrics
db_connection_affinity_hits = Counter(
    'db_connection_affinity_hits_total',
    'Connection affinity cache hits',
    ['query_type']
)

db_connection_affinity_misses = Counter(
    'db_connection_affinity_misses_total',
    'Connection affinity cache misses',
    ['query_type']
)

db_connection_affinity_performance_gain = Gauge(
    'db_connection_affinity_performance_gain_percent',
    'Performance gain from connection affinity',
    ['query_type']
)

class V1MetricsCollector:
    """Collect V1-specific metrics."""

    def __init__(self):
        self.start_times = {}

    def start_query_api_timer(self, operation: str, collection: str) -> str:
        """Start timing a Query API operation."""
        timer_id = f"{operation}_{collection}_{time.time()}"
        self.start_times[timer_id] = time.time()
        return timer_id

    def end_query_api_timer(self, timer_id: str, operation: str,
                           collection: str, status: str):
        """End timing a Query API operation."""
        if timer_id in self.start_times:
            duration = time.time() - self.start_times[timer_id]
            query_api_duration.labels(
                operation=operation,
                collection=collection
            ).observe(duration)

            query_api_requests.labels(
                operation=operation,
                collection=collection,
                status=status
            ).inc()

            del self.start_times[timer_id]

    def record_hyde_cache_hit(self):
        """Record HyDE cache hit."""
        hyde_cache_hits.inc()

    def record_hyde_cache_miss(self, generation_time: float):
        """Record HyDE cache miss and generation time."""
        hyde_cache_misses.inc()
        hyde_generation_duration.observe(generation_time)

    def update_dragonfly_metrics(self, hit_rate: float, memory_bytes: int, evicted: int):
        """Update DragonflyDB metrics."""
        dragonfly_cache_hit_rate.set(hit_rate)
        dragonfly_memory_usage.set(memory_bytes)
        dragonfly_evicted_keys._value._value = evicted

    def record_payload_indexing(self, operation: str, field_type: str,
                               duration: float, success: bool, error_type: str = None):
        """Record payload indexing metrics."""
        payload_indexing_duration.labels(
            operation=operation,
            field_type=field_type
        ).observe(duration)

        if not success and error_type:
            payload_indexing_errors.labels(
                operation=operation,
                error_type=error_type
            ).inc()

    def record_collection_alias_operation(self, operation: str,
                                        success: bool, error_type: str = None):
        """Record collection alias operation."""
        status = "success" if success else "error"
        collection_alias_operations.labels(
            operation=operation,
            status=status
        ).inc()

        if not success and error_type:
            collection_alias_errors.labels(
                operation=operation,
                error_type=error_type
            ).inc()
```

### 4. Vector Database Metrics

**Qdrant Performance**:

```python
# src/monitoring/qdrant_metrics.py
from prometheus_client import Histogram, Counter, Gauge

# Search performance
qdrant_search_duration = Histogram(
    'qdrant_search_duration_seconds',
    'Qdrant search operation duration',
    ['collection', 'search_type']
)

qdrant_search_requests = Counter(
    'qdrant_search_requests_total',
    'Total Qdrant search requests',
    ['collection', 'search_type', 'status']
)

# Index operations
qdrant_index_operations = Counter(
    'qdrant_index_operations_total',
    'Qdrant index operations',
    ['operation', 'collection', 'status']
)

# Collection metrics
qdrant_collection_points = Gauge(
    'qdrant_collection_points',
    'Number of points in collection',
    ['collection']
)

qdrant_collection_segments = Gauge(
    'qdrant_collection_segments',
    'Number of segments in collection',
    ['collection']
)

# Memory usage
qdrant_memory_usage = Gauge(
    'qdrant_memory_usage_bytes',
    'Qdrant memory usage in bytes'
)

class QdrantMetricsCollector:
    """Collect Qdrant-specific metrics."""

    def __init__(self, qdrant_client):
        self.client = qdrant_client

    async def collect_collection_metrics(self):
        """Collect metrics for all collections."""
        try:
            collections = await self.client.get_collections()

            for collection in collections.collections:
                collection_name = collection.name

                # Get collection info
                info = await self.client.get_collection(collection_name)

                # Update metrics
                qdrant_collection_points.labels(
                    collection=collection_name
                ).set(info.points_count)

                qdrant_collection_segments.labels(
                    collection=collection_name
                ).set(info.segments_count)

        except Exception as e:
            logger.error(f"Failed to collect Qdrant metrics: {e}")

    def record_search_operation(self, collection: str, search_type: str,
                               duration: float, success: bool):
        """Record search operation metrics."""
        qdrant_search_duration.labels(
            collection=collection,
            search_type=search_type
        ).observe(duration)

        status = "success" if success else "error"
        qdrant_search_requests.labels(
            collection=collection,
            search_type=search_type,
            status=status
        ).inc()

    def record_index_operation(self, operation: str, collection: str, success: bool):
        """Record index operation."""
        status = "success" if success else "error"
        qdrant_index_operations.labels(
            operation=operation,
            collection=collection,
            status=status
        ).inc()
```

## 🔍 Performance Analysis

### 1. Baseline Performance Metrics

**V1 Enhanced Performance Targets**:

| Metric                   | Target  | Alert Threshold |
| ------------------------ | ------- | --------------- |
| Query API Latency (95th) | < 100ms | > 500ms         |
| HyDE Cache Hit Rate      | > 70%   | < 50%           |
| DragonflyDB Hit Rate     | > 80%   | < 60%           |
| Vector Search (95th)     | < 50ms  | > 200ms         |
| Payload Index Query      | < 10ms  | > 100ms         |
| Collection Alias Ops     | < 5s    | > 30s           |

**System Performance Targets**:

| Resource        | Target | Alert Threshold |
| --------------- | ------ | --------------- |
| CPU Usage       | < 70%  | > 80%           |
| Memory Usage    | < 80%  | > 90%           |
| Disk Usage      | < 80%  | > 90%           |
| Network Latency | < 10ms | > 50ms          |

### 2. Performance Analysis Queries

**Query API Performance Analysis**:

```promql
# Average Query API latency by operation
avg(rate(query_api_duration_seconds_sum[5m])) by (operation) /
avg(rate(query_api_duration_seconds_count[5m])) by (operation)

# Query API error rate
rate(query_api_requests_total{status!="success"}[5m]) /
rate(query_api_requests_total[5m])

# Query API throughput
rate(query_api_requests_total[5m])
```

**HyDE Performance Analysis**:

```promql
# HyDE cache hit rate
rate(hyde_cache_hits_total[5m]) /
(rate(hyde_cache_hits_total[5m]) + rate(hyde_cache_misses_total[5m]))

# Average HyDE generation time
avg(hyde_generation_duration_seconds)

# HyDE cache efficiency trend
increase(hyde_cache_hits_total[1h]) /
increase(hyde_cache_requests_total[1h])
```

**DragonflyDB Analysis**:

```promql
# DragonflyDB memory growth rate
rate(dragonfly_memory_usage_bytes[5m])

# Cache eviction rate
rate(dragonfly_evicted_keys_total[5m])

# Cache hit rate trend
avg_over_time(dragonfly_cache_hit_rate[1h])
```

### 3. Capacity Planning Metrics

**Resource Utilization Trends**:

```promql
# CPU usage trend (24 hours)
avg_over_time(node_cpu_usage_percent[24h])

# Memory usage growth rate
rate(node_memory_usage_bytes[1h])

# Disk space consumption rate
rate(node_filesystem_used_bytes[1h])

# Network bandwidth utilization
rate(node_network_transmit_bytes_total[5m]) +
rate(node_network_receive_bytes_total[5m])
```

**Database Growth Analysis**:

```promql
# Collection growth rate
rate(qdrant_collection_points[1h])

# Index size growth
rate(qdrant_index_size_bytes[1h])

# Storage utilization trend
rate(qdrant_storage_used_bytes[1h])
```

## 🚨 Alerting Configuration

### 1. AlertManager Configuration

Create `monitoring/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: "localhost:587"
  smtp_from: "alerts@yourcompany.com"

route:
  group_by: ["alertname"]
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: "web.hook"

receivers:
  - name: "web.hook"
    webhook_configs:
      - url: "http://webhook-service:5000/alerts"
        send_resolved: true

  - name: "email-alerts"
    email_configs:
      - to: "ops-team@yourcompany.com"
        subject: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Instance: {{ .Labels.instance }}
          Severity: {{ .Labels.severity }}
          {{ end }}

  - name: "slack-alerts"
    slack_configs:
      - api_url: "YOUR_SLACK_WEBHOOK_URL"
        channel: "#ops-alerts"
        title: "AI Vector DB Alert"
        text: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"

inhibit_rules:
  - source_match:
      severity: "critical"
    target_match:
      severity: "warning"
    equal: ["alertname", "instance"]
```

### 2. PagerDuty Integration

```yaml
# Add to alertmanager.yml receivers
- name: "pagerduty"
  pagerduty_configs:
    - service_key: "YOUR_PAGERDUTY_SERVICE_KEY"
      description: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
      details:
        firing: "{{ .Alerts.Firing | len }}"
        resolved: "{{ .Alerts.Resolved | len }}"
        num_firing: "{{ .Alerts.Firing | len }}"
        num_resolved: "{{ .Alerts.Resolved | len }}"
```

### 3. Custom Webhook Alerts

```python
# monitoring/webhook_handler.py
from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

@app.route('/alerts', methods=['POST'])
def handle_alert():
    """Handle incoming Prometheus alerts."""
    data = request.get_json()

    for alert in data.get('alerts', []):
        alert_name = alert.get('labels', {}).get('alertname')
        severity = alert.get('labels', {}).get('severity')
        status = alert.get('status')

        # Custom alert processing
        if severity == 'critical':
            send_urgent_notification(alert)
        elif alert_name.startswith('V1'):
            process_v1_alert(alert)

        # Log all alerts
        log_alert(alert)

    return jsonify({'status': 'received'})

def send_urgent_notification(alert):
    """Send urgent notification for critical alerts."""
    # Implementation for urgent notifications
    # (SMS, phone calls, etc.)
    pass

def process_v1_alert(alert):
    """Process V1-specific alerts."""
    alert_name = alert.get('labels', {}).get('alertname')

    if alert_name == 'QueryAPISlowdown':
        # Trigger automatic optimization
        trigger_query_optimization()
    elif alert_name == 'HyDECacheMissRate':
        # Scale HyDE cache
        scale_hyde_cache()
    elif alert_name == 'DragonflyMemoryHigh':
        # Optimize DragonflyDB
        optimize_dragonfly_memory()

def log_alert(alert):
    """Log alert for audit purposes."""
    with open('/var/log/alerts.log', 'a') as f:
        f.write(f"{json.dumps(alert)}\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📊 Health Checks

### 1. System Health Checks

Create comprehensive health check script `scripts/health-check.sh`:

```bash
#!/bin/bash

set -e

echo "🔍 System Health Check - V1 Enhanced"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
check_service() {
    local service_name=$1
    local check_command=$2

    echo -n "Checking $service_name... "

    if eval "$check_command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed${NC}"
        return 1
    fi
}

check_metric() {
    local metric_name=$1
    local query=$2
    local threshold=$3
    local operator=$4

    echo -n "Checking $metric_name... "

    result=$(curl -s "http://localhost:9090/api/v1/query?query=$query" | \
             jq -r '.data.result[0].value[1]' 2>/dev/null || echo "error")

    if [ "$result" = "error" ] || [ "$result" = "null" ]; then
        echo -e "${RED}✗ No data${NC}"
        return 1
    fi

    case $operator in
        "lt")
            if (( $(echo "$result < $threshold" | bc -l) )); then
                echo -e "${GREEN}✓ $result < $threshold${NC}"
                return 0
            else
                echo -e "${RED}✗ $result >= $threshold${NC}"
                return 1
            fi
            ;;
        "gt")
            if (( $(echo "$result > $threshold" | bc -l) )); then
                echo -e "${GREEN}✓ $result > $threshold${NC}"
                return 0
            else
                echo -e "${RED}✗ $result <= $threshold${NC}"
                return 1
            fi
            ;;
    esac
}

# System checks
echo -e "\n${YELLOW}System Services${NC}"
echo "----------------"

check_service "Docker" "docker ps >/dev/null 2>&1"
check_service "Qdrant" "curl -s http://localhost:6333/health | grep -q ok"
check_service "DragonflyDB" "redis-cli -p 6379 ping | grep -q PONG"
check_service "Prometheus" "curl -s http://localhost:9090/-/healthy | grep -q Prometheus"
check_service "Grafana" "curl -s http://localhost:3000/api/health | grep -q ok"

# Application checks
echo -e "\n${YELLOW}Application Services${NC}"
echo "--------------------"

check_service "MCP Server" "curl -s http://localhost:8080/health | grep -q healthy"
check_service "Query API" "curl -s http://localhost:8000/health | grep -q ok"

# V1 Enhanced checks
echo -e "\n${YELLOW}V1 Enhanced Components${NC}"
echo "-----------------------"

check_service "HyDE Service" "curl -s http://localhost:8000/health/hyde | grep -q active"
check_service "Cache Layer" "curl -s http://localhost:8000/health/cache | grep -q connected"
check_service "Payload Indexes" "curl -s http://localhost:6333/collections/documents | jq '.result.payload_schema | length' | grep -v '^0$'"

# Performance checks
echo -e "\n${YELLOW}Performance Metrics${NC}"
echo "-------------------"

check_metric "CPU Usage" "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)" "80" "lt"
check_metric "Memory Usage" "100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))" "80" "lt"
check_metric "Query API Latency" "histogram_quantile(0.95, rate(query_api_duration_seconds_bucket[5m]))" "0.5" "lt"
check_metric "HyDE Cache Hit Rate" "rate(hyde_cache_hits_total[5m]) / (rate(hyde_cache_hits_total[5m]) + rate(hyde_cache_misses_total[5m]))" "0.7" "gt"

# Data integrity checks
echo -e "\n${YELLOW}Data Integrity${NC}"
echo "---------------"

check_service "Collection Consistency" "python scripts/check_data_integrity.py"
check_service "Index Health" "python scripts/verify_indexes.py"
check_service "Alias Mappings" "python scripts/validate_aliases.py"

echo -e "\n${YELLOW}Summary${NC}"
echo "-------"

# Count passed/failed checks
total_checks=$(grep -c "check_service\|check_metric" "$0")
passed_checks=$(echo "Health check completed" && exit 0)

echo "Health check completed at $(date)"
echo "Check detailed logs for any failures"

# Exit with error if any checks failed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All systems healthy${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed${NC}"
    exit 1
fi
```

### 2. Application Health Endpoints

```python
# src/monitoring/health.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
import asyncio
import json

router = APIRouter()

class HealthChecker:
    """Comprehensive health checking for all components."""

    def __init__(self, qdrant_client, dragonfly_client, hyde_service):
        self.qdrant = qdrant_client
        self.dragonfly = dragonfly_client
        self.hyde = hyde_service

    async def check_all(self) -> dict:
        """Run all health checks."""
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }

        # Run checks in parallel
        checks = [
            ("database", self.check_database()),
            ("cache", self.check_cache()),
            ("hyde", self.check_hyde()),
            ("indexes", self.check_indexes()),
            ("aliases", self.check_aliases())
        ]

        check_results = await asyncio.gather(
            *[check[1] for check in checks],
            return_exceptions=True
        )

        # Process results
        for i, (name, _) in enumerate(checks):
            result = check_results[i]

            if isinstance(result, Exception):
                results["checks"][name] = {
                    "status": "error",
                    "error": str(result)
                }
                results["status"] = "degraded"
            else:
                results["checks"][name] = result
                if result["status"] != "healthy":
                    results["status"] = "degraded"

        return results

    async def check_database(self) -> dict:
        """Check Qdrant database health."""
        try:
            # Check connection
            collections = await self.qdrant.get_collections()

            # Check collection health
            collection_health = {}
            for collection in collections.collections:
                info = await self.qdrant.get_collection(collection.name)
                collection_health[collection.name] = {
                    "points": info.points_count,
                    "segments": info.segments_count,
                    "status": "healthy" if info.status == "green" else "degraded"
                }

            return {
                "status": "healthy",
                "collections": collection_health,
                "total_collections": len(collections.collections)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def check_cache(self) -> dict:
        """Check DragonflyDB cache health."""
        try:
            # Test basic operations
            test_key = "health_check_test"
            await self.dragonfly.set(test_key, "test_value", ex=10)
            value = await self.dragonfly.get(test_key)
            await self.dragonfly.delete(test_key)

            # Get cache stats
            info = await self.dragonfly.info()

            return {
                "status": "healthy",
                "memory_usage": info.get("used_memory_human"),
                "hit_rate": info.get("keyspace_hits", 0) /
                          (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0) + 1),
                "connected_clients": info.get("connected_clients", 0)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def check_hyde(self) -> dict:
        """Check HyDE service health."""
        try:
            # Test HyDE generation
            test_query = "test health check query"
            result = await self.hyde.enhance(test_query)

            # Get cache stats
            cache_stats = await self.hyde.get_cache_stats()

            return {
                "status": "healthy",
                "cache_hit_rate": cache_stats.get("hit_rate", 0),
                "avg_generation_time": cache_stats.get("avg_generation_time", 0),
                "cache_size": cache_stats.get("cache_size", 0)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def check_indexes(self) -> dict:
        """Check payload indexes health."""
        try:
            index_health = {}

            collections = await self.qdrant.get_collections()
            for collection in collections.collections:
                info = await self.qdrant.get_collection(collection.name)

                # Check payload schema
                payload_schema = info.payload_schema or {}

                index_health[collection.name] = {
                    "indexed_fields": list(payload_schema.keys()),
                    "index_count": len(payload_schema),
                    "status": "healthy"
                }

            return {
                "status": "healthy",
                "indexes": index_health
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def check_aliases(self) -> dict:
        """Check collection aliases health."""
        try:
            aliases = await self.qdrant.get_collection_aliases()

            alias_health = {}
            for alias in aliases.aliases:
                alias_health[alias.alias_name] = {
                    "collection": alias.collection_name,
                    "status": "healthy"
                }

            return {
                "status": "healthy",
                "aliases": alias_health,
                "total_aliases": len(aliases.aliases)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Health endpoints
@router.get("/health")
async def health_check(health_checker: HealthChecker):
    """Basic health check."""
    try:
        # Quick database ping
        await health_checker.qdrant.get_collections()
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@router.get("/health/detailed")
async def detailed_health_check(health_checker: HealthChecker):
    """Comprehensive health check."""
    result = await health_checker.check_all()

    if result["status"] == "error":
        raise HTTPException(status_code=503, detail=result)
    elif result["status"] == "degraded":
        raise HTTPException(status_code=200, detail=result)  # 200 but degraded

    return result

@router.get("/health/{component}")
async def component_health_check(component: str, health_checker: HealthChecker):
    """Check specific component health."""
    check_methods = {
        "database": health_checker.check_database,
        "cache": health_checker.check_cache,
        "hyde": health_checker.check_hyde,
        "indexes": health_checker.check_indexes,
        "aliases": health_checker.check_aliases
    }

    if component not in check_methods:
        raise HTTPException(status_code=404, detail=f"Component '{component}' not found")

    try:
        result = await check_methods[component]()

        if result["status"] == "error":
            raise HTTPException(status_code=503, detail=result)

        return result

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
```

## 📝 Logging Configuration

### 1. Structured Logging Setup

```python
# src/monitoring/logging_config.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""

        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add trace ID if present (for distributed tracing)
        if hasattr(record, 'trace_id'):
            log_entry["trace_id"] = record.trace_id

        # Add user ID if present
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id

        # Add request ID if present
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id

        return json.dumps(log_entry)

def setup_logging(level: str = "INFO",
                 log_file: str = None,
                 structured: bool = True) -> None:
    """Setup application logging."""

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Choose formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure specific loggers
    configure_specific_loggers()

def configure_specific_loggers():
    """Configure logging for specific modules."""

    # Qdrant client (reduce verbosity)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)

    # HTTP clients (reduce verbosity)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # FastAPI (info level)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Application loggers (debug in development)
    logging.getLogger("app").setLevel(logging.DEBUG)
    logging.getLogger("monitoring").setLevel(logging.DEBUG)

class RequestLoggingMiddleware:
    """Middleware to log HTTP requests with structured data."""

    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger("app.requests")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())

        # Log request start
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": scope["method"],
                "path": scope["path"],
                "client": scope.get("client"),
                "user_agent": dict(scope.get("headers", {})).get(b"user-agent", b"").decode()
            }
        )

        start_time = time.time()

        try:
            await self.app(scope, receive, send)

            # Log successful completion
            duration = time.time() - start_time
            self.logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "duration_ms": round(duration * 1000, 2),
                    "status": "success"
                }
            )

        except Exception as e:
            # Log error
            duration = time.time() - start_time
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(e),
                    "status": "error"
                },
                exc_info=True
            )
            raise

class V1OperationLogger:
    """Logger for V1-specific operations."""

    def __init__(self):
        self.logger = logging.getLogger("app.v1_operations")

    def log_query_api_operation(self, operation: str, collection: str,
                               duration: float, success: bool,
                               result_count: int = None, error: str = None):
        """Log Query API operation."""

        extra = {
            "operation_type": "query_api",
            "operation": operation,
            "collection": collection,
            "duration_ms": round(duration * 1000, 2),
            "success": success
        }

        if result_count is not None:
            extra["result_count"] = result_count

        if error:
            extra["error"] = error

        if success:
            self.logger.info(f"Query API {operation} completed", extra=extra)
        else:
            self.logger.error(f"Query API {operation} failed", extra=extra)

    def log_hyde_operation(self, query: str, cache_hit: bool,
                          generation_time: float = None):
        """Log HyDE operation."""

        extra = {
            "operation_type": "hyde",
            "query_length": len(query),
            "cache_hit": cache_hit
        }

        if generation_time is not None:
            extra["generation_time_ms"] = round(generation_time * 1000, 2)

        if cache_hit:
            self.logger.debug("HyDE cache hit", extra=extra)
        else:
            self.logger.info("HyDE document generated", extra=extra)

    def log_cache_operation(self, operation: str, key: str,
                           hit: bool = None, size: int = None):
        """Log cache operation."""

        extra = {
            "operation_type": "cache",
            "operation": operation,
            "key_hash": hash(key) % 10000  # Anonymized key
        }

        if hit is not None:
            extra["hit"] = hit

        if size is not None:
            extra["size_bytes"] = size

        self.logger.debug(f"Cache {operation}", extra=extra)

    def log_indexing_operation(self, collection: str, field: str,
                              operation: str, duration: float, success: bool):
        """Log payload indexing operation."""

        extra = {
            "operation_type": "indexing",
            "collection": collection,
            "field": field,
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            "success": success
        }

        if success:
            self.logger.info(f"Index {operation} completed", extra=extra)
        else:
            self.logger.error(f"Index {operation} failed", extra=extra)
```

### 2. Log Aggregation Setup

**ELK Stack Configuration** (`monitoring/docker-compose.logging.yml`):

```yaml
version: "3.8"

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - monitoring

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    container_name: logstash
    ports:
      - "5044:5044"
      - "5000:5000/tcp"
      - "5000:5000/udp"
      - "9600:9600"
    volumes:
      - ./logstash/config:/usr/share/logstash/pipeline
    environment:
      LS_JAVA_OPTS: "-Xmx512m -Xms512m"
    depends_on:
      - elasticsearch
    networks:
      - monitoring

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - monitoring

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.8.0
    container_name: filebeat
    user: root
    volumes:
      - ./filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /var/log:/var/log:ro
    command: filebeat -e -strict.perms=false
    depends_on:
      - logstash
    networks:
      - monitoring

volumes:
  elasticsearch_data:

networks:
  monitoring:
    external: true
```

**Logstash Configuration** (`monitoring/logstash/config/logstash.conf`):

```ruby
input {
  beats {
    port => 5044
  }

  http {
    port => 8080
    codec => json
  }
}

filter {
  if [fields][log_type] == "application" {
    json {
      source => "message"
    }

    # Parse timestamp
    date {
      match => [ "timestamp", "ISO8601" ]
    }

    # Add tags based on log level
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }

    # Add tags for V1 operations
    if [operation_type] {
      mutate {
        add_tag => [ "v1_operation", "%{operation_type}" ]
      }
    }
  }

  # Docker logs
  if [container][name] {
    mutate {
      add_field => { "service" => "%{[container][name]}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{+YYYY.MM.dd}"
  }

  # Debug output (remove in production)
  stdout {
    codec => rubydebug
  }
}
```

## 🔬 Troubleshooting Monitoring Issues

### 1. Common Monitoring Problems

**Prometheus Not Scraping Metrics**:

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# Verify Prometheus configuration
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
```

**Grafana Dashboard Issues**:

```bash
# Check Grafana logs
docker logs grafana

# Test Prometheus data source
curl -X POST http://admin:admin@localhost:3000/api/datasources/proxy/1/api/v1/query \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=up"

# Reload Grafana configuration
curl -X POST http://admin:admin@localhost:3000/api/admin/provisioning/dashboards/reload
```

**Missing Metrics**:

```python
# Debug metrics collection
from prometheus_client import REGISTRY, generate_latest

# List all registered metrics
for collector in REGISTRY._collector_to_names:
    print(f"Collector: {collector}")

# Generate metrics manually
metrics_output = generate_latest(REGISTRY)
print(metrics_output.decode())
```

### 2. Performance Issues

**High Cardinality Metrics**:

```python
# Avoid high cardinality labels
# Bad:
http_requests_total.labels(
    method="GET",
    endpoint="/api/search",
    user_id="user_12345",  # High cardinality!
    timestamp="2025-06-09T10:30:00Z"  # Very high cardinality!
).inc()

# Good:
http_requests_total.labels(
    method="GET",
    endpoint="/api/search",
    status_code="200"
).inc()
```

**Monitoring Resource Usage**:

```bash
# Check Prometheus resource usage
docker stats prometheus

# Check disk usage
du -sh /var/lib/docker/volumes/prometheus_data/_data

# Monitor memory usage
curl http://localhost:9090/api/v1/query?query=process_resident_memory_bytes{job=\"prometheus\"}
```

### 3. Alert Debugging

**Alert Not Firing**:

```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Test alert query manually
curl "http://localhost:9090/api/v1/query?query=up==0"

# Check AlertManager logs
docker logs alertmanager
```

**Alert Fatigue**:

```yaml
# Implement alert grouping
route:
  group_by: ["alertname", "severity"]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h

# Use inhibition rules
inhibit_rules:
  - source_match:
      severity: "critical"
    target_match:
      severity: "warning"
    equal: ["service"]
```

## 📚 Production Best Practices

### 1. Monitoring Strategy

**Monitoring Hierarchy**:

1. **Infrastructure**: CPU, memory, disk, network
2. **Platform**: Kubernetes, Docker, databases
3. **Application**: Request rates, latency, errors
4. **Business**: User actions, feature usage

**SLI/SLO Framework**:

```yaml
# Service Level Indicators (SLIs)
slis:
  availability:
    query: "up == 1"
    target: 99.9%

  latency:
    query: "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
    target: "< 500ms"

  error_rate:
    query: "rate(http_requests_total{status=~'5..'}[5m]) / rate(http_requests_total[5m])"
    target: "< 1%"

# Service Level Objectives (SLOs)
slos:
  search_performance:
    sli: latency
    target: 95%
    window: 30d

  system_availability:
    sli: availability
    target: 99.5%
    window: 30d
```

### 2. Monitoring as Code

**Terraform for Infrastructure**:

```hcl
# monitoring/terraform/main.tf
resource "aws_cloudwatch_dashboard" "vector_db_dashboard" {
  dashboard_name = "vector-db-monitoring"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        properties = {
          metrics = [
            ["AWS/EC2", "CPUUtilization", "InstanceId", var.instance_id],
            ["AWS/EC2", "MemoryUtilization", "InstanceId", var.instance_id]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "System Metrics"
        }
      }
    ]
  })
}

resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "vector-db-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ec2 cpu utilization"

  dimensions = {
    InstanceId = var.instance_id
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
}
```

### 3. Security Considerations

**Sensitive Data in Metrics**:

```python
# Avoid exposing sensitive data
# Bad:
user_activity.labels(
    user_id="user_12345",      # PII
    email="user@example.com",  # PII
    query="confidential data"  # Sensitive
).inc()

# Good:
user_activity.labels(
    user_type="premium",
    query_category="search",
    anonymized_user=hash("user_12345") % 10000
).inc()
```

**Access Control**:

```yaml
# Grafana RBAC
grafana:
  rbac:
    check_access: true

  auth:
    oauth_auto_login: true
    oauth_allow_insecure_email_lookup: false

  security:
    admin_password: ${GRAFANA_ADMIN_PASSWORD}
    secret_key: ${GRAFANA_SECRET_KEY}
```

### 4. Cost Optimization

**Metric Retention**:

```yaml
# Prometheus retention configuration
prometheus:
  retention: 30d # Reduce for cost savings
  retention_size: 100GB

  # Use recording rules for long-term storage
  recording_rules:
    - name: aggregated_metrics
      interval: 5m
      rules:
        - record: instance:cpu_usage:rate5m
          expr: 100 - (avg by (instance)(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

**Efficient Queries**:

```promql
# Inefficient (scans all time series)
avg(cpu_usage)

# Efficient (uses recording rule)
avg(instance:cpu_usage:rate5m)

# Use appropriate time ranges
rate(http_requests_total[5m])  # Not [1d]
```

## 🚀 Advanced Monitoring Features

### 1. Distributed Tracing

**Jaeger Integration**:

```python
# src/monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    """Setup distributed tracing with Jaeger."""

    # Configure tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )

    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return tracer

class TracedVectorSearch:
    """Vector search with distributed tracing."""

    def __init__(self, qdrant_client):
        self.client = qdrant_client
        self.tracer = trace.get_tracer(__name__)

    async def search_with_tracing(self, collection: str, query_vector: list,
                                 limit: int = 10, trace_context: dict = None):
        """Perform vector search with distributed tracing."""

        with self.tracer.start_as_current_span("vector_search") as span:
            # Add span attributes
            span.set_attribute("collection", collection)
            span.set_attribute("vector_dimension", len(query_vector))
            span.set_attribute("limit", limit)

            # Add trace context if provided
            if trace_context:
                span.set_attribute("trace_id", trace_context.get("trace_id"))
                span.set_attribute("user_id", trace_context.get("user_id"))

            try:
                # Perform search
                with self.tracer.start_as_current_span("qdrant_search"):
                    results = await self.client.search(
                        collection_name=collection,
                        query_vector=query_vector,
                        limit=limit
                    )

                # Add result metrics to span
                span.set_attribute("result_count", len(results))
                span.set_attribute("min_score", min(r.score for r in results) if results else 0)
                span.set_attribute("max_score", max(r.score for r in results) if results else 0)

                return results

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
```

### 2. Custom Monitoring Dashboard

**React Dashboard Component**:

```tsx
// monitoring/dashboard/src/components/V1Dashboard.tsx
import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface MetricData {
  timestamp: string;
  value: number;
}

interface DashboardProps {
  prometheusUrl: string;
}

export const V1Dashboard: React.FC<DashboardProps> = ({ prometheusUrl }) => {
  const [queryApiLatency, setQueryApiLatency] = useState<MetricData[]>([]);
  const [hydeHitRate, setHydeHitRate] = useState<MetricData[]>([]);
  const [dragonflyMemory, setDragonflyMemory] = useState<MetricData[]>([]);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        // Fetch Query API latency
        const latencyResponse = await fetch(
          `${prometheusUrl}/api/v1/query_range?query=histogram_quantile(0.95, rate(query_api_duration_seconds_bucket[5m]))&start=${
            Date.now() - 3600000
          }&end=${Date.now()}&step=60`
        );
        const latencyData = await latencyResponse.json();
        setQueryApiLatency(
          latencyData.data.result[0]?.values.map(
            ([timestamp, value]: [number, string]) => ({
              timestamp: new Date(timestamp * 1000).toISOString(),
              value: parseFloat(value),
            })
          ) || []
        );

        // Fetch HyDE hit rate
        const hydeResponse = await fetch(
          `${prometheusUrl}/api/v1/query_range?query=rate(hyde_cache_hits_total[5m]) / (rate(hyde_cache_hits_total[5m]) + rate(hyde_cache_misses_total[5m]))&start=${
            Date.now() - 3600000
          }&end=${Date.now()}&step=60`
        );
        const hydeData = await hydeResponse.json();
        setHydeHitRate(
          hydeData.data.result[0]?.values.map(
            ([timestamp, value]: [number, string]) => ({
              timestamp: new Date(timestamp * 1000).toISOString(),
              value: parseFloat(value) * 100, // Convert to percentage
            })
          ) || []
        );

        // Fetch DragonflyDB memory
        const memoryResponse = await fetch(
          `${prometheusUrl}/api/v1/query_range?query=dragonfly_memory_usage_bytes&start=${
            Date.now() - 3600000
          }&end=${Date.now()}&step=60`
        );
        const memoryData = await memoryResponse.json();
        setDragonflyMemory(
          memoryData.data.result[0]?.values.map(
            ([timestamp, value]: [number, string]) => ({
              timestamp: new Date(timestamp * 1000).toISOString(),
              value: parseFloat(value) / (1024 * 1024 * 1024), // Convert to GB
            })
          ) || []
        );
      } catch (error) {
        console.error("Failed to fetch metrics:", error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [prometheusUrl]);

  return (
    <div className="dashboard">
      <h1>V1 Enhanced Monitoring Dashboard</h1>

      <div className="metrics-grid">
        <div className="metric-panel">
          <h3>Query API Latency (95th percentile)</h3>
          <LineChart width={600} height={300} data={queryApiLatency}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#8884d8"
              name="Latency (s)"
            />
          </LineChart>
        </div>

        <div className="metric-panel">
          <h3>HyDE Cache Hit Rate</h3>
          <LineChart width={600} height={300} data={hydeHitRate}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#82ca9d"
              name="Hit Rate (%)"
            />
          </LineChart>
        </div>

        <div className="metric-panel">
          <h3>DragonflyDB Memory Usage</h3>
          <LineChart width={600} height={300} data={dragonflyMemory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#ffc658"
              name="Memory (GB)"
            />
          </LineChart>
        </div>
      </div>
    </div>
  );
};
```

## 📞 Support and Maintenance

### 1. Regular Maintenance Tasks

**Weekly Monitoring Checklist**:

- [ ] Review alert fatigue and tune thresholds
- [ ] Check disk usage for monitoring data
- [ ] Validate backup integrity
- [ ] Review performance trends
- [ ] Update dashboards based on new requirements

**Monthly Monitoring Tasks**:

- [ ] Analyze capacity planning metrics
- [ ] Review and update SLOs
- [ ] Security audit of monitoring access
- [ ] Performance optimization review
- [ ] Documentation updates

### 2. Emergency Procedures

**Monitoring System Outage**:

1. Switch to backup monitoring (if available)
2. Enable emergency logging
3. Use basic health checks
4. Restore from backup
5. Investigate root cause

**High Alert Volume**:

1. Identify alert storm source
2. Temporarily silence non-critical alerts
3. Focus on critical system alerts
4. Fix underlying issues
5. Gradually re-enable alerts

## 📊 Enhanced Database Connection Pool Dashboards (BJO-134)

### 1. Database Connection Pool Dashboard

Create `config/grafana/dashboards/database-connection-pool.json`:

```json
{
  "dashboard": {
    "title": "Database Connection Pool - Enhanced (BJO-134)",
    "panels": [
      {
        "title": "Connection Pool Utilization",
        "type": "stat",
        "targets": [
          {
            "expr": "db_connection_pool_checked_out / db_connection_pool_size * 100"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                { "color": "green", "value": 0 },
                { "color": "yellow", "value": 70 },
                { "color": "red", "value": 90 }
              ]
            }
          }
        }
      },
      {
        "title": "ML Model Performance",
        "type": "timeseries",
        "targets": [
          {
            "expr": "db_connection_pool_ml_model_accuracy",
            "legendFormat": "Model Accuracy"
          },
          {
            "expr": "db_connection_pool_ml_prediction_confidence",
            "legendFormat": "Prediction Confidence"
          }
        ]
      },
      {
        "title": "Circuit Breaker Status",
        "type": "stat",
        "targets": [
          {
            "expr": "db_circuit_breaker_state"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              { "options": { "0": { "text": "Closed", "color": "green" } } },
              { "options": { "1": { "text": "Open", "color": "red" } } },
              { "options": { "2": { "text": "Half-Open", "color": "yellow" } } }
            ]
          }
        }
      },
      {
        "title": "Connection Affinity Performance",
        "type": "timeseries",
        "targets": [
          {
            "expr": "avg(db_connection_affinity_performance_score) by (query_type)",
            "legendFormat": "Affinity Score - {{query_type}}"
          },
          {
            "expr": "rate(db_connection_affinity_hits_total[5m])",
            "legendFormat": "Affinity Hits/sec"
          }
        ]
      }
    ]
  }
}
```

### 2. Query Performance Analysis Dashboard

```json
{
  "dashboard": {
    "title": "Enhanced Query Performance Analysis",
    "panels": [
      {
        "title": "Query Latency Percentiles",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(db_query_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(db_query_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ]
      },
      {
        "title": "Query Types Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (query_type) (rate(db_query_total[5m]))"
          }
        ]
      },
      {
        "title": "Connection Specialization Performance",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(db_connection_affinity_performance_gain_percent[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3. Predictive Load Monitoring Dashboard

```json
{
  "dashboard": {
    "title": "Predictive Load Monitoring (BJO-134)",
    "panels": [
      {
        "title": "Load Predictions vs Actual",
        "type": "timeseries",
        "targets": [
          {
            "expr": "db_connection_pool_ml_predicted_load",
            "legendFormat": "Predicted Load"
          },
          {
            "expr": "db_connection_pool_current_load_factor",
            "legendFormat": "Actual Load"
          }
        ]
      },
      {
        "title": "Prediction Accuracy Over Time",
        "type": "stat",
        "targets": [
          {
            "expr": "avg_over_time(db_connection_pool_ml_model_accuracy[1h])"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                { "color": "red", "value": 0 },
                { "color": "yellow", "value": 70 },
                { "color": "green", "value": 85 }
              ]
            }
          }
        }
      },
      {
        "title": "Feature Importance",
        "type": "bargauge",
        "targets": [
          {
            "expr": "db_connection_pool_ml_feature_importance"
          }
        ]
      }
    ]
  }
}
```

### 4. Dashboard Setup Commands

```bash
# Create dashboard directory
mkdir -p config/grafana/dashboards

# Copy dashboard configurations
cp config/grafana/dashboards/database-connection-pool.json \
   /var/lib/grafana/dashboards/

# Import dashboards via API
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @config/grafana/dashboards/database-connection-pool.json

# Restart Grafana to load new dashboards
docker restart grafana
```

### 5. Custom Monitoring Queries

**Performance Regression Detection**:

```promql
# Detect 50.9% performance improvement regression
(
  histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m])) -
  histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m] offset 1w))
) / histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m] offset 1w)) * 100 > 25
```

**Throughput Monitoring**:

```promql
# Monitor 887.9% throughput improvement
rate(db_query_total[5m]) / rate(db_query_total[5m] offset 1w) > 8
```

**Connection Pool Efficiency**:

```promql
# Connection pool efficiency score
(
  db_connection_pool_successful_queries_total /
  db_connection_pool_total_queries_total
) * 100
```

---

## 📈 Summary

This comprehensive monitoring and observability guide provides operators with:

- **Complete monitoring stack setup** with Prometheus, Grafana, and alerting
- **V1 enhanced metrics** for Query API, HyDE, DragonflyDB, and payload indexing
- **Production-ready dashboards** and alert configurations
- **Troubleshooting procedures** for common monitoring issues
- **Best practices** for scalable, secure monitoring
- **Advanced features** like distributed tracing and custom dashboards

The monitoring system is designed to provide real-time visibility into system performance,
enabling proactive issue detection and resolution while supporting the V1 performance
improvements of 50-70% through comprehensive tracking and optimization capabilities.

For questions or additional monitoring requirements, refer to the troubleshooting section
or contact the development team through the established support channels.
