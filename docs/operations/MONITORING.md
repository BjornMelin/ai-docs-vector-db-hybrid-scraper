# Monitoring & Observability Guide

**Status**: Current  
**Last Updated**: 2025-05-26

## Overview

This guide covers monitoring, metrics collection, and observability for the AI Documentation Vector DB system in production environments.

## Metrics Collection

### System Metrics

The unified architecture provides comprehensive metrics through various service layers:

#### 1. API Performance Metrics
```python
# Automatically collected by service layer
- Request count and rate
- Response times (p50, p95, p99)
- Error rates by endpoint
- Concurrent request count
```

#### 2. Embedding Generation Metrics
```python
# EmbeddingManager metrics
- Embeddings generated per second
- Batch processing efficiency
- Model-specific latencies
- Token usage and costs
```

#### 3. Vector Database Metrics
```python
# QdrantService metrics
- Search query latency
- Collection sizes and growth
- Index performance
- Memory usage
```

#### 4. Cache Performance
```python
# CacheManager metrics
- Hit/miss rates
- TTL effectiveness
- Memory usage
- Eviction rates
```

## Monitoring Implementation

### 1. Structured Logging

All services use structured logging with correlation IDs:

```python
import structlog

logger = structlog.get_logger()

# Automatic context propagation
logger.info("search_request", 
    query=query,
    collection=collection_name,
    correlation_id=request_id,
    user_id=user_id
)
```

### 2. Performance Tracking

Built-in performance tracking for critical operations:

```python
from src.services.base import track_performance

@track_performance("embedding_generation")
async def generate_embeddings(self, texts: List[str]):
    # Automatically tracks:
    # - Execution time
    # - Success/failure
    # - Resource usage
    pass
```

### 3. Health Checks

Comprehensive health check endpoints:

```bash
# Service health
GET /health

# Detailed health with dependencies
GET /health/detailed

# Response includes:
{
  "status": "healthy",
  "services": {
    "qdrant": "connected",
    "redis": "connected",
    "openai": "available"
  },
  "metrics": {
    "uptime": 86400,
    "requests_total": 15000,
    "error_rate": 0.001
  }
}
```

## Key Metrics to Monitor

### 1. Search Performance
- **Target**: < 100ms p95 latency
- **Alert**: > 200ms sustained
- **Dashboard**: Query latency histogram

### 2. Embedding Generation
- **Target**: > 1000 embeddings/second
- **Alert**: < 500 embeddings/second
- **Dashboard**: Throughput graph

### 3. Cache Effectiveness
- **Target**: > 80% hit rate
- **Alert**: < 60% hit rate
- **Dashboard**: Hit/miss ratio

### 4. Error Rates
- **Target**: < 0.1% error rate
- **Alert**: > 1% error rate
- **Dashboard**: Error rate by endpoint

## Monitoring Stack Setup

### 1. Prometheus Integration

Export metrics in Prometheus format:

```python
# Automatic metrics export
from prometheus_client import start_http_server

# Start metrics server
start_http_server(9090)
```

### 2. Grafana Dashboards

Pre-built dashboards available:
- System Overview
- Search Performance
- Embedding Analytics
- Cache Statistics
- Error Analysis

### 3. Alerting Rules

Example Prometheus alert rules:

```yaml
groups:
  - name: ai_docs_alerts
    rules:
      - alert: HighSearchLatency
        expr: histogram_quantile(0.95, search_latency_seconds) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Search latency is high"
          
      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.6
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below threshold"
```

## Resource Monitoring

### 1. Memory Usage
```bash
# Monitor service memory
docker stats qdrant-vector-db

# Alert thresholds
- Warning: > 80% memory usage
- Critical: > 95% memory usage
```

### 2. Disk Usage
```bash
# Monitor vector storage
du -sh data/qdrant/storage

# Alert thresholds
- Warning: > 80% disk usage
- Critical: > 90% disk usage
```

### 3. CPU Usage
```bash
# Monitor CPU across services
top -p $(pgrep -f unified_mcp_server)

# Alert thresholds
- Warning: > 80% sustained CPU
- Critical: > 95% sustained CPU
```

## Performance Analysis

### 1. Slow Query Analysis
```python
# Automatic slow query logging
SLOW_QUERY_THRESHOLD = 0.5  # seconds

# Logs include:
- Query text
- Execution time
- Collection accessed
- Result count
```

### 2. Bottleneck Detection
```python
# Performance profiling enabled
ENABLE_PROFILING = true

# Profiles include:
- Function call graphs
- Memory allocations
- I/O wait times
```

### 3. Cost Analysis
```python
# API usage tracking
{
  "openai": {
    "embeddings": {
      "requests": 10000,
      "tokens": 5000000,
      "cost_usd": 2.50
    }
  },
  "firecrawl": {
    "pages": 1000,
    "cost_usd": 10.00
  }
}
```

## Debugging Tools

### 1. Request Tracing
```python
# Enable detailed tracing
LOG_LEVEL=DEBUG
ENABLE_REQUEST_TRACING=true

# Trace includes:
- Full request/response
- Service call chain
- Timing breakdowns
```

### 2. Memory Profiling
```python
# Enable memory profiling
import tracemalloc
tracemalloc.start()

# Analyze memory usage
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
```

### 3. Query Explain Plans
```python
# Analyze vector search performance
result = await qdrant_service.explain_query(
    query_vector=embedding,
    collection_name="docs"
)
```

## Monitoring Best Practices

### 1. Dashboard Organization
- **Overview**: System health at a glance
- **Service-Specific**: Deep dive per service
- **Business Metrics**: User-facing KPIs
- **Cost Tracking**: API usage and expenses

### 2. Alert Fatigue Prevention
- Set meaningful thresholds
- Use alert grouping
- Implement alert suppression
- Regular threshold review

### 3. Retention Policies
- Metrics: 30 days high-res, 1 year downsampled
- Logs: 7 days verbose, 30 days errors
- Traces: 24 hours detailed, 7 days sampled

## Integration Examples

### 1. CloudWatch Integration
```python
# AWS CloudWatch metrics
import boto3
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='AIDocsVectorDB',
    MetricData=[{
        'MetricName': 'SearchLatency',
        'Value': latency_ms,
        'Unit': 'Milliseconds'
    }]
)
```

### 2. Datadog Integration
```python
# Datadog APM
from ddtrace import tracer

@tracer.wrap()
async def search_documents(query: str):
    # Automatic tracing
    pass
```

### 3. Custom Metrics Export
```python
# Export to any backend
class MetricsExporter:
    async def export(self, metrics: Dict):
        # Send to your monitoring system
        pass
```

## Troubleshooting Monitoring Issues

### 1. Missing Metrics
- Check service health endpoints
- Verify metrics server is running
- Confirm network connectivity

### 2. High Cardinality
- Review label usage
- Implement label limits
- Use metric aggregation

### 3. Storage Growth
- Implement retention policies
- Use metric downsampling
- Archive old data

## Related Documentation

- [Performance Guide](./PERFORMANCE_GUIDE.md) - Optimization strategies
- [Troubleshooting](./TROUBLESHOOTING.md) - Common issues
- [System Overview](../architecture/SYSTEM_OVERVIEW.md) - Architecture details