# ML Application Monitoring & Observability

Comprehensive monitoring infrastructure using Prometheus, Grafana, and Alertmanager for production-ready observability.

## Quick Start

### 1. Start Monitoring Stack
```bash
# Start Prometheus, Grafana, and Alertmanager
./scripts/start-monitoring.sh
```

### 2. Start Main Application
```bash
# Start main application with monitoring enabled
docker-compose up -d
```

### 3. Access Dashboards
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

## Available Dashboards

### 1. Application Overview
- Service health status
- Dependency status (Qdrant, Redis)
- High-level performance metrics
- System resource usage

### 2. Vector Search Performance
- Search request rates and latency percentiles
- Concurrent request tracking
- Search quality scores by collection
- HNSW performance metrics

### 3. Cache Performance  
- Hit/miss rates across cache types
- Cache latency and memory usage
- Eviction rates and patterns
- DragonflyDB vs Local cache comparison

### 4. Embedding Generation & Costs
- Cost tracking per provider and model
- Embedding generation latency
- Request rates and batch sizes
- Queue depth monitoring

## Metrics Collected

### Vector Search
- `ml_app_vector_search_requests_total`: Total search requests
- `ml_app_vector_search_duration_seconds`: Search latency histogram
- `ml_app_vector_search_quality_score`: Search quality metrics
- `ml_app_vector_search_concurrent_requests`: Active requests

### Embeddings
- `ml_app_embedding_requests_total`: Embedding generation requests  
- `ml_app_embedding_generation_duration_seconds`: Generation latency
- `ml_app_embedding_cost_total`: Cumulative costs (USD)
- `ml_app_embedding_batch_size`: Batch size distribution
- `ml_app_embedding_queue_depth`: Queue depth gauge

### Cache Performance
- `ml_app_cache_hits_total`/`ml_app_cache_misses_total`: Hit/miss counts
- `ml_app_cache_operation_duration_seconds`: Cache operation latency
- `ml_app_cache_memory_usage_bytes`: Memory usage per cache
- `ml_app_cache_evictions_total`: Eviction counts

### System Health
- `ml_app_service_health_status`: Service health (1=healthy, 0=unhealthy)
- `ml_app_dependency_health_status`: Dependency health
- `ml_app_system_cpu_usage_percent`: CPU utilization
- `ml_app_system_memory_usage_bytes`: Memory usage

## Alerting Rules

### Critical Alerts
- **ServiceUnhealthy**: Service reporting unhealthy status
- **DependencyUnhealthy**: Critical dependency failure

### Warning Alerts  
- **HighSearchErrorRate**: >5% search error rate for 2+ minutes
- **SlowSearchResponseTime**: 95th percentile >10s for 5+ minutes
- **HighEmbeddingCosts**: >$10/hour embedding costs
- **LowCacheHitRate**: <70% cache hit rate for 10+ minutes
- **HighMemoryUsage**: >8GB system memory usage
- **HighCPUUsage**: >80% CPU usage for 5+ minutes
- **TaskQueueBackup**: >100 pending tasks for 5+ minutes

### Info Alerts
- **QdrantCollectionGrowthRate**: Rapid collection growth (>10k vectors/hour)

## Configuration

### Environment Variables
```bash
# Enable monitoring in application
MONITORING_ENABLED=true
PROMETHEUS_MULTIPROC_DIR=/tmp

# Service URLs
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379
```

### Monitoring Configuration
Edit `src/config/models.py` to customize monitoring settings:
```python
monitoring = MonitoringConfig(
    enabled=True,
    metrics_port=8000,
    health_check_interval=30,
    collect_system_metrics=True
)
```

## Commands

```bash
# Start monitoring stack
./scripts/start-monitoring.sh

# Stop monitoring stack  
./scripts/stop-monitoring.sh

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Check metrics endpoint
curl http://localhost:8000/metrics

# Check health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/live  
curl http://localhost:8000/health/ready
```

## Development

### Adding New Metrics
1. Add metric definition to `src/services/monitoring/metrics.py`
2. Use monitoring decorators in your services
3. Update Grafana dashboards as needed

### Custom Dashboards
1. Create dashboard JSON in `config/grafana/dashboards/`
2. Restart Grafana to auto-provision new dashboard

### Alert Configuration
1. Add alert rules to `config/prometheus/alert_rules.yml`
2. Configure notifications in `config/alertmanager/alertmanager.yml`
3. Restart Prometheus and Alertmanager

## Troubleshooting

### Common Issues

**Metrics not appearing**:
- Check `MONITORING_ENABLED=true` environment variable
- Verify `/metrics` endpoint returns data
- Check Prometheus target status at http://localhost:9090/targets

**Dashboard empty**:
- Verify Prometheus data source in Grafana
- Check metric names match dashboard queries
- Ensure time range includes recent data

**Alerts not firing**:
- Check alert rules syntax in Prometheus UI
- Verify Alertmanager configuration
- Check notification channels are configured

### Performance Impact
- Metrics collection adds <1ms latency overhead
- Memory overhead: ~50MB for metrics registry
- Storage: ~1GB/month for 15s scrape intervals
- Monitoring can be disabled via `MONITORING_ENABLED=false`

## Security

- Grafana admin password: Change from default `admin123`
- Prometheus: No authentication by default (add reverse proxy for production)
- Alertmanager: Configure proper notification channels
- Network: Uses isolated Docker networks for security