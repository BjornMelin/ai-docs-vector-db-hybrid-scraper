# Performance Testing Framework

This directory contains comprehensive performance testing for the AI Documentation Vector DB Hybrid Scraper, focusing on system performance optimization, resource utilization monitoring, and scalability validation.

## Framework Overview

The performance testing framework provides:

- **Multi-dimensional performance analysis** across CPU, memory, network, and database layers
- **API latency measurement** with percentile analysis and SLA validation
- **Throughput testing** for concurrent request handling and batch processing
- **Resource utilization monitoring** with bottleneck identification
- **Performance regression detection** with baseline comparisons

## Directory Structure

- **memory/**: Memory usage analysis and leak detection
- **cpu/**: CPU utilization and optimization testing  
- **network/**: Network performance and latency testing
- **database/**: Database performance optimization and query analysis
- **api_latency/**: API response time validation and SLA monitoring
- **throughput/**: System throughput measurement and scaling analysis

## Core Performance Categories

### Memory Performance Testing

```python
# Memory usage patterns and optimization
@pytest.mark.performance
@pytest.mark.memory
def test_memory_efficiency():
    """Test memory usage during document processing."""
    pass
```

**Key Areas:**
- Memory leak detection during long-running operations
- Peak memory usage during batch processing
- Memory efficiency of embedding generation
- Garbage collection impact on performance
- Memory usage patterns across different workloads

### CPU Performance Testing

```python
# CPU utilization and processing efficiency
@pytest.mark.performance  
@pytest.mark.cpu
def test_cpu_optimization():
    """Test CPU usage during intensive operations."""
    pass
```

**Key Areas:**
- CPU utilization during document processing
- Multi-threading efficiency for parallel operations
- CPU bottleneck identification in pipelines
- Processing efficiency optimization
- Async operation CPU overhead analysis

### Network Performance Testing

```python
# Network latency and bandwidth utilization
@pytest.mark.performance
@pytest.mark.network
def test_network_efficiency():
    """Test network performance for external services."""
    pass
```

**Key Areas:**
- Network latency for external API calls
- Bandwidth utilization during bulk operations
- Connection pooling efficiency
- Network timeout handling
- External service dependency performance

### Database Performance Testing

```python
# Database query and operation optimization
@pytest.mark.performance
@pytest.mark.database
def test_database_performance():
    """Test database operation efficiency."""
    pass
```

**Key Areas:**
- Vector search query performance
- Database connection pool efficiency
- Index optimization and query planning
- Bulk operation performance
- Database scaling and sharding analysis

### API Latency Testing

```python
# API response time and SLA validation
@pytest.mark.performance
@pytest.mark.api_latency
def test_api_response_times():
    """Test API endpoint response times."""
    pass
```

**Key Areas:**
- P50, P95, P99 response time analysis
- API endpoint performance comparison
- SLA compliance validation
- Response time under load
- Cold start vs. warm performance

### Throughput Testing

```python
# System throughput and scaling analysis
@pytest.mark.performance
@pytest.mark.throughput
def test_system_throughput():
    """Test system throughput capabilities."""
    pass
```

**Key Areas:**
- Requests per second capabilities
- Concurrent user handling
- Batch processing throughput
- Scaling efficiency analysis
- Throughput bottleneck identification

## Usage Commands

### Quick Start

```bash
# Run all performance tests
uv run pytest tests/performance/ -v

# Run specific performance category
uv run pytest tests/performance/memory/ -v
uv run pytest tests/performance/api_latency/ -v
uv run pytest tests/performance/database/ -v

# Run with performance markers
uv run pytest -m "performance" -v
```

### Performance Analysis

```bash
# Run performance tests with profiling
uv run pytest tests/performance/ -v --profile

# Generate performance report
uv run pytest tests/performance/ --benchmark-only --benchmark-json=performance_report.json

# Run specific performance markers
uv run pytest -m "performance and memory" -v
uv run pytest -m "performance and not slow" -v
```

### Baseline Comparison

```bash
# Run performance tests with baseline comparison
uv run pytest tests/performance/ --benchmark-compare=baseline.json

# Save performance baseline
uv run pytest tests/performance/ --benchmark-save=baseline

# Performance regression detection
uv run pytest tests/performance/ --benchmark-compare-fail=min:5% --benchmark-compare-fail=mean:10%
```

### CI/CD Integration

```bash
# Fast performance checks for CI
uv run pytest tests/performance/ -m "fast and performance" --maxfail=3

# Full performance validation with reporting
uv run pytest tests/performance/ --benchmark-json=ci_performance.json --tb=short
```

## Performance Metrics and Targets

### Response Time Targets

| Operation Type | P50 Target | P95 Target | P99 Target |
|---------------|------------|------------|------------|
| Document Search | < 100ms | < 300ms | < 500ms |
| Document Ingestion | < 500ms | < 1s | < 2s |
| Embedding Generation | < 200ms | < 500ms | < 1s |
| Bulk Operations | < 2s | < 5s | < 10s |

### Throughput Targets

| Operation Type | Target RPS | Concurrent Users | Notes |
|---------------|------------|------------------|-------|
| Search Queries | 1000+ | 100+ | With caching |
| Document Uploads | 100+ | 50+ | Batch processing |
| API Endpoints | 500+ | 100+ | Average load |

### Resource Utilization Targets

| Resource | Normal Load | Peak Load | Critical Threshold |
|----------|-------------|-----------|-------------------|
| CPU Usage | < 60% | < 80% | > 90% |
| Memory Usage | < 70% | < 85% | > 95% |
| Database Connections | < 70% | < 85% | > 95% |
| Network Bandwidth | < 50% | < 70% | > 85% |

## Performance Testing Strategies

### Benchmark Testing

```python
# Benchmark critical operations
def test_search_performance(benchmark):
    """Benchmark search operation performance."""
    result = benchmark(search_function, query="test")
    assert result.response_time < 100  # milliseconds
```

### Load Profile Testing

```python
# Test with realistic load profiles
@pytest.mark.parametrize("concurrent_users", [10, 50, 100])
def test_concurrent_load(concurrent_users):
    """Test performance under concurrent load."""
    pass
```

### Resource Monitoring

```python
# Monitor resource usage during tests
@pytest.fixture
def resource_monitor():
    """Monitor CPU, memory, and network during tests."""
    monitor = ResourceMonitor()
    monitor.start()
    yield monitor
    report = monitor.stop()
    assert report.max_cpu_usage < 80
    assert report.max_memory_usage < 85
```

## Integration with Load Testing

The performance testing framework integrates with the load testing framework (`tests/load/`) to provide:

- **Performance validation** during load tests
- **Resource monitoring** under stress conditions
- **Baseline establishment** for load test scenarios
- **Regression detection** across performance dimensions

## Monitoring and Alerting

### Performance Dashboards

- Real-time performance metrics visualization
- Historical trend analysis and reporting
- Performance regression alerts
- Resource utilization monitoring
- SLA compliance tracking

### Automated Analysis

- Performance bottleneck identification
- Optimization recommendation generation
- Trend analysis and forecasting
- Anomaly detection for performance issues
- Root cause analysis for performance degradation

## Tools and Frameworks

- **pytest-benchmark**: Performance benchmarking
- **memory_profiler**: Memory usage analysis
- **psutil**: System resource monitoring
- **cProfile**: CPU profiling and analysis
- **py-spy**: Production profiling
- **locust**: Load testing integration

## Best Practices

### Test Design
- Use realistic data sizes and complexity
- Test both warm and cold performance scenarios
- Include edge cases and boundary conditions
- Validate performance under various load conditions
- Implement proper baseline establishment

### Measurement Accuracy
- Use statistical analysis for consistent results
- Account for system variability and noise
- Implement proper warmup periods
- Use multiple measurement iterations
- Consider external factors affecting performance

### Continuous Improvement
- Establish performance baselines and targets
- Implement automated performance regression detection
- Regular performance review and optimization
- Track performance trends over time
- Document performance optimization decisions

This performance testing framework ensures optimal system performance, efficient resource utilization, and scalable operation of the AI Documentation Vector DB Hybrid Scraper.