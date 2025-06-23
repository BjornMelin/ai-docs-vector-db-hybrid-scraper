# Stress Testing Suite

This directory contains comprehensive stress testing infrastructure for the AI Documentation Vector DB Hybrid Scraper. The stress testing suite is designed to identify system breaking points, validate resource handling under extreme conditions, and ensure graceful degradation and recovery.

## Overview

The stress testing framework implements several types of stress tests:

1. **Resource Exhaustion Tests** - Test behavior when system resources are exhausted
2. **Breaking Point Identification** - Find the maximum load the system can handle
3. **Circuit Breaker & Rate Limiter Tests** - Validate protection mechanisms
4. **System Monitoring** - Comprehensive monitoring during stress conditions

## Test Files

### `test_resource_exhaustion.py`
Tests system behavior under various resource constraints:
- **Memory exhaustion** through large document processing
- **CPU saturation** via parallel embedding generation  
- **Network bandwidth stress** with large payload transfers
- **Database connection pool exhaustion**
- **File descriptor limit testing**
- **Cascading resource failure scenarios**

Key features:
- Real-time resource monitoring with `ResourceMonitor`
- Configurable resource constraints via `ResourceConstraints`
- Cross-platform resource limiting with context managers
- Memory leak detection over extended periods
- Graceful degradation validation

### `test_breaking_points.py`
Identifies system breaking points through systematic load increases:
- **Gradual load increase** to find performance cliff
- **Sudden spike handling** to test burst capacity
- **Recovery time measurement** after overload conditions
- **Performance degradation analysis** with trend detection

Key features:
- `BreakingPointAnalyzer` for automated breaking point detection
- Graceful vs catastrophic failure analysis
- Performance curve generation and analysis
- Multi-phase testing (baseline → stress → recovery)
- Service degradation simulation

### `test_circuit_breakers.py`
Tests effectiveness of circuit breakers and rate limiters:
- **Circuit breaker trigger points** under varying failure rates
- **Recovery behavior** after service restoration
- **Rate limiter effectiveness** against different load patterns
- **Thundering herd protection** scenarios

Key features:
- Mock circuit breaker with realistic state transitions
- Mock rate limiter with burst capacity and time windows
- Configurable failure injection for testing triggers
- Recovery validation and timing measurement

### `test_system_monitoring.py`
Comprehensive system monitoring during stress tests:
- **Real-time metrics collection** of system and application metrics
- **Performance degradation detection** with automated alerting
- **Resource usage analysis** across multiple dimensions
- **Custom metrics tracking** for application-specific data

Key features:
- `SystemMonitor` with configurable collection intervals
- Performance alert system with thresholds
- Metrics aggregation and statistical analysis
- Degradation pattern detection over time

### `conftest.py`
Shared fixtures and utilities for stress testing:
- **Resource constraint management** with automatic cleanup
- **Chaos engineering scenarios** for failure injection
- **Stress test orchestration** for complex multi-phase tests
- **Predefined test profiles** for different stress levels

## Usage Examples

### Basic Resource Exhaustion Test
```python
@pytest.mark.stress
async def test_memory_stress(load_test_runner, resource_monitor):
    """Test system under memory pressure."""
    config = LoadTestConfig(
        test_type=LoadTestType.STRESS,
        concurrent_users=100,
        requests_per_second=50,
        duration_seconds=120,
        data_size_mb=20.0,  # Large documents
    )
    
    result = await load_test_runner.run_load_test(
        config=config,
        target_function=memory_intensive_operation,
        data_size_mb=config.data_size_mb,
    )
    
    peak_usage = resource_monitor.get_peak_usage()
    assert peak_usage["peak_memory_mb"] > 500
```

### Breaking Point Detection
```python
@pytest.mark.stress
async def test_find_breaking_point(load_test_runner):
    """Find system breaking point through gradual load increase."""
    analyzer = BreakingPointAnalyzer()
    
    # Test increasing load levels
    for users in [50, 100, 200, 500, 1000]:
        config = LoadTestConfig(
            test_type=LoadTestType.STRESS,
            concurrent_users=users,
            requests_per_second=users / 2,
            duration_seconds=60,
        )
        
        result = await load_test_runner.run_load_test(
            config=config,
            target_function=test_operation,
        )
        
        # Add performance point
        analyzer.add_performance_point(PerformancePoint(
            users=users,
            rps=result.metrics.throughput_rps,
            avg_response_time=statistics.mean(result.metrics.response_times) * 1000,
            error_rate=(result.metrics.failed_requests / result.metrics.total_requests) * 100,
            cpu_usage=50,  # Would come from monitoring
            memory_usage=users * 2,  # Would come from monitoring
            timestamp=time.time(),
        ))
    
    # Analyze breaking point
    breaking_point = analyzer.identify_breaking_point()
    assert breaking_point.breaking_point_users is not None
```

### Chaos Engineering
```python
@pytest.mark.chaos
async def test_chaos_scenario(failure_injector, load_test_runner):
    """Test system resilience with chaos engineering."""
    # Start chaos scenario
    chaos_task = asyncio.create_task(
        failure_injector.inject_memory_pressure(
            pressure_mb=200,
            duration=60.0
        )
    )
    
    # Run load test during chaos
    config = LoadTestConfig(
        test_type=LoadTestType.STRESS,
        concurrent_users=50,
        requests_per_second=25,
        duration_seconds=90,
    )
    
    result = await load_test_runner.run_load_test(
        config=config,
        target_function=resilient_operation,
    )
    
    await chaos_task  # Wait for chaos to complete
    
    # Verify system remained functional
    error_rate = (result.metrics.failed_requests / result.metrics.total_requests) * 100
    assert error_rate < 50  # System should handle some chaos
```

## Configuration

### Stress Test Profiles
Predefined profiles for different stress levels:
- `light_stress`: 50 users, 25 RPS, 2 minutes
- `moderate_stress`: 200 users, 100 RPS, 5 minutes  
- `heavy_stress`: 500 users, 250 RPS, 10 minutes
- `extreme_stress`: 1000 users, 500 RPS, 5 minutes

### Resource Constraints
Available resource constraints:
- Memory limits (MB)
- File descriptor limits
- CPU time limits
- Network bandwidth limits

### Chaos Scenarios
Predefined chaos scenarios:
- `memory_pressure`: Allocate memory to create pressure
- `cpu_spike`: Generate CPU load spikes
- `network_failure`: Inject network failures
- `disk_io_stress`: Create disk I/O stress

## Monitoring and Metrics

### System Metrics Collected
- CPU usage (percentage and load average)
- Memory usage (percentage and absolute)
- Disk I/O (read/write MB/s)
- Network I/O (sent/received MB/s)
- Open file descriptors
- TCP connections
- Process count

### Application Metrics
- Active request count
- Request queue size
- Response time percentiles
- Error rates by type
- Cache hit rates
- Database connection utilization
- Embedding queue size
- Garbage collection statistics

### Performance Alerts
Automatic alerts for:
- CPU usage > 80% (warning), > 95% (critical)
- Memory usage > 85% (warning), > 95% (critical)
- Response time P95 > 2s (warning), > 5s (critical)
- Error rate > 5% (warning), > 15% (critical)
- Open files > 1000 (warning), > 2000 (critical)

## Running Tests

### Individual Test Files
```bash
# Resource exhaustion tests
uv run pytest tests/load/stress_testing/test_resource_exhaustion.py -v

# Breaking point tests  
uv run pytest tests/load/stress_testing/test_breaking_points.py -v

# Circuit breaker tests
uv run pytest tests/load/stress_testing/test_circuit_breakers.py -v

# System monitoring tests
uv run pytest tests/load/stress_testing/test_system_monitoring.py -v
```

### By Test Markers
```bash
# All stress tests
uv run pytest -m stress

# Resource exhaustion only
uv run pytest -m resource_exhaustion

# Breaking point identification
uv run pytest -m breaking_point

# Chaos engineering tests
uv run pytest -m chaos

# Recovery validation tests
uv run pytest -m recovery

# Slow-running tests
uv run pytest -m slow
```

### With Specific Profiles
```bash
# Light stress testing
uv run pytest tests/load/stress_testing/ --stress-profile=light_stress

# Extreme stress testing
uv run pytest tests/load/stress_testing/ --stress-profile=extreme_stress
```

## Best Practices

### Resource Management
- Always use context managers for resource constraints
- Clean up allocated memory after tests
- Reset system state between test runs
- Monitor for resource leaks

### Test Isolation
- Use separate processes for resource-intensive tests
- Clean up background tasks and threads
- Reset global state between tests
- Use temporary files that are automatically cleaned up

### Failure Handling
- Expect and plan for test failures under extreme conditions
- Validate graceful degradation rather than complete success
- Measure recovery time and effectiveness
- Test circuit breaker and rate limiter behavior

### Monitoring
- Start monitoring before applying stress
- Stop monitoring after tests complete
- Collect baseline metrics for comparison
- Generate comprehensive reports

## Troubleshooting

### Common Issues
1. **Resource limits too restrictive**: Adjust limits based on test environment
2. **Tests timing out**: Increase timeouts for resource-constrained tests  
3. **Memory errors**: Reduce memory pressure or increase available memory
4. **File descriptor errors**: Increase FD limits or reduce concurrent operations
5. **Network failures**: Check for actual network issues vs injected failures

### Debug Mode
Enable debug logging for detailed information:
```bash
uv run pytest tests/load/stress_testing/ -v -s --log-cli-level=DEBUG
```

### System Requirements
- Minimum 4GB RAM for moderate stress testing
- Minimum 8GB RAM for heavy stress testing
- SSD recommended for disk I/O tests
- Multiple CPU cores for CPU saturation tests
- Adequate file descriptor limits (check with `ulimit -n`)

## Integration

The stress testing suite integrates with:
- **Load testing framework** (`tests/load/`) for traffic generation
- **Performance monitoring** for real-time metrics
- **Chaos engineering** for failure injection
- **CI/CD pipeline** for automated stress validation
- **Alerting systems** for performance threshold monitoring

## Future Enhancements

Planned improvements:
- Distributed stress testing across multiple nodes
- Cloud resource integration (AWS/GCP stress testing)
- Machine learning-based performance prediction
- Automated capacity planning recommendations
- Integration with APM tools (Prometheus, Grafana)
- Real-time performance visualization
- Automated report generation and analysis