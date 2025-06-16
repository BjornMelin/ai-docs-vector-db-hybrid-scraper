# Database Performance Benchmarks

Clean, pytest-benchmark based performance tests that validate our enterprise database infrastructure achievements from BJO-134.

## Performance Targets

- **887.9% throughput improvement** over baseline
- **50.9% P95 latency reduction**
- **95% ML prediction accuracy** for load monitoring
- **99.9% uptime** with circuit breaker protection

## Quick Start

```bash
# Run quick benchmarks (excludes slow tests)
python scripts/run_benchmarks.py --quick

# Run full benchmark suite
python scripts/run_benchmarks.py

# Save results to JSON
python scripts/run_benchmarks.py --save-results

# Verbose output
python scripts/run_benchmarks.py --verbose
```

## Direct pytest Usage

```bash
# Run all benchmarks
uv run pytest tests/benchmarks/ --benchmark-only

# Run only fast benchmarks
uv run pytest tests/benchmarks/ --benchmark-only -m "not slow"

# Save results
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json
```

## What's Tested

### Core Performance Tests
- **Database session creation/cleanup** - Tests connection overhead
- **Concurrent session handling** - Tests scalability under load
- **Sustained throughput** - Validates QPS targets under sustained load

### Enterprise Monitoring Tests
- **ML prediction accuracy** - Tests load monitor performance
- **Circuit breaker response** - Tests resilience patterns
- **Monitoring system overhead** - Validates lightweight monitoring

### Enterprise Features Tests
- **Connection affinity** - Tests query optimization hit rates
- **Enterprise monitoring overhead** - Validates performance impact

## Results

Benchmark results are saved to:
- `benchmark_results/database_performance.json` - Detailed JSON results
- `.benchmarks/` - Historical comparison data

## Comparison with Previous Implementation

This clean pytest-benchmark implementation replaces our previous 395-line custom script with:
- ✅ **Industry standard** pytest-benchmark for reliable statistical analysis
- ✅ **Clean test patterns** following pytest conventions
- ✅ **Real database operations** (not mocked) for accurate results
- ✅ **Automatic comparison** and regression detection
- ✅ **Easy CI/CD integration** with standard pytest workflow
- ✅ **Statistical reliability** with proper outlier detection and confidence intervals

The benchmarks validate the same enterprise features and performance targets while being more trustable, maintainable, and accurate.