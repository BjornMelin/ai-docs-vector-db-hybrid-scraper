---
name: Performance Regression Report
about: Report a performance regression in database operations or query processing
title: '[PERF] Performance regression in [component]'
labels: ['bug', 'performance', 'needs-investigation']
assignees: ''
---

## Performance Regression Summary

**Component**: [e.g., Database connection pool, Vector search, Embedding generation]
**Expected Performance**: [e.g., <100ms query latency, >1000 QPS throughput]
**Actual Performance**: [e.g., 250ms query latency, 400 QPS throughput]
**Performance Impact**: [e.g., 150% latency increase, 60% throughput decrease]

## Environment Details

**System Configuration**
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 14]
- Python Version: [e.g., 3.13.1]
- Package Version: [output of `uv pip show ai-docs-vector-db-hybrid-scraper`]
- Database: [e.g., Qdrant 1.7.4, PostgreSQL 15.2]
- Hardware: [e.g., 8 CPU cores, 32GB RAM, SSD storage]

**Configuration**
```yaml
# Include relevant configuration sections
database:
  connection_pool:
    min_size: 5
    max_size: 20
  # ... other config
```

## Benchmark Results

### Before (Expected Performance)
```bash
# Include benchmark command and results
uv run python scripts/benchmark_query_api.py
# Results:
# Average latency: 95ms
# 95th percentile: 140ms
# Throughput: 1250 QPS
```

### After (Current Performance)
```bash
# Same benchmark command
uv run python scripts/benchmark_query_api.py
# Results:
# Average latency: 238ms
# 95th percentile: 340ms
# Throughput: 502 QPS
```

## Reproduction Steps

1. **Setup Environment**
   ```bash
   # Commands to reproduce the environment
   git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
   cd ai-docs-vector-db-hybrid-scraper
   git checkout [specific-commit-or-branch]
   ```

2. **Run Benchmark**
   ```bash
   # Exact commands to reproduce the regression
   ./scripts/start-services.sh
   uv run python scripts/benchmark_query_api.py --config regression_test.json
   ```

3. **Expected vs Actual Results**
   - Expected: [specific metrics]
   - Actual: [specific metrics]

## Performance Analysis

**Profiling Data**
```bash
# Commands used for profiling
uv run python -m cProfile -o regression.prof scripts/benchmark_query_api.py
uv run python -c "import pstats; pstats.Stats('regression.prof').sort_stats('cumulative').print_stats(20)"
```

**Memory Usage**
```bash
# Memory profiling results
uv run python -m memory_profiler scripts/benchmark_query_api.py
```

**System Resources**
- CPU utilization during regression: [e.g., 85% average]
- Memory usage: [e.g., 24GB peak, 18GB sustained]
- Disk I/O: [e.g., 150MB/s read, 80MB/s write]
- Network I/O: [if applicable]

## Suspected Causes

**Recent Changes**
- [ ] Database schema modifications
- [ ] Connection pool configuration changes
- [ ] Query optimization modifications
- [ ] ML model updates
- [ ] Caching strategy changes
- [ ] Infrastructure changes

**Hypotheses**
1. [Most likely cause based on analysis]
2. [Second most likely cause]
3. [Other potential causes]

## Impact Assessment

**Severity**: [Critical/High/Medium/Low]

**Business Impact**
- User experience degradation: [description]
- Resource cost increase: [estimated additional costs]
- System stability concerns: [any stability issues]

**Affected Features**
- [ ] Vector search queries
- [ ] Document embedding generation
- [ ] Bulk import operations
- [ ] Real-time search suggestions
- [ ] MCP server operations

## Additional Context

**Related Issues**
- #[issue-number]: [brief description]
- #[issue-number]: [brief description]

**Logs and Traces**
```bash
# Include relevant log snippets
[timestamp] ERROR: Connection pool exhausted after 5000ms
[timestamp] WARN: Query timeout exceeded 10000ms threshold
```

**Monitoring Data**
- Grafana dashboard links: [if available]
- Prometheus metrics: [relevant metric snapshots]
- Custom monitoring data: [application-specific metrics]

## Proposed Investigation Steps

- [ ] Bisect git history to identify regression commit
- [ ] Profile specific components under load
- [ ] Compare database query plans before/after
- [ ] Analyze memory allocation patterns
- [ ] Review recent configuration changes
- [ ] Test with different hardware configurations

## Additional Information

**Urgency**: [How quickly does this need to be resolved?]
**Workarounds**: [Any temporary solutions or configurations that help]
**Testing**: [Specific test cases that can verify the fix]

---

**For Maintainers:**
- [ ] Regression confirmed and reproduced
- [ ] Root cause identified
- [ ] Fix implemented and tested
- [ ] Performance benchmarks validate resolution
- [ ] Documentation updated if needed