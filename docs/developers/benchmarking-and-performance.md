# Benchmarking and Performance Testing

The system includes comprehensive benchmarking capabilities for performance testing, optimization, and system validation across all components.

## Overview

The benchmarking suite provides:

- **Multi-dimensional Testing**: Embedding models, HNSW optimization, crawling performance, lightweight tier efficiency
- **Anti-detection Performance**: Browser automation stealth capabilities
- **Payload Indexing**: Vector database indexing performance
- **Query API**: Search and retrieval performance benchmarks
- **Smart Model Selection**: Automated model selection based on performance profiles

## Benchmark Scripts

### Core Benchmarking Scripts

#### Embedding Model Benchmarks

```bash
# Run comprehensive embedding model benchmarks
uv run python scripts/benchmark_embedding_models.py \
  --models "text-embedding-3-small,BAAI/bge-small-en-v1.5" \
  --test-sizes "100,1000,5000" \
  --iterations 5 \
  --output benchmarks/embedding-results.json

# Compare cost vs performance
uv run python scripts/benchmark_embedding_models.py \
  --cost-analysis \
  --budget-limit 100.0 \
  --quality-threshold 0.8
```

#### HNSW Optimization Benchmarks

```bash
# Benchmark HNSW parameter optimization
uv run python scripts/benchmark_hnsw_optimization.py \
  --collection-types "api_reference,tutorials,blog_posts" \
  --ef-range "50,100,150,200" \
  --m-range "12,16,20,24" \
  --test-queries 1000

# Test adaptive ef selection
uv run python scripts/benchmark_hnsw_optimization.py \
  --adaptive-ef \
  --time-budgets "50,100,200,500" \
  --accuracy-targets "0.8,0.9,0.95"
```

#### Crawling Performance Benchmarks

```bash
# Benchmark Crawl4AI performance
uv run python scripts/benchmark_crawl4ai_performance.py \
  --urls-file test_urls.txt \
  --concurrent-workers "1,5,10,20" \
  --memory-monitoring \
  --output crawl-performance.json

# Test memory-adaptive dispatcher
uv run python scripts/benchmark_crawl4ai_performance.py \
  --memory-adaptive \
  --memory-thresholds "50,70,85" \
  --session-limits "5,10,20"
```

#### Lightweight Tier Benchmarks

```bash
# Test lightweight scraping efficiency
uv run python scripts/benchmark_lightweight_tier.py \
  --site-types "documentation,github,simple-html" \
  --comparison-mode \
  --metrics "speed,accuracy,resource-usage"

# Pattern matching optimization
uv run python scripts/benchmark_lightweight_tier.py \
  --pattern-optimization \
  --url-patterns "*.md,*/docs/*,*/api/*"
```

#### Anti-detection Performance

```bash
# Benchmark anti-detection capabilities
uv run python scripts/benchmark_anti_detection_performance.py \
  --detection-tests "cloudflare,recaptcha,fingerprinting" \
  --browser-profiles "stealth,standard,aggressive" \
  --success-rate-target 0.95
```

#### Payload Indexing Performance

```bash
# Test vector indexing performance
uv run python scripts/benchmark_payload_indexing.py \
  --document-sizes "1k,10k,100k" \
  --batch-sizes "10,50,100,500" \
  --quantization-modes "none,scalar,binary"

# Benchmark prefetch optimization
uv run python scripts/benchmark_payload_indexing.py \
  --prefetch-testing \
  --vector-types "dense,sparse,hyde" \
  --multipliers "1.5,2.0,3.0,5.0"
```

#### Query API Benchmarks

```bash
# Comprehensive query performance testing
uv run python scripts/benchmark_query_api.py \
  --query-types "simple,complex,hybrid,reranked" \
  --result-sizes "10,50,100" \
  --accuracy-levels "fast,balanced,accurate"

# Test RRF and DBSF fusion algorithms
uv run python scripts/benchmark_query_api.py \
  --fusion-algorithms "rrf,dbsf" \
  --weight-combinations "0.3:0.7,0.5:0.5,0.7:0.3" \
  --quality-metrics
```

## Performance Configuration

### Model Benchmarks

```python
# Configure model benchmarks in config/models.py
model_benchmarks = {
    "text-embedding-3-small": ModelBenchmark(
        model_name="text-embedding-3-small",
        provider="openai",
        avg_latency_ms=78,
        quality_score=85,
        tokens_per_second=12800,
        cost_per_million_tokens=20.0,
        max_context_length=8191,
        embedding_dimensions=1536,
    ),
    "BAAI/bge-small-en-v1.5": ModelBenchmark(
        model_name="BAAI/bge-small-en-v1.5", 
        provider="fastembed",
        avg_latency_ms=45,
        quality_score=78,
        tokens_per_second=22000,
        cost_per_million_tokens=0.0,
        max_context_length=512,
        embedding_dimensions=384,
    ),
}
```

### Smart Selection Configuration

```python
# Configure smart model selection in SmartSelectionConfig
smart_selection = SmartSelectionConfig(
    quality_weight=0.4,  # Prioritize quality
    speed_weight=0.3,    # Balance speed
    cost_weight=0.3,     # Consider cost
    
    # Quality thresholds (0-100 scale)
    quality_fast_threshold=60.0,
    quality_balanced_threshold=75.0, 
    quality_best_threshold=85.0,
    
    # Speed thresholds (tokens/second)
    speed_fast_threshold=500.0,
    speed_balanced_threshold=200.0,
    
    # Cost thresholds (per million tokens)
    cost_cheap_threshold=50.0,
    cost_moderate_threshold=100.0,
)
```

### HNSW Optimization Settings

```python
# Collection-specific HNSW configurations
collection_hnsw_configs = CollectionHNSWConfigs(
    api_reference=HNSWConfig(
        m=20,  # High accuracy for API docs
        ef_construct=300,
        min_ef=100,
        max_ef=200,
    ),
    tutorials=HNSWConfig(
        m=16,  # Balanced for tutorials
        ef_construct=200,
        min_ef=75,
        max_ef=150,
    ),
    blog_posts=HNSWConfig(
        m=12,  # Fast for blog content
        ef_construct=150,
        min_ef=50,
        max_ef=100,
    ),
)
```

## Continuous Benchmarking

### Automated Performance Testing

```bash
# Set up continuous benchmarking
cat > .github/workflows/performance.yml << 'EOF'
name: Performance Benchmarks
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run benchmarks
        run: |
          uv run python scripts/benchmark_query_api.py --ci-mode
          uv run python scripts/benchmark_embedding_models.py --ci-mode
          uv run python scripts/benchmark_hnsw_optimization.py --ci-mode
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmarks/
EOF
```

### Performance Monitoring

```python
# Monitor performance metrics
from src.services.monitoring.metrics import performance_monitor

@performance_monitor.track_performance("embedding_generation")
async def generate_embeddings(texts: list[str]):
    # Track latency, throughput, cost
    pass

@performance_monitor.track_performance("vector_search")  
async def search_vectors(query_vector: list[float]):
    # Monitor search performance
    pass

# Get performance reports
metrics = performance_monitor.get_metrics()
print(f"Average embedding latency: {metrics['embedding_generation']['avg_latency_ms']}ms")
print(f"Search throughput: {metrics['vector_search']['throughput_per_sec']}/sec")
```

## Performance Optimization

### Memory Management

```python
# Configure memory limits and optimization
performance_config = PerformanceConfig(
    max_memory_usage_mb=1000.0,
    gc_threshold=0.8,
    max_concurrent_requests=10,
    
    # DragonflyDB optimization
    dragonfly_pipeline_size=100,
    dragonfly_scan_count=1000,
    enable_dragonfly_compression=True,
)
```

### Rate Limiting

```python
# Configure provider rate limits
default_rate_limits = {
    "openai": {"max_calls": 500, "time_window": 60},     # 500/min
    "firecrawl": {"max_calls": 100, "time_window": 60},  # 100/min  
    "crawl4ai": {"max_calls": 50, "time_window": 1},     # 50/sec
    "qdrant": {"max_calls": 100, "time_window": 1},      # 100/sec
}
```

### Caching Optimization

```python
# Optimize cache performance
cache_config = CacheConfig(
    enable_caching=True,
    enable_local_cache=True,
    enable_dragonfly_cache=True,
    
    # Cache TTL settings
    cache_ttl_seconds={
        CacheType.EMBEDDINGS: 86400,  # 24 hours
        CacheType.CRAWL: 3600,        # 1 hour
        CacheType.SEARCH: 7200,       # 2 hours
        CacheType.HYDE: 3600,         # 1 hour
    },
    
    # Local cache limits
    local_max_size=1000,
    local_max_memory_mb=100.0,
)
```

## Performance Targets

### Latency Targets

- **Embedding Generation**: <100ms for text-embedding-3-small
- **Vector Search**: <50ms for balanced accuracy queries
- **Content Intelligence**: <200ms for full analysis
- **Crawling**: <5s for standard documentation pages
- **Cache Hit Rate**: >80% for repeated operations

### Throughput Targets

- **Concurrent Requests**: 50+ req/sec sustained
- **Batch Processing**: 1000+ documents/minute
- **Memory Usage**: <1GB for standard workloads
- **CPU Utilization**: <70% under normal load

### Quality Targets

- **Search Accuracy**: >90% relevance for balanced mode
- **Content Extraction**: >95% success rate
- **Anti-detection**: >95% success rate for stealth mode
- **Test Coverage**: >90% across all components

## Troubleshooting Performance

### Common Performance Issues

**High Memory Usage**
- Enable memory-adaptive dispatcher
- Reduce batch sizes
- Increase GC threshold
- Monitor memory leaks

**Slow Search Performance**
- Optimize HNSW parameters
- Enable prefetch optimization
- Use adaptive ef selection
- Check quantization settings

**Poor Cache Performance**
- Verify DragonflyDB connection
- Enable compression
- Optimize cache key patterns
- Monitor TTL settings

**Crawling Bottlenecks**
- Enable lightweight tier for simple content
- Optimize concurrent worker counts
- Use memory-adaptive dispatching
- Monitor resource usage

For detailed performance analysis and optimization strategies, see the [Developer API Reference](api-reference.md).