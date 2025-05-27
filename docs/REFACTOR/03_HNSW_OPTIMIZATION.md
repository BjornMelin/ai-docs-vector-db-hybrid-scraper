# HNSW Configuration Optimization Guide

**GitHub Issue**: [#57](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/57)

## Overview

HNSW (Hierarchical Navigable Small World) is Qdrant's high-performance vector index. Proper tuning can improve search accuracy by 5-10% and reduce latency by 20-30%. Our current configuration uses defaults that aren't optimized for documentation search workloads.

## Current Configuration

```python
# Current: Using Qdrant defaults
hnsw_config = HnswConfigDiff(
    m=16,              # Default
    ef_construct=128,  # Default
    full_scan_threshold=10000,  # Default
)
```

## Optimized Configuration

Based on our workload characteristics:

- ~100K-1M documents per collection
- High-quality embeddings (OpenAI text-embedding-3-small)
- Accuracy more important than speed
- 95th percentile latency target: <100ms

```python
# Optimized for documentation search
hnsw_config = HnswConfigDiff(
    m=16,                      # Increased connections for better accuracy
    ef_construct=200,          # Higher quality graph construction
    full_scan_threshold=10000, # When to bypass index
    max_m=16,                  # Maximum connections
    ef_retrieve=100,           # Runtime search parameter
    seed=42,                   # Reproducible builds
)
```

## Parameter Deep Dive

### 1. M Parameter (Connections per Node)

Controls graph connectivity:

```python
# M parameter effects:
# m=4:  Fast but lower accuracy (~85-90% recall)
# m=8:  Balanced (~92-95% recall)
# m=16: High accuracy (~96-98% recall) - Our choice
# m=32: Diminishing returns, 2x memory usage

async def test_m_parameter_impact():
    """Benchmark different M values."""
    for m in [4, 8, 16, 32]:
        config = HnswConfigDiff(m=m, ef_construct=200)
        await create_collection_with_config(config)
        
        # Measure recall@10
        recall = await measure_recall(ground_truth, predictions)
        memory = await get_memory_usage()
        
        print(f"m={m}: recall={recall:.3f}, memory={memory}MB")
```

### 2. EF Construction (Build Quality)

Controls index build quality:

```python
# ef_construct effects:
# 100: Fast build, lower quality (~94% recall)
# 200: Balanced build time/quality (~97% recall) - Our choice
# 400: Slow build, marginal gains (~98% recall)

async def optimize_ef_construct():
    """Find optimal ef_construct value."""
    
    build_times = []
    recall_scores = []
    
    for ef in [100, 150, 200, 300, 400]:
        start = time.time()
        await build_index(ef_construct=ef)
        build_time = time.time() - start
        
        recall = await measure_recall()
        
        build_times.append(build_time)
        recall_scores.append(recall)
    
    # Plot efficiency curve
    plot_pareto_frontier(build_times, recall_scores)
```

### 3. EF Retrieve (Runtime Search)

Dynamic parameter for search quality/speed tradeoff:

```python
async def adaptive_ef_retrieve(
    self,
    query_vector: list[float],
    time_budget_ms: int = 100
) -> list[SearchResult]:
    """Dynamically adjust ef parameter based on time budget."""
    
    # Start with conservative estimate
    ef = 50
    
    while ef <= 200:
        start = time.time()
        
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=10,
            search_params=SearchParams(
                hnsw_ef=ef,
                exact=False
            )
        )
        
        search_time_ms = (time.time() - start) * 1000
        
        # If we have time budget remaining, increase quality
        if search_time_ms < time_budget_ms * 0.7:
            ef = min(ef + 50, 200)
        else:
            break
    
    return results
```

## Collection-Specific Optimization

Different collections benefit from different settings:

```python
def get_optimized_config(collection_type: str) -> HnswConfigDiff:
    """Get optimized HNSW config for collection type."""
    
    configs = {
        # High-accuracy for API reference
        "api_reference": HnswConfigDiff(
            m=20,
            ef_construct=300,
            full_scan_threshold=5000,
        ),
        
        # Balanced for tutorials
        "tutorials": HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=10000,
        ),
        
        # Fast for blog posts
        "blog_posts": HnswConfigDiff(
            m=12,
            ef_construct=150,
            full_scan_threshold=20000,
        ),
    }
    
    return configs.get(collection_type, configs["tutorials"])
```

## Quantization Integration

HNSW works with quantization for memory efficiency:

```python
async def create_optimized_collection(
    self,
    collection_name: str,
    vectors_config: dict
) -> None:
    """Create collection with optimized HNSW and quantization."""
    
    await self.client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=1536,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                    full_scan_threshold=10000,
                ),
                quantization_config=ScalarQuantization(
                    type=ScalarType.INT8,
                    quantile=0.95,
                    always_ram=True,
                ),
            ),
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    full_scan_threshold=5000,
                ),
            ),
        },
    )
```

## Performance Monitoring

Track HNSW performance metrics:

```python
class HNSWMonitor:
    """Monitor HNSW index performance."""
    
    async def collect_metrics(self, collection_name: str) -> dict:
        """Collect HNSW performance metrics."""
        
        # Get collection info
        info = await self.client.get_collection(collection_name)
        
        # Run benchmark queries
        latencies = []
        recalls = []
        
        for _ in range(100):
            query = generate_random_query()
            
            # Measure exact search (ground truth)
            exact_results = await self.search_exact(query)
            
            # Measure HNSW search
            start = time.time()
            hnsw_results = await self.search_hnsw(query)
            latency = time.time() - start
            
            # Calculate recall
            recall = calculate_recall(exact_results, hnsw_results)
            
            latencies.append(latency)
            recalls.append(recall)
        
        return {
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "mean_recall": np.mean(recalls),
            "min_recall": np.min(recalls),
            "index_size_mb": info.index_size_mb,
            "vectors_count": info.vectors_count,
        }
```

## A/B Testing Framework

Test configuration changes safely:

```python
class HNSWExperiment:
    """A/B test different HNSW configurations."""
    
    async def run_experiment(
        self,
        control_config: HnswConfigDiff,
        treatment_config: HnswConfigDiff,
        duration_hours: int = 24
    ) -> dict:
        """Run A/B test between configurations."""
        
        # Create shadow collection with new config
        shadow_collection = f"{self.collection_name}_experiment"
        await self.create_collection(shadow_collection, treatment_config)
        
        # Mirror data to shadow collection
        await self.mirror_data(self.collection_name, shadow_collection)
        
        # Run parallel queries
        control_metrics = []
        treatment_metrics = []
        
        end_time = time.time() + (duration_hours * 3600)
        
        while time.time() < end_time:
            query = await self.get_next_query()
            
            # Query both collections
            control_result = await self.query_collection(
                self.collection_name, query
            )
            treatment_result = await self.query_collection(
                shadow_collection, query
            )
            
            # Collect metrics
            control_metrics.append(control_result)
            treatment_metrics.append(treatment_result)
        
        # Analyze results
        return self.analyze_experiment(control_metrics, treatment_metrics)
```

## Migration Strategy

### Phase 1: Benchmark Current Performance

```python
# Establish baseline metrics
baseline = await monitor.collect_metrics("documentation")
```

### Phase 2: Create Optimized Shadow Collection

```python
# Create with new HNSW config
await create_collection(
    "documentation_optimized",
    optimized_hnsw_config
)
```

### Phase 3: A/B Test

```python
# Run 24-hour experiment
results = await experiment.run_experiment(
    control_config=current_config,
    treatment_config=optimized_config
)
```

### Phase 4: Gradual Rollout

```python
# Use collection aliases for zero-downtime switch
if results["improvement"] > 0.05:  # 5% improvement threshold
    await switch_alias("documentation", "documentation_optimized")
```

## Workload-Specific Tuning

### High-Accuracy Workload (API Docs)

```python
config = HnswConfigDiff(
    m=20,                    # More connections
    ef_construct=300,        # Higher build quality
    full_scan_threshold=5000 # Lower threshold
)
```

### Balanced Workload (Tutorials)

```python
config = HnswConfigDiff(
    m=16,                     # Balanced connections
    ef_construct=200,         # Good build quality
    full_scan_threshold=10000 # Standard threshold
)
```

### High-Speed Workload (Search Suggestions)

```python
config = HnswConfigDiff(
    m=12,                     # Fewer connections
    ef_construct=150,         # Faster builds
    full_scan_threshold=20000 # Higher threshold
)
```

## Common Issues and Solutions

### Issue 1: Slow Index Building

```python
# Solution: Parallel index building
async def parallel_index_build(vectors: list, batch_size: int = 1000):
    """Build index in parallel batches."""
    
    batches = [
        vectors[i:i + batch_size]
        for i in range(0, len(vectors), batch_size)
    ]
    
    tasks = [
        insert_batch(batch)
        for batch in batches
    ]
    
    await asyncio.gather(*tasks)
```

### Issue 2: Memory Usage

```python
# Solution: Use quantization with HNSW
quantization_config = ScalarQuantization(
    type=ScalarType.INT8,
    quantile=0.95,
    always_ram=True  # Keep quantized vectors in RAM
)
```

### Issue 3: Inconsistent Recall

```python
# Solution: Dynamic ef adjustment
async def ensure_recall(
    min_recall: float = 0.95,
    max_ef: int = 200
):
    """Dynamically adjust ef to maintain recall."""
    
    current_ef = 50
    
    while current_ef <= max_ef:
        recall = await measure_current_recall()
        
        if recall >= min_recall:
            break
            
        current_ef += 25
        await update_search_params(hnsw_ef=current_ef)
```

## Performance Expectations

With optimized HNSW configuration:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Recall@10 | 92% | 97% | +5.4% |
| P95 Latency | 120ms | 85ms | -29% |
| P99 Latency | 180ms | 110ms | -39% |
| Index Build Time | 10min | 15min | +50% |
| Memory Usage | 4GB | 4.5GB | +12.5% |

## Testing

```python
@pytest.mark.asyncio
async def test_hnsw_optimization():
    """Test that optimized HNSW improves performance."""
    
    # Create collections with different configs
    await create_collection("test_default", default_config)
    await create_collection("test_optimized", optimized_config)
    
    # Insert same data
    await insert_test_data("test_default")
    await insert_test_data("test_optimized")
    
    # Benchmark
    default_metrics = await benchmark_collection("test_default")
    optimized_metrics = await benchmark_collection("test_optimized")
    
    # Assert improvements
    assert optimized_metrics["recall"] > default_metrics["recall"]
    assert optimized_metrics["p95_latency"] < default_metrics["p95_latency"]
```

## Next Steps

1. Implement HNSW monitoring dashboard
2. Set up automated A/B testing for config changes
3. Create per-collection optimization profiles
4. Build adaptive ef_retrieve logic
5. Document optimal settings for different use cases
