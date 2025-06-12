# Query API Optimization and Advanced Features

This document covers the advanced Query API capabilities including prefetch
optimization, RRF/DBSF fusion algorithms, and performance tuning strategies.

## Query API Overview

The Query API provides high-performance search capabilities with intelligent
prefetch optimization and advanced fusion algorithms for combining multiple
search strategies.

### Key Features

- **Prefetch Optimization**: Intelligent candidate retrieval with
  vector-type-specific multipliers
- **RRF/DBSF Fusion**: Advanced algorithms for combining dense and sparse
  search results
- **Adaptive Strategies**: Dynamic selection of optimal search approaches
- **Performance Monitoring**: Built-in metrics and optimization feedback

## Prefetch Optimization

### Configuration

```python
from src.config.models import VectorSearchConfig, VectorType

# Configure prefetch multipliers by vector type
vector_search_config = VectorSearchConfig(
    prefetch_multipliers={
        VectorType.DENSE: 2.0,   # Dense vectors: retrieve 2x candidates
        VectorType.SPARSE: 5.0,  # Sparse vectors: retrieve 5x candidates
        VectorType.HYDE: 3.0,    # HyDE vectors: retrieve 3x candidates
    },
    max_prefetch_limits={
        VectorType.DENSE: 200,   # Maximum 200 dense candidates
        VectorType.SPARSE: 500,  # Maximum 500 sparse candidates
        VectorType.HYDE: 150,    # Maximum 150 HyDE candidates
    },
    default_search_limit=10,
    max_search_limit=100
)
```

### Optimized Search Execution

```python
from src.services.vector_db.search import VectorSearchService

search_service = VectorSearchService(vector_search_config)

# Automatic prefetch optimization
results = await search_service.search_with_prefetch(
    query="How to implement microservices authentication",
    collection_name="architecture_docs",
    limit=10,
    vector_type=VectorType.DENSE,
    accuracy=SearchAccuracy.BALANCED
)

# Prefetch calculation:
# - Requested: 10 results
# - Dense multiplier: 2.0
# - Prefetch size: min(10 * 2.0, 200) = 20 candidates
# - Returns top 10 after reranking
```

### Prefetch Performance Benefits

- **Improved Relevance**: Larger candidate pool enables better reranking
- **Cost Efficiency**: Optimized for each vector type's characteristics
- **Reduced Latency**: Smart caching of prefetch results
- **Memory Optimization**: Configurable limits prevent excessive memory usage

## Fusion Algorithms

### Reciprocal Rank Fusion (RRF)

RRF combines multiple search results by focusing on ranking positions rather
than raw scores.

```python
from src.services.vector_db.search import FusionAlgorithm

# RRF fusion configuration
results = await search_service.hybrid_search(
    query="Python authentication best practices",
    collection_name="security_docs",
    dense_weight=0.6,
    sparse_weight=0.4,
    fusion_algorithm=FusionAlgorithm.RRF,
    rrf_constant=60,  # RRF ranking constant (typically 60)
    limit=10
)

# RRF score calculation:
# For each result: 1 / (rank + rrf_constant)
# Final score = dense_weight * rrf_dense_score + sparse_weight * rrf_sparse_score
```

**RRF Benefits:**

- **Rank-based**: Less sensitive to score distribution differences
- **Robust**: Handles varying score ranges between search methods
- **Proven**: Well-established algorithm with strong research backing
- **Tunable**: RRF constant allows fine-tuning for different domains

### Distribution-Based Score Fusion (DBSF)

DBSF normalizes score distributions before fusion to handle different scoring
ranges.

```python
# DBSF fusion with score normalization
results = await search_service.hybrid_search(
    query="FastAPI security implementation patterns",
    collection_name="api_docs",
    dense_weight=0.7,
    sparse_weight=0.3,
    fusion_algorithm=FusionAlgorithm.DBSF,
    normalize_scores=True,  # Enable z-score normalization
    score_distribution_method="zscore",  # or "minmax"
    limit=10
)

# DBSF process:
# 1. Collect scores from both dense and sparse search
# 2. Normalize score distributions (z-score or min-max)
# 3. Apply weighted fusion: dense_weight * norm_dense + sparse_weight * norm_sparse
```

**DBSF Benefits:**

- **Distribution-aware**: Handles different score ranges intelligently
- **Flexible normalization**: Supports z-score and min-max normalization
- **Better calibration**: More accurate fusion when score ranges differ significantly
- **Domain-adaptive**: Adjusts to different content types automatically

### Adaptive Fusion

Automatically selects the optimal fusion algorithm based on query characteristics.

```python
# Adaptive fusion with automatic optimization
results = await search_service.adaptive_hybrid_search(
    query="authentication security patterns",
    collection_name="security_docs",
    auto_weight_selection=True,    # Automatically determine optimal weights
    fusion_strategy="adaptive",    # Choose best fusion algorithm
    performance_target="balanced", # balance accuracy and speed
    limit=10
)

# Adaptive selection criteria:
# - Query complexity (simple queries may skip fusion)
# - Score distribution characteristics (chooses RRF vs DBSF)
# - Historical performance data
# - Collection characteristics
```

## Advanced Search Strategies

### Multi-Vector Fusion

Combine multiple vector types for comprehensive search coverage.

```python
# Multi-vector search with different strategies per vector type
results = await search_service.multi_vector_search(
    query="implement OAuth2 with JWT tokens",
    collection_name="auth_docs",
    vector_strategies={
        VectorType.DENSE: {
            "weight": 0.5,
            "prefetch_multiplier": 2.0,
            "accuracy": SearchAccuracy.BALANCED
        },
        VectorType.SPARSE: {
            "weight": 0.3,
            "prefetch_multiplier": 4.0,
            "accuracy": SearchAccuracy.FAST
        },
        VectorType.HYDE: {
            "weight": 0.2,
            "prefetch_multiplier": 3.0,
            "accuracy": SearchAccuracy.ACCURATE
        }
    },
    fusion_algorithm=FusionAlgorithm.ADAPTIVE,
    limit=10
)
```

### Query-Specific Optimization

Optimize search parameters based on query characteristics.

```python
# Automatic optimization based on query analysis
from src.services.query_processing.pipeline import QueryProcessingPipeline

pipeline = QueryProcessingPipeline()

# Query analysis informs search optimization
optimized_results = await pipeline.process_with_optimization(
    query="troubleshoot slow database queries in production",
    collection_name="performance_docs",
    optimization_strategy="query_adaptive",  # Optimize based on query intent
    limit=10
)

# Optimization factors:
# - Query intent (troubleshooting -> prioritize specific solutions)
# - Query complexity (complex -> higher prefetch, better fusion)
# - Domain specificity (technical -> favor precise matches)
# - Urgency indicators (production issues -> prioritize recent content)
```

## Performance Monitoring and Tuning

### Search Performance Metrics

```python
# Enable detailed performance monitoring
search_service = VectorSearchService(
    config=vector_search_config,
    enable_performance_tracking=True
)

# Get performance metrics
metrics = await search_service.get_performance_metrics()

print(f"Average search latency: {metrics['avg_search_latency_ms']}ms")
print(f"Prefetch efficiency: {metrics['prefetch_hit_rate']:.1%}")
print(f"Fusion algorithm usage: {metrics['fusion_algorithm_usage']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")

# Per-vector-type performance
for vector_type, perf in metrics['vector_type_performance'].items():
    print(f"{vector_type}: {perf['avg_latency_ms']}ms, {perf['accuracy_score']:.2f}")
```

### Optimization Recommendations

```python
# Get optimization recommendations based on usage patterns
recommendations = await search_service.get_optimization_recommendations()

for rec in recommendations:
    print(f"Component: {rec['component']}")
    print(f"Current performance: {rec['current_metrics']}")
    print(f"Recommendation: {rec['suggestion']}")
    print(f"Expected improvement: {rec['expected_improvement']}")
    print("---")

# Example recommendations:
# - Increase prefetch multiplier for dense vectors (low reranking benefit)
# - Switch to RRF fusion for this collection (better score calibration)
# - Enable caching for repeated query patterns
# - Adjust HNSW parameters for better speed/accuracy balance
```

### A/B Testing Framework

```python
# A/B testing for fusion algorithms
from src.services.vector_db.search import ABTestConfig

ab_config = ABTestConfig(
    name="rrf_vs_dbsf_comparison",
    control_strategy={
        "fusion_algorithm": FusionAlgorithm.RRF,
        "rrf_constant": 60
    },
    experimental_strategy={
        "fusion_algorithm": FusionAlgorithm.DBSF,
        "normalize_scores": True
    },
    traffic_split=0.1,  # 10% experimental traffic
    metrics=["relevance_score", "search_latency", "user_satisfaction"]
)

# Run A/B test
ab_results = await search_service.run_ab_test(ab_config, duration_days=7)

print(f"Control performance: {ab_results['control']['avg_relevance']:.3f}")
print(f"Experimental performance: {ab_results['experimental']['avg_relevance']:.3f}")
print(f"Statistical significance: {ab_results['significance_test']['p_value']:.4f}")
```

## Configuration Best Practices

### Production Configuration

```python
# Optimized production configuration
production_config = VectorSearchConfig(
    # Conservative prefetch to balance accuracy and performance
    prefetch_multipliers={
        VectorType.DENSE: 1.8,
        VectorType.SPARSE: 4.0,
        VectorType.HYDE: 2.5,
    },
    # Reasonable limits for production workloads
    max_prefetch_limits={
        VectorType.DENSE: 150,
        VectorType.SPARSE: 400,
        VectorType.HYDE: 120,
    },
    # Accuracy tuned for production use
    search_accuracy_params={
        SearchAccuracy.FAST: {"ef": 60, "exact": False},
        SearchAccuracy.BALANCED: {"ef": 120, "exact": False},
        SearchAccuracy.ACCURATE: {"ef": 200, "exact": False},
    },
    default_search_limit=10,
    max_search_limit=50  # Prevent excessive resource usage
)
```

### Development/Testing Configuration

```python
# Development configuration for experimentation
dev_config = VectorSearchConfig(
    # Higher prefetch for better accuracy during testing
    prefetch_multipliers={
        VectorType.DENSE: 3.0,
        VectorType.SPARSE: 6.0,
        VectorType.HYDE: 4.0,
    },
    # Higher limits for comprehensive testing
    max_prefetch_limits={
        VectorType.DENSE: 300,
        VectorType.SPARSE: 600,
        VectorType.HYDE: 200,
    },
    # More aggressive accuracy settings
    search_accuracy_params={
        SearchAccuracy.FAST: {"ef": 80, "exact": False},
        SearchAccuracy.BALANCED: {"ef": 150, "exact": False},
        SearchAccuracy.ACCURATE: {"ef": 250, "exact": False},
    },
    default_search_limit=20,
    max_search_limit=100
)
```

## Troubleshooting

### Common Performance Issues

#### High Latency

- Reduce prefetch multipliers
- Lower accuracy requirements
- Enable result caching
- Check HNSW parameter tuning

#### Poor Relevance

- Increase prefetch limits
- Experiment with fusion algorithms
- Adjust vector type weights
- Review query preprocessing

#### Memory Usage

- Reduce max prefetch limits
- Enable vector quantization
- Implement result streaming
- Monitor cache size

#### Inconsistent Results

- Ensure stable fusion weights
- Check score normalization settings
- Verify vector index consistency
- Monitor system resource availability

### Debug Tools

```python
# Enable debug mode for detailed analysis
search_service = VectorSearchService(
    config=vector_search_config,
    debug_mode=True
)

# Get detailed search execution info
debug_results = await search_service.search_with_debug(
    query="authentication implementation",
    collection_name="docs",
    limit=10
)

print(f"Prefetch details: {debug_results['prefetch_info']}")
print(f"Fusion algorithm used: {debug_results['fusion_algorithm']}")
print(f"Score distributions: {debug_results['score_distributions']}")
print(f"Reranking impact: {debug_results['reranking_metrics']}")
```

For additional configuration options and advanced use cases,
see the [API Reference](../developers/api-reference.md) and
[Performance Benchmarking](../developers/benchmarking-and-performance.md) guides.
