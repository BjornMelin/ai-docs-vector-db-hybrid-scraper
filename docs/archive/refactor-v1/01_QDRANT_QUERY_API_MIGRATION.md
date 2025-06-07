# Qdrant Query API Migration Guide

**GitHub Issue**: [#55](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/55)

## Overview

This guide covers migrating from basic `search()` to Qdrant's advanced `query_points()` API, enabling multi-stage retrieval and native fusion algorithms.

## Why Query API?

The Query API provides:

- **15-30% performance improvement** through optimized execution
- **Native fusion algorithms** (RRF, DBSFusion)
- **Multi-stage retrieval** in a single request
- **Reduced network overhead** with prefetch
- **Better integration** with HyDE and reranking

## Migration Steps

### 1. Update Search Interface

#### Before (Basic Search)

```python
async def search(
    self,
    collection_name: str,
    query_vector: list[float],
    limit: int = 10,
    score_threshold: float = 0.0,
) -> list[dict]:
    results = await self.client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
    )
    return [self._format_result(r) for r in results]
```

#### After (Query API)

```python
async def search(
    self,
    collection_name: str,
    query_vector: list[float],
    sparse_vector: dict[int, float] | None = None,
    limit: int = 10,
    score_threshold: float = 0.0,
    fusion_type: str = "rrf",
) -> list[dict]:
    # Build prefetch configuration
    prefetch = []
    
    # Add sparse vector prefetch if available
    if sparse_vector:
        prefetch.append(
            Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=limit * 3,  # Cast wider net for sparse
            )
        )
    
    # Use Query API with prefetch
    results = await self.client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense",
        prefetch=prefetch,
        fusion=Fusion.RRF if fusion_type == "rrf" else Fusion.DBSF,
        limit=limit,
        score_threshold=score_threshold,
    )
    
    return [self._format_result(r) for r in results]
```

### 2. Implement Multi-Stage Retrieval

```python
async def multi_stage_search(
    self,
    collection_name: str,
    stages: list[SearchStage],
    limit: int = 10,
) -> list[dict]:
    """
    Perform multi-stage retrieval with different strategies.
    
    Example stages:
    - Stage 1: Wide sparse search (100 results)
    - Stage 2: Precise dense search (50 results)
    - Stage 3: Rerank top results
    """
    prefetch = []
    
    for stage in stages[:-1]:  # All but final stage
        prefetch.append(
            Prefetch(
                query=stage.query_vector,
                using=stage.vector_name,
                limit=stage.limit,
                filter=stage.filter,
                params=stage.search_params,
            )
        )
    
    # Final stage query
    final_stage = stages[-1]
    results = await self.client.query_points(
        collection_name=collection_name,
        query=final_stage.query_vector,
        using=final_stage.vector_name,
        prefetch=prefetch,
        fusion=Fusion.RRF,
        limit=limit,
    )
    
    return results
```

### 3. HyDE Integration with Query API

```python
async def hyde_search(
    self,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Search using HyDE with Query API prefetch."""
    
    # Generate hypothetical documents
    hypothetical_docs = await self.generate_hypothetical_docs(query, n=5)
    hypothetical_embeddings = await self.embed_texts(hypothetical_docs)
    hypothetical_vector = np.mean(hypothetical_embeddings, axis=0)
    
    # Original query embedding
    query_embedding = await self.embed_text(query)
    
    # Use Query API with both embeddings
    results = await self.client.query_points(
        collection_name="documents",
        prefetch=[
            # HyDE embedding - cast wider net
            Prefetch(
                query=hypothetical_vector.tolist(),
                using="dense",
                limit=50,
            ),
            # Original query - for precision
            Prefetch(
                query=query_embedding,
                using="dense",
                limit=30,
            ),
        ],
        query=query_embedding,  # Final fusion query
        using="dense",
        fusion=Fusion.RRF,
        limit=limit,
    )
    
    return results
```

### 4. Filtered Search Optimization

```python
async def filtered_search(
    self,
    query_vector: list[float],
    filters: dict[str, Any],
    limit: int = 10,
) -> list[dict]:
    """Optimized filtered search using indexed payload fields."""
    
    # Build Qdrant filter
    must_conditions = []
    
    if "doc_type" in filters:
        must_conditions.append(
            FieldCondition(
                key="doc_type",
                match=MatchValue(value=filters["doc_type"])
            )
        )
    
    if "language" in filters:
        must_conditions.append(
            FieldCondition(
                key="language",
                match=MatchValue(value=filters["language"])
            )
        )
    
    if "created_after" in filters:
        must_conditions.append(
            FieldCondition(
                key="created_at",
                range=Range(gte=filters["created_after"])
            )
        )
    
    filter_obj = Filter(must=must_conditions) if must_conditions else None
    
    # Use Query API with filter
    results = await self.client.query_points(
        collection_name="documents",
        query=query_vector,
        using="dense",
        filter=filter_obj,
        limit=limit,
    )
    
    return results
```

### 5. Update MCP Server Tools

```python
@mcp.tool()
async def advanced_search(request: AdvancedSearchRequest) -> list[SearchResult]:
    """Advanced search with Query API and filtering."""
    
    # Build prefetch configs
    prefetch_configs = []
    
    if request.use_hyde:
        hyde_embedding = await generate_hyde_embedding(request.query)
        prefetch_configs.append(
            Prefetch(query=hyde_embedding, using="dense", limit=50)
        )
    
    if request.sparse_query:
        prefetch_configs.append(
            Prefetch(query=request.sparse_query, using="sparse", limit=100)
        )
    
    # Perform search
    results = await qdrant_service.query_points(
        collection_name=request.collection,
        query=request.query_vector,
        prefetch=prefetch_configs,
        filter=request.filter,
        fusion=Fusion[request.fusion_type.upper()],
        params=SearchParams(
            hnsw_ef=request.search_accuracy or 100,
            exact=request.exact_search or False,
        ),
        limit=request.limit,
    )
    
    return [format_result(r) for r in results]
```

## Performance Tuning

### 1. Prefetch Optimization

```python
# Optimal prefetch limits based on research
PREFETCH_LIMITS = {
    "sparse": lambda final_limit: final_limit * 5,  # Cast wider
    "hyde": lambda final_limit: final_limit * 3,    # Moderate expansion
    "dense": lambda final_limit: final_limit * 2,   # Precision focus
}
```

### 2. Fusion Algorithm Selection

```python
def select_fusion_algorithm(query_type: str) -> Fusion:
    """Select optimal fusion algorithm based on query type."""
    
    fusion_map = {
        "hybrid": Fusion.RRF,      # Best for combining dense+sparse
        "multi_stage": Fusion.RRF, # Good for multiple strategies
        "reranking": Fusion.DBSF,  # Better for similar vectors
    }
    
    return fusion_map.get(query_type, Fusion.RRF)
```

### 3. Search Parameter Optimization

```python
def get_search_params(accuracy_level: str) -> SearchParams:
    """Get optimized search parameters."""
    
    params_map = {
        "fast": SearchParams(hnsw_ef=50, exact=False),
        "balanced": SearchParams(hnsw_ef=100, exact=False),
        "accurate": SearchParams(hnsw_ef=200, exact=False),
        "exact": SearchParams(exact=True),
    }
    
    return params_map.get(accuracy_level, params_map["balanced"])
```

## Migration Checklist

- [ ] Update all `search()` calls to `query_points()`
- [ ] Implement prefetch for multi-stage retrieval
- [ ] Add fusion algorithm selection
- [ ] Update search interfaces with new parameters
- [ ] Add performance monitoring for Query API
- [ ] Update documentation with Query API examples
- [ ] Benchmark performance improvements
- [ ] Test with various query types
- [ ] Implement gradual rollout with feature flags

## Expected Results

After migration, you should see:

- **15-30% faster search latency**
- **Better relevance** with native fusion
- **Simplified code** for complex searches
- **Reduced API calls** with prefetch
- **Improved HyDE integration**

## Troubleshooting

### Common Issues

1. **Prefetch limit too high**
   - Symptom: Slow searches
   - Solution: Reduce prefetch limits

2. **Fusion algorithm mismatch**
   - Symptom: Poor relevance
   - Solution: Test different fusion types

3. **Missing vector names**
   - Symptom: Query errors
   - Solution: Ensure vector names match collection config

## Next Steps

1. Complete Query API migration
2. Add payload indexing (see [Payload Indexing Guide](./01_PAYLOAD_INDEXING.md))
3. Integrate HyDE (see [HyDE Implementation](./04_HYDE_IMPLEMENTATION.md))
4. Monitor performance improvements
