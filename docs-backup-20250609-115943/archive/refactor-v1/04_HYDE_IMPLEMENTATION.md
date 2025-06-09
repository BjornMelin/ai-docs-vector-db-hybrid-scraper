# HyDE (Hypothetical Document Embeddings) Implementation Guide

> **Status**: Deprecated  
> **Last Updated**: 2025-06-09  
> **Purpose**: 04_Hyde_Implementation archived documentation  
> **Audience**: Historical reference

**GitHub Issue**: [#60](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/60)

## Overview

This guide covers implementing HyDE to improve search accuracy by 15-25% through generating synthetic documents that bridge user queries and actual content.

## What is HyDE?

HyDE improves retrieval by:

1. **Generating hypothetical answers** to queries using an LLM
2. **Embedding these synthetic documents**
3. **Averaging multiple generations** for robustness
4. **Using the result for semantic search** via Query API prefetch

## Implementation Architecture

```plaintext
Query → LLM Generation → Multiple Hypothetical Docs → Embeddings → Average → Search
   ↓                                                                    ↓
Cache Check ←────────────────── DragonflyDB ──────────────────→ Cache Store
```

## Step-by-Step Implementation

### 1. HyDE Query Engine Core

```python
from typing import List, Optional
import numpy as np
import hashlib
from openai import AsyncOpenAI

class HyDEQueryEngine:
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        embedding_manager: EmbeddingManager,
        qdrant_service: QdrantService,
        cache_manager: DragonflyCache,
        config: HyDEConfig
    ):
        self.llm = llm_client
        self.embeddings = embedding_manager
        self.qdrant = qdrant_service
        self.cache = cache_manager
        self.config = config
        
    async def enhanced_search(
        self,
        query: str,
        collection: str = "documents",
        limit: int = 10,
        filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """Perform HyDE-enhanced search with caching."""
        
        # 1. Get or generate HyDE embedding
        hyde_embedding = await self._get_hyde_embedding(query)
        
        # 2. Get original query embedding
        query_embedding = await self.embeddings.generate_embedding(query)
        
        # 3. Use Query API with prefetch
        results = await self.qdrant.query_points(
            collection_name=collection,
            prefetch=[
                # HyDE embedding - broader semantic understanding
                Prefetch(
                    query=hyde_embedding,
                    using="dense",
                    limit=self.config.hyde_prefetch_limit  # 50
                ),
                # Original query - precision matching
                Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=self.config.query_prefetch_limit  # 30
                )
            ],
            query=query_embedding,  # Final fusion query
            using="dense",
            fusion=Fusion.RRF,
            filter=filters,
            limit=limit
        )
        
        # 4. Optional reranking
        if self.config.enable_reranking:
            results = await self.embeddings.rerank_results(query, results)
            
        return results
```

### 2. Hypothetical Document Generation

```python
async def _get_hyde_embedding(self, query: str) -> List[float]:
    """Get or generate HyDE embedding with caching."""
    
    # Check cache first
    cache_key = f"hyde:{hashlib.md5(query.encode()).hexdigest()}"
    cached = await self.cache.get(cache_key)
    
    if cached:
        return np.frombuffer(cached["embedding"], dtype=np.float32).tolist()
    
    # Generate hypothetical documents
    hypothetical_docs = await self._generate_hypothetical_docs(query)
    
    # Embed all documents
    embeddings_result = await self.embeddings.generate_embeddings(
        texts=hypothetical_docs,
        provider_name="openai",  # Use high-quality provider
        auto_select=False
    )
    
    # Average embeddings
    embeddings_array = np.array(embeddings_result["embeddings"])
    averaged_embedding = np.mean(embeddings_array, axis=0)
    
    # Cache for future use
    await self.cache.hset(cache_key, {
        "embedding": averaged_embedding.tobytes(),
        "hypothetical_docs": json.dumps(hypothetical_docs),
        "query": query,
        "timestamp": time.time()
    })
    await self.cache.expire(cache_key, self.config.cache_ttl_seconds)
    
    return averaged_embedding.tolist()

async def _generate_hypothetical_docs(self, query: str) -> List[str]:
    """Generate multiple hypothetical documents."""
    
    prompt = self._build_hyde_prompt(query)
    
    # Generate multiple completions
    response = await self.llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        n=self.config.num_generations,  # 5
        temperature=self.config.generation_temperature,  # 0.7
        max_tokens=self.config.max_generation_tokens  # 200
    )
    
    # Extract generated texts
    hypothetical_docs = [
        choice.message.content.strip()
        for choice in response.choices
    ]
    
    # Log for debugging
    logger.debug(f"Generated {len(hypothetical_docs)} hypothetical documents for query: {query}")
    
    return hypothetical_docs
```

### 3. Prompt Engineering

```python
def _build_hyde_prompt(self, query: str) -> str:
    """Build effective prompt for hypothetical document generation."""
    
    # Domain-specific prompts perform better
    if self._is_technical_query(query):
        return f"""You are a technical documentation expert. 
Answer this question with a detailed, accurate response:

Question: {query}

Provide a comprehensive answer that would appear in high-quality technical documentation:"""
    
    elif self._is_code_query(query):
        return f"""You are a code documentation expert.
Answer this programming question:

Question: {query}

Provide a detailed answer with code examples as would appear in API documentation:"""
    
    else:
        # General prompt
        return f"""Answer the following question accurately and comprehensively:

Question: {query}

Answer:"""

def _is_technical_query(self, query: str) -> bool:
    """Detect technical queries."""
    technical_keywords = [
        "api", "function", "method", "class", "parameter",
        "configuration", "setup", "install", "error", "debug"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in technical_keywords)

def _is_code_query(self, query: str) -> bool:
    """Detect code-related queries."""
    code_indicators = [
        "how to", "example", "code", "implement", "python",
        "javascript", "function", "syntax", "library"
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in code_indicators)
```

### 4. Configuration

```python
from pydantic import BaseModel, Field

class HyDEConfig(BaseModel):
    """HyDE configuration with defaults."""
    
    # Feature flags
    enable_hyde: bool = True
    enable_fallback: bool = True
    enable_reranking: bool = True
    
    # Generation settings
    num_generations: int = Field(5, ge=1, le=10)
    generation_temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_generation_tokens: int = Field(200, ge=50, le=500)
    generation_model: str = "gpt-3.5-turbo"
    
    # Search settings
    hyde_prefetch_limit: int = 50
    query_prefetch_limit: int = 30
    hyde_weight_in_fusion: float = Field(0.6, ge=0.0, le=1.0)
    
    # Caching
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_hypothetical_docs: bool = True
    
    # Performance
    parallel_generation: bool = True
    generation_timeout_seconds: int = 10
```

### 5. Integration with Query API

```python
class HyDESearchRequest(BaseModel):
    """Enhanced search request with HyDE options."""
    query: str
    collection: str = "documents"
    limit: int = 10
    use_hyde: bool = True
    hyde_config: Optional[dict] = None
    filters: Optional[dict] = None
    
@mcp.tool()
async def hyde_search(request: HyDESearchRequest) -> List[SearchResult]:
    """Perform HyDE-enhanced search via MCP."""
    
    # Override config if provided
    if request.hyde_config:
        config = HyDEConfig(**request.hyde_config)
    else:
        config = app_config.hyde
    
    # Create engine with config
    engine = HyDEQueryEngine(
        llm_client=llm_client,
        embedding_manager=embedding_manager,
        qdrant_service=qdrant_service,
        cache_manager=cache_manager,
        config=config
    )
    
    # Perform search
    if request.use_hyde:
        results = await engine.enhanced_search(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            filters=request.filters
        )
    else:
        # Fallback to regular search
        results = await qdrant_service.search(
            collection_name=request.collection,
            query_text=request.query,
            limit=request.limit,
            filters=request.filters
        )
    
    return results
```

### 6. Performance Optimization

```python
class OptimizedHyDEEngine(HyDEQueryEngine):
    """Performance-optimized HyDE implementation."""
    
    async def _generate_hypothetical_docs_parallel(
        self, 
        query: str
    ) -> List[str]:
        """Generate documents in parallel for speed."""
        
        # Create multiple prompts with variation
        prompts = [
            self._build_hyde_prompt(query),
            self._build_hyde_prompt(query + " Explain in detail."),
            self._build_hyde_prompt(query + " Provide examples."),
            self._build_hyde_prompt(query + " Include best practices."),
            self._build_hyde_prompt(query + " Focus on implementation.")
        ]
        
        # Generate in parallel
        tasks = [
            self._generate_single_doc(prompt)
            for prompt in prompts[:self.config.num_generations]
        ]
        
        hypothetical_docs = await asyncio.gather(*tasks)
        return hypothetical_docs
    
    async def _generate_single_doc(self, prompt: str) -> str:
        """Generate single document with timeout."""
        try:
            response = await asyncio.wait_for(
                self.llm.chat.completions.create(
                    model=self.config.generation_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.generation_temperature,
                    max_tokens=self.config.max_generation_tokens
                ),
                timeout=self.config.generation_timeout_seconds
            )
            return response.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            logger.warning("HyDE generation timeout")
            return ""  # Will be filtered out
```

### 7. A/B Testing Support

```python
class HyDEABTestManager:
    """A/B testing for HyDE effectiveness."""
    
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        self.test_groups = {
            "control": 0.5,  # Regular search
            "treatment": 0.5  # HyDE search
        }
        
    async def search_with_ab_test(
        self,
        query: str,
        user_id: str,
        **kwargs
    ) -> dict:
        """Perform search with A/B test tracking."""
        
        # Determine test group
        group = self._assign_group(user_id)
        
        # Track query
        await self.metrics.track_query(
            query=query,
            user_id=user_id,
            group=group
        )
        
        # Perform search
        start_time = time.time()
        
        if group == "treatment":
            results = await self.hyde_engine.enhanced_search(query, **kwargs)
            search_type = "hyde"
        else:
            results = await self.qdrant_service.search(query, **kwargs)
            search_type = "regular"
            
        # Track metrics
        await self.metrics.track_search_metrics(
            query=query,
            user_id=user_id,
            group=group,
            latency_ms=(time.time() - start_time) * 1000,
            result_count=len(results),
            search_type=search_type
        )
        
        return {
            "results": results,
            "search_type": search_type,
            "test_group": group
        }
```

## Monitoring and Metrics

### Key Metrics to Track

1. **Generation Metrics**
   - Average generation time
   - Generation failures/timeouts
   - Token usage and costs

2. **Cache Metrics**
   - HyDE cache hit rate
   - Cache storage size
   - TTL effectiveness

3. **Search Quality**
   - Click-through rate comparison
   - Result relevance scores
   - User satisfaction metrics

4. **Performance Impact**
   - Search latency with/without HyDE
   - Resource usage
   - Cost per search

### Monitoring Implementation

```python
class HyDEMetrics:
    def __init__(self, prometheus_client):
        self.generation_time = Histogram(
            'hyde_generation_seconds',
            'Time to generate hypothetical documents'
        )
        self.cache_hits = Counter(
            'hyde_cache_hits_total',
            'Number of HyDE cache hits'
        )
        self.search_accuracy = Histogram(
            'hyde_search_accuracy',
            'Search accuracy with HyDE',
            buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        )
```

## Best Practices

1. **Prompt Engineering**
   - Use domain-specific prompts
   - Include context about expected format
   - Test different prompt variations

2. **Generation Parameters**
   - Start with 5 generations, adjust based on results
   - Use temperature 0.7 for diversity
   - Limit tokens to control costs

3. **Caching Strategy**
   - Cache for 1 hour (typical session length)
   - Include query normalization
   - Monitor cache effectiveness

4. **Error Handling**
   - Fallback to regular search on HyDE failure
   - Set generation timeouts
   - Log failures for analysis

5. **Cost Management**
   - Use GPT-3.5-turbo for generation
   - Monitor token usage
   - Implement daily limits if needed

## Expected Results

- **Accuracy**: 15-25% improvement in search relevance
- **Coverage**: Better handling of ambiguous queries
- **Latency**: +5-10ms with caching (acceptable tradeoff)
- **Cost**: ~$0.002 per query with 5 generations

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check cache hit rate
   - Reduce generation count
   - Use parallel generation

2. **Poor Quality**
   - Improve prompts
   - Increase generation diversity
   - Check embedding quality

3. **High Costs**
   - Reduce token limits
   - Increase cache TTL
   - Use cheaper models

## Conclusion

HyDE significantly improves search accuracy by bridging the semantic gap between queries and documents. With proper caching and integration with Query API, the performance impact is minimal while accuracy gains are substantial.
