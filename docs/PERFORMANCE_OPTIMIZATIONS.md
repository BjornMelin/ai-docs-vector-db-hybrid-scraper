# Performance Optimizations

> **Status:** Strategies Documented  
> **Priority:** High  
> **Estimated Effort:** 3-4 weeks  
> **Documentation Created:** 2025-05-22

## Overview

Comprehensive performance optimization strategies targeting embedding generation, vector operations, search latency, and resource utilization. These optimizations build on our research-backed foundation to achieve production-scale performance.

## Current Performance Baseline

### Measured Performance (2025-05-22)

- **Embedding Speed**: 45ms (FastEmbed) / 78ms (OpenAI) per chunk
- **Search Latency**: 23ms (quantized) / 41ms (full precision)
- **Storage Efficiency**: 83-99% reduction with quantization
- **Search Accuracy**: 89.3% (hybrid + reranking) vs 71.2% (dense-only)
- **Memory Usage**: ~2GB for 1M vectors (quantized)

### Performance Targets

- **Embedding Speed**: <20ms per chunk (2x improvement)
- **Search Latency**: <10ms at scale (2x improvement)
- **Throughput**: 10,000+ documents/hour
- **Concurrent Users**: 100+ simultaneous searches
- **Memory Efficiency**: <1GB for 1M vectors

## Optimization Categories

### 1. Embedding Generation Optimization

#### Batch Processing with OpenAI Batch API

```python
# src/services/batch_embedding_service.py
from openai import AsyncOpenAI
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime

class BatchEmbeddingService:
    """OpenAI Batch API for 50% cost reduction."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.batch_jobs: Dict[str, Dict] = {}
    
    async def submit_batch_job(
        self, 
        texts: List[str],
        custom_id_prefix: str = "embed"
    ) -> str:
        """Submit batch embedding job."""
        
        # Create batch requests
        requests = []
        for i, text in enumerate(texts):
            requests.append({
                "custom_id": f"{custom_id_prefix}_{i}",
                "method": "POST", 
                "url": "/v1/embeddings",
                "body": {
                    "model": "text-embedding-3-small",
                    "input": text,
                    "dimensions": 1536
                }
            })
        
        # Upload batch file
        batch_file = await self.client.files.create(
            file=("\n".join(json.dumps(req) for req in requests)).encode(),
            purpose="batch"
        )
        
        # Create batch job
        batch_job = await self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h"
        )
        
        self.batch_jobs[batch_job.id] = {
            "status": "submitted",
            "text_count": len(texts),
            "submitted_at": datetime.utcnow()
        }
        
        return batch_job.id
    
    async def get_batch_results(self, batch_id: str) -> List[List[float]]:
        """Get batch job results."""
        batch = await self.client.batches.retrieve(batch_id)
        
        if batch.status == "completed":
            # Download results
            result_file = await self.client.files.content(
                batch.output_file_id
            )
            
            results = []
            for line in result_file.text.strip().split('\n'):
                result = json.loads(line)
                if result.get("response", {}).get("status_code") == 200:
                    embedding = result["response"]["body"]["data"][0]["embedding"]
                    results.append(embedding)
            
            return results
        
        return []
    
    async def wait_for_completion(
        self, 
        batch_id: str, 
        max_wait: int = 3600
    ) -> List[List[float]]:
        """Wait for batch completion with exponential backoff."""
        wait_time = 10
        total_waited = 0
        
        while total_waited < max_wait:
            batch = await self.client.batches.retrieve(batch_id)
            
            if batch.status == "completed":
                return await self.get_batch_results(batch_id)
            elif batch.status == "failed":
                raise Exception(f"Batch job failed: {batch.errors}")
            
            await asyncio.sleep(wait_time)
            total_waited += wait_time
            wait_time = min(wait_time * 1.5, 300)  # Max 5 min
        
        raise TimeoutError(f"Batch job didn't complete in {max_wait}s")
```

#### Parallel Embedding Generation

```python
# src/services/parallel_embedding_service.py
import asyncio
from typing import List, Callable
from src.utils.concurrency import ConcurrencyLimiter

class ParallelEmbeddingService:
    """Parallel embedding generation with rate limiting."""
    
    def __init__(self, max_concurrent: int = 10):
        self.limiter = ConcurrencyLimiter(max_concurrent)
    
    async def generate_parallel_embeddings(
        self,
        text_batches: List[List[str]],
        embedding_func: Callable
    ) -> List[List[float]]:
        """Generate embeddings in parallel with rate limiting."""
        
        async def process_batch(batch: List[str]) -> List[List[float]]:
            async with self.limiter:
                return await embedding_func(batch)
        
        # Process all batches concurrently
        tasks = [process_batch(batch) for batch in text_batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_embeddings = []
        for result in batch_results:
            if isinstance(result, Exception):
                raise result
            all_embeddings.extend(result)
        
        return all_embeddings
    
    async def adaptive_batch_processing(
        self,
        texts: List[str],
        embedding_func: Callable,
        target_latency: float = 1.0
    ) -> List[List[float]]:
        """Adaptive batching based on performance feedback."""
        
        if len(texts) <= 10:
            return await embedding_func(texts)
        
        # Start with small batch to measure performance
        test_batch_size = min(10, len(texts))
        start_time = time.time()
        
        test_embeddings = await embedding_func(texts[:test_batch_size])
        test_latency = time.time() - start_time
        
        # Calculate optimal batch size
        per_item_latency = test_latency / test_batch_size
        optimal_batch_size = max(
            1, 
            int(target_latency / per_item_latency)
        )
        
        # Process remaining texts with optimal batch size
        remaining_texts = texts[test_batch_size:]
        batches = [
            remaining_texts[i:i + optimal_batch_size]
            for i in range(0, len(remaining_texts), optimal_batch_size)
        ]
        
        remaining_embeddings = await self.generate_parallel_embeddings(
            batches, embedding_func
        )
        
        return test_embeddings + remaining_embeddings
```

### 2. Vector Database Optimization

#### Connection Pooling for Qdrant

```python
# src/infrastructure/qdrant_pool.py
from qdrant_client import QdrantClient
from typing import Optional, Dict, Any
import asyncio
from dataclasses import dataclass
import time

@dataclass
class PooledConnection:
    """Pooled Qdrant connection."""
    client: QdrantClient
    created_at: float
    last_used: float
    in_use: bool = False

class QdrantConnectionPool:
    """Connection pool for Qdrant clients."""
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        min_connections: int = 2,
        max_connections: int = 10,
        max_idle_time: float = 300.0
    ):
        self.url = url
        self.api_key = api_key
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        
        self.pool: List[PooledConnection] = []
        self.pool_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize connection pool."""
        if self._initialized:
            return
        
        async with self.pool_lock:
            if not self._initialized:
                # Create minimum connections
                for _ in range(self.min_connections):
                    conn = await self._create_connection()
                    self.pool.append(conn)
                
                self._initialized = True
                
                # Start cleanup task
                asyncio.create_task(self._cleanup_idle_connections())
    
    async def _create_connection(self) -> PooledConnection:
        """Create new Qdrant connection."""
        client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30.0,
            prefer_grpc=True  # Use gRPC for better performance
        )
        
        now = time.time()
        return PooledConnection(
            client=client,
            created_at=now,
            last_used=now
        )
    
    async def get_connection(self) -> QdrantClient:
        """Get connection from pool."""
        await self.initialize()
        
        async with self.pool_lock:
            # Find available connection
            for conn in self.pool:
                if not conn.in_use:
                    conn.in_use = True
                    conn.last_used = time.time()
                    return conn.client
            
            # Create new connection if under limit
            if len(self.pool) < self.max_connections:
                conn = await self._create_connection()
                conn.in_use = True
                self.pool.append(conn)
                return conn.client
            
            # Wait for available connection
            while True:
                await asyncio.sleep(0.1)
                for conn in self.pool:
                    if not conn.in_use:
                        conn.in_use = True
                        conn.last_used = time.time()
                        return conn.client
    
    async def return_connection(self, client: QdrantClient):
        """Return connection to pool."""
        async with self.pool_lock:
            for conn in self.pool:
                if conn.client == client:
                    conn.in_use = False
                    conn.last_used = time.time()
                    break
    
    async def _cleanup_idle_connections(self):
        """Remove idle connections periodically."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            async with self.pool_lock:
                now = time.time()
                active_connections = []
                
                for conn in self.pool:
                    if (
                        not conn.in_use and 
                        now - conn.last_used > self.max_idle_time and
                        len(active_connections) >= self.min_connections
                    ):
                        # Close idle connection
                        try:
                            conn.client.close()
                        except:
                            pass
                    else:
                        active_connections.append(conn)
                
                self.pool = active_connections
```

#### Optimized Bulk Operations

```python
# src/services/optimized_vector_service.py
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import asyncio

class OptimizedVectorService:
    """Optimized vector operations for Qdrant."""
    
    def __init__(self, connection_pool: QdrantConnectionPool):
        self.pool = connection_pool
    
    async def bulk_upsert_optimized(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]],
        batch_size: int = 1000,
        parallel_batches: int = 4
    ) -> bool:
        """Optimized bulk upsert with parallel processing."""
        
        # Split into batches
        batches = [
            vectors[i:i + batch_size]
            for i in range(0, len(vectors), batch_size)
        ]
        
        # Process batches in parallel
        async def process_batch(batch: List[Dict[str, Any]]) -> bool:
            client = await self.pool.get_connection()
            try:
                points = [
                    models.PointStruct(
                        id=item["id"],
                        vector=item["vector"],
                        payload=item.get("payload", {})
                    )
                    for item in batch
                ]
                
                await client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=False  # Don't wait for indexing
                )
                
                return True
            finally:
                await self.pool.return_connection(client)
        
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(parallel_batches)
        
        async def bounded_process(batch):
            async with semaphore:
                return await process_batch(batch)
        
        tasks = [bounded_process(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for errors
        for result in results:
            if isinstance(result, Exception):
                raise result
        
        return True
    
    async def optimized_search(
        self,
        collection_name: str,
        query_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        limit: int = 10,
        ef_search: int = 128  # HNSW parameter for quality/speed
    ) -> List[Dict[str, Any]]:
        """Optimized hybrid search with HNSW tuning."""
        
        client = await self.pool.get_connection()
        try:
            if sparse_vector:
                # Hybrid search with Query API
                query = models.Query(
                    fusion=models.Fusion.RRF,
                    prefetch=[
                        models.Prefetch(
                            query=query_vector,
                            using="dense",
                            limit=limit * 2
                        ),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=list(sparse_vector.keys()),
                                values=list(sparse_vector.values())
                            ),
                            using="sparse",
                            limit=limit * 2
                        )
                    ]
                )
                
                search_params = models.SearchParams(
                    hnsw_ef=ef_search  # Optimize HNSW search
                )
                
                results = await client.query_points(
                    collection_name=collection_name,
                    query=query,
                    limit=limit,
                    search_params=search_params,
                    with_payload=True,
                    with_vectors=False
                )
            else:
                # Dense-only search
                results = await client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    search_params=models.SearchParams(hnsw_ef=ef_search),
                    with_payload=True,
                    with_vectors=False
                )
            
            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                for point in results
            ]
            
        finally:
            await self.pool.return_connection(client)
```

### 3. Intelligent Caching System

#### Multi-Layer Caching

```python
# src/services/intelligent_cache.py
import hashlib
import asyncio
import time
from typing import Any, Optional, Dict, Union
from dataclasses import dataclass
import redis.asyncio as redis
import pickle

@dataclass
class CacheConfig:
    """Cache configuration."""
    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 3600
    embedding_ttl: int = 86400  # 24 hours
    search_ttl: int = 1800      # 30 minutes
    max_memory_items: int = 1000

class IntelligentCache:
    """Multi-layer cache with Redis and in-memory."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "redis_hits": 0
        }
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
        except Exception:
            # Fall back to memory-only cache
            self.redis_client = None
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, (str, int, float)):
            key_data = str(data)
        else:
            key_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        hash_obj = hashlib.md5(key_data.encode() if isinstance(key_data, str) else key_data)
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback layers."""
        
        # Check memory cache first
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if item["expires"] > time.time():
                self.cache_stats["hits"] += 1
                self.cache_stats["memory_hits"] += 1
                return item["value"]
            else:
                del self.memory_cache[key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    value = pickle.loads(data)
                    
                    # Store in memory for faster access
                    if len(self.memory_cache) < self.config.max_memory_items:
                        self.memory_cache[key] = {
                            "value": value,
                            "expires": time.time() + 300  # 5 min in memory
                        }
                    
                    self.cache_stats["hits"] += 1
                    self.cache_stats["redis_hits"] += 1
                    return value
            except Exception:
                pass
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ):
        """Set in cache with TTL."""
        ttl = ttl or self.config.default_ttl
        
        # Store in memory cache
        if len(self.memory_cache) < self.config.max_memory_items:
            self.memory_cache[key] = {
                "value": value,
                "expires": time.time() + min(ttl, 300)  # Max 5 min in memory
            }
        
        # Store in Redis
        if self.redis_client:
            try:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                await self.redis_client.setex(key, ttl, data)
            except Exception:
                pass
    
    async def cache_embeddings(
        self, 
        texts: List[str], 
        embeddings: List[List[float]]
    ):
        """Cache embeddings with content-based keys."""
        for text, embedding in zip(texts, embeddings):
            key = self._generate_key("embedding", text)
            await self.set(key, embedding, self.config.embedding_ttl)
    
    async def get_cached_embeddings(
        self, 
        texts: List[str]
    ) -> Dict[str, List[float]]:
        """Get cached embeddings."""
        cached = {}
        
        for text in texts:
            key = self._generate_key("embedding", text)
            embedding = await self.get(key)
            if embedding:
                cached[text] = embedding
        
        return cached
    
    async def cache_search_results(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ):
        """Cache search results."""
        cache_data = {
            "query_vector": query_vector,
            "filters": filters or {}
        }
        key = self._generate_key("search", cache_data)
        await self.set(key, results, self.config.search_ttl)
    
    async def get_cached_search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        cache_data = {
            "query_vector": query_vector,
            "filters": filters or {}
        }
        key = self._generate_key("search", cache_data)
        return await self.get(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests 
            if total_requests > 0 else 0
        )
        
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "memory_size": len(self.memory_cache)
        }
```

### 4. Advanced Search Optimization

#### Smart Reranking Pipeline

```python
# src/services/optimized_reranking.py
from sentence_transformers import CrossEncoder
import torch
from typing import List, Dict, Any, Optional
import asyncio

class OptimizedReranker:
    """Optimized reranking with model caching and batching."""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.max_batch_size = max_batch_size
        self.model: Optional[CrossEncoder] = None
        self._model_lock = asyncio.Lock()
    
    def _get_device(self, device: str) -> str:
        """Determine optimal device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def _load_model(self):
        """Load reranking model lazily."""
        if self.model is None:
            async with self._model_lock:
                if self.model is None:
                    # Load in thread to avoid blocking
                    def load_model():
                        return CrossEncoder(
                            self.model_name,
                            device=self.device,
                            max_length=512
                        )
                    
                    loop = asyncio.get_event_loop()
                    self.model = await loop.run_in_executor(None, load_model)
    
    async def rerank_batch(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents efficiently."""
        if not documents:
            return []
        
        await self._load_model()
        
        # Prepare pairs for reranking
        pairs = [(query, doc) for doc in documents]
        
        # Process in batches
        all_rerank_scores = []
        for i in range(0, len(pairs), self.max_batch_size):
            batch = pairs[i:i + self.max_batch_size]
            
            # Run reranking in thread
            def rerank_batch():
                return self.model.predict(batch, convert_to_numpy=True)
            
            loop = asyncio.get_event_loop()
            batch_scores = await loop.run_in_executor(None, rerank_batch)
            all_rerank_scores.extend(batch_scores.tolist())
        
        # Combine with original scores
        results = []
        for i, (doc, rerank_score) in enumerate(zip(documents, all_rerank_scores)):
            original_score = scores[i] if scores else 0.0
            
            results.append({
                "document": doc,
                "original_score": original_score,
                "rerank_score": float(rerank_score),
                "combined_score": float(rerank_score) * 0.7 + original_score * 0.3,
                "index": i
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return results
    
    async def adaptive_reranking(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        top_k: int = 10,
        rerank_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Adaptive reranking based on score distribution."""
        
        if len(search_results) <= top_k:
            # No need to rerank if few results
            return search_results
        
        # Check score distribution
        scores = [result.get("score", 0.0) for result in search_results]
        score_variance = np.var(scores) if len(scores) > 1 else 0
        
        if score_variance < rerank_threshold:
            # High variance means clear ranking, skip reranking
            return search_results[:top_k]
        
        # Extract documents for reranking
        documents = [
            result.get("payload", {}).get("text", "")
            for result in search_results
        ]
        
        # Rerank top candidates
        rerank_count = min(len(documents), top_k * 3)
        rerank_docs = documents[:rerank_count]
        rerank_scores = scores[:rerank_count]
        
        reranked = await self.rerank_batch(query, rerank_docs, rerank_scores)
        
        # Update original results with rerank scores
        for i, rerank_result in enumerate(reranked[:top_k]):
            original_idx = rerank_result["index"]
            search_results[original_idx]["rerank_score"] = rerank_result["rerank_score"]
            search_results[original_idx]["combined_score"] = rerank_result["combined_score"]
        
        # Sort by combined score
        return sorted(
            search_results[:rerank_count], 
            key=lambda x: x.get("combined_score", x.get("score", 0.0)),
            reverse=True
        )[:top_k]
```

### 5. Memory and Resource Optimization

#### Memory-Efficient Processing

```python
# src/utils/memory_optimization.py
import gc
import psutil
import asyncio
from typing import Iterator, List, Any, Callable
from contextlib import asynccontextmanager

class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }
    
    @staticmethod
    @asynccontextmanager
    async def memory_monitor(threshold_mb: float = 1000.0):
        """Monitor memory usage and trigger GC if needed."""
        initial_memory = MemoryOptimizer.get_memory_usage()
        
        try:
            yield
        finally:
            current_memory = MemoryOptimizer.get_memory_usage()
            memory_increase = current_memory["rss"] - initial_memory["rss"]
            
            if memory_increase > threshold_mb:
                gc.collect()
                
                # Force garbage collection for PyTorch if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    
    @staticmethod
    def chunked_processing(
        items: List[Any], 
        chunk_size: int,
        max_memory_mb: float = 500.0
    ) -> Iterator[List[Any]]:
        """Process items in memory-aware chunks."""
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            
            # Check memory before processing
            memory = MemoryOptimizer.get_memory_usage()
            if memory["rss"] > max_memory_mb:
                gc.collect()
            
            yield chunk

class StreamingProcessor:
    """Process large datasets with streaming."""
    
    def __init__(self, batch_size: int = 100, max_memory_mb: float = 500.0):
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
    
    async def process_stream(
        self,
        items: Iterator[Any],
        processor: Callable[[List[Any]], Any],
        callback: Optional[Callable[[Any], None]] = None
    ) -> List[Any]:
        """Process stream with memory management."""
        
        results = []
        batch = []
        
        for item in items:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                async with MemoryOptimizer.memory_monitor(self.max_memory_mb):
                    result = await processor(batch)
                    results.append(result)
                    
                    if callback:
                        callback(result)
                
                batch = []
        
        # Process remaining items
        if batch:
            async with MemoryOptimizer.memory_monitor(self.max_memory_mb):
                result = await processor(batch)
                results.append(result)
                
                if callback:
                    callback(result)
        
        return results
```

## Performance Monitoring

### Real-Time Metrics Collection

```python
# src/monitoring/performance_monitor.py
import time
import asyncio
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

@dataclass
class PerformanceMetric:
    """Performance metric data."""
    operation: str
    duration: float
    memory_delta: float
    timestamp: float
    metadata: Dict[str, Any]

class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.alert_thresholds: Dict[str, float] = {}
        self.alert_callbacks: List[Callable] = []
    
    def set_alert_threshold(self, operation: str, threshold_ms: float):
        """Set performance alert threshold."""
        self.alert_thresholds[operation] = threshold_ms
    
    def add_alert_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    async def track_operation(
        self,
        operation: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Track operation performance."""
        
        start_time = time.time()
        start_memory = MemoryOptimizer.get_memory_usage()["rss"]
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = MemoryOptimizer.get_memory_usage()["rss"]
            
            duration = (end_time - start_time) * 1000  # ms
            memory_delta = end_memory - start_memory
            
            metric = PerformanceMetric(
                operation=operation,
                duration=duration,
                memory_delta=memory_delta,
                timestamp=end_time,
                metadata={"args_count": len(args)}
            )
            
            self.metrics.append(metric)
            self.operation_stats[operation].append(duration)
            
            # Check alerts
            if (
                operation in self.alert_thresholds and 
                duration > self.alert_thresholds[operation]
            ):
                for callback in self.alert_callbacks:
                    callback(metric)
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for operation."""
        durations = self.operation_stats.get(operation, [])
        
        if not durations:
            return {}
        
        return {
            "count": len(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "p95": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations),
            "p99": statistics.quantiles(durations, n=100)[98] if len(durations) > 100 else max(durations),
            "min": min(durations),
            "max": max(durations)
        }
    
    def get_recent_performance(self, minutes: int = 5) -> Dict[str, Any]:
        """Get recent performance summary."""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            metric for metric in self.metrics 
            if metric.timestamp > cutoff_time
        ]
        
        operations = {}
        for metric in recent_metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration)
        
        summary = {}
        for operation, durations in operations.items():
            summary[operation] = {
                "count": len(durations),
                "avg_duration": statistics.mean(durations),
                "total_time": sum(durations)
            }
        
        return {
            "time_window_minutes": minutes,
            "total_operations": len(recent_metrics),
            "operations": summary
        }
```

## Official Documentation References

### Performance Optimization

- **Qdrant Performance**: <https://qdrant.tech/documentation/guides/optimization/>
- **OpenAI Batch API**: <https://platform.openai.com/docs/guides/batch>
- **FastEmbed Performance**: <https://qdrant.github.io/fastembed/benchmarks/>
- **Python asyncio**: <https://docs.python.org/3/library/asyncio-task.html>

### Caching & Memory

- **Redis Performance**: <https://redis.io/docs/manual/optimization/>
- **Python Memory Profiling**: <https://docs.python.org/3/library/tracemalloc.html>
- **PyTorch Memory Management**: <https://pytorch.org/docs/stable/notes/cuda.html#memory-management>

### Monitoring

- **Prometheus Python**: <https://prometheus.io/docs/prometheus/latest/configuration/configuration/>
- **psutil Documentation**: <https://psutil.readthedocs.io/>
- **Performance Best Practices**: <https://docs.python.org/3/howto/perf_profiling.html>

## Success Criteria

### Performance Targets (2025)

- [ ] <20ms embedding generation per chunk
- [ ] <10ms search latency at scale
- [ ] 10,000+ documents/hour throughput
- [ ] 100+ concurrent users supported
- [ ] <1GB memory for 1M vectors

### Resource Efficiency

- [ ] 50% reduction in API costs
- [ ] 80%+ cache hit rate for common operations
- [ ] 90% reduction in memory allocation churn
- [ ] Linear scalability with document count

### Monitoring & Observability

- [ ] Real-time performance dashboards
- [ ] Automated alerting for degradation
- [ ] Comprehensive operation metrics
- [ ] Memory usage optimization

These optimizations will deliver production-scale performance while maintaining accuracy and reliability.
