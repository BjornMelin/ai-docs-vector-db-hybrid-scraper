# DragonflyDB Cache Implementation Guide

**GitHub Issue**: [#59](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/59)

## Overview

DragonflyDB is a modern Redis-compatible in-memory data store that offers 4.5x better throughput and 38% less memory usage than Redis. It's designed for modern cloud workloads with better vertical scaling and lower latency.

## Why DragonflyDB?

### Performance Comparison

| Metric | Redis | DragonflyDB | Improvement |
|--------|--------|-------------|-------------|
| Throughput (ops/sec) | 200K | 900K | 4.5x |
| Memory efficiency | Baseline | 38% less | 1.6x |
| P99 latency | 2.5ms | 0.8ms | 3.1x |
| Vertical scaling | 16 cores | 64+ cores | 4x |
| Snapshot time (10GB) | 30s | <1s | 30x |

### Key Advantages

1. **Drop-in Redis replacement** - No code changes needed
2. **Better multi-core utilization** - Scales vertically
3. **Lower memory footprint** - Advanced data structures
4. **Faster persistence** - Non-blocking snapshots
5. **Cloud-native** - Designed for Kubernetes

## Implementation Plan

### 1. Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Keep existing Qdrant
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
  
  # Add DragonflyDB
  dragonfly:
    image: docker.dragonflydb.io/dragonflydb/dragonfly:latest
    ports:
      - "6379:6379"
    volumes:
      - ./data/dragonfly:/data
    environment:
      - DRAGONFLY_THREADS=8
      - DRAGONFLY_MEMORY_LIMIT=4gb
      - DRAGONFLY_SNAPSHOT_INTERVAL=3600
      - DRAGONFLY_SAVE_SCHEDULE="0 */1 * * *"
    command: >
      --logtostderr
      --cache_mode
      --maxmemory_policy=allkeys-lru
      --compression=zstd
    ulimits:
      memlock:
        soft: -1
        hard: -1
    restart: unless-stopped
```

### 2. Cache Service Implementation

```python
# src/services/cache/dragonfly_cache.py
import asyncio
import json
import hashlib
from typing import Any, Optional
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from .base import BaseCache
from ..logging_config import get_logger

logger = get_logger(__name__)

class DragonflyCache(BaseCache):
    """High-performance cache using DragonflyDB."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
        # Connection settings
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        
        # Performance settings
        self.max_connections = config.get("max_connections", 50)
        self.socket_keepalive = config.get("socket_keepalive", True)
        self.socket_connect_timeout = config.get("socket_connect_timeout", 5)
        
        # Retry configuration
        self.retry = Retry(
            retries=3,
            backoff=ExponentialBackoff(base=0.1, cap=1.0),
            supported_errors=(ConnectionError, TimeoutError),
        )
        
        # Connection pool
        self.pool = None
        self.client = None
    
    async def initialize(self):
        """Initialize DragonflyDB connection pool."""
        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=self.max_connections,
            socket_keepalive=self.socket_keepalive,
            socket_connect_timeout=self.socket_connect_timeout,
            retry=self.retry,
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
        
        # Test connection
        await self.client.ping()
        logger.info("DragonflyDB cache initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic deserialization."""
        try:
            value = await self.client.get(key)
            if value:
                # Try JSON deserialization
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Return as string if not JSON
                    return value.decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in cache with automatic serialization."""
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, (str, bytes)):
                value = str(value)
            
            # Set with options
            return await self.client.set(
                key,
                value,
                ex=ttl,
                nx=nx,  # Only set if not exists
                xx=xx,  # Only set if exists
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            return bool(await self.client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(await self.client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def mget(self, keys: list[str]) -> list[Optional[Any]]:
        """Get multiple values efficiently."""
        try:
            values = await self.client.mget(keys)
            results = []
            
            for value in values:
                if value:
                    try:
                        results.append(json.loads(value))
                    except json.JSONDecodeError:
                        results.append(value.decode('utf-8'))
                else:
                    results.append(None)
            
            return results
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            return [None] * len(keys)
    
    async def mset(self, mapping: dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values efficiently."""
        try:
            # Serialize values
            serialized = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized[key] = json.dumps(value)
                elif not isinstance(value, (str, bytes)):
                    serialized[key] = str(value)
                else:
                    serialized[key] = value
            
            # Use pipeline for atomic operation
            async with self.client.pipeline() as pipe:
                pipe.mset(serialized)
                
                # Set TTL if provided
                if ttl:
                    for key in serialized:
                        pipe.expire(key, ttl)
                
                await pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            return False
    
    async def cleanup(self):
        """Close connections."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
```

### 3. Advanced Caching Patterns

```python
# src/services/cache/patterns.py
from typing import Any, Callable, Optional
import asyncio
import hashlib

class CachePatterns:
    """Advanced caching patterns for DragonflyDB."""
    
    def __init__(self, cache: DragonflyCache):
        self.cache = cache
    
    async def cache_aside(
        self,
        key: str,
        fetch_func: Callable,
        ttl: int = 3600,
        stale_while_revalidate: int = 60,
    ) -> Any:
        """Cache-aside pattern with stale-while-revalidate."""
        
        # Try to get from cache
        cached = await self.cache.get(key)
        
        if cached is not None:
            # Check if stale
            ttl_remaining = await self.cache.client.ttl(key)
            
            if ttl_remaining < stale_while_revalidate:
                # Return stale data and refresh in background
                asyncio.create_task(self._refresh_cache(key, fetch_func, ttl))
            
            return cached
        
        # Fetch and cache
        return await self._refresh_cache(key, fetch_func, ttl)
    
    async def _refresh_cache(self, key: str, fetch_func: Callable, ttl: int) -> Any:
        """Refresh cache with distributed lock."""
        
        lock_key = f"lock:{key}"
        lock_acquired = await self.cache.set(lock_key, "1", ttl=10, nx=True)
        
        if lock_acquired:
            try:
                # Fetch fresh data
                data = await fetch_func()
                
                # Cache it
                await self.cache.set(key, data, ttl=ttl)
                
                return data
            finally:
                # Release lock
                await self.cache.delete(lock_key)
        else:
            # Wait for other process to populate cache
            for _ in range(10):
                await asyncio.sleep(0.5)
                cached = await self.cache.get(key)
                if cached is not None:
                    return cached
            
            # Fallback to fetching
            return await fetch_func()
    
    async def batch_cache(
        self,
        keys: list[str],
        fetch_func: Callable[[list[str]], dict[str, Any]],
        ttl: int = 3600,
    ) -> dict[str, Any]:
        """Efficient batch caching."""
        
        # Get existing values
        cached_values = await self.cache.mget(keys)
        
        # Find missing keys
        results = {}
        missing_keys = []
        
        for key, value in zip(keys, cached_values):
            if value is not None:
                results[key] = value
            else:
                missing_keys.append(key)
        
        # Fetch missing values
        if missing_keys:
            fresh_data = await fetch_func(missing_keys)
            
            # Cache fresh data
            if fresh_data:
                await self.cache.mset(fresh_data, ttl=ttl)
                results.update(fresh_data)
        
        return results
    
    async def cached_computation(
        self,
        func: Callable,
        *args,
        cache_key: Optional[str] = None,
        ttl: int = 3600,
        **kwargs,
    ) -> Any:
        """Cache expensive computation results."""
        
        # Generate cache key from function and arguments
        if not cache_key:
            key_data = f"{func.__name__}:{args}:{kwargs}"
            cache_key = f"compute:{hashlib.md5(key_data.encode()).hexdigest()}"
        
        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Compute and cache
        result = await func(*args, **kwargs)
        await self.cache.set(cache_key, result, ttl=ttl)
        
        return result
```

### 4. Embedding Cache Layer

```python
# src/services/cache/embedding_cache.py
import numpy as np
from typing import Optional

class EmbeddingCache:
    """Specialized cache for embeddings using DragonflyDB."""
    
    def __init__(self, cache: DragonflyCache):
        self.cache = cache
        self.ttl = 86400 * 7  # 7 days for embeddings
    
    async def get_embedding(self, text: str, model: str) -> Optional[list[float]]:
        """Get cached embedding."""
        key = self._get_key(text, model)
        
        cached = await self.cache.get(key)
        if cached:
            # Convert back to list of floats
            return [float(x) for x in cached]
        
        return None
    
    async def set_embedding(self, text: str, model: str, embedding: list[float]):
        """Cache embedding vector."""
        key = self._get_key(text, model)
        
        # Store as compact format
        await self.cache.set(key, embedding, ttl=self.ttl)
    
    async def get_batch_embeddings(
        self,
        texts: list[str],
        model: str
    ) -> tuple[dict[str, list[float]], list[str]]:
        """Get batch embeddings, return cached and missing."""
        
        keys = [self._get_key(text, model) for text in texts]
        cached_values = await self.cache.mget(keys)
        
        cached = {}
        missing = []
        
        for text, value in zip(texts, cached_values):
            if value is not None:
                cached[text] = [float(x) for x in value]
            else:
                missing.append(text)
        
        return cached, missing
    
    async def set_batch_embeddings(
        self,
        embeddings: dict[str, list[float]],
        model: str
    ):
        """Cache multiple embeddings efficiently."""
        
        mapping = {
            self._get_key(text, model): embedding
            for text, embedding in embeddings.items()
        }
        
        await self.cache.mset(mapping, ttl=self.ttl)
    
    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key for embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"emb:{model}:{text_hash}"
    
    async def warm_cache(self, common_queries: list[str], model: str):
        """Pre-warm cache with common queries."""
        missing = []
        
        for query in common_queries:
            if not await self.cache.exists(self._get_key(query, model)):
                missing.append(query)
        
        if missing:
            logger.info(f"Warming cache with {len(missing)} embeddings")
            # Generate embeddings for missing queries
            # This would call the embedding service
```

### 5. Search Result Cache

```python
# src/services/cache/search_cache.py
class SearchResultCache:
    """Cache search results with intelligent invalidation."""
    
    def __init__(self, cache: DragonflyCache):
        self.cache = cache
        self.ttl = 3600  # 1 hour for search results
    
    async def get_search_results(
        self,
        query: str,
        filters: dict,
        limit: int
    ) -> Optional[list[dict]]:
        """Get cached search results."""
        
        key = self._get_search_key(query, filters, limit)
        return await self.cache.get(key)
    
    async def set_search_results(
        self,
        query: str,
        filters: dict,
        limit: int,
        results: list[dict]
    ):
        """Cache search results."""
        
        key = self._get_search_key(query, filters, limit)
        
        # Store with shorter TTL for popular queries
        popularity = await self._get_query_popularity(query)
        ttl = self.ttl // 2 if popularity > 10 else self.ttl
        
        await self.cache.set(key, results, ttl=ttl)
        
        # Track query popularity
        await self._increment_query_popularity(query)
    
    async def invalidate_by_collection(self, collection_name: str):
        """Invalidate all cached searches for a collection."""
        
        pattern = f"search:{collection_name}:*"
        
        # Use SCAN for efficient pattern matching
        cursor = 0
        while True:
            cursor, keys = await self.cache.client.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                await self.cache.client.delete(*keys)
            
            if cursor == 0:
                break
    
    def _get_search_key(self, query: str, filters: dict, limit: int) -> str:
        """Generate deterministic cache key."""
        
        # Sort filters for consistency
        sorted_filters = json.dumps(filters, sort_keys=True)
        
        key_data = f"{query}:{sorted_filters}:{limit}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"search:results:{key_hash}"
    
    async def _get_query_popularity(self, query: str) -> int:
        """Get query popularity for cache strategy."""
        
        key = f"popular:{hashlib.md5(query.encode()).hexdigest()}"
        count = await self.cache.get(key)
        return int(count) if count else 0
    
    async def _increment_query_popularity(self, query: str):
        """Track query popularity."""
        
        key = f"popular:{hashlib.md5(query.encode()).hexdigest()}"
        await self.cache.client.incr(key)
        await self.cache.client.expire(key, 86400)  # Reset daily
```

### 6. Integration with Services

```python
# Update EmbeddingManager to use DragonflyDB
class EmbeddingManager:
    def __init__(self, config: dict[str, Any]):
        # Initialize DragonflyDB cache
        self.cache = EmbeddingCache(
            DragonflyCache(config.get("cache", {}))
        )
    
    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small"
    ) -> list[list[float]]:
        """Generate embeddings with caching."""
        
        # Check cache first
        cached, missing = await self.cache.get_batch_embeddings(texts, model)
        
        if not missing:
            return [cached[text] for text in texts]
        
        # Generate missing embeddings
        new_embeddings = await self._generate_uncached(missing, model)
        
        # Cache new embeddings
        embedding_map = dict(zip(missing, new_embeddings))
        await self.cache.set_batch_embeddings(embedding_map, model)
        
        # Combine results
        results = []
        for text in texts:
            if text in cached:
                results.append(cached[text])
            else:
                results.append(embedding_map[text])
        
        return results
```

## Performance Optimization

### 1. Pipeline Operations

```python
async def bulk_operations_example():
    """Use pipelines for atomic bulk operations."""
    
    async with cache.client.pipeline() as pipe:
        # Multiple operations in single round trip
        pipe.set("key1", "value1")
        pipe.set("key2", "value2")
        pipe.expire("key1", 3600)
        pipe.expire("key2", 3600)
        
        results = await pipe.execute()
```

### 2. Lua Scripts

```python
# Custom Lua script for atomic operations
RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])

local current = redis.call('incr', key)
if current == 1 then
    redis.call('expire', key, window)
end

if current > limit then
    return 0
else
    return 1
end
"""

async def check_rate_limit(user_id: str, limit: int = 100, window: int = 3600):
    """Atomic rate limiting with Lua script."""
    
    key = f"rate:{user_id}"
    allowed = await cache.client.eval(
        RATE_LIMIT_SCRIPT,
        keys=[key],
        args=[limit, window]
    )
    return bool(allowed)
```

## Migration Strategy

1. **Deploy DragonflyDB alongside Redis**
2. **Implement dual-write pattern**
3. **Gradually migrate reads to DragonflyDB**
4. **Monitor performance metrics**
5. **Complete cutover and remove Redis**

## Monitoring

```python
class CacheMonitor:
    """Monitor DragonflyDB performance."""
    
    async def collect_metrics(self) -> dict:
        """Collect cache metrics."""
        
        info = await self.cache.client.info()
        
        return {
            "memory_used": info["used_memory"],
            "memory_peak": info["used_memory_peak"],
            "connected_clients": info["connected_clients"],
            "total_commands": info["total_commands_processed"],
            "cache_hit_ratio": info.get("cache_hit_ratio", 0),
            "evicted_keys": info.get("evicted_keys", 0),
            "ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
        }
```

## Testing

```python
@pytest.mark.asyncio
async def test_dragonfly_performance():
    """Test DragonflyDB performance vs Redis."""
    
    # Benchmark set operations
    start = time.time()
    for i in range(10000):
        await dragonfly_cache.set(f"key{i}", f"value{i}")
    dragonfly_time = time.time() - start
    
    # Assert DragonflyDB is faster
    assert dragonfly_time < redis_time * 0.7  # At least 30% faster
```

## Expected Performance Gains

| Operation | Redis | DragonflyDB | Improvement |
|-----------|--------|-------------|-------------|
| Embedding cache hit | 2.5ms | 0.8ms | 3.1x |
| Batch get (100 keys) | 15ms | 3ms | 5x |
| Search result cache | 5ms | 1.5ms | 3.3x |
| Cache warming (10K) | 45s | 8s | 5.6x |
