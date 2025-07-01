"""Intelligent caching system with TTL, LRU eviction, and multi-level storage.

This module provides smart caching strategies with performance optimization,
intelligent cache warming, and memory usage optimization for ML operations.
"""

import asyncio
import functools
import gzip
import hashlib
import json
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheConfig:
    """Configuration for intelligent caching system."""
    
    max_memory_mb: int = 256
    max_items: int = 10000
    default_ttl_seconds: int = 3600
    enable_persistence: bool = True
    persistence_path: str = ".cache"
    enable_compression: bool = True
    enable_cache_warming: bool = True
    warming_batch_size: int = 100
    memory_pressure_threshold: float = 0.8
    eviction_batch_size: int = 50


@dataclass
class CacheEntry:
    """Entry in the intelligent cache."""
    
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: int = 3600
    size_bytes: int = 0
    is_compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() > (self.created_at + self.ttl_seconds)
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    item_count: int = 0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    compression_ratio: float = 0.0
    avg_access_time_ms: float = 0.0
    cache_warming_success_rate: float = 0.0


class IntelligentCache(Generic[K, V]):
    """Multi-level intelligent cache with TTL, LRU eviction, and compression."""
    
    def __init__(self, config: CacheConfig | None = None):
        """Initialize intelligent cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._memory_cache: OrderedDict[K, CacheEntry] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._stats = CacheStats()
        self._background_tasks: set[asyncio.Task] = set()
        
        # Setup persistence if enabled
        if self.config.enable_persistence:
            self._persistence_path = Path(self.config.persistence_path)
            self._persistence_path.mkdir(exist_ok=True)
    
    async def get(
        self, 
        key: K, 
        default: V | None = None,
        update_stats: bool = True,
    ) -> V | None:
        """Get value from cache with intelligent access tracking.
        
        Args:
            key: Cache key
            default: Default value if not found
            update_stats: Whether to update access statistics
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        async with self._cache_lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self._memory_cache[key]
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    return default
                
                # Update access statistics
                if update_stats:
                    entry.update_access()
                    # Move to end (LRU)
                    self._memory_cache.move_to_end(key)
                
                self._stats.hits += 1
                
                # Decompress if needed
                value = self._decompress_value(entry.value, entry.is_compressed)
                
                # Update access time statistics
                access_time_ms = (time.time() - start_time) * 1000
                self._update_access_time_stats(access_time_ms)
                
                return value
            
            # Check persistent cache if enabled
            if self.config.enable_persistence:
                persisted_value = await self._get_from_persistence(key)
                if persisted_value is not None:
                    # Add back to memory cache
                    await self.set(key, persisted_value)
                    self._stats.hits += 1
                    return persisted_value
        
        self._stats.misses += 1
        return default
    
    async def set(
        self, 
        key: K, 
        value: V, 
        ttl_seconds: int | None = None,
        compress: bool | None = None,
    ) -> None:
        """Set value in cache with intelligent storage optimization.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            compress: Whether to compress the value
        """
        ttl = ttl_seconds or self.config.default_ttl_seconds
        should_compress = compress if compress is not None else self.config.enable_compression
        
        async with self._cache_lock:
            # Compress value if enabled
            stored_value, is_compressed = self._compress_value(value, should_compress)
            
            # Calculate size
            size_bytes = self._calculate_size(stored_value)
            
            # Check memory pressure and evict if necessary
            await self._check_memory_pressure(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                value=stored_value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl_seconds=ttl,
                size_bytes=size_bytes,
                is_compressed=is_compressed,
            )
            
            # Add to memory cache
            self._memory_cache[key] = entry
            self._memory_cache.move_to_end(key)  # Mark as most recently used
            
            # Update statistics
            self._stats.size_bytes += size_bytes
            self._stats.item_count += 1
            
            # Persist if enabled
            if self.config.enable_persistence:
                await self._persist_to_disk(key, value)
    
    async def delete(self, key: K) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        async with self._cache_lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                self._stats.size_bytes -= entry.size_bytes
                self._stats.item_count -= 1
                del self._memory_cache[key]
                
                # Remove from persistence
                if self.config.enable_persistence:
                    await self._delete_from_persistence(key)
                
                return True
            
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._cache_lock:
            self._memory_cache.clear()
            self._stats = CacheStats()
            
            if self.config.enable_persistence:
                await self._clear_persistence()
    
    async def warm_cache(
        self, 
        key_value_pairs: list[tuple[K, V]],
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Warm cache with predefined key-value pairs.
        
        Args:
            key_value_pairs: List of (key, value) pairs to warm
            batch_size: Batch size for warming
            
        Returns:
            Warming results statistics
        """
        batch_size = batch_size or self.config.warming_batch_size
        total_pairs = len(key_value_pairs)
        successful_warming = 0
        
        # Process in batches to avoid memory pressure
        for i in range(0, total_pairs, batch_size):
            batch = key_value_pairs[i:i + batch_size]
            
            # Warm batch
            for key, value in batch:
                try:
                    await self.set(key, value)
                    successful_warming += 1
                except Exception as e:
                    logger.warning(f"Cache warming failed for key {key}: {e}")
            
            # Allow other operations to proceed
            await asyncio.sleep(0.001)
        
        # Update warming statistics
        success_rate = successful_warming / max(total_pairs, 1)
        self._stats.cache_warming_success_rate = success_rate
        
        return {
            "total_pairs": total_pairs,
            "successful_warming": successful_warming,
            "success_rate": success_rate,
            "batches_processed": (total_pairs + batch_size - 1) // batch_size,
        }
    
    async def _check_memory_pressure(self, new_item_size: int) -> None:
        """Check memory pressure and evict items if necessary.
        
        Args:
            new_item_size: Size of new item to be added
        """
        current_memory_mb = self._stats.size_bytes / (1024 * 1024)
        new_total_memory = (self._stats.size_bytes + new_item_size) / (1024 * 1024)
        
        # Check if we'll exceed memory limit
        if (new_total_memory > self.config.max_memory_mb or 
            len(self._memory_cache) >= self.config.max_items):
            
            await self._evict_items()
    
    async def _evict_items(self) -> None:
        """Evict items using intelligent LRU + TTL strategy."""
        items_to_evict = min(
            self.config.eviction_batch_size,
            len(self._memory_cache) // 4  # Evict 25% when pressure occurs
        )
        
        if items_to_evict <= 0:
            return
        
        # Sort items by eviction priority (expired first, then LRU)
        eviction_candidates = []
        
        for key, entry in self._memory_cache.items():
            priority_score = self._calculate_eviction_priority(entry)
            eviction_candidates.append((priority_score, key, entry))
        
        # Sort by priority (higher score = higher priority for eviction)
        eviction_candidates.sort(reverse=True)
        
        # Evict items
        for i in range(min(items_to_evict, len(eviction_candidates))):
            _, key, entry = eviction_candidates[i]
            self._stats.size_bytes -= entry.size_bytes
            self._stats.item_count -= 1
            self._stats.evictions += 1
            del self._memory_cache[key]
    
    def _calculate_eviction_priority(self, entry: CacheEntry) -> float:
        """Calculate eviction priority for cache entry.
        
        Args:
            entry: Cache entry to evaluate
            
        Returns:
            Priority score (higher = more likely to evict)
        """
        current_time = time.time()
        
        # High priority for expired items
        if entry.is_expired():
            return 1000.0
        
        # Priority based on last access time (older = higher priority)
        time_since_access = current_time - entry.last_accessed
        access_priority = time_since_access / 3600  # Normalize to hours
        
        # Priority based on access frequency (less accessed = higher priority)
        frequency_priority = 1.0 / max(entry.access_count, 1)
        
        # Priority based on size (larger = higher priority)
        size_priority = entry.size_bytes / (1024 * 1024)  # Normalize to MB
        
        return access_priority + frequency_priority + (size_priority * 0.1)
    
    def _compress_value(self, value: V, should_compress: bool) -> tuple[Any, bool]:
        """Compress value if compression is enabled and beneficial.
        
        Args:
            value: Value to potentially compress
            should_compress: Whether compression should be attempted
            
        Returns:
            Tuple of (processed_value, is_compressed)
        """
        if not should_compress:
            return value, False
        
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Only compress if value is large enough to benefit
            if len(serialized) > 1024:  # 1KB threshold
                compressed = gzip.compress(serialized)
                
                # Only use compression if it actually reduces size
                if len(compressed) < len(serialized) * 0.9:  # 10% reduction minimum
                    return compressed, True
            
            return serialized, False
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value, False
    
    def _decompress_value(self, value: Any, is_compressed: bool) -> V:
        """Decompress value if it was compressed.
        
        Args:
            value: Value to decompress
            is_compressed: Whether value is compressed
            
        Returns:
            Decompressed value
        """
        if not is_compressed:
            if isinstance(value, bytes):
                return pickle.loads(value)
            return value
        
        try:
            decompressed = gzip.decompress(value)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return value
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes.
        
        Args:
            value: Value to calculate size for
            
        Returns:
            Size in bytes
        """
        try:
            if isinstance(value, bytes):
                return len(value)
            else:
                return len(pickle.dumps(value))
        except (IOError, OSError, PermissionError, asyncio.TimeoutError) as e:
            # Fallback estimation
            return len(str(value))
    
    async def _get_from_persistence(self, key: K) -> V | None:
        """Get value from persistent storage.
        
        Args:
            key: Cache key
            
        Returns:
            Persisted value if found
        """
        try:
            key_hash = self._hash_key(key)
            cache_file = self._persistence_path / f"{key_hash}.cache"
            
            if cache_file.exists():
                with cache_file.open('rb') as f:
                    data = pickle.load(f)
                    
                # Check if persisted data is expired
                if time.time() > data['expires_at']:
                    cache_file.unlink()
                    return None
                    
                return data['value']
                
        except Exception as e:
            logger.warning(f"Failed to read from persistence: {e}")
        
        return None
    
    async def _persist_to_disk(self, key: K, value: V) -> None:
        """Persist value to disk.
        
        Args:
            key: Cache key
            value: Value to persist
        """
        try:
            key_hash = self._hash_key(key)
            cache_file = self._persistence_path / f"{key_hash}.cache"
            
            data = {
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + self.config.default_ttl_seconds,
            }
            
            with cache_file.open('wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.warning(f"Failed to persist to disk: {e}")
    
    async def _delete_from_persistence(self, key: K) -> None:
        """Delete key from persistent storage.
        
        Args:
            key: Cache key to delete
        """
        try:
            key_hash = self._hash_key(key)
            cache_file = self._persistence_path / f"{key_hash}.cache"
            
            if cache_file.exists():
                cache_file.unlink()
                
        except Exception as e:
            logger.warning(f"Failed to delete from persistence: {e}")
    
    async def _clear_persistence(self) -> None:
        """Clear all persistent cache files."""
        try:
            for cache_file in self._persistence_path.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear persistence: {e}")
    
    def _hash_key(self, key: K) -> str:
        """Generate hash for cache key.
        
        Args:
            key: Cache key to hash
            
        Returns:
            Hash string
        """
        key_str = json.dumps(key, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _update_access_time_stats(self, access_time_ms: float) -> None:
        """Update average access time statistics.
        
        Args:
            access_time_ms: Access time in milliseconds
        """
        # Exponential moving average
        alpha = 0.1
        if self._stats.avg_access_time_ms == 0:
            self._stats.avg_access_time_ms = access_time_ms
        else:
            self._stats.avg_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self._stats.avg_access_time_ms
            )
    
    def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics.
        
        Returns:
            Current cache statistics
        """
        total_accesses = self._stats.hits + self._stats.misses
        hit_rate = self._stats.hits / max(total_accesses, 1)
        memory_usage_mb = self._stats.size_bytes / (1024 * 1024)
        
        # Update computed stats
        self._stats.hit_rate = hit_rate
        self._stats.memory_usage_mb = memory_usage_mb
        
        return self._stats
    
    def get_memory_usage(self) -> dict[str, Any]:
        """Get detailed memory usage information.
        
        Returns:
            Memory usage breakdown
        """
        total_size = sum(entry.size_bytes for entry in self._memory_cache.values())
        compressed_size = sum(
            entry.size_bytes for entry in self._memory_cache.values() 
            if entry.is_compressed
        )
        
        return {
            "total_memory_mb": total_size / (1024 * 1024),
            "compressed_memory_mb": compressed_size / (1024 * 1024),
            "compression_ratio": compressed_size / max(total_size, 1),
            "item_count": len(self._memory_cache),
            "avg_item_size_kb": total_size / max(len(self._memory_cache), 1) / 1024,
            "memory_limit_mb": self.config.max_memory_mb,
            "memory_utilization": (total_size / (1024 * 1024)) / self.config.max_memory_mb,
        }


class EmbeddingCache(IntelligentCache[str, list[float]]):
    """Specialized cache for embedding vectors with optimized storage."""
    
    def __init__(self, config: CacheConfig | None = None):
        """Initialize embedding cache with optimized settings."""
        if config is None:
            config = CacheConfig(
                max_memory_mb=512,  # Higher memory for embeddings
                enable_compression=True,  # Embeddings compress well
                default_ttl_seconds=7200,  # 2 hours default
                enable_cache_warming=True,
            )
        
        super().__init__(config)
    
    async def get_embedding(
        self,
        text: str,
        provider: str,
        model: str,
        dimensions: int,
    ) -> list[float] | None:
        """Get embedding from cache with provider/model context.
        
        Args:
            text: Text that was embedded
            provider: Embedding provider
            model: Model name
            dimensions: Expected dimensions
            
        Returns:
            Cached embedding if found
        """
        cache_key = f"{provider}:{model}:{dimensions}:{hashlib.sha256(text.encode()).hexdigest()}"
        return await self.get(cache_key)
    
    async def set_embedding(
        self,
        text: str,
        provider: str,
        model: str,
        dimensions: int,
        embedding: list[float],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache embedding with provider/model context.
        
        Args:
            text: Text that was embedded
            provider: Embedding provider
            model: Model name
            dimensions: Embedding dimensions
            embedding: Embedding vector
            ttl_seconds: Time to live
        """
        cache_key = f"{provider}:{model}:{dimensions}:{hashlib.sha256(text.encode()).hexdigest()}"
        await self.set(cache_key, embedding, ttl_seconds)


class SearchResultCache(IntelligentCache[str, dict[str, Any]]):
    """Specialized cache for search results with query optimization."""
    
    def __init__(self, config: CacheConfig | None = None):
        """Initialize search result cache."""
        if config is None:
            config = CacheConfig(
                max_memory_mb=128,
                default_ttl_seconds=1800,  # 30 minutes
                enable_compression=True,
                enable_cache_warming=False,  # Search results are dynamic
            )
        
        super().__init__(config)
    
    async def get_search_results(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any] | None:
        """Get cached search results.
        
        Args:
            query: Search query
            filters: Search filters
            limit: Result limit
            
        Returns:
            Cached search results
        """
        cache_key = self._generate_search_key(query, filters, limit)
        return await self.get(cache_key)
    
    async def set_search_results(
        self,
        query: str,
        results: dict[str, Any],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache search results.
        
        Args:
            query: Search query
            results: Search results
            filters: Search filters
            limit: Result limit
            ttl_seconds: Time to live
        """
        cache_key = self._generate_search_key(query, filters, limit)
        await self.set(cache_key, results, ttl_seconds)
    
    def _generate_search_key(
        self,
        query: str,
        filters: dict[str, Any] | None,
        limit: int,
    ) -> str:
        """Generate cache key for search query.
        
        Args:
            query: Search query
            filters: Search filters
            limit: Result limit
            
        Returns:
            Cache key
        """
        key_data = {
            "query": query.lower().strip(),
            "filters": filters or {},
            "limit": limit,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


# Decorator for caching function results
def cached_with_ttl(ttl_seconds: int = 3600, cache_size: int = 1000):
    """Decorator for caching function results with TTL.
    
    Args:
        ttl_seconds: Time to live for cached results
        cache_size: Maximum cache size
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        cache: dict[str, tuple[Any, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Check cache
            current_time = time.time()
            if key in cache:
                value, cached_time = cache[key]
                if current_time - cached_time < ttl_seconds:
                    return value
                else:
                    del cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result (with size limit)
            if len(cache) >= cache_size:
                # Remove oldest entry
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[key] = (result, current_time)
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "max_size": cache_size,
            "ttl_seconds": ttl_seconds,
        }
        
        return wrapper
    
    return decorator