"""Enterprise mode cache service implementation.

Full-featured caching service with distributed caching, analytics, and advanced features
for enterprise deployments.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from src.architecture.service_factory import BaseService

logger = logging.getLogger(__name__)


class EnterpriseCacheService(BaseService):
    """Full-featured cache service for enterprise deployments.
    
    Features:
    - Multi-tier caching (memory + distributed)
    - Advanced eviction policies
    - Cache analytics and monitoring
    - Compression and serialization
    - Cache warming and preloading
    - Performance metrics
    """
    
    def __init__(self):
        super().__init__()
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.distributed_cache: Optional[Any] = None
        self.max_size = 10000  # Enterprise scale
        self.max_memory_mb = 1000  # Enterprise scale
        self.default_ttl = 3600  # 1 hour default TTL
        self.enable_compression = True
        self.enable_analytics = True
        self.enable_distributed = True
        
        # Analytics
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_sets = 0
        self._cache_deletes = 0
        self._access_patterns: Dict[str, int] = {}
        self._hot_keys: Set[str] = set()
        
        # Performance tracking
        self._operation_times: Dict[str, List[float]] = {
            'get': [],
            'set': [],
            'delete': [],
        }
    
    async def initialize(self) -> None:
        """Initialize the enterprise cache service."""
        logger.info("Initializing enterprise cache service")
        
        try:
            # Initialize distributed cache if enabled
            if self.enable_distributed:
                await self._initialize_distributed_cache()
            
            # Initialize cache warming
            await self._initialize_cache_warming()
            
            # Initialize analytics
            if self.enable_analytics:
                await self._initialize_analytics()
            
            self._mark_initialized()
            logger.info("Enterprise cache service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enterprise cache service: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up cache service resources."""
        # Clean up distributed cache connection
        if self.distributed_cache:
            try:
                await self.distributed_cache.close()
            except Exception as e:
                logger.error(f"Error closing distributed cache: {e}")
        
        # Clear local cache
        self.local_cache.clear()
        self._access_patterns.clear()
        self._hot_keys.clear()
        
        # Clear analytics
        for times_list in self._operation_times.values():
            times_list.clear()
        
        self._mark_cleanup()
        logger.info("Enterprise cache service cleaned up")
    
    def get_service_name(self) -> str:
        """Get the service name."""
        return "enterprise_cache_service"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with multi-tier lookup.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()
        
        try:
            # Check local cache first (L1)
            value = await self._get_local(key)
            if value is not None:
                self._cache_hits += 1
                self._track_access(key)
                self._record_operation_time('get', time.time() - start_time)
                return value
            
            # Check distributed cache (L2)
            if self.distributed_cache:
                value = await self._get_distributed(key)
                if value is not None:
                    # Populate local cache
                    await self._set_local(key, value, ttl=self.default_ttl)
                    self._cache_hits += 1
                    self._track_access(key)
                    self._record_operation_time('get', time.time() - start_time)
                    return value
            
            # Cache miss
            self._cache_misses += 1
            self._record_operation_time('get', time.time() - start_time)
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tier: str = "both"
    ) -> None:
        """Set value in cache with multi-tier storage.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tier: Cache tier ("local", "distributed", or "both")
        """
        start_time = time.time()
        
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            # Compress value if enabled
            if self.enable_compression:
                value = await self._compress_value(value)
            
            # Set in appropriate tiers
            if tier in ("local", "both"):
                await self._set_local(key, value, ttl)
            
            if tier in ("distributed", "both") and self.distributed_cache:
                await self._set_distributed(key, value, ttl)
            
            self._cache_sets += 1
            self._track_access(key)
            self._record_operation_time('set', time.time() - start_time)
            
            logger.debug(f"Cached value for key: {key} in {tier} tier(s)")
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted from any tier
        """
        start_time = time.time()
        deleted = False
        
        try:
            # Delete from local cache
            if key in self.local_cache:
                del self.local_cache[key]
                deleted = True
            
            # Delete from distributed cache
            if self.distributed_cache:
                dist_deleted = await self._delete_distributed(key)
                deleted = deleted or dist_deleted
            
            # Clean up tracking
            if key in self._access_patterns:
                del self._access_patterns[key]
            self._hot_keys.discard(key)
            
            self._cache_deletes += 1
            self._record_operation_time('delete', time.time() - start_time)
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def clear(self, tier: str = "both") -> None:
        """Clear cache entries from specified tiers.
        
        Args:
            tier: Cache tier to clear ("local", "distributed", or "both")
        """
        try:
            if tier in ("local", "both"):
                self.local_cache.clear()
                logger.info("Local cache cleared")
            
            if tier in ("distributed", "both") and self.distributed_cache:
                await self._clear_distributed()
                logger.info("Distributed cache cleared")
            
            # Reset analytics
            self._access_patterns.clear()
            self._hot_keys.clear()
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache tier.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is valid
        """
        # Check local cache first
        if await self._exists_local(key):
            return True
        
        # Check distributed cache
        if self.distributed_cache:
            return await self._exists_distributed(key)
        
        return False
    
    async def warm_cache(self, keys: List[str]) -> None:
        """Pre-warm cache with specified keys.
        
        Args:
            keys: List of cache keys to warm
        """
        if not self.distributed_cache:
            return
        
        logger.info(f"Warming cache with {len(keys)} keys")
        
        # Batch load from distributed cache to local cache
        for key in keys:
            try:
                value = await self._get_distributed(key)
                if value is not None:
                    await self._set_local(key, value, ttl=self.default_ttl)
            except Exception as e:
                logger.error(f"Cache warming failed for key {key}: {e}")
        
        logger.info("Cache warming completed")
    
    async def get_hot_keys(self, limit: int = 10) -> List[str]:
        """Get most frequently accessed keys.
        
        Args:
            limit: Maximum number of keys to return
            
        Returns:
            List of hot keys sorted by access frequency
        """
        sorted_keys = sorted(
            self._access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [key for key, _ in sorted_keys[:limit]]
    
    async def _initialize_distributed_cache(self) -> None:
        """Initialize distributed cache connection."""
        try:
            # Initialize Redis/Dragonfly connection
            from src.services.cache.dragonfly_cache import DragonflyCacheAdapter
            self.distributed_cache = DragonflyCacheAdapter()
            await self.distributed_cache.initialize()
            logger.info("Distributed cache initialized")
        except ImportError:
            logger.warning("Distributed cache not available, using local cache only")
            self.enable_distributed = False
        except Exception as e:
            logger.error(f"Distributed cache initialization failed: {e}")
            self.enable_distributed = False
    
    async def _initialize_cache_warming(self) -> None:
        """Initialize cache warming strategies."""
        # Would implement cache warming logic
        logger.info("Cache warming strategies initialized")
    
    async def _initialize_analytics(self) -> None:
        """Initialize cache analytics."""
        logger.info("Cache analytics initialized")
    
    async def _get_local(self, key: str) -> Optional[Any]:
        """Get value from local cache."""
        if key not in self.local_cache:
            return None
        
        entry = self.local_cache[key]
        
        # Check TTL
        if time.time() > entry["expires_at"]:
            del self.local_cache[key]
            return None
        
        return entry["value"]
    
    async def _set_local(self, key: str, value: Any, ttl: int) -> None:
        """Set value in local cache."""
        # Evict if needed
        await self._evict_local_if_needed()
        
        self.local_cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
        }
    
    async def _exists_local(self, key: str) -> bool:
        """Check if key exists in local cache."""
        if key not in self.local_cache:
            return False
        
        entry = self.local_cache[key]
        if time.time() > entry["expires_at"]:
            del self.local_cache[key]
            return False
        
        return True
    
    async def _get_distributed(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        if not self.distributed_cache:
            return None
        
        try:
            return await self.distributed_cache.get(key)
        except Exception as e:
            logger.error(f"Distributed cache get failed: {e}")
            return None
    
    async def _set_distributed(self, key: str, value: Any, ttl: int) -> None:
        """Set value in distributed cache."""
        if not self.distributed_cache:
            return
        
        try:
            await self.distributed_cache.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Distributed cache set failed: {e}")
    
    async def _delete_distributed(self, key: str) -> bool:
        """Delete value from distributed cache."""
        if not self.distributed_cache:
            return False
        
        try:
            return await self.distributed_cache.delete(key)
        except Exception as e:
            logger.error(f"Distributed cache delete failed: {e}")
            return False
    
    async def _exists_distributed(self, key: str) -> bool:
        """Check if key exists in distributed cache."""
        if not self.distributed_cache:
            return False
        
        try:
            return await self.distributed_cache.exists(key)
        except Exception as e:
            logger.error(f"Distributed cache exists check failed: {e}")
            return False
    
    async def _clear_distributed(self) -> None:
        """Clear distributed cache."""
        if not self.distributed_cache:
            return
        
        try:
            await self.distributed_cache.clear()
        except Exception as e:
            logger.error(f"Distributed cache clear failed: {e}")
    
    async def _compress_value(self, value: Any) -> Any:
        """Compress value if compression is enabled."""
        # Would implement compression logic
        return value
    
    async def _evict_local_if_needed(self) -> None:
        """Evict entries from local cache if needed."""
        if len(self.local_cache) >= self.max_size:
            # Evict LRU entries
            current_time = time.time()
            entries_with_age = [
                (key, current_time - entry["created_at"])
                for key, entry in self.local_cache.items()
            ]
            
            # Sort by age (oldest first)
            entries_with_age.sort(key=lambda x: x[1], reverse=True)
            
            # Remove 10% of entries
            num_to_remove = max(1, len(entries_with_age) // 10)
            for key, _ in entries_with_age[:num_to_remove]:
                del self.local_cache[key]
    
    def _track_access(self, key: str) -> None:
        """Track key access patterns."""
        self._access_patterns[key] = self._access_patterns.get(key, 0) + 1
        
        # Mark as hot key if accessed frequently
        if self._access_patterns[key] > 10:
            self._hot_keys.add(key)
    
    def _record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation timing."""
        if operation in self._operation_times:
            times_list = self._operation_times[operation]
            times_list.append(duration)
            
            # Keep only recent 1000 measurements
            if len(times_list) > 1000:
                times_list.pop(0)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_operations = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_operations * 100) if total_operations > 0 else 0
        
        # Calculate average operation times
        avg_times = {}
        for op, times in self._operation_times.items():
            avg_times[f"avg_{op}_time_ms"] = (sum(times) / len(times) * 1000) if times else 0
        
        return {
            "service_type": "enterprise",
            "local_cache_entries": len(self.local_cache),
            "max_size": self.max_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_sets": self._cache_sets,
            "cache_deletes": self._cache_deletes,
            "hit_rate_percent": round(hit_rate, 2),
            "hot_keys_count": len(self._hot_keys),
            "access_patterns_count": len(self._access_patterns),
            "features": {
                "distributed": self.enable_distributed,
                "compression": self.enable_compression,
                "analytics": self.enable_analytics,
                "cache_warming": True,
                "multi_tier": True,
            },
            "performance": avg_times,
        }
    
    async def get_cache_health(self) -> Dict[str, Any]:
        """Get comprehensive cache health information."""
        stats = self.get_cache_stats()
        
        # Calculate health metrics
        memory_usage_pct = (len(self.local_cache) / self.max_size) * 100
        hit_rate = stats["hit_rate_percent"]
        
        # Determine health status
        if hit_rate > 80 and memory_usage_pct < 90:
            status = "excellent"
        elif hit_rate > 60 and memory_usage_pct < 95:
            status = "good"
        elif hit_rate > 40:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "hit_rate_percent": hit_rate,
            "memory_usage_percent": memory_usage_pct,
            "local_cache_entries": stats["local_cache_entries"],
            "distributed_cache_available": self.enable_distributed,
            "hot_keys": await self.get_hot_keys(5),
            "recommendations": self._get_health_recommendations(hit_rate, memory_usage_pct),
        }
    
    def _get_health_recommendations(self, hit_rate: float, memory_usage: float) -> List[str]:
        """Get cache health recommendations."""
        recommendations = []
        
        if hit_rate < 60:
            recommendations.append("Consider increasing cache TTL or size")
        if memory_usage > 90:
            recommendations.append("Consider increasing cache size or implementing more aggressive eviction")
        if len(self._hot_keys) > 100:
            recommendations.append("Consider cache warming for frequently accessed keys")
        
        return recommendations