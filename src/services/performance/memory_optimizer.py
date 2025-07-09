"""Memory Optimization Module.

This module provides memory optimization strategies including
object pooling, garbage collection tuning, and memory profiling.
"""

import asyncio
import contextlib
import gc
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from weakref import WeakValueDictionary

import psutil


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    available_mb: float
    percent_used: float
    gc_stats: dict[str, int]


class ObjectPool:
    """Generic object pool for reusing expensive objects."""

    def __init__(self, factory: Any, max_size: int = 100):
        """Initialize object pool.

        Args:
            factory: Callable to create new objects
            max_size: Maximum pool size

        """
        self.factory = factory
        self.max_size = max_size
        self._pool: list[Any] = []
        self._in_use: WeakValueDictionary = WeakValueDictionary()
        self._created_count = 0
        self._pool_available = asyncio.Event()

    async def acquire(self) -> Any:
        """Acquire an object from the pool."""
        # Try to get from pool
        if self._pool:
            obj = self._pool.pop()
        if not self._pool:
            self._pool_available.clear()
            self._in_use[id(obj)] = obj
            return obj

        # Create new if under limit
        if self._created_count < self.max_size:
            obj = (
                await self.factory()
                if asyncio.iscoroutinefunction(self.factory)
                else self.factory()
            )
            self._created_count += 1
            self._in_use[id(obj)] = obj
            return obj

        # Wait for available object
        if not self._pool:
            await self._pool_available.wait()

        obj = self._pool.pop()
        if not self._pool:
            self._pool_available.clear()
        self._in_use[id(obj)] = obj
        return obj

    def release(self, obj: Any) -> None:
        """Release an object back to the pool."""
        if id(obj) in self._in_use:
            del self._in_use[id(obj)]

        if len(self._pool) < self.max_size:
            # Reset object state if it has a reset method
            if hasattr(obj, "reset"):
                obj.reset()
            self._pool.append(obj)
            self._pool_available.set()
        else:
            # Let it be garbage collected
            self._created_count -= 1

    def clear(self) -> None:
        """Clear the pool."""
        self._pool.clear()
        self._in_use.clear()
        self._created_count = 0
        self._pool_available = asyncio.Event()


class MemoryOptimizer:
    """Central memory optimization coordinator."""

    def __init__(self):
        """Initialize memory optimizer."""
        self._object_pools: dict[str, ObjectPool] = {}
        self._memory_thresholds = {
            "warning": 80,  # Warn at 80% memory usage
            "critical": 90,  # Critical at 90% memory usage
        }
        self._gc_disabled_contexts = 0

    def create_object_pool(
        self, name: str, factory: Any, max_size: int = 100
    ) -> ObjectPool:
        """Create a new object pool.

        Args:
            name: Pool name
            factory: Object factory function
            max_size: Maximum pool size

        Returns:
            Created object pool

        """
        pool = ObjectPool(factory, max_size)
        self._object_pools[name] = pool
        return pool

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics.

        Returns:
            MemoryStats object

        """
        process = psutil.Process()
        memory_info = process.memory_info()

        # System memory
        system_memory = psutil.virtual_memory()

        # GC stats
        gc_stats = {
            f"generation_{i}": gc.get_count()[i]
            for i in range(gc.get_count().__len__())
        }

        return MemoryStats(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            available_mb=system_memory.available / 1024 / 1024,
            percent_used=system_memory.percent,
            gc_stats=gc_stats,
        )

    def optimize_gc_settings(self) -> dict[str, Any]:
        """Optimize garbage collection settings.

        Returns:
            Previous GC settings

        """
        # Get current thresholds
        old_thresholds = gc.get_threshold()

        # Set more aggressive thresholds for memory-constrained environments
        stats = self.get_memory_stats()

        if stats.percent_used > self._memory_thresholds["critical"]:
            # Very aggressive GC
            gc.set_threshold(100, 5, 5)
        elif stats.percent_used > self._memory_thresholds["warning"]:
            # Moderately aggressive GC
            gc.set_threshold(500, 10, 10)
        else:
            # Default settings
            gc.set_threshold(700, 10, 10)

        # Force collection
        gc.collect()

        return {
            "old_thresholds": old_thresholds,
            "new_thresholds": gc.get_threshold(),
            "memory_percent": stats.percent_used,
        }

    def disable_gc_context(self):
        """Context manager to temporarily disable GC."""

        class GCContext:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def __enter__(self):
                self.optimizer._gc_disabled_contexts += 1  # noqa: SLF001
                if self.optimizer._gc_disabled_contexts == 1:  # noqa: SLF001
                    gc.disable()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.optimizer._gc_disabled_contexts -= 1  # noqa: SLF001
                if self.optimizer._gc_disabled_contexts == 0:  # noqa: SLF001
                    gc.enable()
                    gc.collect()

        return GCContext(self)

    def profile_memory_usage(self, top_n: int = 10) -> dict[str, Any]:
        """Profile memory usage by object type.

        Args:
            top_n: Number of top memory consumers to return

        Returns:
            Memory usage profile

        """
        # Count objects by type
        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)

        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            type_counts[obj_type] += 1
            with contextlib.suppress(TypeError):
                type_sizes[obj_type] += sys.getsizeof(obj)
            # Some objects don't support getsizeof

        # Sort by size
        sorted_types = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        return {
            "top_consumers": [
                {
                    "type": type_name,
                    "count": type_counts[type_name],
                    "size_mb": size / 1024 / 1024,
                }
                for type_name, size in sorted_types
            ],
            "total_objects": sum(type_counts.values()),
            "memory_stats": self.get_memory_stats(),
        }

    async def free_memory_if_needed(self, threshold_mb: float = 100) -> bool:
        """Free memory if available memory is below threshold.

        Args:
            threshold_mb: Minimum available memory in MB

        Returns:
            True if memory was freed

        """
        stats = self.get_memory_stats()

        if stats.available_mb < threshold_mb:
            logger.warning(f"Low memory: {stats.available_mb:.1f}MB available")

            # Clear object pools
            for pool in self._object_pools.values():
                pool.clear()

            # Force garbage collection
            gc.collect()

            # Get new stats
            new_stats = self.get_memory_stats()
            freed_mb = new_stats.available_mb - stats.available_mb

            logger.info(f"Freed {freed_mb:.1f}MB of memory")
            return True

        return False
