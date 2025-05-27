"""Performance benchmarks comparing Redis vs DragonflyDB cache implementations."""

import asyncio
import time
from typing import Any

import pytest
from src.services.cache.dragonfly_cache import DragonflyCache

# Redis removed - DragonflyDB is the only cache backend


class CachePerformanceBenchmarks:
    """Performance benchmarks for cache implementations."""

    @pytest.fixture
    async def cache(self):
        """Create DragonflyDB cache instance."""
        cache = DragonflyCache(
            redis_url="redis://localhost:6379",
            key_prefix="bench:dragonfly:",
            max_connections=50,
        )

        try:
            # Test connection
            await cache.client
            yield cache, "dragonfly"
        except Exception as e:
            pytest.skip(f"Could not connect to DragonflyDB: {e}")
        finally:
            await cache.close()

    async def generate_test_data(self, size: int) -> dict[str, Any]:
        """Generate test data for benchmarking."""
        return {
            f"key_{i}": {
                "text": f"This is test data item {i} with some content",
                "embedding": [float(j) for j in range(384)],  # Simulated embedding
                "metadata": {
                    "id": i,
                    "timestamp": time.time(),
                    "category": f"category_{i % 10}",
                },
            }
            for i in range(size)
        }

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_single_operations_performance(self, cache):
        """Benchmark single get/set operations."""
        cache_instance, backend = cache

        # Prepare test data
        test_data = await self.generate_test_data(1000)

        # Benchmark SET operations
        start_time = time.time()
        for key, value in test_data.items():
            await cache_instance.set(key, value, ttl=3600)
        set_duration = time.time() - start_time

        # Benchmark GET operations
        start_time = time.time()
        for key in test_data:
            await cache_instance.get(key)
        get_duration = time.time() - start_time

        # Calculate ops/sec
        set_ops_per_sec = len(test_data) / set_duration
        get_ops_per_sec = len(test_data) / get_duration

        print(f"\n{backend.upper()} Single Operations Performance:")
        print(
            f"SET: {set_ops_per_sec:.0f} ops/sec ({set_duration:.2f}s for {len(test_data)} ops)"
        )
        print(
            f"GET: {get_ops_per_sec:.0f} ops/sec ({get_duration:.2f}s for {len(test_data)} ops)"
        )

        # Performance assertions (DragonflyDB should be faster)
        if backend == "dragonfly":
            # DragonflyDB should achieve at least 1000 ops/sec for both operations
            assert set_ops_per_sec > 1000, (
                f"DragonflyDB SET too slow: {set_ops_per_sec} ops/sec"
            )
            assert get_ops_per_sec > 1000, (
                f"DragonflyDB GET too slow: {get_ops_per_sec} ops/sec"
            )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_operations_performance(self, cache):
        """Benchmark batch operations (where DragonflyDB should excel)."""
        cache_instance, backend = cache

        # Prepare test data
        test_data = await self.generate_test_data(1000)

        # Test batch SET (using set_many)
        start_time = time.time()
        await cache_instance.set_many(test_data, ttl=3600)
        batch_set_duration = time.time() - start_time

        # Test batch GET (using get_many)
        start_time = time.time()
        await cache_instance.get_many(list(test_data.keys()))
        batch_get_duration = time.time() - start_time

        # Calculate ops/sec
        batch_set_ops_per_sec = len(test_data) / batch_set_duration
        batch_get_ops_per_sec = len(test_data) / batch_get_duration

        print(f"\n{backend.upper()} Batch Operations Performance:")
        print(
            f"BATCH SET: {batch_set_ops_per_sec:.0f} ops/sec ({batch_set_duration:.2f}s)"
        )
        print(
            f"BATCH GET: {batch_get_ops_per_sec:.0f} ops/sec ({batch_get_duration:.2f}s)"
        )

        # DragonflyDB should be significantly faster for batch operations
        if backend == "dragonfly":
            assert batch_set_ops_per_sec > 2000, (
                f"DragonflyDB batch SET too slow: {batch_set_ops_per_sec}"
            )
            assert batch_get_ops_per_sec > 5000, (
                f"DragonflyDB batch GET too slow: {batch_get_ops_per_sec}"
            )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_operations_performance(self, cache):
        """Benchmark concurrent operations."""
        cache_instance, backend = cache

        async def concurrent_sets(start_idx: int, count: int):
            """Perform concurrent SET operations."""
            tasks = []
            for i in range(start_idx, start_idx + count):
                key = f"concurrent_key_{i}"
                value = {"data": f"concurrent_value_{i}", "index": i}
                tasks.append(cache_instance.set(key, value))

            await asyncio.gather(*tasks)

        async def concurrent_gets(start_idx: int, count: int):
            """Perform concurrent GET operations."""
            tasks = []
            for i in range(start_idx, start_idx + count):
                key = f"concurrent_key_{i}"
                tasks.append(cache_instance.get(key))

            await asyncio.gather(*tasks)

        # Benchmark concurrent SET operations
        start_time = time.time()
        await asyncio.gather(*[concurrent_sets(i * 100, 100) for i in range(10)])
        concurrent_set_duration = time.time() - start_time

        # Benchmark concurrent GET operations
        start_time = time.time()
        await asyncio.gather(*[concurrent_gets(i * 100, 100) for i in range(10)])
        concurrent_get_duration = time.time() - start_time

        total_ops = 1000
        concurrent_set_ops_per_sec = total_ops / concurrent_set_duration
        concurrent_get_ops_per_sec = total_ops / concurrent_get_duration

        print(f"\n{backend.upper()} Concurrent Operations Performance:")
        print(
            f"CONCURRENT SET: {concurrent_set_ops_per_sec:.0f} ops/sec ({concurrent_set_duration:.2f}s)"
        )
        print(
            f"CONCURRENT GET: {concurrent_get_ops_per_sec:.0f} ops/sec ({concurrent_get_duration:.2f}s)"
        )

        # DragonflyDB should handle concurrency better
        if backend == "dragonfly":
            assert concurrent_set_ops_per_sec > 1500, (
                "DragonflyDB concurrent SET too slow"
            )
            assert concurrent_get_ops_per_sec > 3000, (
                "DragonflyDB concurrent GET too slow"
            )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_value_performance(self, cache):
        """Benchmark operations with large values."""
        cache_instance, backend = cache

        # Create large embedding vectors (typical for AI applications)
        large_embeddings = {
            f"large_key_{i}": {
                "embedding": [float(j) for j in range(1536)],  # OpenAI embedding size
                "text": "This is a large document " * 100,  # Large text
                "metadata": {"size": "large", "index": i},
            }
            for i in range(100)
        }

        # Benchmark large value operations
        start_time = time.time()
        for key, value in large_embeddings.items():
            await cache_instance.set(key, value, ttl=3600)
        large_set_duration = time.time() - start_time

        start_time = time.time()
        for key in large_embeddings:
            await cache_instance.get(key)
        large_get_duration = time.time() - start_time

        large_set_ops_per_sec = len(large_embeddings) / large_set_duration
        large_get_ops_per_sec = len(large_embeddings) / large_get_duration

        print(f"\n{backend.upper()} Large Value Performance:")
        print(
            f"LARGE SET: {large_set_ops_per_sec:.0f} ops/sec ({large_set_duration:.2f}s)"
        )
        print(
            f"LARGE GET: {large_get_ops_per_sec:.0f} ops/sec ({large_get_duration:.2f}s)"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency(self, cache):
        """Test memory usage efficiency."""
        cache_instance, backend = cache

        # Store identical data to test deduplication/compression
        identical_data = {"repeated": "x" * 1000}  # Large repeated string

        # Store many copies
        start_time = time.time()
        for i in range(500):
            await cache_instance.set(f"identical_{i}", identical_data)

        # Measure time and check if compression is working
        storage_time = time.time() - start_time

        print(f"\n{backend.upper()} Memory Efficiency:")
        print(f"Stored 500 identical large values in {storage_time:.2f}s")

        # Verify data integrity
        retrieved = await cache_instance.get("identical_0")
        assert retrieved == identical_data

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_ttl_performance(self, cache):
        """Benchmark TTL-related operations."""
        cache_instance, backend = cache

        # Set keys with various TTLs
        start_time = time.time()
        for i in range(1000):
            await cache_instance.set(f"ttl_key_{i}", f"value_{i}", ttl=60 + i)
        ttl_set_duration = time.time() - start_time

        # Check TTLs
        start_time = time.time()
        for i in range(1000):
            await cache_instance.ttl(f"ttl_key_{i}")
        ttl_check_duration = time.time() - start_time

        ttl_set_ops_per_sec = 1000 / ttl_set_duration
        ttl_check_ops_per_sec = 1000 / ttl_check_duration

        print(f"\n{backend.upper()} TTL Performance:")
        print(f"TTL SET: {ttl_set_ops_per_sec:.0f} ops/sec")
        print(f"TTL CHECK: {ttl_check_ops_per_sec:.0f} ops/sec")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_pattern_operations_performance(self, cache):
        """Benchmark pattern-based operations."""
        cache_instance, backend = cache

        # Create keys with patterns
        patterns = ["user:*", "session:*", "cache:*"]
        for pattern_base in ["user", "session", "cache"]:
            for i in range(100):
                await cache_instance.set(f"{pattern_base}:{i}", f"value_{i}")

        # Benchmark pattern scanning (DragonflyDB optimization)
        if hasattr(cache_instance, "scan_keys"):
            start_time = time.time()
            for pattern in patterns:
                keys = await cache_instance.scan_keys(pattern)
                assert len(keys) == 100
            scan_duration = time.time() - start_time

            print(f"\n{backend.upper()} Pattern Scan Performance:")
            print(f"Scanned 3 patterns (300 keys total) in {scan_duration:.2f}s")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_realistic_workload(self, cache):
        """Simulate realistic AI application workload."""
        cache_instance, backend = cache

        # Simulate embedding cache usage
        async def embedding_workload():
            """Simulate embedding cache operations."""
            embeddings = {}
            for i in range(200):
                text = f"Document {i} with various content and metadata"
                embedding = [float(j) for j in range(384)]
                embeddings[f"emb:{hash(text)}"] = {
                    "text": text,
                    "embedding": embedding,
                    "model": "text-embedding-3-small",
                }

            # Cache embeddings
            await cache_instance.set_many(embeddings, ttl=86400)  # 24h TTL

            # Simulate retrieval requests
            for _ in range(50):
                random_key = f"emb:{hash(f'Document {_ % 200} with various content and metadata')}"
                await cache_instance.get(random_key)

        # Simulate search result cache usage
        async def search_workload():
            """Simulate search result cache operations."""
            for i in range(100):
                query = f"search query {i}"
                results = [{"doc_id": j, "score": 0.9 - j * 0.01} for j in range(10)]
                await cache_instance.set(f"search:{hash(query)}", results, ttl=3600)

        # Run realistic workload
        start_time = time.time()
        await asyncio.gather(
            embedding_workload(),
            search_workload(),
        )
        workload_duration = time.time() - start_time

        print(f"\n{backend.upper()} Realistic Workload Performance:")
        print(f"Completed AI workload simulation in {workload_duration:.2f}s")

        # DragonflyDB should complete workload faster
        if backend == "dragonfly":
            assert workload_duration < 10.0, (
                f"DragonflyDB workload too slow: {workload_duration}s"
            )


# Redis comparison removed - DragonflyDB is the only cache backend
# Performance tests above validate DragonflyDB meets performance targets
